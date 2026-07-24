"""
Unified memory with semantic recall for Mangaba AI v3.0

A single :class:`Memory` facade over the three memory concerns the framework
already models separately (short-term recency, long-term persistence and
entity tracking).  It is **additive** — ``ShortTermMemory``, ``LongTermMemory``
and ``EntityMemory`` keep working exactly as before and this class implements
the same :class:`~mangaba.memory.base.BaseMemory` contract, so it can be handed
straight to ``Agent(memory=...)``.

What it adds:

* **Composite scoring** — ``score = w_sim * similarity + w_recency * recency +
  w_importance * importance`` (see :class:`MemoryWeights`).
* **Semantic recall** through an injected :class:`~mangaba.embeddings.base.BaseEmbedding`
  (cosine), degrading gracefully to the same lexical keyword scoring the
  existing memory classes use when no embedding provider is available.
* **Automatic fact extraction** from a user/assistant exchange, LLM-driven when
  an ``llm`` client is injected, regex/heuristic otherwise.
* **Consolidation** of near-duplicate entries, LLM-driven when possible.
* **Non-blocking writes** on a background worker thread, with ``flush()``.
* **Hierarchical scope** (``agent`` / ``crew`` / ``global``) for filtered recall.
* **Pluggable storage** via the :class:`StorageBackend` protocol
  (:class:`InMemoryBackend` and :class:`SQLiteBackend` ship in the box).

Zero external services are required: with no embedding provider, no LLM and the
default in-memory backend the class is fully functional.

Example::

    from mangaba.memory import Memory

    memory = Memory(db_path="./.mangaba/memory.db")
    memory.add("The user prefers dark mode", metadata={"importance": 0.9})
    memory.add_interaction("I live in Maceio", "Noted, you live in Maceio.")
    memory.flush()

    print(memory.get_relevant("where does the user live?", max_results=3))
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import queue
import re
import sqlite3
import threading
import time
import uuid
import weakref
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from pydantic import BaseModel, Field, field_validator

from mangaba.memory.base import BaseMemory
from mangaba.memory.entity import EntityMemory

log = logging.getLogger(__name__)


DEFAULT_DB_PATH = "./.mangaba/memory.db"

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Sentences carrying one of these markers are treated as more important by the
# heuristic importance scorer (see ``Memory.infer_importance``).
_IMPORTANCE_MARKERS = (
    "always", "never", "must", "should", "required", "requires", "important",
    "critical", "remember", "prefer", "prefers", "preference", "deadline",
    "policy", "rule", "warning", "do not", "don't",
)

# Verbs that make a sentence look like a stand-alone statement of fact.
_FACT_VERB_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|prefers?|likes?|dislikes?|uses?|works?|lives?|"
    r"needs?|wants?|owns?|costs?|must|should|will|can|runs?|supports?|requires?)\b"
)

# Openers that are conversational filler rather than facts.
_FILLER_PREFIXES = (
    "thanks", "thank you", "ok", "okay", "sure", "hello", "hi ", "hey",
    "sorry", "please", "yes", "no ", "got it", "understood", "of course",
)

_FACT_EXTRACTION_PROMPT = """Extract the discrete, standalone facts contained in the exchange below.

Rules:
- One fact per line, no numbering, no bullet characters.
- Write each fact in the third person ("The user ...", "The system ...").
- Only facts that would still be useful days later. Skip pleasantries and questions.
- If there are no durable facts, output the single word NONE.

User: {user_text}
Assistant: {assistant_text}

Facts:"""

_MERGE_PROMPT = """Two memory entries describe the same thing and may conflict.
Merge them into ONE canonical statement. Prefer the newer entry when they
disagree. Answer with the merged statement only, no preamble.

Older entry: {older}
Newer entry: {newer}

Merged statement:"""


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

class MemoryScope(str, Enum):
    """Hierarchical visibility of a memory entry.

    ``AGENT`` is private to one agent, ``CREW`` is shared by the agents of a
    crew run and ``GLOBAL`` is visible to everything.
    """

    AGENT = "agent"
    CREW = "crew"
    GLOBAL = "global"


ScopeLike = Union[str, MemoryScope]


def _coerce_scope(value: Optional[ScopeLike], default: MemoryScope = MemoryScope.GLOBAL) -> MemoryScope:
    if value is None:
        return default
    if isinstance(value, MemoryScope):
        return value
    try:
        return MemoryScope(str(value).lower())
    except ValueError:
        log.warning("Unknown memory scope %r — falling back to %s", value, default.value)
        return default


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MemoryWeights(BaseModel):
    """Weights of the composite recall score.

    ``score = similarity * w_similarity + recency * w_recency + importance * w_importance``

    All three components are normalised to ``0..1`` before weighting, so the
    weights are directly comparable.  Defaults favour relevance while still
    letting a fresh or explicitly important memory overtake a slightly better
    lexical/semantic match.

    Example::

        Memory(weights=MemoryWeights(similarity=0.5, recency=0.3, importance=0.2))
    """

    similarity: float = Field(default=0.6, ge=0.0)
    recency: float = Field(default=0.25, ge=0.0)
    importance: float = Field(default=0.15, ge=0.0)

    @field_validator("similarity", "recency", "importance")
    @classmethod
    def finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Weights must be finite numbers")
        return v

    def normalized(self) -> "MemoryWeights":
        """Return an equivalent set of weights summing to 1.0."""
        total = self.similarity + self.recency + self.importance
        if total <= 0:
            return MemoryWeights()
        return MemoryWeights(
            similarity=self.similarity / total,
            recency=self.recency / total,
            importance=self.importance / total,
        )


class MemoryEntry(BaseModel):
    """A single stored memory."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    scope: MemoryScope = MemoryScope.GLOBAL
    kind: str = "fact"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_result(self, score: Optional[float] = None, parts: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Shape the entry like the dicts the other memory classes return."""
        out: Dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "scope": self.scope.value,
            "kind": self.kind,
            "importance": self.importance,
            "entities": list(self.entities),
        }
        if score is not None:
            out["score"] = score
        if parts:
            out.update(parts)
        return out


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------

@runtime_checkable
class StorageBackend(Protocol):
    """Durable store for :class:`MemoryEntry` objects.

    Implementations must be safe to call from the memory writer thread.

    Example::

        class RedisBackend:
            def upsert(self, entry): ...
            def delete(self, ids): ...
            def load_all(self): ...
            def clear(self): ...
            def close(self): ...
    """

    def upsert(self, entry: MemoryEntry) -> None:
        """Insert or replace *entry*."""
        ...

    def delete(self, ids: Sequence[str]) -> None:
        """Remove the entries with the given ids."""
        ...

    def load_all(self) -> List[MemoryEntry]:
        """Return every persisted entry."""
        ...

    def clear(self) -> None:
        """Erase everything."""
        ...

    def close(self) -> None:
        """Release resources (no-op for volatile backends)."""
        ...


class InMemoryBackend:
    """Volatile backend — entries live only for the process lifetime.

    Example::

        memory = Memory(storage=InMemoryBackend())
    """

    def __init__(self) -> None:
        self._rows: Dict[str, MemoryEntry] = {}
        self._lock = threading.RLock()

    def upsert(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._rows[entry.id] = entry.model_copy(deep=True)

    def delete(self, ids: Sequence[str]) -> None:
        with self._lock:
            for i in ids:
                self._rows.pop(i, None)

    def load_all(self) -> List[MemoryEntry]:
        with self._lock:
            return [e.model_copy(deep=True) for e in self._rows.values()]

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS unified_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    embedding TEXT DEFAULT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    kind TEXT NOT NULL DEFAULT 'fact',
    importance REAL NOT NULL DEFAULT 0.5,
    entities TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_unified_memories_scope ON unified_memories(scope);"


class SQLiteBackend:
    """SQLite backend, following ``LongTermMemory``'s conventions.

    JSON-encodes ``metadata``/``embedding``/``entities`` into TEXT columns and
    shares one ``check_same_thread=False`` connection guarded by a lock, so the
    writer thread and the caller thread can both use it.

    Example::

        memory = Memory(storage=SQLiteBackend("./.mangaba/memory.db"))
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            parent = Path(db_path).expanduser().parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    def upsert(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO unified_memories "
                "(id, content, metadata, embedding, scope, kind, importance, entities, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.id,
                    entry.content,
                    json.dumps(entry.metadata, default=str),
                    json.dumps(entry.embedding) if entry.embedding else None,
                    entry.scope.value,
                    entry.kind,
                    float(entry.importance),
                    json.dumps(entry.entities),
                    entry.created_at,
                    entry.updated_at,
                ),
            )
            self._conn.commit()

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        with self._lock:
            self._conn.executemany("DELETE FROM unified_memories WHERE id = ?", [(i,) for i in ids])
            self._conn.commit()

    def load_all(self) -> List[MemoryEntry]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, metadata, embedding, scope, kind, importance, entities, created_at, updated_at "
                "FROM unified_memories ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM unified_memories")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover - defensive
                log.debug("SQLiteBackend: connection already closed")

    @staticmethod
    def _row_to_entry(row: tuple) -> MemoryEntry:
        return MemoryEntry(
            id=row[0],
            content=row[1],
            metadata=json.loads(row[2]) if row[2] else {},
            embedding=json.loads(row[3]) if row[3] else None,
            scope=_coerce_scope(row[4]),
            kind=row[5] or "fact",
            importance=float(row[6]),
            entities=json.loads(row[7]) if row[7] else [],
            created_at=row[8],
            updated_at=row[9],
        )


# ---------------------------------------------------------------------------
# Background writer
# ---------------------------------------------------------------------------

class _Writer:
    """Single worker thread draining a queue of ``(op, payload)`` jobs."""

    def __init__(self, backend: StorageBackend, name: str) -> None:
        self._backend = backend
        self._queue: "queue.Queue[Optional[Tuple[str, Any]]]" = queue.Queue()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def submit(self, op: str, payload: Any) -> None:
        if self._stopped.is_set():
            self._apply(op, payload)
            return
        self._queue.put((op, payload))

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until the queue drains. Returns ``False`` on timeout."""
        deadline = timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                if deadline <= 0:
                    return False
                start = time.monotonic()
                self._queue.all_tasks_done.wait(deadline)
                deadline -= time.monotonic() - start
        return True

    def stop(self, timeout: float = 5.0) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._apply(item[0], item[1])
            finally:
                self._queue.task_done()

    def _apply(self, op: str, payload: Any) -> None:
        try:
            if op == "upsert":
                self._backend.upsert(payload)
            elif op == "delete":
                self._backend.delete(payload)
            elif op == "clear":
                self._backend.clear()
            elif op == "call":
                payload()
            else:  # pragma: no cover - defensive
                log.warning("Memory writer: unknown op %r", op)
        except Exception:
            log.exception("Memory writer failed on op %r", op)


# Instances are flushed on interpreter shutdown so queued writes are not lost.
_LIVE_MEMORIES: "weakref.WeakSet[Memory]" = weakref.WeakSet()


def _atexit_flush() -> None:  # pragma: no cover - exercised at interpreter exit
    for mem in list(_LIVE_MEMORIES):
        try:
            mem.close(timeout=2.0)
        except Exception:
            log.debug("Memory atexit flush failed", exc_info=True)


atexit.register(_atexit_flush)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class Memory(BaseMemory):
    """Unified memory with composite semantic recall.

    Args:
        embedding: Optional :class:`~mangaba.embeddings.base.BaseEmbedding`.
            When given, similarity is cosine over embeddings; otherwise the
            lexical keyword score used by ``ShortTermMemory``/``LongTermMemory``
            is used (normalised to ``0..1``).
        llm: Optional LLM client (anything exposing ``generate_text(prompt)``,
            ``generate(prompt).text`` or being callable). Only ever used to
            *improve* fact extraction and consolidation — never required.
        storage: A :class:`StorageBackend`. Defaults to :class:`SQLiteBackend`
            when ``db_path`` is given, otherwise :class:`InMemoryBackend`.
        db_path: Shorthand for ``storage=SQLiteBackend(db_path)``.
        weights: Composite score weights (see :class:`MemoryWeights`).
        half_life_hours: Recency half-life. ``recency = 0.5 ** (age_h / half_life)``.
        default_scope: Scope applied to entries that do not declare one.
        dedup_threshold: Similarity above which ``consolidate()`` treats two
            entries as near-duplicates.
        async_writes: When ``True`` (default) durability happens on a worker
            thread; the in-process index is updated synchronously so recall is
            immediately consistent. Call :meth:`flush` to await durability.
        max_entries: Soft cap; the lowest-scoring oldest entries are evicted
            past this many. ``0`` disables eviction.

    Example::

        memory = Memory(embedding=my_embedder, llm=my_llm, db_path="./.mangaba/memory.db")
        memory.add("Deploys happen on Fridays", metadata={"scope": "crew", "importance": 0.8})
        hits = memory.search("when do we deploy?", top_k=3, scope="crew")
        memory.flush()
    """

    def __init__(
        self,
        embedding: Optional[Any] = None,
        llm: Optional[Any] = None,
        storage: Optional[StorageBackend] = None,
        db_path: Optional[str] = None,
        weights: Optional[MemoryWeights] = None,
        half_life_hours: float = 72.0,
        default_scope: ScopeLike = MemoryScope.GLOBAL,
        dedup_threshold: float = 0.85,
        async_writes: bool = True,
        max_entries: int = 0,
    ) -> None:
        if storage is not None:
            self.storage: StorageBackend = storage
        elif db_path is not None:
            self.storage = SQLiteBackend(db_path)
        else:
            self.storage = InMemoryBackend()

        self.embedding = embedding
        self.llm = llm
        self.weights = (weights or MemoryWeights()).normalized()
        self.half_life_hours = max(1e-6, float(half_life_hours))
        self.default_scope = _coerce_scope(default_scope)
        self.dedup_threshold = float(dedup_threshold)
        self.async_writes = bool(async_writes)
        self.max_entries = int(max_entries)

        self._lock = threading.RLock()
        self._entries: Dict[str, MemoryEntry] = {}
        self._closed = False

        for entry in self.storage.load_all():
            self._entries[entry.id] = entry

        self._writer: Optional[_Writer] = None
        if self.async_writes:
            self._writer = _Writer(self.storage, name=f"mangaba-memory-{uuid.uuid4().hex[:6]}")

        _LIVE_MEMORIES.add(self)

    # ── BaseMemory API ─────────────────────────────────────────────────

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store *content* and return its id.

        Recognised ``metadata`` keys (all optional): ``importance`` (0..1),
        ``scope`` (``agent``/``crew``/``global``), ``kind``, ``entities``
        (list of names) and ``created_at`` (ISO timestamp override, useful for
        backfills and tests). Everything else is preserved verbatim.
        """
        meta = dict(metadata or {})
        text = (content or "").strip()
        if not text:
            return ""

        scope = _coerce_scope(meta.pop("scope", None), self.default_scope)
        kind = str(meta.pop("kind", "fact"))
        created_at = str(meta.pop("created_at", None) or datetime.now().isoformat())
        entities = list(meta.pop("entities", None) or EntityMemory._extract_entities(text))

        raw_importance = meta.pop("importance", None)
        importance = self._coerce_importance(raw_importance, text)

        entry = MemoryEntry(
            content=text,
            metadata=meta,
            scope=scope,
            kind=kind,
            importance=importance,
            entities=entities,
            created_at=created_at,
            updated_at=created_at,
        )

        with self._lock:
            self._entries[entry.id] = entry
            evicted = self._evict_if_needed()

        if self.async_writes and self._writer is not None:
            self._writer.submit("call", lambda e=entry: self._embed_and_persist(e))
        else:
            self._embed_and_persist(entry)

        if evicted:
            self._submit("delete", evicted)

        return entry.id

    def search(
        self,
        query: str,
        top_k: int = 5,
        scope: Optional[Union[ScopeLike, Sequence[ScopeLike]]] = None,
        kind: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return the ``top_k`` entries ranked by the composite score.

        ``scope`` accepts a single scope or a sequence of scopes; ``None``
        searches every scope. Results carry ``score``, ``similarity``,
        ``recency`` and ``importance`` so callers can explain a ranking.
        """
        scopes = self._coerce_scope_filter(scope)
        now = datetime.now()
        query_vec = self._embed(query) if self.embedding is not None else None
        query_tokens = _tokenize(query)

        with self._lock:
            candidates = list(self._entries.values())

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for entry in candidates:
            if scopes is not None and entry.scope not in scopes:
                continue
            if kind is not None and entry.kind != kind:
                continue

            similarity = self._similarity_to_query(entry, query_vec, query_tokens)
            recency = self._recency(entry, now)
            score = (
                self.weights.similarity * similarity
                + self.weights.recency * recency
                + self.weights.importance * entry.importance
            )
            if similarity <= 0.0 and query_tokens:
                # Nothing in common with the query — never surface it.
                continue
            if score < min_score:
                continue
            scored.append((
                score,
                entry.to_result(score, {
                    "similarity": similarity,
                    "recency": recency,
                }),
            ))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def get_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda e: e.created_at, reverse=True)
        return [e.to_result() for e in entries]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        self._submit("clear", None)

    def get_relevant(
        self,
        query: str,
        max_results: int = 5,
        scope: Optional[Union[ScopeLike, Sequence[ScopeLike]]] = None,
    ) -> str:
        """Formatted recall block for prompt injection (same shape as ``BaseMemory``)."""
        results = self.search(query, top_k=max_results, scope=scope)
        if not results:
            return ""
        lines = [f"- {r.get('content', '')}" for r in results]
        return "Relevant memories:\n" + "\n".join(lines)

    # ── interactions & facts ───────────────────────────────────────────

    def add_interaction(
        self,
        user_text: str,
        assistant_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Store an exchange plus the discrete facts extracted from it.

        The verbatim exchange is stored with ``kind="interaction"``; every
        extracted fact is stored separately with ``kind="fact"``. Returns all
        created ids, the interaction id first.
        """
        meta = dict(metadata or {})
        exchange = f"User: {user_text}\nAssistant: {assistant_text}".strip()

        ids: List[str] = []
        interaction_meta = dict(meta)
        interaction_meta["kind"] = "interaction"
        ids.append(self.add(exchange, metadata=interaction_meta))

        for fact in self.extract_facts(user_text, assistant_text):
            fact_meta = dict(meta)
            fact_meta["kind"] = "fact"
            fact_meta.setdefault("source", "add_interaction")
            ids.append(self.add(fact, metadata=fact_meta))

        return [i for i in ids if i]

    def extract_facts(self, user_text: str, assistant_text: str = "") -> List[str]:
        """Pull standalone facts out of an exchange.

        With an ``llm`` injected the model is asked for one fact per line. With
        no LLM (or when the call fails) the heuristic fallback is used:

        1. Split both turns into sentences.
        2. Drop questions, filler openers and sentences shorter than 15 chars.
        3. Keep sentences containing a statement verb (``is``, ``prefers``,
           ``uses``, ``must``, ...).
        4. Rewrite first person from the user turn into the third person
           (``I am`` -> ``The user is``, ``my`` -> ``the user's``).
        5. De-duplicate case-insensitively, preserving order.
        """
        if self.llm is not None:
            facts = self._extract_facts_llm(user_text, assistant_text)
            if facts:
                return facts
        return self._extract_facts_heuristic(user_text, assistant_text)

    def _extract_facts_llm(self, user_text: str, assistant_text: str) -> List[str]:
        prompt = _FACT_EXTRACTION_PROMPT.format(user_text=user_text, assistant_text=assistant_text)
        raw = self._llm_complete(prompt)
        if not raw:
            return []
        facts: List[str] = []
        for line in raw.splitlines():
            cleaned = line.strip().lstrip("-*•").strip()
            cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
            if not cleaned or cleaned.upper() == "NONE":
                continue
            facts.append(cleaned)
        return _dedupe_preserving_order(facts)

    @staticmethod
    def _extract_facts_heuristic(user_text: str, assistant_text: str) -> List[str]:
        facts: List[str] = []
        for text, is_user in ((user_text or "", True), (assistant_text or "", False)):
            for sentence in _SENTENCE_RE.split(text):
                cleaned = sentence.strip()
                if len(cleaned) < 15 or cleaned.endswith("?"):
                    continue
                lowered = cleaned.lower()
                if lowered.startswith(_FILLER_PREFIXES):
                    continue
                if not _FACT_VERB_RE.search(lowered):
                    continue
                facts.append(_to_third_person(cleaned) if is_user else cleaned.rstrip("."))
        return _dedupe_preserving_order(facts)

    # ── consolidation ──────────────────────────────────────────────────

    def consolidate(self, threshold: Optional[float] = None) -> int:
        """Merge near-duplicate entries. Returns how many entries were dropped.

        Two entries are near-duplicates when their similarity (cosine over
        embeddings when both have one, Jaccard over word sets otherwise) is at
        or above ``threshold``. The newest of the pair survives, inheriting the
        highest importance and the union of entities/metadata. With an ``llm``
        injected the survivor's content is replaced by a merged canonical
        statement; without one the older text is dropped and logged.
        """
        limit = self.dedup_threshold if threshold is None else float(threshold)

        with self._lock:
            entries = sorted(self._entries.values(), key=lambda e: e.created_at)

        dropped: List[str] = []
        merged: List[MemoryEntry] = []
        survivors: List[MemoryEntry] = []
        for entry in entries:
            match: Optional[MemoryEntry] = None
            for kept in survivors:
                if kept.scope != entry.scope:
                    continue
                if self._entry_similarity(kept, entry) >= limit:
                    match = kept
                    break

            if match is None:
                survivors.append(entry)
                continue

            older, newer = (match, entry) if match.created_at <= entry.created_at else (entry, match)
            merged_text = newer.content
            if self.llm is not None:
                llm_text = self._llm_complete(_MERGE_PROMPT.format(older=older.content, newer=newer.content))
                if llm_text:
                    merged_text = llm_text.strip()
            else:
                log.info(
                    "consolidate: dropping duplicate %s (kept %s) — no llm, keeping most recent text",
                    older.id, newer.id,
                )

            match.content = merged_text
            match.importance = max(match.importance, entry.importance)
            match.entities = _dedupe_preserving_order(list(match.entities) + list(entry.entities))
            merged_meta = dict(entry.metadata)
            merged_meta.update(match.metadata)
            match.metadata = merged_meta
            match.created_at = newer.created_at
            match.updated_at = datetime.now().isoformat()
            match.embedding = None  # content changed — re-embed lazily
            dropped.append(entry.id)
            merged.append(match)

        if not dropped:
            return 0

        with self._lock:
            for did in dropped:
                self._entries.pop(did, None)
            still_here = [e for e in merged if e.id in self._entries]

        self._submit("delete", dropped)
        for survivor in still_here:
            if self.async_writes and self._writer is not None:
                self._writer.submit("call", lambda e=survivor: self._embed_and_persist(e))
            else:
                self._embed_and_persist(survivor)

        log.info("consolidate: merged %d duplicate entries", len(dropped))
        return len(dropped)

    # ── entity view ────────────────────────────────────────────────────

    def entities(self) -> Dict[str, List[str]]:
        """Map of entity name -> contents mentioning it."""
        out: Dict[str, List[str]] = {}
        with self._lock:
            entries = list(self._entries.values())
        for entry in entries:
            for name in entry.entities:
                out.setdefault(name.lower(), []).append(entry.content)
        return out

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Return ``{"name", "facts"}`` for an entity, or ``None``."""
        facts = self.entities().get(name.lower())
        if not facts:
            return None
        return {"name": name, "facts": facts}

    # ── lifecycle ──────────────────────────────────────────────────────

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for queued background writes to reach the backend.

        Returns ``True`` when the queue drained within ``timeout``.
        """
        if self._writer is None:
            return True
        return self._writer.flush(timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Flush, stop the worker thread and close the backend."""
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.flush(timeout)
            self._writer.stop(timeout)
        try:
            self.storage.close()
        except Exception:  # pragma: no cover - defensive
            log.debug("Memory: backend close failed", exc_info=True)

    def stats(self) -> Dict[str, Any]:
        """Counts by scope and kind, useful for debugging and dashboards."""
        with self._lock:
            entries = list(self._entries.values())
        by_scope: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        for e in entries:
            by_scope[e.scope.value] = by_scope.get(e.scope.value, 0) + 1
            by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
        return {
            "total": len(entries),
            "by_scope": by_scope,
            "by_kind": by_kind,
            "embedded": sum(1 for e in entries if e.embedding),
            "async_writes": self.async_writes,
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ── scoring internals ──────────────────────────────────────────────

    @staticmethod
    def infer_importance(text: str) -> float:
        """Heuristic importance in ``0..1`` when none was supplied.

        Starts at ``0.5`` and adjusts:

        * ``+0.20`` when the text carries a directive/preference marker
          (``always``, ``never``, ``must``, ``remember``, ``prefers``, ...);
        * ``+0.10`` when it contains a number or a date (concrete facts);
        * ``+0.10`` when it names a capitalised entity beyond the first word;
        * ``-0.20`` when it is shorter than 15 characters (likely chatter).

        The result is clamped to ``0..1``.
        """
        stripped = (text or "").strip()
        score = 0.5
        lowered = stripped.lower()
        if any(marker in lowered for marker in _IMPORTANCE_MARKERS):
            score += 0.20
        if re.search(r"\d", stripped):
            score += 0.10
        if re.search(r"(?<!^)\b[A-Z][a-z]{2,}", stripped):
            score += 0.10
        if len(stripped) < 15:
            score -= 0.20
        return max(0.0, min(1.0, score))

    def _coerce_importance(self, raw: Any, text: str) -> float:
        if raw is None:
            return self.infer_importance(text)
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            log.warning("Invalid importance %r — falling back to the heuristic", raw)
            return self.infer_importance(text)

    def _recency(self, entry: MemoryEntry, now: datetime) -> float:
        """Exponential decay: ``0.5 ** (age_hours / half_life_hours)``."""
        try:
            created = datetime.fromisoformat(entry.created_at)
        except (TypeError, ValueError):
            return 0.0
        age_hours = max(0.0, (now - created).total_seconds() / 3600.0)
        return 0.5 ** (age_hours / self.half_life_hours)

    def _similarity_to_query(
        self,
        entry: MemoryEntry,
        query_vec: Optional[List[float]],
        query_tokens: List[str],
    ) -> float:
        if query_vec is not None and entry.embedding:
            return max(0.0, _cosine(query_vec, entry.embedding))
        return _lexical_overlap(query_tokens, entry.content)

    def _entry_similarity(self, a: MemoryEntry, b: MemoryEntry) -> float:
        if a.embedding and b.embedding:
            return max(0.0, _cosine(a.embedding, b.embedding))
        if self.embedding is not None:
            vec_a = a.embedding or self._embed(a.content)
            vec_b = b.embedding or self._embed(b.content)
            if vec_a and vec_b:
                return max(0.0, _cosine(vec_a, vec_b))
        return _jaccard(_tokenize(a.content), _tokenize(b.content))

    def _embed(self, text: str) -> Optional[List[float]]:
        if self.embedding is None:
            return None
        try:
            if hasattr(self.embedding, "embed_text"):
                return list(self.embedding.embed_text(text))
            return list(self.embedding(text))
        except Exception:
            log.warning("Embedding provider failed — falling back to lexical scoring", exc_info=True)
            return None

    # ── persistence internals ──────────────────────────────────────────

    def _embed_and_persist(self, entry: MemoryEntry) -> None:
        if entry.embedding is None and self.embedding is not None:
            vec = self._embed(entry.content)
            if vec:
                entry.embedding = vec
        self.storage.upsert(entry)

    def _submit(self, op: str, payload: Any) -> None:
        if self.async_writes and self._writer is not None:
            self._writer.submit(op, payload)
            return
        if op == "upsert":
            self.storage.upsert(payload)
        elif op == "delete":
            self.storage.delete(payload)
        elif op == "clear":
            self.storage.clear()

    def _evict_if_needed(self) -> List[str]:
        """Drop the oldest, least important entries past ``max_entries``."""
        if self.max_entries <= 0 or len(self._entries) <= self.max_entries:
            return []
        ordered = sorted(self._entries.values(), key=lambda e: (e.importance, e.created_at))
        excess = len(self._entries) - self.max_entries
        evicted = [e.id for e in ordered[:excess]]
        for eid in evicted:
            self._entries.pop(eid, None)
        log.debug("Memory: evicted %d entries over max_entries=%d", len(evicted), self.max_entries)
        return evicted

    # ── llm helper ─────────────────────────────────────────────────────

    def _llm_complete(self, prompt: str) -> Optional[str]:
        """Best-effort text completion; never raises."""
        if self.llm is None:
            return None
        try:
            if hasattr(self.llm, "generate_text"):
                return str(self.llm.generate_text(prompt))
            if hasattr(self.llm, "generate"):
                resp = self.llm.generate(prompt)
                return str(getattr(resp, "text", resp))
            if callable(self.llm):
                return str(self.llm(prompt))
        except Exception:
            log.warning("Memory: LLM call failed — using the heuristic fallback", exc_info=True)
        return None

    @staticmethod
    def _coerce_scope_filter(
        scope: Optional[Union[ScopeLike, Sequence[ScopeLike]]],
    ) -> Optional[set]:
        if scope is None:
            return None
        if isinstance(scope, (str, MemoryScope)):
            return {_coerce_scope(scope)}
        return {_coerce_scope(s) for s in scope}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1]


def _lexical_overlap(query_tokens: List[str], text: str) -> float:
    """Fraction of query tokens present in *text* (``0..1``).

    Same idea as the keyword scoring in ``ShortTermMemory``/``LongTermMemory``,
    normalised so it can be mixed with cosine similarity.
    """
    if not query_tokens:
        return 0.0
    haystack = (text or "").lower()
    hits = sum(1 for t in query_tokens if t in haystack)
    return hits / float(len(query_tokens))


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _dedupe_preserving_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


_THIRD_PERSON_RULES = (
    (re.compile(r"^i am\b", re.IGNORECASE), "The user is"),
    (re.compile(r"^i'm\b", re.IGNORECASE), "The user is"),
    (re.compile(r"^i have\b", re.IGNORECASE), "The user has"),
    (re.compile(r"^i\b", re.IGNORECASE), "The user"),
    (re.compile(r"^my\b", re.IGNORECASE), "The user's"),
)


def _to_third_person(sentence: str) -> str:
    """Rewrite a first-person user sentence into the third person."""
    out = sentence.strip()
    for pattern, replacement in _THIRD_PERSON_RULES:
        if pattern.search(out):
            out = pattern.sub(replacement, out, count=1)
            break
    out = re.sub(r"\bmy\b", "the user's", out)
    out = re.sub(r"\bI am\b", "the user is", out)
    out = re.sub(r"\bI\b", "the user", out)
    return out.rstrip(".")
