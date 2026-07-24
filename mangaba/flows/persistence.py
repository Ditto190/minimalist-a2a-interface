"""
Flow persistence — SQLite-backed checkpointing, resume and fork.

A checkpoint is written after every completed step of a persisted flow, so an
interrupted run can be resumed in a new process: already-completed steps are
skipped and their recorded outputs replayed to trigger downstream listeners.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from mangaba.core.exceptions import FlowPersistenceError

log = logging.getLogger(__name__)

#: Default on-disk location of the checkpoint database.
DEFAULT_DB_PATH = "./.mangaba_flows.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flow_states (
    flow_id         TEXT PRIMARY KEY,
    flow_class      TEXT NOT NULL DEFAULT '',
    state_json      TEXT NOT NULL DEFAULT '{}',
    completed_steps TEXT NOT NULL DEFAULT '[]',
    method_outputs  TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_flow_states_updated ON flow_states (updated_at DESC);
"""


def _now() -> str:
    return datetime.now().isoformat()


def _dumps(value: Any) -> str:
    """JSON-encode best-effort — unserialisable objects fall back to ``str``."""
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover — default=str covers almost all
        log.debug("Falling back to string encoding for checkpoint payload: %s", exc)
        return json.dumps(str(value))


class FlowRecord(BaseModel):
    """A single persisted flow checkpoint.

    Example::

        record = store.load("a1b2c3")
        print(record.completed_steps)
    """

    flow_id: str
    flow_class: str = ""
    state: Dict[str, Any] = Field(default_factory=dict)
    completed_steps: List[str] = Field(default_factory=list)
    method_outputs: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Store ABC
# ---------------------------------------------------------------------------

class BaseFlowStore(ABC):
    """Abstract checkpoint store for flows."""

    @abstractmethod
    def save(self, record: FlowRecord) -> None:
        ...

    @abstractmethod
    def load(self, flow_id: str) -> Optional[FlowRecord]:
        ...

    @abstractmethod
    def delete(self, flow_id: str) -> bool:
        ...

    @abstractmethod
    def list_flows(self, limit: int = 50) -> List[FlowRecord]:
        ...

    def resume(self, flow_id: str) -> FlowRecord:
        """Load a checkpoint, raising if it does not exist.

        Args:
            flow_id: Identifier previously used to save the flow.

        Returns:
            The stored :class:`FlowRecord`.

        Raises:
            FlowPersistenceError: If no checkpoint exists for ``flow_id``.
        """
        record = self.load(flow_id)
        if record is None:
            raise FlowPersistenceError(f"No persisted flow found with id '{flow_id}'")
        return record

    def fork(self, flow_id: str, new_id: Optional[str] = None, include_completed_steps: bool = True) -> FlowRecord:
        """Branch a saved checkpoint into a brand-new flow id.

        Args:
            flow_id: The checkpoint to branch from.
            new_id: Identifier for the branch (random when omitted).
            include_completed_steps: Keep the completed-step list so the fork
                continues where the parent stopped. Set ``False`` to replay the
                whole graph against the copied state.

        Returns:
            The newly saved :class:`FlowRecord`.

        Raises:
            FlowPersistenceError: If the source checkpoint is missing.

        Example::

            branch = store.fork("a1b2c3")
            flow = MyFlow(flow_id=branch.flow_id).resume(branch.flow_id)
        """
        source = self.resume(flow_id)
        forked_id = new_id or uuid.uuid4().hex

        state = dict(source.state)
        if "id" in state:
            state["id"] = forked_id

        record = FlowRecord(
            flow_id=forked_id,
            flow_class=source.flow_class,
            state=state,
            completed_steps=list(source.completed_steps) if include_completed_steps else [],
            method_outputs=dict(source.method_outputs) if include_completed_steps else {},
            created_at=_now(),
            updated_at=_now(),
        )
        self.save(record)
        log.info("Forked flow %s -> %s", flow_id, forked_id)
        return record


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------

class SQLiteFlowStore(BaseFlowStore):
    """SQLite checkpoint store (stdlib only, no external dependencies).

    The database file is created on first use, together with its parent
    directory. Access is guarded by a lock so concurrent steps of the same
    process cannot interleave writes.

    Example::

        store = SQLiteFlowStore("./.mangaba_flows.db")
        store.save(FlowRecord(flow_id="abc", state={"step": 1}))
        record = store.resume("abc")
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ── public API ─────────────────────────────────────────────────────

    def save(self, record: FlowRecord) -> None:
        """Insert or update a checkpoint (upsert on ``flow_id``)."""
        record.updated_at = _now()
        sql = (
            "INSERT INTO flow_states "
            "(flow_id, flow_class, state_json, completed_steps, method_outputs, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(flow_id) DO UPDATE SET "
            "flow_class=excluded.flow_class, state_json=excluded.state_json, "
            "completed_steps=excluded.completed_steps, method_outputs=excluded.method_outputs, "
            "updated_at=excluded.updated_at"
        )
        params = (
            record.flow_id,
            record.flow_class,
            _dumps(record.state),
            _dumps(record.completed_steps),
            _dumps(record.method_outputs),
            record.created_at,
            record.updated_at,
        )
        with self._connect() as conn:
            conn.execute(sql, params)

    def load(self, flow_id: str) -> Optional[FlowRecord]:
        """Return the checkpoint for ``flow_id``, or ``None`` if absent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT flow_id, flow_class, state_json, completed_steps, method_outputs, created_at, updated_at "
                "FROM flow_states WHERE flow_id = ?",
                (flow_id,),
            ).fetchone()
        return self._to_record(row) if row else None

    def delete(self, flow_id: str) -> bool:
        """Delete a checkpoint. Returns ``True`` when a row was removed."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM flow_states WHERE flow_id = ?", (flow_id,))
            return cur.rowcount > 0

    def list_flows(self, limit: int = 50) -> List[FlowRecord]:
        """Return the most recently updated checkpoints."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT flow_id, flow_class, state_json, completed_steps, method_outputs, created_at, updated_at "
                "FROM flow_states ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._to_record(row) for row in rows]

    # ── internal ───────────────────────────────────────────────────────

    def _init_db(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> _LockedConnection:
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
        except sqlite3.Error as exc:
            raise FlowPersistenceError(f"Could not open flow database '{self.db_path}': {exc}", cause=exc) from exc
        return _LockedConnection(conn, self._lock)

    @staticmethod
    def _to_record(row: Any) -> FlowRecord:
        try:
            return FlowRecord(
                flow_id=row[0],
                flow_class=row[1] or "",
                state=json.loads(row[2] or "{}"),
                completed_steps=json.loads(row[3] or "[]"),
                method_outputs=json.loads(row[4] or "{}"),
                created_at=row[5] or "",
                updated_at=row[6] or "",
            )
        except (ValueError, TypeError) as exc:
            raise FlowPersistenceError(f"Corrupted checkpoint row for flow '{row[0]}': {exc}", cause=exc) from exc

    def __repr__(self) -> str:
        return f"SQLiteFlowStore(db_path={self.db_path!r})"


class _LockedConnection:
    """Context manager wrapping a sqlite3 connection with a process lock."""

    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock) -> None:
        self._conn = conn
        self._lock = lock

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        return self._conn

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()
            self._lock.release()


# ---------------------------------------------------------------------------
# Default store
# ---------------------------------------------------------------------------

_default_store: Optional[SQLiteFlowStore] = None


def get_default_store(db_path: str = DEFAULT_DB_PATH) -> SQLiteFlowStore:
    """Return (creating on first call) the process-wide default store.

    Example::

        store = get_default_store()
        for record in store.list_flows(limit=5):
            print(record.flow_id, record.updated_at)
    """
    global _default_store
    if _default_store is None or _default_store.db_path != db_path:
        _default_store = SQLiteFlowStore(db_path)
    return _default_store
