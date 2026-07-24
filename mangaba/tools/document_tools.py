"""
Document tools for Mangaba AI — semantic search inside a file, and grep across a tree.

* :class:`DocumentSearchTool` answers questions about **one** document. It
  loads the file with the matching loader from :mod:`mangaba.rag.loaders`
  (PDF, DOCX, XLSX, JSON, CSV, plain text), splits it with
  :class:`~mangaba.rag.splitters.RecursiveTextSplitter`, embeds every chunk
  once, and then returns the passages closest to each query.
* :class:`FileSearchTool` is the literal/regex counterpart: it walks a
  directory tree and reports ``path:line`` matches, like ``grep -rn``.

Example::

    from mangaba.embeddings.openai_embed import OpenAIEmbedding
    from mangaba.tools.document_tools import DocumentSearchTool, FileSearchTool

    handbook = DocumentSearchTool(file_path="handbook.pdf", embedding=OpenAIEmbedding())
    print(handbook.run(query="What is the parental leave policy?"))

    grep = FileSearchTool(root_dir="./src")
    print(grep.run(pattern="TODO", glob="*.py"))
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from mangaba.tools.base import BaseTool

log = logging.getLogger(__name__)


#: Files larger than this are skipped by :class:`FileSearchTool`.
DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024

#: Directories never descended into during a file search.
SKIPPED_DIRS = frozenset(
    {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".tox", "dist", "build"}
)


# ---------------------------------------------------------------------------
# Semantic search inside one document
# ---------------------------------------------------------------------------

class DocumentSearchInput(BaseModel):
    """Arguments accepted by a file-bound :class:`DocumentSearchTool`."""

    query: str = Field(..., description="What to look for, in plain language")
    top_k: Optional[int] = Field(default=None, description="How many passages to return")


class DocumentSearchAnyFileInput(BaseModel):
    """Arguments accepted by an unbound :class:`DocumentSearchTool`."""

    query: str = Field(..., description="What to look for, in plain language")
    file_path: str = Field(..., description="Path of the document to search")
    top_k: Optional[int] = Field(default=None, description="How many passages to return")


class DocumentSearchTool(BaseTool):
    """Semantic search *inside* a single document.

    The file is loaded, chunked and embedded on first use and the index is
    kept in memory, so repeated questions about the same document cost only
    one embedding call each.

    Bind the tool to a file (recommended — the agent can then only read that
    document) or leave ``file_path`` unset to let the caller choose per query::

        # Bound: the agent can only search this file
        tool = DocumentSearchTool(file_path="contract.pdf", embedding=emb)
        tool.run(query="What is the termination notice period?")

        # Unbound: file_path becomes a required argument
        anyfile = DocumentSearchTool(embedding=emb)
        anyfile.run(query="revenue", file_path="report.xlsx")

    Formats follow :mod:`mangaba.rag.loaders`: ``.txt``/``.md``/``.rst``/
    ``.log``/``.csv``/``.json`` need nothing extra; ``.pdf``/``.docx``/
    ``.xlsx`` need ``pip install mangaba[documents]``.
    """

    name = "document_search"
    description = "Search inside a document and return the passages most relevant to a question"

    def __init__(
        self,
        embedding: Any,
        file_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        max_chars_per_passage: int = 1200,
        loader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if embedding is None or not hasattr(embedding, "embed_text"):
            raise ValueError(
                "DocumentSearchTool needs an embedding provider exposing embed_text(), "
                "e.g. mangaba.embeddings.openai_embed.OpenAIEmbedding()"
            )
        self.embedding = embedding
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.max_chars_per_passage = max_chars_per_passage
        self.loader_kwargs = dict(loader_kwargs or {})
        self.args_schema = DocumentSearchInput if file_path else DocumentSearchAnyFileInput

        if file_path:
            self.name = "document_search"
            self.description = (
                f"Search inside the document '{os.path.basename(file_path)}' and return "
                "the passages most relevant to a question"
            )

        #: path → (chunk texts, chunk vectors, chunk metadata)
        self._indexes: Dict[str, Tuple[List[str], List[List[float]], List[Dict[str, Any]]]] = {}

    # -- execution -----------------------------------------------------------

    def _run(self, query: str, file_path: Optional[str] = None, top_k: Optional[int] = None) -> str:
        target = self.file_path or file_path
        if not target:
            return "Error: no file_path given and the tool is not bound to a document."
        if self.file_path and file_path and file_path != self.file_path:
            log.debug("Ignoring file_path=%r — this tool is bound to %r", file_path, self.file_path)

        try:
            texts, vectors, metas = self._index(target)
        except FileNotFoundError:
            return f"Error: File '{target}' not found"
        except ImportError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface loader problems as tool output
            return f"Error indexing '{target}': {exc}"

        if not texts:
            return f"'{target}' produced no searchable text."

        try:
            query_vector = self.embedding.embed_text(query)
        except Exception as exc:  # noqa: BLE001 - provider failures are tool output
            return f"Error embedding the query: {exc}"

        k = max(1, top_k or self.top_k)
        scored = sorted(
            ((cosine_similarity(query_vector, vec), i) for i, vec in enumerate(vectors)),
            key=lambda pair: pair[0],
            reverse=True,
        )[:k]

        blocks: List[str] = []
        for rank, (score, index) in enumerate(scored, 1):
            passage = texts[index]
            if len(passage) > self.max_chars_per_passage:
                passage = passage[: self.max_chars_per_passage] + " …"
            blocks.append(f"[{rank}] score={score:.3f} {self._locate(metas[index])}\n{passage}")
        return "\n\n".join(blocks)

    # -- indexing ------------------------------------------------------------

    def _index(self, path: str) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """Load, split and embed *path* once, caching the result."""
        key = str(Path(path).expanduser().resolve())
        cached = self._indexes.get(key)
        if cached is not None:
            return cached

        if not os.path.isfile(key):
            raise FileNotFoundError(key)

        from mangaba.rag.loaders import load_file
        from mangaba.rag.splitters import RecursiveTextSplitter

        documents = load_file(key, **self.loader_kwargs)
        splitter = RecursiveTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = splitter.split_documents(documents)

        texts = [c.content for c in chunks if c.content.strip()]
        metas = [dict(c.metadata) for c in chunks if c.content.strip()]
        vectors = self._embed_batch(texts)

        log.debug("DocumentSearchTool indexed %s into %d chunks", key, len(texts))
        self._indexes[key] = (texts, vectors, metas)
        return self._indexes[key]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embed_batch = getattr(self.embedding, "embed_batch", None)
        if callable(embed_batch):
            return list(embed_batch(texts))
        return [self.embedding.embed_text(t) for t in texts]

    def clear_index(self, file_path: Optional[str] = None) -> None:
        """Drop the cached index so the next query re-reads the file."""
        if file_path is None:
            self._indexes.clear()
            return
        self._indexes.pop(str(Path(file_path).expanduser().resolve()), None)

    @staticmethod
    def _locate(metadata: Dict[str, Any]) -> str:
        """Render the positional metadata a loader attached to a chunk."""
        source = metadata.get("source", "")
        extra = [f"{k}={metadata[k]}" for k in ("page", "sheet", "row", "index") if k in metadata]
        return f"{source}" + (f" ({', '.join(extra)})" if extra else "")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors, using numpy when available.

    Example::

        cosine_similarity([1.0, 0.0], [1.0, 0.0])   # 1.0
    """
    if not a or not b:
        return 0.0
    try:
        import numpy as np  # type: ignore

        va, vb = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(va.dot(vb) / denom) if denom else 0.0
    except ImportError:
        pass

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ---------------------------------------------------------------------------
# Literal / regex search across a tree
# ---------------------------------------------------------------------------

class FileSearchInput(BaseModel):
    """Arguments accepted by :class:`FileSearchTool`."""

    pattern: str = Field(..., description="Text or regular expression to look for")
    directory: Optional[str] = Field(default=None, description="Directory to search (defaults to the tool's root)")
    glob: str = Field(default="*", description="Filename glob, e.g. '*.py' or 'test_*.md'")
    regex: bool = Field(default=False, description="Treat 'pattern' as a regular expression")
    case_sensitive: bool = Field(default=False, description="Match case exactly")
    max_results: Optional[int] = Field(default=None, description="Stop after this many matching lines")


class FileSearchTool(BaseTool):
    """Search a directory tree for lines matching a pattern.

    Returns ``path:line: text`` rows, so an agent can quote the exact location
    of what it found. Version-control and build directories are skipped, and
    binary files are ignored.

    Constrain the reachable area with ``root_dir``: any ``directory`` argument
    that escapes it is refused. Without a root the tool can read anything the
    process can read — give it a root when the pattern comes from a model.

    Example::

        tool = FileSearchTool(root_dir="./src")
        print(tool.run(pattern="def run", glob="*.py"))
        print(tool.run(pattern=r"TODO\\(\\w+\\)", regex=True))
    """

    name = "file_search"
    description = (
        "Search files under a directory for a literal string or regular expression "
        "and return matching 'path:line: text' rows"
    )
    args_schema = FileSearchInput

    def __init__(
        self,
        root_dir: Optional[str] = None,
        max_results: int = 50,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        include_hidden: bool = False,
        recursive: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        self.root_dir = str(Path(root_dir).expanduser().resolve()) if root_dir else None
        self.max_results = max_results
        self.max_file_bytes = max_file_bytes
        self.include_hidden = include_hidden
        self.recursive = recursive
        self.encoding = encoding

    def _run(
        self,
        pattern: str,
        directory: Optional[str] = None,
        glob: str = "*",
        regex: bool = False,
        case_sensitive: bool = False,
        max_results: Optional[int] = None,
    ) -> str:
        try:
            root = self._resolve_directory(directory)
        except ValueError as exc:
            return f"Error: {exc}"
        if not root.is_dir():
            return f"Error: Directory '{root}' not found"

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            matcher = re.compile(pattern if regex else re.escape(pattern), flags)
        except re.error as exc:
            return f"Error: invalid regular expression {pattern!r}: {exc}"

        limit = max(1, max_results or self.max_results)
        hits: List[str] = []
        scanned = 0
        for file_path in self._iter_files(root, glob):
            scanned += 1
            for line_no, line in self._search_file(file_path, matcher):
                hits.append(f"{file_path}:{line_no}: {line}")
                if len(hits) >= limit:
                    header = f"{len(hits)} match(es) (limit reached, {scanned} files scanned):"
                    return "\n".join([header, *hits])

        if not hits:
            return f"No matches for {pattern!r} under '{root}' ({scanned} files scanned)."
        return "\n".join([f"{len(hits)} match(es) in {scanned} files scanned:", *hits])

    # -- internals -----------------------------------------------------------

    def _resolve_directory(self, directory: Optional[str]) -> Path:
        if directory is None:
            return Path(self.root_dir or ".").expanduser().resolve()

        candidate = Path(directory).expanduser()
        if not candidate.is_absolute() and self.root_dir:
            candidate = Path(self.root_dir) / candidate
        resolved = candidate.resolve()

        if self.root_dir:
            root = Path(self.root_dir)
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"Refusing to search '{resolved}': outside the tool root '{root}'")
        return resolved

    def _iter_files(self, root: Path, glob: str) -> List[Path]:
        matches = root.rglob(glob) if self.recursive else root.glob(glob)
        files: List[Path] = []
        for path in sorted(matches):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative_parts = path.relative_to(root).parts
            except ValueError:  # pragma: no cover - defensive
                relative_parts = path.parts
            if any(part in SKIPPED_DIRS for part in relative_parts):
                continue
            if not self.include_hidden and any(part.startswith(".") for part in relative_parts):
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            files.append(path)
        return files

    def _search_file(self, path: Path, matcher: "re.Pattern[str]") -> List[Tuple[int, str]]:
        try:
            with path.open("r", encoding=self.encoding, errors="strict") as fh:
                found: List[Tuple[int, str]] = []
                for line_no, line in enumerate(fh, 1):
                    if "\x00" in line:  # binary file — stop reading it
                        return found
                    if matcher.search(line):
                        found.append((line_no, line.rstrip("\n")[:400]))
                return found
        except (UnicodeDecodeError, OSError) as exc:
            log.debug("FileSearchTool skipped %s: %s", path, exc)
            return []


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "SKIPPED_DIRS",
    "DocumentSearchAnyFileInput",
    "DocumentSearchInput",
    "DocumentSearchTool",
    "FileSearchInput",
    "FileSearchTool",
    "cosine_similarity",
]
