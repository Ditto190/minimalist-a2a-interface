"""
Data tools for Mangaba AI — read-only SQL against SQLite and PostgreSQL.

:class:`SQLQueryTool` lets an agent read a database without being able to
change it. Three rules make that hold:

1. **Only reads.** The statement must be a single ``SELECT`` (or a ``WITH``
   CTE that ends in a ``SELECT``). Comments are stripped first, so a
   ``SELECT 1 -- ; DROP TABLE users`` smuggling attempt cannot hide a second
   statement, and stacked statements are rejected outright.
2. **Read-only connections.** SQLite is opened through a
   ``file:…?mode=ro`` URI; PostgreSQL runs inside a ``READ ONLY``
   transaction. Even a statement that slips past rule 1 cannot write.
3. **No string interpolation.** Values are always passed to the driver as
   bound parameters — the tool never builds SQL by concatenation.

SQLite needs nothing (stdlib :mod:`sqlite3`); PostgreSQL needs
``pip install mangaba[postgres]``.

Example::

    from mangaba.tools.data_tools import SQLQueryTool

    tool = SQLQueryTool(connection_string="sqlite:///analytics.db", max_rows=200)
    result = tool.run(
        query="SELECT name, revenue FROM sales WHERE region = ? ORDER BY revenue DESC",
        parameters=["Nordeste"],
    )
    print(result["columns"], result["row_count"])
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from mangaba.tools.base import BaseTool

log = logging.getLogger(__name__)


#: Rows returned before the result set is truncated.
DEFAULT_MAX_ROWS = 100

#: Keywords that must never appear anywhere in the statement, matched as whole
#: words. The list is deliberately narrow: the first-keyword check already
#: rejects ``INSERT``/``COPY``/``PRAGMA``-style statements, so what remains here
#: guards against writes *hidden inside* a read — a data-modifying CTE
#: (``WITH x AS (DELETE … RETURNING *) SELECT …``) or a ``SELECT … INTO``.
#: Words that are also common identifiers or functions (``replace``, ``set``,
#: ``call``) are left out on purpose to avoid refusing legitimate queries.
FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "merge", "upsert", "grant", "revoke", "attach", "detach", "vacuum",
    "pragma", "reindex", "into",
)

_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")
_WHITESPACE_RE = re.compile(r"\s+")


class UnsafeQueryError(ValueError):
    """The submitted SQL is not a plain read."""


def strip_sql_comments(sql: str) -> str:
    """Remove ``--`` and ``/* */`` comments without touching string literals.

    Example::

        strip_sql_comments("SELECT 1 -- ; DROP TABLE t")   # 'SELECT 1'
    """
    placeholders: List[str] = []

    def _stash(match: "re.Match[str]") -> str:
        placeholders.append(match.group(0))
        return f"\x00{len(placeholders) - 1}\x00"

    stashed = _STRING_LITERAL_RE.sub(_stash, sql)
    stashed = _BLOCK_COMMENT_RE.sub(" ", stashed)
    stashed = _LINE_COMMENT_RE.sub(" ", stashed)

    def _restore(match: "re.Match[str]") -> str:
        return placeholders[int(match.group(1))]

    return re.sub(r"\x00(\d+)\x00", _restore, stashed).strip()


def ensure_read_only(sql: str) -> str:
    """Validate that *sql* is a single read-only statement and return it cleaned.

    Raises:
        UnsafeQueryError: If the statement writes, is stacked with another
            statement, or does not start with SELECT/WITH.

    Example::

        ensure_read_only("SELECT 1")             # 'SELECT 1'
        ensure_read_only("DROP TABLE users")     # UnsafeQueryError
    """
    cleaned = strip_sql_comments(sql or "")
    if not cleaned:
        raise UnsafeQueryError("Empty query.")

    body = cleaned.rstrip().rstrip(";").strip()
    if not body:
        raise UnsafeQueryError("Empty query.")

    # Stacked statements: any ';' left outside a string literal is a second statement.
    without_strings = _STRING_LITERAL_RE.sub("''", body)
    if ";" in without_strings:
        raise UnsafeQueryError(
            "Only one statement per call is allowed — found ';' inside the query."
        )

    normalized = _WHITESPACE_RE.sub(" ", without_strings).strip().lower()
    first = normalized.split(" ", 1)[0]
    if first not in ("select", "with"):
        raise UnsafeQueryError(
            f"Refusing '{first.upper()}': this tool only runs SELECT (or WITH … SELECT) queries."
        )
    if first == "with" and not re.search(r"\bselect\b", normalized):
        raise UnsafeQueryError("A WITH clause must end in a SELECT.")

    words = set(re.findall(r"[a-z_]+", normalized))
    offenders = sorted(words & set(FORBIDDEN_KEYWORDS))
    if offenders:
        raise UnsafeQueryError(
            f"Refusing query: it contains write/DDL keyword(s) {', '.join(k.upper() for k in offenders)}."
        )
    return body


class SQLQueryInput(BaseModel):
    """Arguments accepted by :class:`SQLQueryTool`."""

    query: str = Field(
        ...,
        description=(
            "A single read-only SQL statement (SELECT, or WITH … SELECT). Use "
            "placeholders ('?' on SQLite, '%s' on PostgreSQL) for values and pass "
            "them in 'parameters' — never inline them into the SQL."
        ),
    )
    parameters: Optional[List[Any]] = Field(
        default=None, description="Values bound to the placeholders, in order"
    )
    max_rows: Optional[int] = Field(default=None, description="Cap on the number of rows returned")


class SQLQueryTool(BaseTool):
    """Run a read-only SQL query against SQLite or PostgreSQL.

    Accepts ``sqlite:///path/to.db``, a bare path to a ``.db``/``.sqlite``
    file, or ``postgresql://user:pass@host/dbname``. Writes, DDL and stacked
    statements are rejected before the driver ever sees them, the connection
    itself is opened read-only, and values are always bound as parameters.

    Returns a dict with ``columns``, ``rows``, ``row_count``, ``truncated``
    and ``error``.

    Example::

        tool = SQLQueryTool(connection_string="sqlite:///shop.db")
        out = tool.run(
            query="SELECT id, total FROM orders WHERE total > ? LIMIT 10",
            parameters=[100],
        )
        out["columns"]      # ['id', 'total']

        tool.run(query="DROP TABLE orders")["error"]
        # "Refusing 'DROP': this tool only runs SELECT ..."

    PostgreSQL requires ``pip install mangaba[postgres]``.
    """

    name = "sql_query"
    description = (
        "Run a read-only SQL SELECT query against the configured database and "
        "return the rows. Writes and DDL are rejected."
    )
    args_schema = SQLQueryInput

    def __init__(
        self,
        connection_string: str,
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout: float = 30.0,
    ) -> None:
        if not connection_string:
            raise ValueError("SQLQueryTool needs a connection_string")
        self.connection_string = connection_string
        self.max_rows = max_rows
        self.timeout = timeout
        self.dialect = self._detect_dialect(connection_string)

    # -- execution -----------------------------------------------------------

    def _run(
        self,
        query: str,
        parameters: Optional[Sequence[Any]] = None,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            statement = ensure_read_only(query)
        except UnsafeQueryError as exc:
            return self._error(str(exc))

        limit = max(1, max_rows or self.max_rows)
        args = list(parameters or [])

        try:
            if self.dialect == "sqlite":
                columns, rows = self._run_sqlite(statement, args, limit + 1)
            else:
                columns, rows = self._run_postgres(statement, args, limit + 1)
        except ImportError as exc:
            return self._error(str(exc))
        except Exception as exc:  # noqa: BLE001 - driver errors are tool output
            return self._error(f"Query failed: {exc}")

        truncated = len(rows) > limit
        return {
            "columns": columns,
            "rows": rows[:limit],
            "row_count": min(len(rows), limit),
            "truncated": truncated,
            "dialect": self.dialect,
            "error": None,
        }

    def _run_sqlite(
        self, statement: str, args: List[Any], fetch: int
    ) -> Tuple[List[str], List[List[Any]]]:
        connection = sqlite3.connect(self._sqlite_uri(), uri=True, timeout=self.timeout)
        try:
            connection.row_factory = None
            cursor = connection.execute(statement, args)
            columns = [d[0] for d in (cursor.description or [])]
            rows = [list(r) for r in cursor.fetchmany(fetch)]
            return columns, rows
        finally:
            connection.close()

    def _run_postgres(
        self, statement: str, args: List[Any], fetch: int
    ) -> Tuple[List[str], List[List[Any]]]:
        try:
            import psycopg  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "psycopg package is required for PostgreSQL queries. "
                "Install with: pip install mangaba[postgres]"
            ) from exc

        with psycopg.connect(self.connection_string, connect_timeout=int(self.timeout)) as connection:
            connection.read_only = True
            with connection.cursor() as cursor:
                cursor.execute(statement, args or None)
                columns = [d.name for d in (cursor.description or [])]
                rows = [list(r) for r in cursor.fetchmany(fetch)]
        return columns, rows

    # -- helpers -------------------------------------------------------------

    def _sqlite_uri(self) -> str:
        """Build a read-only SQLite URI from the configured connection string."""
        raw = self.connection_string
        if raw.startswith("sqlite:///"):
            raw = raw[len("sqlite:///") :]
        elif raw.startswith("sqlite://"):
            raw = raw[len("sqlite://") :]

        if raw == ":memory:":
            raise ValueError("An in-memory SQLite database cannot be opened read-only from a tool.")

        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {path}")
        return f"file:{path.as_posix()}?mode=ro"

    @staticmethod
    def _detect_dialect(connection_string: str) -> str:
        lowered = connection_string.lower()
        if lowered.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
            return "postgres"
        if lowered.startswith("sqlite:") or lowered.endswith((".db", ".sqlite", ".sqlite3")):
            return "sqlite"
        raise ValueError(
            f"Unsupported connection string {connection_string!r}. Use 'sqlite:///path.db', "
            "a path to a .db/.sqlite file, or 'postgresql://user:pass@host/dbname'."
        )

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        log.debug("SQLQueryTool refused or failed: %s", message)
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "dialect": None,
            "error": message,
        }


__all__ = [
    "DEFAULT_MAX_ROWS",
    "FORBIDDEN_KEYWORDS",
    "SQLQueryInput",
    "SQLQueryTool",
    "UnsafeQueryError",
    "ensure_read_only",
    "strip_sql_comments",
]
