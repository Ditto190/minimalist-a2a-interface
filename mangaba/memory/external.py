"""External memory providers (Fase 2 — CrewAI-parity).

CrewAI production deployments keep memory outside the container
(Mem0 Cloud / self-hosted with Qdrant / pgvector) so restarts don't wipe
it and every user gets an isolated namespace via ``user_id``.

This module defines a tiny protocol plus a ``Mem0Memory`` wrapper that
lazily imports the real ``mem0`` package. Without ``mem0`` installed it
raises a helpful error instead of breaking ``import mangaba``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class BaseExternalMemory(Protocol):
    """Minimal interface an external memory provider must satisfy."""

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str: ...
    def search(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Dict[str, Any]]: ...
    def clear(self) -> None: ...


class Mem0Memory:
    """Mem0-backed memory with per-user isolation.

    Example::

        from mangaba.memory.external import Mem0Memory

        mem = Mem0Memory(api_key="...", config={"user_id": "user-123"})
        crew = Crew(agents=[...], tasks=[...], memory=mem)

    Requires ``pip install mem0ai``. For self-hosted Qdrant::

        Mem0Memory(config={
            "user_id": current_user_id,
            "vector_store": {"provider": "qdrant", "config": {"host": "...", "port": 6333}},
        })
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.api_key = api_key
        self.config: Dict[str, Any] = dict(config or {})
        self.user_id: Optional[str] = self.config.get("user_id")
        self._client: Optional[Any] = None

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import mem0  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "mem0 is not installed. Install with: pip install mem0ai "
                "or use mangaba.memory.Memory (SQLite) instead."
            ) from exc
        kwargs: Dict[str, Any] = dict(self.config)
        if self.api_key:
            kwargs["api_key"] = self.api_key
        # mem0 exposes Memory (self-hosted) and MemoryClient (cloud).
        factory = getattr(mem0, "Memory", None)
        if self.api_key:
            try:
                from mem0 import MemoryClient as _MC  # type: ignore
                factory = _MC
            except Exception:
                pass
        self._client = factory(**kwargs) if kwargs else factory()
        return self._client

    # ── BaseExternalMemory API ─────────────────────────────────────
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        client = self._client_or_raise()
        meta = dict(metadata or {})
        if self.user_id and "user_id" not in meta:
            meta["user_id"] = self.user_id
        res = client.add(content, metadata=meta, user_id=meta.get("user_id", self.user_id))
        if isinstance(res, dict):
            return str(res.get("id", ""))
        return str(res)

    def search(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        client = self._client_or_raise()
        uid = kwargs.get("user_id", self.user_id)
        res = client.search(query, limit=top_k, user_id=uid)
        if isinstance(res, dict) and "results" in res:
            return list(res["results"])
        return list(res) if isinstance(res, list) else []

    def clear(self) -> None:
        client = self._client_or_raise()
        if hasattr(client, "clear"):
            client.clear()


__all__ = ["BaseExternalMemory", "Mem0Memory"]
