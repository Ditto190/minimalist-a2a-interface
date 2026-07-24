"""
Langfuse tracing for Mangaba AI v3.0

Maps the framework's spans onto Langfuse traces/spans/generations. LLM spans
become *generations* (so token usage shows up in Langfuse's cost dashboards),
everything else becomes a plain span.

Optional dependency::

    pip install langfuse

Configuration (env or constructor)::

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com

Both the v2 (``client.trace(...)``/``span.span(...)``) and v3
(``client.start_span(...)``) SDK shapes are supported; the v3 API is preferred
when present. Langfuse v3 is itself OTel-based, so
:class:`~mangaba.observability.otel.OpenTelemetryCallback` pointed at the
Langfuse OTLP endpoint is a valid alternative.

Example::

    from mangaba.observability import LangfuseCallback, configure_observability

    configure_observability(LangfuseCallback(session_id="run-42"))
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from mangaba.observability.base import Span, SpanTrackingCallback

log = logging.getLogger(__name__)

#: Env vars that make ``auto_configure_from_env`` enable this integration.
ENV_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


class LangfuseCallback(SpanTrackingCallback):
    """Send Mangaba spans to Langfuse.

    Args:
        public_key: Defaults to ``LANGFUSE_PUBLIC_KEY``.
        secret_key: Defaults to ``LANGFUSE_SECRET_KEY``.
        host: Defaults to ``LANGFUSE_HOST``.
        session_id: Optional session/thread id attached to every root span.
        user_id: Optional end-user id attached to every root span.
        client: Bring your own configured ``Langfuse`` client.

    No-ops with a warning when the SDK is missing or the keys are absent.

    Example::

        cb = LangfuseCallback()
        cb.register()
        crew.kickoff()
        cb.flush()
    """

    integration_name = "LangfuseCallback"

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.user_id = user_id
        self.client: Any = client
        self._v3 = False

        if client is not None:
            self._v3 = hasattr(client, "start_span")
            return

        public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        host = host or os.getenv("LANGFUSE_HOST")

        try:
            from langfuse import Langfuse  # type: ignore
        except ImportError:
            self._disable("the 'langfuse' package is required. Install with: pip install langfuse")
            return

        if not public_key or not secret_key:
            self._disable("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set")
            return

        try:
            kwargs: Dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
            if host:
                kwargs["host"] = host
            self.client = Langfuse(**kwargs)
            self._v3 = hasattr(self.client, "start_span")
        except Exception as exc:
            self._disable(f"could not build the Langfuse client ({exc})")

    # ── span hooks ─────────────────────────────────────────────────────

    def _start_span(self, span: Span, parent: Optional[Span]) -> None:
        if self.client is None:
            return
        is_generation = span.kind == "llm"
        metadata = dict(span.attributes)

        if self._v3:
            factory_owner = parent.native if parent is not None and parent.native is not None else self.client
            factory = getattr(
                factory_owner,
                "start_generation" if is_generation else "start_span",
                None,
            ) or getattr(factory_owner, "start_span", None)
            if factory is None:  # pragma: no cover - unexpected SDK shape
                return
            span.native = factory(name=span.name, metadata=metadata)
            if parent is None:
                self._tag_root(span.native)
            return

        # v2: an explicit trace object owns the span tree.
        if parent is None:
            trace_kwargs: Dict[str, Any] = {"name": span.name, "metadata": metadata}
            if self.session_id:
                trace_kwargs["session_id"] = self.session_id
            if self.user_id:
                trace_kwargs["user_id"] = self.user_id
            span.context = self.client.trace(**trace_kwargs)
            span.native = span.context
            return

        owner = parent.native if parent.native is not None else self.client
        factory = getattr(owner, "generation" if is_generation else "span", None)
        if factory is None:  # pragma: no cover - unexpected SDK shape
            return
        span.native = factory(name=span.name, metadata=metadata)
        span.context = parent.context

    def _end_span(self, span: Span, status: str, attributes: Dict[str, Any]) -> None:
        native = span.native
        if native is None:
            return
        payload: Dict[str, Any] = {"metadata": attributes}
        if status == "error":
            payload["level"] = "ERROR"
            payload["status_message"] = str(attributes.get("mangaba.error", "error"))
        if span.kind == "llm" and span.tokens:
            payload["usage_details"] = {"total": span.tokens}

        update = getattr(native, "update", None)
        if update is not None:
            try:
                update(**payload)
            except TypeError:  # older/newer SDKs accept a narrower signature
                log.debug("%s: update(**payload) rejected — updating metadata only",
                          self.integration_name, exc_info=True)
                try:
                    update(metadata=attributes)
                except Exception:  # pragma: no cover - defensive
                    log.debug("%s: update() failed", self.integration_name, exc_info=True)

        end = getattr(native, "end", None)
        if end is not None:
            try:
                end()
            except Exception:  # pragma: no cover - defensive
                log.debug("%s: end() failed", self.integration_name, exc_info=True)

    def _add_event(self, span: Optional[Span], name: str, attributes: Dict[str, Any]) -> None:
        if span is None or span.native is None:
            return
        factory = getattr(span.native, "event", None) or getattr(span.native, "create_event", None)
        if factory is None:
            return
        factory(name=name, metadata=attributes)

    def _tag_root(self, native: Any) -> None:
        update = getattr(native, "update_trace", None) or getattr(native, "update", None)
        if update is None or not (self.session_id or self.user_id):
            return
        kwargs: Dict[str, Any] = {}
        if self.session_id:
            kwargs["session_id"] = self.session_id
        if self.user_id:
            kwargs["user_id"] = self.user_id
        try:
            update(**kwargs)
        except Exception:  # pragma: no cover - SDK version differences
            log.debug("%s: could not tag the root span", self.integration_name, exc_info=True)

    # ── lifecycle ──────────────────────────────────────────────────────

    def flush(self, timeout: float = 5.0) -> bool:
        if not self.enabled or self.client is None:
            return True
        fn = getattr(self.client, "flush", None)
        if fn is None:
            return True
        try:
            fn()
            return True
        except Exception:
            log.debug("%s: flush failed", self.integration_name, exc_info=True)
            return False

    def shutdown(self) -> None:
        if not self.enabled or self.client is None:
            return
        self.flush()
        fn = getattr(self.client, "shutdown", None)
        if fn is not None:
            try:
                fn()
            except Exception:  # pragma: no cover - defensive
                log.debug("%s: shutdown failed", self.integration_name, exc_info=True)
