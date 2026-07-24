"""
OpenTelemetry tracing for Mangaba AI v3.0

Turns the framework's event stream into real OTel spans. Because many
platforms (Grafana Tempo, Honeycomb, Datadog, Jaeger, Langfuse, Arize Phoenix,
New Relic, ...) ingest OTLP, this is the integration to reach for first.

Optional dependency::

    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

Standard OTel environment variables are honoured, so no code change is needed
to point the traces somewhere else::

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer <token>"
    OTEL_SERVICE_NAME=my-crew

Example::

    from mangaba.observability import OpenTelemetryCallback, configure_observability

    configure_observability(OpenTelemetryCallback(service_name="my-crew"))
    crew.kickoff()          # produces crew -> task -> agent -> llm spans
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from mangaba.observability.base import Span, SpanTrackingCallback

log = logging.getLogger(__name__)

#: Env vars that make ``auto_configure_from_env`` enable this integration.
ENV_VARS = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
)


class OpenTelemetryCallback(SpanTrackingCallback):
    """Export Mangaba events as OpenTelemetry spans.

    Args:
        service_name: ``service.name`` resource attribute. Falls back to
            ``OTEL_SERVICE_NAME`` then ``"mangaba"``.
        endpoint: OTLP endpoint. Falls back to ``OTEL_EXPORTER_OTLP_ENDPOINT``.
            When neither is set and no provider is already installed, spans are
            still created but only exported if the host application configured
            a provider itself.
        headers: Extra OTLP headers (e.g. auth tokens).
        tracer_provider: Bring your own provider; nothing is installed globally
            when you do.
        install_provider: When ``True`` (default) and no global provider exists,
            install one with an OTLP exporter.

    The callback degrades to a warning + no-op when ``opentelemetry`` is not
    installed — it never raises and never breaks a crew run.

    Example::

        cb = OpenTelemetryCallback(endpoint="http://localhost:4318")
        cb.register()
        ...
        cb.flush()
    """

    integration_name = "OpenTelemetryCallback"

    def __init__(
        self,
        service_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        tracer_provider: Optional[Any] = None,
        install_provider: bool = True,
    ) -> None:
        super().__init__()
        self.service_name = service_name or os.getenv("OTEL_SERVICE_NAME") or "mangaba"
        self.endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )
        self.headers = headers
        self._provider = tracer_provider
        self._trace: Any = None
        self._status_cls: Any = None
        self._status_code: Any = None
        self._tracer: Any = None

        try:
            from opentelemetry import trace  # type: ignore
            from opentelemetry.trace import Status, StatusCode  # type: ignore
        except ImportError:
            self._disable(
                "the 'opentelemetry-api'/'opentelemetry-sdk' packages are required. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-http"
            )
            return

        self._trace = trace
        self._status_cls = Status
        self._status_code = StatusCode

        try:
            if self._provider is None:
                self._provider = self._resolve_provider(install_provider)
            self._tracer = self._provider.get_tracer("mangaba") if self._provider else trace.get_tracer("mangaba")
        except Exception as exc:
            self._disable(f"could not build a tracer provider ({exc})")

    # ── provider setup ─────────────────────────────────────────────────

    def _resolve_provider(self, install_provider: bool) -> Any:
        """Reuse an already-installed provider, or install one from env config."""
        from opentelemetry import trace  # type: ignore

        current = trace.get_tracer_provider()
        already_installed = type(current).__name__ != "ProxyTracerProvider"
        if already_installed or not install_provider:
            log.debug("%s: reusing the existing tracer provider", self.integration_name)
            return current

        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

        provider = TracerProvider(resource=Resource.create({"service.name": self.service_name}))
        exporter = self._build_exporter()
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return provider

    def _build_exporter(self) -> Any:
        """OTLP/HTTP exporter, falling back to gRPC then to no exporter."""
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
                OTLPSpanExporter,
            )
        except ImportError:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore
                    OTLPSpanExporter,
                )
            except ImportError:
                log.warning(
                    "%s: no OTLP exporter installed (pip install "
                    "opentelemetry-exporter-otlp-proto-http) — spans are created but not exported",
                    self.integration_name,
                )
                return None
        kwargs: Dict[str, Any] = {}
        if self.endpoint:
            kwargs["endpoint"] = self.endpoint
        if self.headers:
            kwargs["headers"] = self.headers
        try:
            return OTLPSpanExporter(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on SDK version
            log.warning("%s: could not build the OTLP exporter (%s)", self.integration_name, exc)
            return None

    # ── span hooks ─────────────────────────────────────────────────────

    def _start_span(self, span: Span, parent: Optional[Span]) -> None:
        if self._tracer is None:
            return
        context = None
        if parent is not None and parent.native is not None:
            context = self._trace.set_span_in_context(parent.native)
        native = self._tracer.start_span(span.name, context=context, attributes=_clean(span.attributes))
        span.native = native
        span.context = context

    def _end_span(self, span: Span, status: str, attributes: Dict[str, Any]) -> None:
        native = span.native
        if native is None:
            return
        for key, value in _clean(attributes).items():
            native.set_attribute(key, value)
        if status == "error":
            native.set_status(self._status_cls(self._status_code.ERROR, str(attributes.get("mangaba.error", ""))))
        else:
            native.set_status(self._status_cls(self._status_code.OK))
        native.end()

    def _add_event(self, span: Optional[Span], name: str, attributes: Dict[str, Any]) -> None:
        if span is None or span.native is None:
            return
        span.native.add_event(name, attributes=_clean(attributes))

    # ── lifecycle ──────────────────────────────────────────────────────

    def flush(self, timeout: float = 5.0) -> bool:
        if not self.enabled or self._provider is None:
            return True
        force_flush = getattr(self._provider, "force_flush", None)
        if force_flush is None:
            return True
        try:
            return bool(force_flush(timeout_millis=int(timeout * 1000)))
        except Exception:
            log.debug("%s: force_flush failed", self.integration_name, exc_info=True)
            return False

    def shutdown(self) -> None:
        if not self.enabled or self._provider is None:
            return
        self.flush()
        shutdown = getattr(self._provider, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:  # pragma: no cover - defensive
                log.debug("%s: shutdown failed", self.integration_name, exc_info=True)


def _clean(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """OTel only accepts scalars (and homogeneous sequences) as attributes."""
    out: Dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        if isinstance(value, (str, bool, int, float)):
            out[key] = value
        else:
            out[key] = str(value)
    return out
