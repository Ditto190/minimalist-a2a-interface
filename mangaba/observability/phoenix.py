"""
Arize Phoenix tracing for Mangaba AI v3.0

Phoenix ingests OpenTelemetry, so this integration reuses
:class:`~mangaba.observability.otel.OpenTelemetryCallback` and only swaps in a
tracer provider built by ``phoenix.otel.register()``. Spans additionally carry
OpenInference ``openinference.span.kind`` attributes so the Phoenix UI renders
them as agents / LLM calls / tools instead of anonymous spans.

Optional dependency::

    pip install arize-phoenix-otel        # add 'arize-phoenix' to run the UI locally

Configuration::

    PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
    PHOENIX_API_KEY=...                   # Phoenix Cloud
    PHOENIX_CLIENT_HEADERS="api_key=..."  # alternative header form

Example::

    from mangaba.observability import PhoenixCallback, configure_observability

    configure_observability(PhoenixCallback(project_name="mangaba-dev"))
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from mangaba.observability.base import Span
from mangaba.observability.otel import OpenTelemetryCallback

log = logging.getLogger(__name__)

#: Env vars that make ``auto_configure_from_env`` enable this integration.
ENV_VARS = ("PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_API_KEY", "PHOENIX_CLIENT_HEADERS")

#: Mangaba span kind -> OpenInference span kind (what the Phoenix UI groups by).
_OPENINFERENCE_KINDS = {
    "crew": "CHAIN",
    "task": "CHAIN",
    "agent": "AGENT",
    "llm": "LLM",
    "tool": "TOOL",
}


class PhoenixCallback(OpenTelemetryCallback):
    """Export Mangaba spans to Arize Phoenix over OTLP.

    Args:
        project_name: Phoenix project. Defaults to ``PHOENIX_PROJECT_NAME``
            then ``"mangaba"``.
        endpoint: Collector endpoint. Defaults to ``PHOENIX_COLLECTOR_ENDPOINT``.
        headers: Extra OTLP headers (``PHOENIX_API_KEY`` is picked up by the
            Phoenix SDK itself).

    Falls back to a warning + no-op when ``arize-phoenix-otel`` is absent. If
    only the base OTel SDK is installed you can still send data to Phoenix by
    pointing :class:`OpenTelemetryCallback` at ``http://localhost:6006/v1/traces``.

    Example::

        cb = PhoenixCallback(project_name="mangaba-dev")
        cb.register()
    """

    integration_name = "PhoenixCallback"

    def __init__(
        self,
        project_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.project_name = project_name or os.getenv("PHOENIX_PROJECT_NAME") or "mangaba"
        provider = self._build_phoenix_provider(
            self.project_name,
            endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
        )
        super().__init__(
            service_name=self.project_name,
            endpoint=endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
            headers=headers,
            tracer_provider=provider,
            install_provider=False,
        )
        if provider is None and self.enabled:
            self._disable(
                "the 'arize-phoenix-otel' package is required. Install with: pip install arize-phoenix-otel"
            )

    @staticmethod
    def _build_phoenix_provider(project_name: str, endpoint: Optional[str]) -> Any:
        try:
            from phoenix.otel import register  # type: ignore
        except ImportError:
            return None
        try:
            kwargs: Dict[str, Any] = {"project_name": project_name, "set_global_tracer_provider": False}
            if endpoint:
                kwargs["endpoint"] = endpoint
            return register(**kwargs)
        except TypeError:  # pragma: no cover - older signature
            try:
                return register(project_name=project_name)
            except Exception as exc:
                log.warning("PhoenixCallback: phoenix.otel.register failed (%s)", exc)
                return None
        except Exception as exc:
            log.warning("PhoenixCallback: phoenix.otel.register failed (%s)", exc)
            return None

    def _start_span(self, span: Span, parent: Optional[Span]) -> None:
        span.attributes["openinference.span.kind"] = _OPENINFERENCE_KINDS.get(span.kind, "CHAIN")
        super()._start_span(span, parent)
