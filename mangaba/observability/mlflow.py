"""
MLflow tracing for Mangaba AI v3.0

Uses MLflow's tracing API (``mlflow.start_span``, MLflow >= 2.14) to record a
crew run as a single trace: ``CHAIN`` spans for crew/task/agent, ``LLM`` spans
for model calls and ``TOOL`` spans for tool executions.

Optional dependency::

    pip install mlflow

Configuration::

    MLFLOW_TRACKING_URI=http://localhost:5000
    MLFLOW_EXPERIMENT_NAME=mangaba            # optional

MLflow keeps its own thread-local span context, so nesting is preserved by
entering/exiting the context managers in event order.

Example::

    from mangaba.observability import MLflowCallback, configure_observability

    configure_observability(MLflowCallback(experiment="mangaba-crews"))
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from mangaba.observability.base import Span, SpanTrackingCallback

log = logging.getLogger(__name__)

#: Env vars that make ``auto_configure_from_env`` enable this integration.
ENV_VARS = ("MLFLOW_TRACKING_URI",)

#: Mangaba span kind -> MLflow ``SpanType``.
_SPAN_TYPES = {
    "crew": "CHAIN",
    "task": "CHAIN",
    "agent": "AGENT",
    "llm": "LLM",
    "tool": "TOOL",
}


class MLflowCallback(SpanTrackingCallback):
    """Record Mangaba spans as MLflow traces.

    Args:
        tracking_uri: Defaults to ``MLFLOW_TRACKING_URI``.
        experiment: Experiment name. Defaults to ``MLFLOW_EXPERIMENT_NAME``.
        autolog: When ``True``, also call ``mlflow.<provider>.autolog()`` for any
            provider SDK MLflow can instrument (best effort, failures ignored).

    No-ops with a warning when ``mlflow`` is missing or too old to expose the
    tracing API.

    Example::

        cb = MLflowCallback(tracking_uri="http://localhost:5000")
        cb.register()
    """

    integration_name = "MLflowCallback"

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment: Optional[str] = None,
        autolog: bool = False,
    ) -> None:
        super().__init__()
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.experiment = experiment or os.getenv("MLFLOW_EXPERIMENT_NAME")
        self._mlflow: Any = None

        try:
            import mlflow  # type: ignore
        except ImportError:
            self._disable("the 'mlflow' package is required. Install with: pip install mlflow")
            return

        if not hasattr(mlflow, "start_span"):
            self._disable("this mlflow version has no tracing API — upgrade with: pip install -U 'mlflow>=2.14'")
            return

        self._mlflow = mlflow
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            if self.experiment:
                mlflow.set_experiment(self.experiment)
            if autolog:
                self._enable_autolog()
        except Exception as exc:
            self._disable(f"could not configure mlflow ({exc})")

    def _enable_autolog(self) -> None:
        for provider in ("openai", "anthropic", "gemini", "litellm"):
            module = getattr(self._mlflow, provider, None)
            fn = getattr(module, "autolog", None) if module is not None else None
            if fn is None:
                continue
            try:
                fn()
            except Exception:  # pragma: no cover - provider SDK may be absent
                log.debug("%s: autolog for %s unavailable", self.integration_name, provider, exc_info=True)

    # ── span hooks ─────────────────────────────────────────────────────

    def _start_span(self, span: Span, parent: Optional[Span]) -> None:
        if self._mlflow is None:
            return
        # ``mlflow.start_span`` is a context manager; entering/exiting it manually
        # keeps MLflow's own parent tracking in sync with our event order.
        manager = self._mlflow.start_span(
            name=span.name,
            span_type=_SPAN_TYPES.get(span.kind, "UNKNOWN"),
            attributes=dict(span.attributes),
        )
        span.context = manager
        span.native = manager.__enter__()

    def _end_span(self, span: Span, status: str, attributes: Dict[str, Any]) -> None:
        native = span.native
        if native is not None:
            setter = getattr(native, "set_attributes", None)
            try:
                if setter is not None:
                    setter(dict(attributes))
                if status == "error":
                    set_status = getattr(native, "set_status", None)
                    if set_status is not None:
                        set_status("ERROR")
            except Exception:  # pragma: no cover - defensive
                log.debug("%s: could not annotate span", self.integration_name, exc_info=True)

        manager = span.context
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:  # pragma: no cover - defensive
                log.debug("%s: could not close span %s", self.integration_name, span.name, exc_info=True)

    def _add_event(self, span: Optional[Span], name: str, attributes: Dict[str, Any]) -> None:
        if span is None or span.native is None:
            return
        setter = getattr(span.native, "set_attribute", None)
        if setter is None:
            return
        setter(f"event.{name}", str(attributes))
