"""
External observability integrations for Mangaba AI v3.0

Each integration subscribes to the global
:class:`~mangaba.core.events.EventBus` and forwards the event stream as a
nested span tree (``crew -> task -> agent -> llm|tool``), attaching token usage
where the framework's events carry it.

Every integration is an **optional dependency**: when the SDK is missing or
unconfigured the callback logs a warning, sets ``enabled = False`` and becomes
a no-op. Importing this package never requires any of them.

============================  ===========================================================
Integration                   Optional dependency
============================  ===========================================================
``OpenTelemetryCallback``     ``opentelemetry-api opentelemetry-sdk``
                              ``opentelemetry-exporter-otlp-proto-http`` (to export)
``LangfuseCallback``          ``langfuse``
``MLflowCallback``            ``mlflow>=2.14``
``PhoenixCallback``           ``arize-phoenix-otel`` (plus ``arize-phoenix`` for the UI)
============================  ===========================================================

Example::

    from mangaba.observability import OpenTelemetryCallback, configure_observability

    configure_observability(OpenTelemetryCallback(service_name="my-crew"))
    crew.kickoff()

Or let the environment decide::

    from mangaba.observability import auto_configure_from_env

    active = auto_configure_from_env()   # [] when nothing is configured
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from mangaba.core.events import EventBus
from mangaba.observability.base import (
    POINT_EVENTS,
    SPAN_END_EVENTS,
    SPAN_START_EVENTS,
    Span,
    SpanTrackingCallback,
)
from mangaba.observability.langfuse import LangfuseCallback
from mangaba.observability.mlflow import MLflowCallback
from mangaba.observability.otel import OpenTelemetryCallback
from mangaba.observability.phoenix import PhoenixCallback

log = logging.getLogger(__name__)

__all__ = [
    "Span",
    "SpanTrackingCallback",
    "SPAN_START_EVENTS",
    "SPAN_END_EVENTS",
    "POINT_EVENTS",
    "OpenTelemetryCallback",
    "LangfuseCallback",
    "MLflowCallback",
    "PhoenixCallback",
    "configure_observability",
    "auto_configure_from_env",
    "flush_observability",
    "shutdown_observability",
    "active_callbacks",
]


# Callbacks registered through this module, so they can be flushed/shut down.
_ACTIVE: List[SpanTrackingCallback] = []

#: Integration -> env vars that switch it on in ``auto_configure_from_env``.
_ENV_MATRIX = (
    (OpenTelemetryCallback, ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")),
    (LangfuseCallback, ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")),
    (MLflowCallback, ("MLFLOW_TRACKING_URI",)),
    (PhoenixCallback, ("PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_API_KEY", "PHOENIX_CLIENT_HEADERS")),
)


def configure_observability(*callbacks: Any, skip_disabled: bool = True) -> List[SpanTrackingCallback]:
    """Register observability callbacks on the global :class:`EventBus`.

    Args:
        *callbacks: Instances (usually :class:`SpanTrackingCallback` subclasses).
            ``None`` entries and iterables of callbacks are accepted too.
        skip_disabled: When ``True`` (default), callbacks whose SDK is missing
            are not registered at all — they would only no-op anyway.

    Returns:
        The callbacks that were actually registered.

    Example::

        configure_observability(
            OpenTelemetryCallback(service_name="my-crew"),
            LangfuseCallback(),
        )
    """
    registered: List[SpanTrackingCallback] = []
    for item in _iter_callbacks(callbacks):
        if skip_disabled and not getattr(item, "enabled", True):
            log.info("Skipping disabled observability callback %s", type(item).__name__)
            continue
        EventBus.register(item)
        _ACTIVE.append(item)
        registered.append(item)
        log.info("Observability: %s registered", type(item).__name__)
    return registered


def auto_configure_from_env(env: Optional[dict] = None) -> List[SpanTrackingCallback]:
    """Enable whichever integrations have their environment variables set.

    Nothing configured means nothing registered — this is always safe to call
    at import time of an application, and never raises.

    ==========================  ==================================================
    Integration                 Triggered by
    ==========================  ==================================================
    ``OpenTelemetryCallback``   ``OTEL_EXPORTER_OTLP_ENDPOINT`` /
                                ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``
    ``LangfuseCallback``        ``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY``
    ``MLflowCallback``          ``MLFLOW_TRACKING_URI``
    ``PhoenixCallback``         ``PHOENIX_COLLECTOR_ENDPOINT`` / ``PHOENIX_API_KEY``
                                / ``PHOENIX_CLIENT_HEADERS``
    ==========================  ==================================================

    Example::

        active = auto_configure_from_env()
        print([type(cb).__name__ for cb in active])
    """
    source = os.environ if env is None else env
    candidates: List[SpanTrackingCallback] = []

    for factory, env_vars in _ENV_MATRIX:
        if factory is LangfuseCallback:
            wanted = all(source.get(v) for v in env_vars)
        else:
            wanted = any(source.get(v) for v in env_vars)
        if not wanted:
            continue
        try:
            callback = factory()
        except Exception:
            log.warning("Observability: %s could not be constructed", factory.__name__, exc_info=True)
            continue
        if not callback.enabled:
            log.warning("Observability: %s is configured but its SDK is unavailable", factory.__name__)
            continue
        candidates.append(callback)

    if not candidates:
        log.debug("Observability: no integration environment variables found")
    return configure_observability(*candidates)


def active_callbacks() -> List[SpanTrackingCallback]:
    """Callbacks registered through :func:`configure_observability`."""
    return list(_ACTIVE)


def flush_observability(timeout: float = 5.0) -> bool:
    """Flush every registered integration. Returns ``True`` if all succeeded."""
    ok = True
    for cb in list(_ACTIVE):
        try:
            ok = bool(cb.flush(timeout)) and ok
        except Exception:
            log.warning("Observability: flush failed for %s", type(cb).__name__, exc_info=True)
            ok = False
    return ok


def shutdown_observability() -> None:
    """Flush, shut down and unregister every integration."""
    for cb in list(_ACTIVE):
        try:
            cb.shutdown()
        except Exception:
            log.warning("Observability: shutdown failed for %s", type(cb).__name__, exc_info=True)
        try:
            cb.unregister()
        except Exception:  # pragma: no cover - defensive
            log.debug("Observability: unregister failed for %s", type(cb).__name__, exc_info=True)
    _ACTIVE.clear()


def _iter_callbacks(items: Any) -> List[Any]:
    """Flatten the varargs of :func:`configure_observability`."""
    out: List[Any] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            out.extend(_iter_callbacks(item))
            continue
        out.append(item)
    return out
