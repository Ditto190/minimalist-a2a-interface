"""
Shared span-tracking machinery for Mangaba AI observability integrations.

Every integration in :mod:`mangaba.observability` subscribes to the framework
:class:`~mangaba.core.events.EventBus` and turns the flat event stream into a
nested span tree.  The framework's events do not carry parent ids, so nesting
is reconstructed with a **per-thread span stack**: a ``*_START`` event pushes a
span, the matching ``*_END``/``*_ERROR`` pops it. A crew run therefore produces
``crew -> task -> agent -> llm|tool`` spans.

Integrations only implement three hooks (:meth:`SpanTrackingCallback._start_span`,
:meth:`SpanTrackingCallback._end_span`, :meth:`SpanTrackingCallback._add_event`)
and must never raise: when their SDK is missing they set ``enabled = False`` and
every hook becomes a no-op.

Example::

    class MyCallback(SpanTrackingCallback):
        def _start_span(self, span, parent):
            return my_sdk.span(span.name, parent=parent.native if parent else None)

        def _end_span(self, span, status, attributes):
            span.native.finish(status=status)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from mangaba.core.events import BaseCallback, Event, EventBus, EventType

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event -> span mapping
# ---------------------------------------------------------------------------

#: ``EventType`` that opens a span, mapped to the span kind.
SPAN_START_EVENTS: Dict[EventType, str] = {
    EventType.CREW_START: "crew",
    EventType.TASK_START: "task",
    EventType.AGENT_START: "agent",
    EventType.LLM_START: "llm",
    EventType.TOOL_START: "tool",
}

#: ``EventType`` that closes a span, mapped to ``(kind, status)``.
SPAN_END_EVENTS: Dict[EventType, tuple] = {
    EventType.CREW_END: ("crew", "ok"),
    EventType.CREW_ERROR: ("crew", "error"),
    EventType.TASK_END: ("task", "ok"),
    EventType.TASK_ERROR: ("task", "error"),
    EventType.AGENT_END: ("agent", "ok"),
    EventType.AGENT_ERROR: ("agent", "error"),
    EventType.LLM_END: ("llm", "ok"),
    EventType.LLM_ERROR: ("llm", "error"),
    EventType.TOOL_END: ("tool", "ok"),
    EventType.TOOL_ERROR: ("tool", "error"),
}

#: Everything else is recorded as a point-in-time event on the current span.
POINT_EVENTS = (
    EventType.REACT_STEP,
    EventType.REACT_THOUGHT,
    EventType.REACT_ACTION,
    EventType.REACT_OBSERVATION,
    EventType.LLM_RETRY,
    EventType.LLM_STREAM_CHUNK,
    EventType.MEMORY_ADD,
    EventType.MEMORY_SEARCH,
    EventType.GUARDRAIL_PASS,
    EventType.GUARDRAIL_FAIL,
    EventType.CUSTOM,
)

#: Keys in ``Event.data`` that carry token usage, mapped to OTel GenAI attribute
#: names (``opentelemetry`` semantic conventions for generative AI).
TOKEN_ATTRIBUTES = {
    "tokens": "gen_ai.usage.total_tokens",
    "total_tokens": "gen_ai.usage.total_tokens",
    "prompt_tokens": "gen_ai.usage.input_tokens",
    "input_tokens": "gen_ai.usage.input_tokens",
    "completion_tokens": "gen_ai.usage.output_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
}

_MAX_ATTR_LEN = 2000


class Span:
    """A span in flight, plus whatever native handle the integration created."""

    __slots__ = ("kind", "name", "start_time", "attributes", "native", "context", "tokens")

    def __init__(self, kind: str, name: str, attributes: Dict[str, Any]) -> None:
        self.kind = kind
        self.name = name
        self.start_time = time.time()
        self.attributes: Dict[str, Any] = attributes
        self.native: Any = None
        self.context: Any = None
        self.tokens: int = 0

    @property
    def duration(self) -> float:
        return time.time() - self.start_time

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Span {self.kind}:{self.name}>"


def flatten_event_data(data: Dict[str, Any], prefix: str = "mangaba") -> Dict[str, Any]:
    """Flatten ``Event.data`` into scalar span attributes.

    Token-usage keys are additionally exported under the OTel GenAI names so
    downstream platforms can aggregate cost without knowing Mangaba's schema.

    Example::

        flatten_event_data({"tokens": 42, "provider": "google"})
        # {'mangaba.tokens': 42, 'gen_ai.usage.total_tokens': 42,
        #  'mangaba.provider': 'google'}
    """
    out: Dict[str, Any] = {}
    for key, value in (data or {}).items():
        attr = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, str):
            out[attr] = value[:_MAX_ATTR_LEN]
        elif isinstance(value, (bool, int, float)):
            out[attr] = value
        else:
            out[attr] = str(value)[:_MAX_ATTR_LEN]

        if key in TOKEN_ATTRIBUTES and isinstance(value, (int, float)):
            out[TOKEN_ATTRIBUTES[key]] = value
    return out


def span_name(kind: str, event: Event) -> str:
    """Human-readable span name derived from the event payload."""
    data = event.data or {}
    if kind == "tool":
        return f"tool.{data.get('tool', 'unknown')}"
    if kind == "llm":
        return f"llm.{data.get('provider', 'call')}"
    if kind == "agent":
        return f"agent.{data.get('role', event.source_id or 'unknown')}"
    if kind == "task":
        return f"task.{event.source_id or 'unknown'}"
    if kind == "crew":
        return f"crew.{event.source_id or 'unknown'}"
    return kind  # pragma: no cover - defensive


# ---------------------------------------------------------------------------
# Callback base
# ---------------------------------------------------------------------------

class SpanTrackingCallback(BaseCallback):
    """Base class that reconstructs a span tree from the flat event stream.

    Subclasses set ``self.enabled`` in ``__init__`` (``False`` when their SDK is
    missing or unconfigured) and implement the three ``_*_span`` hooks. Hook
    exceptions are swallowed and logged — observability must never break a run.
    """

    #: Human name used in log messages.
    integration_name = "observability"

    def __init__(self) -> None:
        super().__init__()
        self.enabled: bool = True
        self._local = threading.local()

    # ── stack helpers ──────────────────────────────────────────────────

    @property
    def _stack(self) -> List[Span]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def current_span(self) -> Optional[Span]:
        """Innermost open span on the calling thread, if any."""
        stack = self._stack
        return stack[-1] if stack else None

    # ── event dispatch ─────────────────────────────────────────────────

    def on_event(self, event: Event) -> None:
        if not self.enabled:
            return
        try:
            kind = SPAN_START_EVENTS.get(event.event_type)
            if kind is not None:
                self._open(kind, event)
                return

            end = SPAN_END_EVENTS.get(event.event_type)
            if end is not None:
                self._close(end[0], end[1], event)
                return

            self._point(event)
        except Exception:
            log.warning("%s: failed to handle %s", self.integration_name, event.event_type, exc_info=True)

    def _open(self, kind: str, event: Event) -> None:
        span = Span(kind, span_name(kind, event), flatten_event_data(event.data))
        span.attributes["mangaba.kind"] = kind
        if event.source_id:
            span.attributes["mangaba.source_id"] = event.source_id
        if event.source_type:
            span.attributes["mangaba.source_type"] = event.source_type
        parent = self.current_span()
        try:
            self._start_span(span, parent)
        except Exception:
            log.warning("%s: could not start span %s", self.integration_name, span.name, exc_info=True)
        self._stack.append(span)

    def _close(self, kind: str, status: str, event: Event) -> None:
        stack = self._stack
        index = None
        for i in range(len(stack) - 1, -1, -1):
            if stack[i].kind == kind:
                index = i
                break
        if index is None:
            log.debug("%s: %s end without a matching open span", self.integration_name, kind)
            return

        span = stack[index]
        del stack[index:]  # anything opened inside it is closed implicitly

        attributes = flatten_event_data(event.data)
        tokens = attributes.get("gen_ai.usage.total_tokens")
        if isinstance(tokens, (int, float)):
            span.tokens += int(tokens)
        # Roll token usage up one level; repeated on each close, a crew span
        # therefore ends up carrying the whole run's usage.
        if stack:
            stack[-1].tokens += span.tokens
        if span.tokens:
            attributes["mangaba.tokens.total"] = span.tokens
        attributes["mangaba.duration_seconds"] = round(span.duration, 6)

        try:
            self._end_span(span, status, attributes)
        except Exception:
            log.warning("%s: could not end span %s", self.integration_name, span.name, exc_info=True)

    def _point(self, event: Event) -> None:
        if event.event_type not in POINT_EVENTS:
            return
        try:
            self._add_event(self.current_span(), event.event_type.value, flatten_event_data(event.data))
        except Exception:
            log.warning("%s: could not record event %s", self.integration_name, event.event_type, exc_info=True)

    # ── hooks for subclasses ───────────────────────────────────────────

    def _start_span(self, span: Span, parent: Optional[Span]) -> None:
        """Create the native span, storing handles on ``span.native``/``span.context``."""
        raise NotImplementedError

    def _end_span(self, span: Span, status: str, attributes: Dict[str, Any]) -> None:
        """Finish the native span with ``status`` (``"ok"``/``"error"``)."""
        raise NotImplementedError

    def _add_event(self, span: Optional[Span], name: str, attributes: Dict[str, Any]) -> None:
        """Record a point-in-time event. Default: ignore."""
        return None

    # ── lifecycle ──────────────────────────────────────────────────────

    def register(self) -> "SpanTrackingCallback":
        """Subscribe to the global :class:`EventBus` and return ``self``."""
        EventBus.register(self)
        return self

    def unregister(self) -> None:
        """Detach from the global :class:`EventBus`."""
        EventBus.unregister(self)

    def flush(self, timeout: float = 5.0) -> bool:
        """Push buffered telemetry to the backend. Default: nothing to do."""
        return True

    def shutdown(self) -> None:
        """Release SDK resources. Default: just flush."""
        self.flush()

    def _disable(self, reason: str) -> None:
        """Turn this integration into a harmless no-op, explaining why."""
        self.enabled = False
        log.warning("%s disabled: %s", self.integration_name, reason)
