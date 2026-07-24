"""
Event system and callback infrastructure for Mangaba AI v3.0

Provides an EventBus that decouples producers (agents, tasks, crews) from
consumers (loggers, tracers, UI), and a BaseCallback ABC for building
custom handlers.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace context
# ---------------------------------------------------------------------------

#: Identifies one logical run. A ``contextvar`` rather than a thread-local so
#: it survives into asyncio tasks; threads need an explicit context copy (see
#: ``Task.aexecute``).
_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mangaba_trace_id", default=None
)


def current_trace_id() -> Optional[str]:
    """Return the trace id of the run in progress, if any."""
    return _current_trace_id.get()


@contextmanager
def start_trace(trace_id: Optional[str] = None) -> Iterator[str]:
    """Group everything emitted inside the block under one trace id.

    Nested calls keep the outermost id, so a crew invoked from inside a flow
    stays part of the same trace instead of starting a second one.

    Example::

        with start_trace() as trace_id:
            crew.kickoff()
    """
    existing = _current_trace_id.get()
    if existing is not None and trace_id is None:
        yield existing
        return

    resolved = trace_id or f"trace_{uuid.uuid4().hex[:16]}"
    token = _current_trace_id.set(resolved)
    try:
        yield resolved
    finally:
        _current_trace_id.reset(token)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    # Agent
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    # LLM
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    LLM_ERROR = "llm_error"
    LLM_RETRY = "llm_retry"
    LLM_STREAM_CHUNK = "llm_stream_chunk"

    # Tools
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"

    # ReAct
    REACT_STEP = "react_step"
    REACT_THOUGHT = "react_thought"
    REACT_ACTION = "react_action"
    REACT_OBSERVATION = "react_observation"

    # Task
    TASK_START = "task_start"
    TASK_END = "task_end"
    TASK_ERROR = "task_error"

    # Crew
    CREW_START = "crew_start"
    CREW_END = "crew_end"
    CREW_ERROR = "crew_error"

    # Human-in-the-loop
    HUMAN_INPUT_REQUEST = "human_input_request"
    HUMAN_INPUT_RECEIVED = "human_input_received"

    # Planning
    PLAN_CREATED = "plan_created"

    # Flow
    FLOW_START = "flow_start"
    FLOW_END = "flow_end"
    FLOW_ERROR = "flow_error"
    FLOW_METHOD_START = "flow_method_start"
    FLOW_METHOD_END = "flow_method_end"
    FLOW_METHOD_ERROR = "flow_method_error"
    FLOW_ROUTE = "flow_route"
    FLOW_STATE_SAVED = "flow_state_saved"
    FLOW_RESUMED = "flow_resumed"

    # Memory
    MEMORY_ADD = "memory_add"
    MEMORY_SEARCH = "memory_search"

    # Guardrails
    GUARDRAIL_PASS = "guardrail_pass"
    GUARDRAIL_FAIL = "guardrail_fail"

    # Generic
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

class Event(BaseModel):
    """Immutable event emitted by framework components."""

    event_type: EventType
    data: Dict[str, Any] = Field(default_factory=dict)
    source_id: str = ""
    source_type: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    parent_event_id: Optional[str] = None
    trace_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Callback ABC
# ---------------------------------------------------------------------------

class BaseCallback(ABC):
    """Abstract base for event handlers."""

    event_filter: Optional[Set[EventType]] = None

    def should_handle(self, event: Event) -> bool:
        if self.event_filter is None:
            return True
        return event.event_type in self.event_filter

    @abstractmethod
    def on_event(self, event: Event) -> None:
        ...


# ---------------------------------------------------------------------------
# Callback manager
# ---------------------------------------------------------------------------

class CallbackManager:
    """Manages a collection of callbacks and dispatches events to them."""

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None) -> None:
        self._callbacks: List[BaseCallback] = list(callbacks or [])

    def add(self, callback: BaseCallback) -> None:
        self._callbacks.append(callback)

    def remove(self, callback: BaseCallback) -> None:
        self._callbacks = [cb for cb in self._callbacks if cb is not callback]

    def emit(self, event: Event) -> None:
        for cb in self._callbacks:
            try:
                if cb.should_handle(event):
                    cb.on_event(event)
            except Exception:
                logger.exception("Callback %s raised an error for event %s", type(cb).__name__, event.event_type)

    @property
    def callbacks(self) -> List[BaseCallback]:
        return list(self._callbacks)


# ---------------------------------------------------------------------------
# Global EventBus singleton
# ---------------------------------------------------------------------------

class EventBus:
    """Process-wide EventBus singleton.

    Components can publish events via ``EventBus.emit(event)`` and register
    handlers via ``EventBus.register(callback_or_fn, event_types)``.
    """

    _manager = CallbackManager()

    @classmethod
    def register(
        cls,
        handler: BaseCallback | Callable[[Event], None],
        event_types: Optional[Set[EventType]] = None,
    ) -> None:
        if isinstance(handler, BaseCallback):
            cls._manager.add(handler)
        else:
            cls._manager.add(_FunctionCallback(handler, event_types))

    @classmethod
    def unregister(cls, handler: BaseCallback) -> None:
        cls._manager.remove(handler)

    @classmethod
    def emit(cls, event: Event) -> None:
        # Stamp the ambient trace so consumers can group a run's events even
        # when they arrive from different threads
        if event.trace_id is None:
            event.trace_id = current_trace_id()
        cls._manager.emit(event)

    @classmethod
    def reset(cls) -> None:
        cls._manager = CallbackManager()

    @classmethod
    def manager(cls) -> CallbackManager:
        return cls._manager


class _FunctionCallback(BaseCallback):
    """Wraps a plain function as a BaseCallback."""

    def __init__(self, fn: Callable[[Event], None], event_filter: Optional[Set[EventType]] = None) -> None:
        self._fn = fn
        self.event_filter = event_filter

    def on_event(self, event: Event) -> None:
        self._fn(event)
