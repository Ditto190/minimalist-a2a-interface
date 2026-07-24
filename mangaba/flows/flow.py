"""
Flows — event-driven orchestration for Mangaba AI v3.0

A Flow is a class whose methods are wired together by decorators instead of by
an explicit list of steps: ``@start`` marks entry points, ``@listen`` reacts to
the result of another method, and ``@router`` returns a label that selects which
listeners run next. Combinators ``and_`` / ``or_`` express join semantics, and
``@persist`` checkpoints the flow state to SQLite after every step so a run can
be resumed in a different process.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import textwrap
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field

from mangaba.core.events import BaseCallback, Event, EventBus, EventType
from mangaba.core.exceptions import FlowCycleError, FlowError
from mangaba.core.types import TokenUsage
from mangaba.flows.persistence import BaseFlowStore, FlowRecord, get_default_store
from mangaba.flows.state import (
    StateLike,
    create_state,
    get_state_id,
    new_state_id,
    set_state_id,
    state_from_dict,
    state_to_dict,
    update_state,
)

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Marker attributes written onto decorated methods.
_START_ATTR = "__mangaba_flow_start__"
_LISTEN_ATTR = "__mangaba_flow_listen__"
_ROUTER_ATTR = "__mangaba_flow_router__"
_PERSIST_ATTR = "__mangaba_flow_persist__"
_PERSIST_STORE_ATTR = "__mangaba_flow_persist_store__"

_MARKERS = (_START_ATTR, _LISTEN_ATTR, _ROUTER_ATTR, _PERSIST_ATTR)

_UNSET = object()


# ---------------------------------------------------------------------------
# Trigger conditions
# ---------------------------------------------------------------------------

class FlowCondition:
    """A boolean combination of triggers (method names or router labels).

    Built through :func:`and_` / :func:`or_` rather than directly. Conditions
    nest freely, so ``and_("a", or_("b", "c"))`` is valid.

    Example::

        @listen(and_(fetch_users, fetch_orders))
        def join(self, _):
            ...
    """

    __slots__ = ("kind", "triggers")

    def __init__(self, kind: str, triggers: Sequence[Any]) -> None:
        if kind not in ("AND", "OR"):
            raise FlowError(f"Unknown condition kind '{kind}' (expected 'AND' or 'OR')")
        if not triggers:
            raise FlowError(f"{kind.lower()}_() requires at least one trigger")
        self.kind = kind
        self.triggers: Tuple[Union[str, FlowCondition], ...] = tuple(
            _as_trigger(t) for t in triggers
        )

    # ── evaluation ─────────────────────────────────────────────────────

    def names(self) -> List[str]:
        """Return every leaf trigger name, depth-first, without duplicates."""
        out: List[str] = []
        for trigger in self.triggers:
            for name in ([trigger] if isinstance(trigger, str) else trigger.names()):
                if name not in out:
                    out.append(name)
        return out

    def involves(self, name: str) -> bool:
        """Whether ``name`` participates anywhere in this condition."""
        return name in self.names()

    def is_satisfied(self, fired: Set[str]) -> bool:
        """Evaluate the condition against the set of triggers seen so far."""
        results = [
            (trigger in fired) if isinstance(trigger, str) else trigger.is_satisfied(fired)
            for trigger in self.triggers
        ]
        return all(results) if self.kind == "AND" else any(results)

    def __repr__(self) -> str:
        joiner = " AND " if self.kind == "AND" else " OR "
        return "(" + joiner.join(repr(t) if isinstance(t, FlowCondition) else t for t in self.triggers) + ")"


def _as_trigger(obj: Any) -> Union[str, FlowCondition]:
    """Normalise a trigger spec (string, method object, condition) to a name."""
    if isinstance(obj, FlowCondition):
        return obj
    if isinstance(obj, str):
        if not obj.strip():
            raise FlowError("Trigger name cannot be empty")
        return obj.strip()
    if isinstance(obj, (staticmethod, classmethod)):
        return _as_trigger(obj.__func__)
    name = getattr(obj, "__name__", None)
    if callable(obj) and name:
        return name
    raise FlowError(
        f"Invalid trigger {obj!r} — expected a method, a method name, or and_()/or_()"
    )


def _as_condition(obj: Any) -> FlowCondition:
    """Wrap a single trigger into an OR condition (identity for conditions)."""
    if isinstance(obj, FlowCondition):
        return obj
    return FlowCondition("OR", [obj])


def and_(*triggers: Any) -> FlowCondition:
    """Fire only after **all** the given triggers have completed.

    Args:
        *triggers: Method objects, method names, or nested conditions.

    Returns:
        A :class:`FlowCondition` usable inside ``@listen`` / ``@router``.

    Example::

        @listen(and_(gather_metrics, gather_reviews))
        def summarise(self, result):
            return f"joined: {result}"
    """
    return FlowCondition("AND", triggers)


def or_(*triggers: Any) -> FlowCondition:
    """Fire as soon as **any** of the given triggers completes.

    Args:
        *triggers: Method objects, method names, or nested conditions.

    Returns:
        A :class:`FlowCondition` usable inside ``@listen`` / ``@router``.

    Example::

        @listen(or_(from_cache, from_api))
        def render(self, payload):
            return payload.upper()
    """
    return FlowCondition("OR", triggers)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def _is_flow_method(obj: Any) -> bool:
    return any(hasattr(obj, marker) for marker in _MARKERS)


def start(condition: Any = None) -> Callable:
    """Mark a method as a flow entry point.

    Every ``@start`` method runs when the flow is kicked off, in definition
    order. Passing a condition additionally registers the method as a listener,
    which is how a flow loops back on itself (loop-backs into a start method are
    the only cycles the graph validator allows).

    Args:
        condition: Optional trigger, method name, or ``and_()`` / ``or_()``
            combination that re-triggers this entry point.

    Returns:
        The decorator (or the decorated function when used bare as ``@start``).

    Example::

        class GreetFlow(Flow):
            @start()
            def begin(self):
                return "hello"

            @start(and_("begin", "retry"))
            def begin_again(self):
                return "hello again"
    """
    if condition is not None and callable(condition) and not _is_flow_method(condition):
        # Bare usage: @start (no parentheses)
        setattr(condition, _START_ATTR, True)
        return condition

    cond = _as_condition(condition) if condition is not None else None

    def decorator(fn: Callable) -> Callable:
        setattr(fn, _START_ATTR, True)
        if cond is not None:
            setattr(fn, _LISTEN_ATTR, cond)
        return fn

    return decorator


def listen(trigger: Any) -> Callable:
    """Run this method when ``trigger`` completes.

    The decorated method receives the trigger's return value as its single
    argument when its signature accepts one; otherwise it is called with no
    arguments.

    Args:
        trigger: A method object, a method name, a router label, or an
            ``and_()`` / ``or_()`` combination.

    Returns:
        The decorator.

    Example::

        class ReviewFlow(Flow):
            @start()
            def draft(self):
                return "draft text"

            @listen(draft)
            def review(self, text):
                return f"reviewed: {text}"
    """
    cond = _as_condition(trigger)

    def decorator(fn: Callable) -> Callable:
        setattr(fn, _LISTEN_ATTR, cond)
        return fn

    return decorator


def router(trigger: Any) -> Callable:
    """Run like a listener, but route on the returned label.

    A router's return value is a string label; listeners registered with
    ``@listen("label")`` fire next. Returning ``None`` (or a non-string) ends
    that branch. The router's own method name is *not* a trigger — route by
    label.

    Args:
        trigger: Same accepted forms as :func:`listen`.

    Returns:
        The decorator.

    Example::

        class TriageFlow(Flow):
            @start()
            def score(self):
                return 0.9

            @router(score)
            def branch(self, value):
                return "high" if value > 0.5 else "low"

            @listen("high")
            def escalate(self):
                return "escalated"
    """
    cond = _as_condition(trigger)

    def decorator(fn: Callable) -> Callable:
        setattr(fn, _ROUTER_ATTR, True)
        setattr(fn, _LISTEN_ATTR, cond)
        return fn

    return decorator


def persist(
    target: Any = None,
    *,
    store: Optional[BaseFlowStore] = None,
    db_path: Optional[str] = None,
) -> Any:
    """Checkpoint flow state to SQLite — usable on a class or a method.

    On a class, state is saved after **every** completed step. On a method,
    state is saved only after that method runs. Either way the checkpoint keys
    on the flow id, so ``flow.resume(flow_id)`` restores the state and skips
    steps that already completed.

    Args:
        target: The class or method being decorated (supplied automatically).
        store: Explicit store instance; defaults to the shared SQLite store.
        db_path: Convenience alternative to ``store`` — path of the database.

    Returns:
        The decorated class/method, or the decorator when called with options.

    Example::

        @persist
        class OrderFlow(Flow[OrderState]):
            @start()
            def charge(self):
                self.state.charged = True
                return "charged"
    """

    def apply(obj: Any) -> Any:
        setattr(obj, _PERSIST_ATTR, True)
        if store is not None:
            setattr(obj, _PERSIST_STORE_ATTR, store)
        elif db_path is not None:
            from mangaba.flows.persistence import SQLiteFlowStore

            setattr(obj, _PERSIST_STORE_ATTR, SQLiteFlowStore(db_path))
        return obj

    if target is None:
        return apply
    return apply(target)


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------

class FlowNode(BaseModel):
    """A node of the flow graph — a method, or a router label."""

    name: str
    kind: str = "listener"  # start | listener | router | label
    condition: str = ""
    is_async: bool = False
    doc: str = ""


class FlowEdge(BaseModel):
    """A directed edge of the flow graph."""

    source: str
    target: str
    label: str = ""
    kind: str = "listen"  # listen | and | or | route


class FlowGraph(BaseModel):
    """Static structure of a flow, used for validation and plotting."""

    name: str = "Flow"
    nodes: List[FlowNode] = Field(default_factory=list)
    edges: List[FlowEdge] = Field(default_factory=list)

    def node(self, name: str) -> Optional[FlowNode]:
        for candidate in self.nodes:
            if candidate.name == name:
                return candidate
        return None


def _router_labels(fn: Callable) -> Set[str]:
    """Best-effort extraction of the string labels a router can return.

    Parses the method source and collects every string constant appearing in a
    ``return`` expression, so plain returns and ternaries alike are picked up.
    Labels built at runtime cannot be detected — routers whose labels stay
    unknown are wired to every label node in the plot instead.
    """
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):  # pragma: no cover
        return set()

    labels: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        for inner in ast.walk(node.value):
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str) and inner.value:
                labels.add(inner.value)
    return labels


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

class _UsageCollector(BaseCallback):
    """Counts tokens reported by ``LLM_END`` events while a flow is running."""

    event_filter = {EventType.LLM_END}

    def __init__(self) -> None:
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def reset(self) -> None:
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def on_event(self, event: Event) -> None:
        for field, key in (
            ("total_tokens", "tokens"),
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
        ):
            try:
                setattr(self, field, getattr(self, field) + int(event.data.get(key) or 0))
            except (TypeError, ValueError):  # pragma: no cover
                pass
        self.calls += 1


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

class Flow(Generic[T]):
    """Event-driven orchestration of decorated methods.

    Subclass ``Flow`` and wire methods with :func:`start`, :func:`listen` and
    :func:`router`. State is a plain dict unless a Pydantic model is declared,
    either as ``Flow[MyState]`` or as a ``state_model = MyState`` class
    attribute; in both cases an ``id`` field is guaranteed.

    Args:
        initial_state: Values merged into the freshly created state.
        flow_id: Identifier used for checkpointing (defaults to the state id).
        state_model: Overrides the class-level structured state model.
        persistence: Store used by ``@persist`` / ``resume`` / ``fork``.
        persist_state: Force checkpointing on or off, ignoring ``@persist``.
        verbose: Log every executed step at INFO level.
        max_steps: Safety valve for looping flows (default 1000 executions).

    Example::

        class ArticleState(FlowState):
            topic: str = ""
            draft: str = ""

        class ArticleFlow(Flow[ArticleState]):
            @start()
            def outline(self):
                return f"outline for {self.state.topic}"

            @listen(outline)
            def write(self, outline):
                self.state.draft = f"{outline} -> full text"
                return self.state.draft

        flow = ArticleFlow()
        result = flow.kickoff(inputs={"topic": "mangaba"})
        flow.plot("article_flow.html")
    """

    #: Optional structured-state model (also inferred from ``Flow[MyState]``).
    state_model: Optional[Type[BaseModel]] = None

    #: Default cap on executed steps, guarding runaway loop-back flows.
    max_steps: int = 1000

    # ── class construction ─────────────────────────────────────────────

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("state_model") is None:
            inferred = _infer_state_model(cls)
            if inferred is not None:
                cls.state_model = inferred

    def __init__(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        *,
        flow_id: Optional[str] = None,
        state_model: Optional[Type[BaseModel]] = None,
        persistence: Optional[BaseFlowStore] = None,
        persist_state: Optional[bool] = None,
        verbose: bool = False,
        max_steps: Optional[int] = None,
    ) -> None:
        self._state_model: Optional[Type[BaseModel]] = state_model or type(self).state_model
        self.state: StateLike = create_state(self._state_model, initial_state)

        self.flow_id: str = flow_id or get_state_id(self.state) or new_state_id()
        set_state_id(self.state, self.flow_id)

        self.verbose = verbose
        if max_steps is not None:
            self.max_steps = max_steps

        # ── persistence ──────────────────────────────────────────────
        class_persist = bool(getattr(type(self), _PERSIST_ATTR, False))
        self._persist_all: bool = class_persist if persist_state is None else bool(persist_state)
        self._store: Optional[BaseFlowStore] = persistence or getattr(type(self), _PERSIST_STORE_ATTR, None)

        # ── graph ────────────────────────────────────────────────────
        self._start_methods: List[str] = []
        self._listeners: Dict[str, FlowCondition] = {}
        self._routers: Set[str] = set()
        self._persist_methods: Set[str] = set()
        self._method_stores: Dict[str, BaseFlowStore] = {}
        self._arity_cache: Dict[str, bool] = {}
        self._scan_methods()
        self._graph: FlowGraph = self._build_graph()
        self._detect_cycles(self._graph)

        # ── runtime ──────────────────────────────────────────────────
        self._method_outputs: Dict[str, Any] = {}
        self._completed_steps: List[str] = []
        self._executed: List[str] = []
        self._and_progress: Dict[str, Set[str]] = {}
        self._skip_once: Set[str] = set()
        self._restored: Optional[FlowRecord] = None
        self._created_at: str = datetime.now().isoformat()
        self._usage = _UsageCollector()
        self._usage_sources: List[Any] = []
        self._last_duration: float = 0.0

        if self.verbose:
            log.info(
                "Flow %s (%s): %d start(s), %d listener(s), %d router(s)",
                type(self).__name__, self.flow_id,
                len(self._start_methods), len(self._listeners), len(self._routers),
            )

    # ── public API ─────────────────────────────────────────────────────

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Run the flow to completion and return the last leaf method's result.

        Args:
            inputs: Values merged into the flow state and passed to every
                ``@start`` method that accepts an argument.

        Returns:
            The return value of the last executed method that had no listener.

        Raises:
            FlowError: If the flow has no entry point or exceeds ``max_steps``.

        Example::

            result = MyFlow().kickoff({"topic": "AI"})
        """
        return _run_sync(self.kickoff_async(inputs))

    async def kickoff_async(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Asynchronous version of :meth:`kickoff` (methods may be sync or async)."""
        return await self._run(inputs, resuming=False)

    def resume(self, flow_id: Optional[str] = None, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Restore a checkpoint and continue, skipping completed steps.

        Args:
            flow_id: Checkpoint to restore (defaults to this flow's id).
            inputs: Extra values merged into the restored state.

        Returns:
            The final result, exactly as :meth:`kickoff` returns it.

        Raises:
            FlowPersistenceError: If no checkpoint exists for ``flow_id``.

        Example::

            flow = MyFlow(persistence=store)
            flow.resume("6f1c...")
        """
        return _run_sync(self.resume_async(flow_id, inputs))

    async def resume_async(
        self,
        flow_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Asynchronous version of :meth:`resume`."""
        self.load_state(flow_id or self.flow_id)
        return await self._run(inputs, resuming=True)

    def fork(
        self,
        flow_id: Optional[str] = None,
        new_id: Optional[str] = None,
        include_completed_steps: bool = True,
    ) -> Flow:
        """Branch a checkpoint into a new flow instance with a new id.

        Args:
            flow_id: Checkpoint to branch from (defaults to this flow's id).
            new_id: Identifier of the branch (random when omitted).
            include_completed_steps: Carry over the completed-step list so the
                branch continues instead of replaying from the start.

        Returns:
            A new instance of the same flow class bound to the forked state.

        Example::

            branch = flow.fork()
            branch.resume()
        """
        store = self._require_store()
        record = store.fork(flow_id or self.flow_id, new_id, include_completed_steps)

        kwargs = dict(
            flow_id=record.flow_id,
            state_model=self._state_model,
            persistence=store,
            persist_state=self._persist_all,
            verbose=self.verbose,
            max_steps=self.max_steps,
        )
        try:
            clone = type(self)(**kwargs)  # type: ignore[call-arg]
        except TypeError:
            # Subclass overrides __init__ with an incompatible signature.
            clone = type(self).__new__(type(self))
            Flow.__init__(clone, **kwargs)
        clone.state = state_from_dict(record.state, self._state_model)
        set_state_id(clone.state, record.flow_id)
        clone._restored = record
        log.info("Flow %s forked into %s", self.flow_id, record.flow_id)
        return clone

    def save_state(self) -> FlowRecord:
        """Write the current state and progress to the store and return the record."""
        return self._checkpoint(force=True)

    def load_state(self, flow_id: Optional[str] = None) -> FlowRecord:
        """Load a checkpoint into this instance without running anything.

        Args:
            flow_id: Checkpoint identifier (defaults to this flow's id).

        Returns:
            The loaded :class:`FlowRecord`.

        Raises:
            FlowPersistenceError: If the checkpoint does not exist.
        """
        record = self._require_store().resume(flow_id or self.flow_id)
        self.flow_id = record.flow_id
        self.state = state_from_dict(record.state, self._state_model)
        set_state_id(self.state, record.flow_id)
        self._restored = record

        EventBus.emit(Event(
            event_type=EventType.FLOW_RESUMED,
            source_id=self.flow_id,
            source_type=type(self).__name__,
            data={"completed_steps": list(record.completed_steps)},
        ))
        return record

    def plot(self, filename: str = "flow.html") -> str:
        """Write a self-contained HTML/SVG rendering of the method graph.

        Args:
            filename: Destination path (``.html`` appended when missing).

        Returns:
            The absolute path of the written file.

        Example::

            path = MyFlow().plot("my_flow.html")
        """
        from mangaba.flows.visualization import plot_flow

        return plot_flow(self, filename)

    def register_usage_source(self, source: Any) -> None:
        """Register an Agent/Crew/LLM client so its tokens count towards metrics.

        Only needed for objects that a step creates locally and does not store
        on ``self`` or on the flow state.
        """
        self._usage_sources.append(source)

    # ── introspection ──────────────────────────────────────────────────

    @property
    def graph(self) -> FlowGraph:
        """The static :class:`FlowGraph` of this flow."""
        return self._graph

    @property
    def method_outputs(self) -> Dict[str, Any]:
        """Mapping of method name to its most recent return value."""
        return dict(self._method_outputs)

    @property
    def completed_steps(self) -> List[str]:
        """Names of every method that completed at least once."""
        return list(self._completed_steps)

    @property
    def executed_methods(self) -> List[str]:
        """Execution order of the last run, including repeated executions."""
        return list(self._executed)

    @property
    def duration(self) -> float:
        """Wall-clock seconds taken by the last run."""
        return self._last_duration

    @property
    def usage_metrics(self) -> TokenUsage:
        """Aggregated token usage of everything the flow invoked.

        Usage is collected from every ``LLMClient`` reachable from the flow —
        clients held on the instance or the state, agents, crews, and anything
        passed to :meth:`register_usage_source`. When no client is reachable
        (for example a crew created and discarded inside a step) the property
        falls back to the totals reported by ``LLM_END`` events observed while
        the flow was running.

        Limitation: events emitted by unrelated components running concurrently
        in the same process are also counted by the fallback, since it observes
        the process-wide event bus.
        """
        total = TokenUsage()
        for client in self._discover_llm_clients():
            usage = getattr(client, "total_usage", None)
            if isinstance(usage, TokenUsage):
                total.prompt_tokens += usage.prompt_tokens
                total.completion_tokens += usage.completion_tokens
                total.total_tokens += usage.total_tokens

        if total.total_tokens == 0 and self._usage.total_tokens:
            total.total_tokens = self._usage.total_tokens
            total.prompt_tokens = self._usage.prompt_tokens
            total.completion_tokens = self._usage.completion_tokens
        return total

    # ── execution engine ───────────────────────────────────────────────

    async def _run(self, inputs: Optional[Dict[str, Any]], resuming: bool) -> Any:
        if not self._start_methods:
            raise FlowError(
                f"Flow '{type(self).__name__}' has no @start() method — nothing to run"
            )

        self._prepare_runtime(resuming)
        if inputs:
            update_state(self.state, dict(inputs))

        started = time.monotonic()
        self._usage.reset()
        EventBus.register(self._usage)

        EventBus.emit(Event(
            event_type=EventType.FLOW_START,
            source_id=self.flow_id,
            source_type=type(self).__name__,
            data={
                "starts": list(self._start_methods),
                "listeners": len(self._listeners),
                "resuming": resuming,
            },
        ))

        payload: Dict[str, Any] = dict(inputs or {})
        queue: deque = deque((name, payload) for name in self._start_methods)
        steps = 0
        final_result: Any = _UNSET
        leaf_result: Any = _UNSET

        try:
            while queue:
                name, argument = queue.popleft()
                steps += 1
                if steps > self.max_steps:
                    raise FlowError(
                        f"Flow '{type(self).__name__}' exceeded max_steps={self.max_steps} — "
                        "a loop-back @start condition is probably never settling"
                    )

                result = await self._execute_method(name, argument)
                final_result = result

                trigger = self._resolve_trigger(name, result)
                triggered = self._collect_listeners(trigger) if trigger else []
                queue.extend((listener, result) for listener in triggered)

                if not triggered and name not in self._routers:
                    leaf_result = result

            outcome = leaf_result if leaf_result is not _UNSET else (
                final_result if final_result is not _UNSET else None
            )
            self._last_duration = time.monotonic() - started

            EventBus.emit(Event(
                event_type=EventType.FLOW_END,
                source_id=self.flow_id,
                source_type=type(self).__name__,
                data={
                    "duration": self._last_duration,
                    "steps": len(self._executed),
                    "order": list(self._executed),
                },
            ))
            return outcome

        except Exception as exc:
            self._last_duration = time.monotonic() - started
            EventBus.emit(Event(
                event_type=EventType.FLOW_ERROR,
                source_id=self.flow_id,
                source_type=type(self).__name__,
                data={"error": str(exc), "executed": list(self._executed)},
            ))
            raise
        finally:
            EventBus.unregister(self._usage)

    async def _execute_method(self, name: str, argument: Any) -> Any:
        if name in self._skip_once:
            self._skip_once.discard(name)
            replayed = self._method_outputs.get(name)
            if self.verbose:
                log.info("Flow %s: skipping already-completed step '%s'", self.flow_id, name)
            return replayed

        method = getattr(self, name)
        EventBus.emit(Event(
            event_type=EventType.FLOW_METHOD_START,
            source_id=self.flow_id,
            source_type=type(self).__name__,
            data={"method": name},
        ))

        started = time.monotonic()
        try:
            result = method(argument) if self._accepts_argument(name, method) else method()
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            EventBus.emit(Event(
                event_type=EventType.FLOW_METHOD_ERROR,
                source_id=self.flow_id,
                source_type=type(self).__name__,
                data={"method": name, "error": str(exc)},
            ))
            log.error("Flow %s: step '%s' failed: %s", self.flow_id, name, exc)
            raise

        self._method_outputs[name] = result
        if name not in self._completed_steps:
            self._completed_steps.append(name)
        self._executed.append(name)

        if self.verbose:
            log.info("Flow %s: step '%s' completed", self.flow_id, name)

        EventBus.emit(Event(
            event_type=EventType.FLOW_METHOD_END,
            source_id=self.flow_id,
            source_type=type(self).__name__,
            data={
                "method": name,
                "duration": time.monotonic() - started,
                "result_preview": str(result)[:200],
            },
        ))

        if self._persist_all or name in self._persist_methods:
            self._checkpoint(method_name=name)
        return result

    def _resolve_trigger(self, name: str, result: Any) -> Optional[str]:
        """Return the trigger a completed method emits (route label for routers)."""
        if name not in self._routers:
            return name

        label = result if isinstance(result, str) and result.strip() else None
        EventBus.emit(Event(
            event_type=EventType.FLOW_ROUTE,
            source_id=self.flow_id,
            source_type=type(self).__name__,
            data={"router": name, "route": label},
        ))
        if label is None:
            log.debug("Router '%s' returned %r — branch ends here", name, result)
        return label

    def _collect_listeners(self, trigger: str) -> List[str]:
        """Return listeners whose condition is satisfied by ``trigger``."""
        fired: List[str] = []
        for listener, condition in self._listeners.items():
            if not condition.involves(trigger):
                continue
            progress = self._and_progress.setdefault(listener, set())
            progress.add(trigger)
            if condition.is_satisfied(progress):
                progress.clear()
                fired.append(listener)
        return fired

    def _prepare_runtime(self, resuming: bool) -> None:
        self._executed = []
        self._and_progress = {}
        if resuming and self._restored is not None:
            self._completed_steps = list(self._restored.completed_steps)
            self._method_outputs = dict(self._restored.method_outputs)
            self._skip_once = set(self._restored.completed_steps)
            log.info(
                "Flow %s resuming — %d step(s) already completed",
                self.flow_id, len(self._skip_once),
            )
        else:
            self._completed_steps = []
            self._method_outputs = {}
            self._skip_once = set()

    def _accepts_argument(self, name: str, method: Callable) -> bool:
        cached = self._arity_cache.get(name)
        if cached is not None:
            return cached

        accepts = False
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):  # pragma: no cover
            signature = None
        if signature is not None:
            for parameter in signature.parameters.values():
                if parameter.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    accepts = True
                    break
        self._arity_cache[name] = accepts
        return accepts

    # ── persistence helpers ────────────────────────────────────────────

    def _require_store(self) -> BaseFlowStore:
        if self._store is None:
            self._store = get_default_store()
        return self._store

    def _checkpoint(self, method_name: str = "", force: bool = False) -> FlowRecord:
        store = self._require_store()
        record = FlowRecord(
            flow_id=self.flow_id,
            flow_class=type(self).__name__,
            state=state_to_dict(self.state),
            completed_steps=list(self._completed_steps),
            method_outputs=dict(self._method_outputs),
            created_at=self._restored.created_at if self._restored else self._created_at,
        )
        store.save(record)
        EventBus.emit(Event(
            event_type=EventType.FLOW_STATE_SAVED,
            source_id=self.flow_id,
            source_type=type(self).__name__,
            data={"method": method_name, "forced": force, "steps": len(self._completed_steps)},
        ))
        return record

    # ── usage helpers ──────────────────────────────────────────────────

    def _discover_llm_clients(self) -> List[Any]:
        """Collect LLM clients reachable from the flow instance and its state."""
        found: Dict[int, Any] = {}
        candidates: List[Any] = list(self._usage_sources)
        candidates.extend(self.__dict__.values())
        candidates.extend(self._method_outputs.values())
        if isinstance(self.state, BaseModel):
            candidates.extend(self.state.__dict__.values())
        elif isinstance(self.state, dict):
            candidates.extend(self.state.values())

        seen: Set[int] = set()
        while candidates:
            item = candidates.pop()
            if item is None or id(item) in seen:
                continue
            seen.add(id(item))

            if isinstance(item, (list, tuple, set)):
                candidates.extend(item)
                continue
            if isinstance(item, dict):
                candidates.extend(item.values())
                continue
            if isinstance(item, TokenUsage):
                continue
            if isinstance(item, (str, bytes, int, float, bool)):
                continue

            if isinstance(getattr(item, "total_usage", None), TokenUsage):
                found[id(item)] = item
                continue
            for attribute in ("llm", "agents", "agent"):
                nested = getattr(item, attribute, None)
                if nested is not None:
                    candidates.append(nested)
        return list(found.values())

    # ── graph construction / validation ────────────────────────────────

    def _scan_methods(self) -> None:
        """Walk the MRO in definition order collecting decorated methods."""
        ordered: Dict[str, Any] = {}
        for klass in reversed(type(self).__mro__):
            if klass in (object, Flow, Generic):
                continue
            for name, attribute in vars(klass).items():
                if name.startswith("__"):
                    continue
                ordered[name] = attribute

        for name, attribute in ordered.items():
            function = attribute.__func__ if isinstance(attribute, (staticmethod, classmethod)) else attribute
            if not callable(function) or not _is_flow_method(function):
                continue

            if getattr(function, _START_ATTR, False):
                self._start_methods.append(name)
            condition = getattr(function, _LISTEN_ATTR, None)
            if condition is not None:
                self._listeners[name] = condition
            if getattr(function, _ROUTER_ATTR, False):
                self._routers.add(name)
            if getattr(function, _PERSIST_ATTR, False):
                self._persist_methods.add(name)
                store = getattr(function, _PERSIST_STORE_ATTR, None)
                if store is not None:
                    self._method_stores[name] = store
                    if self._store is None:
                        self._store = store

    def _build_graph(self) -> FlowGraph:
        graph = FlowGraph(name=type(self).__name__)
        method_names = set(self._start_methods) | set(self._listeners)

        # Nodes for every decorated method.
        for name in list(self._start_methods) + [n for n in self._listeners if n not in self._start_methods]:
            if graph.node(name) is not None:
                continue
            function = getattr(type(self), name, None)
            kind = "router" if name in self._routers else ("start" if name in self._start_methods else "listener")
            condition = self._listeners.get(name)
            graph.nodes.append(FlowNode(
                name=name,
                kind=kind,
                condition=repr(condition) if condition else "",
                is_async=inspect.iscoroutinefunction(function),
                doc=(inspect.getdoc(function) or "").split("\n")[0][:120],
            ))

        # Statically declared router labels.
        router_labels: Dict[str, Set[str]] = {
            name: _router_labels(getattr(type(self), name)) for name in self._routers
        }

        for listener, condition in self._listeners.items():
            kind = "listen"
            if isinstance(condition, FlowCondition) and len(condition.triggers) > 1:
                kind = "and" if condition.kind == "AND" else "or"

            for trigger in condition.names():
                if trigger in method_names:
                    graph.edges.append(FlowEdge(source=trigger, target=listener, kind=kind))
                    continue

                # Router label — add a label node and wire the matching routers.
                if graph.node(trigger) is None:
                    graph.nodes.append(FlowNode(name=trigger, kind="label"))
                producers = [n for n, labels in router_labels.items() if trigger in labels]
                if not producers:
                    producers = list(self._routers)
                    if not producers:
                        raise FlowError(
                            f"Method '{listener}' listens to '{trigger}', which is neither a "
                            f"flow method nor a label emitted by a @router in "
                            f"'{type(self).__name__}'"
                        )
                    log.warning(
                        "Could not statically attribute route '%s' to a router in %s — "
                        "assuming every router may emit it",
                        trigger, type(self).__name__,
                    )
                for producer in producers:
                    graph.edges.append(FlowEdge(source=producer, target=trigger, kind="route", label=trigger))
                graph.edges.append(FlowEdge(source=trigger, target=listener, kind=kind))

        return graph

    def _detect_cycles(self, graph: FlowGraph) -> None:
        """Raise :class:`FlowCycleError` on circular listener chains.

        Only chains made exclusively of direct ``@listen`` edges are rejected —
        those can never fire, because every method in the ring waits for another
        member of the same ring. Two kinds of edge are intentionally exempt:

        * edges emitted by a ``@router`` (the loop is data-dependent and ends
          when the router stops returning that label);
        * edges landing on a ``@start`` method (a conditional entry point is the
          idiomatic way to loop a flow back on itself).

        Both escape hatches remain bounded at runtime by ``max_steps``.
        """
        adjacency: Dict[str, List[str]] = {}
        for edge in graph.edges:
            if edge.kind == "route" or edge.target in self._start_methods:
                continue
            adjacency.setdefault(edge.source, []).append(edge.target)

        visiting: Set[str] = set()
        done: Set[str] = set()

        def walk(node: str, path: List[str]) -> None:
            if node in visiting:
                cycle = path[path.index(node):] + [node]
                raise FlowCycleError(
                    f"Circular listener graph in '{type(self).__name__}': "
                    + " -> ".join(cycle)
                )
            if node in done:
                return
            visiting.add(node)
            for neighbour in adjacency.get(node, []):
                walk(neighbour, path + [node])
            visiting.discard(node)
            done.add(node)

        for name in list(adjacency):
            walk(name, [])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(flow_id={self.flow_id!r}, "
            f"starts={len(self._start_methods)}, listeners={len(self._listeners)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_state_model(cls: type) -> Optional[Type[BaseModel]]:
    """Extract ``MyState`` from a ``class MyFlow(Flow[MyState])`` declaration."""
    for base in getattr(cls, "__orig_bases__", ()) or ():
        origin = get_origin(base)
        if origin is None or not (isinstance(origin, type) and issubclass(origin, Flow)):
            continue
        for argument in get_args(base):
            if isinstance(argument, type) and issubclass(argument, BaseModel):
                return argument
    return None


def _run_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, even inside a live event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
