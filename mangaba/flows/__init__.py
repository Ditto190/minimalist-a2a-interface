"""
Flows — event-driven orchestration for Mangaba AI v3.0

Subclass :class:`Flow`, decorate methods with :func:`start`, :func:`listen` and
:func:`router`, and the engine runs the resulting graph to completion. State is
a dict by default or a Pydantic model when declared, ``@persist`` checkpoints
every step to SQLite, and ``flow.plot()`` renders the graph to standalone HTML.

Example::

    from mangaba.flows import Flow, FlowState, start, listen, router, and_, persist

    class ReviewState(FlowState):
        topic: str = ""
        score: float = 0.0

    @persist
    class ReviewFlow(Flow[ReviewState]):
        @start()
        def draft(self):
            return f"draft about {self.state.topic}"

        @router(draft)
        def triage(self, text):
            return "publish" if len(text) > 10 else "revise"

        @listen("publish")
        def publish(self):
            return "published"

    flow = ReviewFlow()
    print(flow.kickoff({"topic": "mangaba"}))
"""

from __future__ import annotations

from mangaba.flows.flow import (
    Flow,
    FlowCondition,
    FlowEdge,
    FlowGraph,
    FlowNode,
    and_,
    listen,
    or_,
    persist,
    router,
    start,
)
from mangaba.flows.persistence import (
    DEFAULT_DB_PATH,
    BaseFlowStore,
    FlowRecord,
    SQLiteFlowStore,
    get_default_store,
)
from mangaba.flows.state import FlowState, create_state, state_from_dict, state_to_dict
from mangaba.flows.visualization import plot_flow, render_graph_html

__all__ = [
    # engine
    "Flow",
    "FlowCondition",
    # decorators / combinators
    "start",
    "listen",
    "router",
    "persist",
    "and_",
    "or_",
    # state
    "FlowState",
    "create_state",
    "state_to_dict",
    "state_from_dict",
    # persistence
    "BaseFlowStore",
    "SQLiteFlowStore",
    "FlowRecord",
    "get_default_store",
    "DEFAULT_DB_PATH",
    # graph / plotting
    "FlowGraph",
    "FlowNode",
    "FlowEdge",
    "plot_flow",
    "render_graph_html",
]
