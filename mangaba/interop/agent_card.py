"""
Agent Card models for the open Agent2Agent (A2A) protocol.

An **Agent Card** is A2A's discovery document: a small JSON file, served at
``/.well-known/agent.json``, that tells any other agent what this agent is
called, where to reach it, what it can do and which content types it speaks.
Because the card is what foreign frameworks read, every field is serialized
with the spec's camelCase names (``defaultInputModes``, ``pushNotifications``,
…) while staying snake_case in Python.

Example::

    from mangaba import Agent
    from mangaba.interop import agent_card_for

    agent = Agent(role="Data Analyst", goal="Explain metrics",
                  backstory="A decade of BI work", tools=[CalculatorTool()])

    card = agent_card_for(agent, url="https://analyst.example.com/")
    print(card.to_json())
    # {"name": "Data Analyst", ..., "defaultInputModes": ["text/plain"], ...}

NOTE: this module implements the *open* A2A standard. It is unrelated to
``protocols/a2a.py``, which is an in-house in-process message bus that happens
to share the "agent-to-agent" name.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)


#: Protocol revision advertised by the cards this module builds.
A2A_PROTOCOL_VERSION = "0.2.0"

#: Content types a Mangaba agent accepts and produces by default.
DEFAULT_INPUT_MODES: List[str] = ["text/plain"]
DEFAULT_OUTPUT_MODES: List[str] = ["text/plain"]


def slugify(text: str) -> str:
    """Turn free text into a stable, URL-safe identifier.

    Example::

        slugify("Senior Data Analyst")   # 'senior_data_analyst'
    """
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower())
    return cleaned.strip("_") or "agent"


# ---------------------------------------------------------------------------
# Card components
# ---------------------------------------------------------------------------

class AgentProvider(BaseModel):
    """Who runs the agent behind the card.

    Example::

        AgentProvider(organization="Mangaba AI", url="https://mangaba.ia.br/")
    """

    model_config = ConfigDict(populate_by_name=True)

    organization: str = Field(..., description="Legal or product name of the operator")
    url: Optional[str] = Field(default=None, description="Homepage of the operator")


class AgentCapabilities(BaseModel):
    """Optional protocol features the agent supports.

    All three default to ``False``: :class:`~mangaba.interop.a2a.A2AServer`
    answers ``message/send`` + ``tasks/get`` and does not push notifications
    or replay state history.

    Example::

        AgentCapabilities(streaming=True).model_dump(by_alias=True)
        # {'streaming': True, 'pushNotifications': False, 'stateTransitionHistory': False}
    """

    model_config = ConfigDict(populate_by_name=True)

    streaming: bool = Field(
        default=False,
        description="Server supports message/stream (SSE) responses",
    )
    push_notifications: bool = Field(
        default=False,
        alias="pushNotifications",
        serialization_alias="pushNotifications",
        description="Server can POST task updates to a client-supplied webhook",
    )
    state_transition_history: bool = Field(
        default=False,
        alias="stateTransitionHistory",
        serialization_alias="stateTransitionHistory",
        description="Server keeps and exposes the full task state history",
    )


class AgentSkill(BaseModel):
    """One capability the agent advertises.

    A skill is the unit other frameworks route on, so give it a description a
    foreign planner can act upon.

    Example::

        AgentSkill(
            id="web_search",
            name="Web Search",
            description="Search the public web and summarise the findings",
            tags=["search", "research"],
            examples=["web_search(query)"],
        )
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Stable identifier, unique within the card")
    name: str = Field(..., description="Human-readable skill name")
    description: str = Field(default="", description="What the skill does")
    tags: List[str] = Field(default_factory=list, description="Keywords used for discovery")
    examples: List[str] = Field(default_factory=list, description="Sample prompts or invocations")
    input_modes: Optional[List[str]] = Field(
        default=None,
        alias="inputModes",
        serialization_alias="inputModes",
        description="Content types accepted by this skill (overrides the card default)",
    )
    output_modes: Optional[List[str]] = Field(
        default=None,
        alias="outputModes",
        serialization_alias="outputModes",
        description="Content types produced by this skill (overrides the card default)",
    )


class AgentCard(BaseModel):
    """The A2A discovery document for a single agent.

    Serialize with :meth:`to_dict` / :meth:`to_json` — both emit the spec's
    camelCase keys. Parsing accepts either spelling, so a card fetched from a
    foreign framework and a card built in Python both validate.

    Example::

        card = AgentCard(
            name="Research Agent",
            description="Answers research questions",
            url="https://research.example.com/",
            version="1.0.0",
            skills=[AgentSkill(id="research", name="Research")],
        )
        card.to_dict()["defaultInputModes"]   # ['text/plain']

        # Round-trip a foreign card
        same = AgentCard.from_dict(card.to_dict())
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., description="Human-readable agent name")
    description: str = Field(default="", description="What the agent is for")
    url: str = Field(..., description="Base URL of the agent's A2A JSON-RPC endpoint")
    version: str = Field(default="1.0.0", description="Version of the agent itself")
    protocol_version: str = Field(
        default=A2A_PROTOCOL_VERSION,
        alias="protocolVersion",
        serialization_alias="protocolVersion",
        description="A2A protocol revision this card conforms to",
    )
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    default_input_modes: List[str] = Field(
        default_factory=lambda: list(DEFAULT_INPUT_MODES),
        alias="defaultInputModes",
        serialization_alias="defaultInputModes",
        description="Content types the agent accepts unless a skill says otherwise",
    )
    default_output_modes: List[str] = Field(
        default_factory=lambda: list(DEFAULT_OUTPUT_MODES),
        alias="defaultOutputModes",
        serialization_alias="defaultOutputModes",
        description="Content types the agent produces unless a skill says otherwise",
    )
    skills: List[AgentSkill] = Field(default_factory=list)
    provider: Optional[AgentProvider] = Field(default=None)
    documentation_url: Optional[str] = Field(
        default=None,
        alias="documentationUrl",
        serialization_alias="documentationUrl",
        description="Where a human can read more about this agent",
    )

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with the spec's camelCase field names."""
        return self.model_dump(by_alias=True, exclude_none=True)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to a JSON string using the spec's camelCase field names."""
        return self.model_dump_json(by_alias=True, exclude_none=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Parse a card served by any A2A implementation."""
        return cls.model_validate(data)

    def skill_ids(self) -> List[str]:
        """Return the ids of every advertised skill."""
        return [s.id for s in self.skills]

    def __repr__(self) -> str:
        return f"AgentCard(name={self.name!r}, url={self.url!r}, skills={len(self.skills)})"


# ---------------------------------------------------------------------------
# Building a card from Mangaba objects
# ---------------------------------------------------------------------------

def skill_for_tool(tool: Any) -> AgentSkill:
    """Derive an :class:`AgentSkill` from a Mangaba tool.

    Example::

        skill_for_tool(CalculatorTool()).id    # 'calculator'
    """
    tool_name = getattr(tool, "name", None) or tool.__class__.__name__
    description = getattr(tool, "description", "") or f"Tool '{tool_name}'"

    examples: List[str] = []
    try:
        schema = tool.get_function_schema()
        params = schema.get("parameters") or {}
        required = list(params.get("required") or [])
        if not required:
            required = list((params.get("properties") or {}).keys())[:2]
        examples.append(f"{tool_name}({', '.join(required)})" if required else f"{tool_name}()")
    except Exception as exc:  # noqa: BLE001 - a tool with a broken schema must not kill discovery
        log.debug("Could not derive an example for tool %r: %s", tool_name, exc)

    return AgentSkill(
        id=slugify(tool_name),
        name=tool_name.replace("_", " ").title(),
        description=description,
        tags=sorted({"tool", *slugify(tool_name).split("_")}),
        examples=examples,
    )


def skill_for_role(role: str, goal: str, backstory: str = "") -> AgentSkill:
    """Derive the fallback skill of a tool-less agent from its role and goal."""
    description = goal.strip() or backstory.strip() or role.strip()
    tags = sorted({t for t in slugify(role).split("_") if len(t) > 2}) or ["agent"]
    return AgentSkill(
        id=slugify(role),
        name=role.strip() or "Agent",
        description=description,
        tags=tags,
        examples=[goal.strip()] if goal.strip() else [],
    )


def agent_card_for(
    agent: Any,
    url: str = "http://localhost:8000/",
    version: str = "1.0.0",
    name: Optional[str] = None,
    description: Optional[str] = None,
    capabilities: Optional[AgentCapabilities] = None,
    provider: Optional[AgentProvider] = None,
    documentation_url: Optional[str] = None,
    default_input_modes: Optional[List[str]] = None,
    default_output_modes: Optional[List[str]] = None,
) -> AgentCard:
    """Build an :class:`AgentCard` for a Mangaba :class:`~mangaba.core.agent.Agent`.

    One skill is derived per tool the agent carries. An agent with no tools
    still advertises a single skill, derived from its role and goal, so it
    remains discoverable and routable.

    A :class:`~mangaba.core.crew.Crew` is also accepted: the card then
    advertises one skill per member agent.

    Example::

        agent = Agent(role="Analyst", goal="Explain metrics",
                      backstory="BI veteran", tools=[CalculatorTool()])
        card = agent_card_for(agent, url="http://localhost:9000/")
        card.skill_ids()          # ['calculator']
        card.to_dict()["defaultOutputModes"]   # ['text/plain']
    """
    if hasattr(agent, "agents") and hasattr(agent, "tasks"):
        return _card_for_crew(
            agent,
            url=url,
            version=version,
            name=name,
            description=description,
            capabilities=capabilities,
            provider=provider,
            documentation_url=documentation_url,
            default_input_modes=default_input_modes,
            default_output_modes=default_output_modes,
        )

    role = str(getattr(agent, "role", "") or "Agent")
    goal = str(getattr(agent, "goal", "") or "")
    backstory = str(getattr(agent, "backstory", "") or "")
    tools = list(getattr(agent, "tools", None) or [])

    skills = [skill_for_tool(t) for t in tools] or [skill_for_role(role, goal, backstory)]

    card_description = description if description is not None else (goal or backstory or role)

    return AgentCard(
        name=name or role,
        description=card_description,
        url=url,
        version=version,
        capabilities=capabilities or AgentCapabilities(),
        default_input_modes=list(default_input_modes or DEFAULT_INPUT_MODES),
        default_output_modes=list(default_output_modes or DEFAULT_OUTPUT_MODES),
        skills=skills,
        provider=provider,
        documentation_url=documentation_url,
    )


def _card_for_crew(
    crew: Any,
    url: str,
    version: str,
    name: Optional[str],
    description: Optional[str],
    capabilities: Optional[AgentCapabilities],
    provider: Optional[AgentProvider],
    documentation_url: Optional[str],
    default_input_modes: Optional[List[str]],
    default_output_modes: Optional[List[str]],
) -> AgentCard:
    """Build a card for a Crew — one skill per member agent."""
    members = list(getattr(crew, "agents", None) or [])
    skills = [
        skill_for_role(
            str(getattr(m, "role", "") or "Agent"),
            str(getattr(m, "goal", "") or ""),
            str(getattr(m, "backstory", "") or ""),
        )
        for m in members
    ]
    if not skills:
        skills = [skill_for_role("crew", "Run a multi-agent crew")]

    roles = ", ".join(str(getattr(m, "role", "")) for m in members if getattr(m, "role", None))
    return AgentCard(
        name=name or "Mangaba Crew",
        description=description if description is not None else (f"Crew of: {roles}" if roles else "Mangaba crew"),
        url=url,
        version=version,
        capabilities=capabilities or AgentCapabilities(),
        default_input_modes=list(default_input_modes or DEFAULT_INPUT_MODES),
        default_output_modes=list(default_output_modes or DEFAULT_OUTPUT_MODES),
        skills=skills,
        provider=provider,
        documentation_url=documentation_url,
    )


__all__ = [
    "A2A_PROTOCOL_VERSION",
    "DEFAULT_INPUT_MODES",
    "DEFAULT_OUTPUT_MODES",
    "AgentCapabilities",
    "AgentCard",
    "AgentProvider",
    "AgentSkill",
    "agent_card_for",
    "skill_for_role",
    "skill_for_tool",
    "slugify",
]
