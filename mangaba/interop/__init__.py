"""
Interoperability with agents built on other frameworks.

This package implements the **open Agent2Agent (A2A) protocol**: Agent Cards
for discovery, a JSON-RPC server that publishes a Mangaba agent, and a client
that consumes any remote A2A agent — including as a native Mangaba tool.

Example::

    from mangaba import Agent
    from mangaba.interop import A2AServer, A2AClient, agent_card_for

    agent = Agent(role="Researcher", goal="Answer questions", backstory="...")

    with A2AServer(agent, port=0) as server:
        client = A2AClient(server.url)
        print(client.get_card().name)
        print(client.ask("What is the open A2A protocol?"))

NOTE: unrelated to ``protocols/a2a.py``, the in-house in-process message bus
that shares the "agent-to-agent" name. This package speaks the public
standard over HTTP; that module passes objects between agents inside one
Python process.
"""

from mangaba.interop.agent_card import (
    A2A_PROTOCOL_VERSION,
    DEFAULT_INPUT_MODES,
    DEFAULT_OUTPUT_MODES,
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    agent_card_for,
    skill_for_role,
    skill_for_tool,
)
from mangaba.interop.a2a import (
    AGENT_CARD_PATHS,
    A2AArtifact,
    A2AClient,
    A2AError,
    A2AInvalidParamsError,
    A2AMessage,
    A2APart,
    A2ARemoteAgentTool,
    A2ARemoteError,
    A2AServer,
    A2ATask,
    A2ATaskNotFoundError,
    A2ATaskState,
    A2ATaskStatus,
    A2ATransportError,
    InMemoryTaskStore,
)

__all__ = [
    # Discovery
    "A2A_PROTOCOL_VERSION",
    "AGENT_CARD_PATHS",
    "DEFAULT_INPUT_MODES",
    "DEFAULT_OUTPUT_MODES",
    "AgentCapabilities",
    "AgentCard",
    "AgentProvider",
    "AgentSkill",
    "agent_card_for",
    "skill_for_role",
    "skill_for_tool",
    # Protocol
    "A2AArtifact",
    "A2AClient",
    "A2AMessage",
    "A2APart",
    "A2ARemoteAgentTool",
    "A2AServer",
    "A2ATask",
    "A2ATaskState",
    "A2ATaskStatus",
    "InMemoryTaskStore",
    # Errors
    "A2AError",
    "A2AInvalidParamsError",
    "A2ARemoteError",
    "A2ATaskNotFoundError",
    "A2ATransportError",
]
