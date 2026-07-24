"""Mangaba AI — Professional multi-agent orchestration framework.

Agents that reason, plan and use tools; crews that coordinate them; flows that
orchestrate the whole thing by event; plus memory, knowledge, guardrails and
tracing around it all.

Example::

    from mangaba import Agent, Task, Crew, Process, tool

    @tool
    def search(query: str) -> str:
        \"\"\"Search the web.\"\"\"
        return f"Results for {query}"

    analyst = Agent(role="Analyst", goal="Explain the numbers",
                    backstory="Ten years in equity research", tools=[search])
    task = Task(description="Summarise Q4", expected_output="A brief", agent=analyst)
    print(Crew(agents=[analyst], tasks=[task]).kickoff().final_output)
"""

# ── Core ───────────────────────────────────────────────────────────────
from mangaba.core.agent import Agent
from mangaba.core.task import Task, TaskOutput
from mangaba.core.crew import Crew, CrewOutput, Process
from mangaba.core.workflow import Pipeline, Stage, ParallelStage, ConditionalStage
from mangaba.core.events import (
    Event,
    EventBus,
    EventType,
    current_trace_id,
    start_trace,
)
from mangaba.core.exceptions import FlowError, MangabaError
from mangaba.core.reasoning import ReActEngine
from mangaba.core.deliberation import Deliberator
from mangaba.core.planner import ExecutionPlan, TaskPlanner

# ── Guardrails, parsers, human review ──────────────────────────────────
from mangaba.core.guardrails import (
    ContentFilterGuardrail,
    FunctionGuardrail,
    GuardrailChain,
    GuardrailValidationError,
    LengthGuardrail,
    LLMGuardrail,
    SchemaGuardrail,
)
from mangaba.core.output_parsers import JSONOutputParser, PydanticOutputParser
from mangaba.core.human import (
    AutoApproveHumanInput,
    BaseHumanInput,
    CallbackHumanInput,
    ConsoleHumanInput,
)

# ── Types ──────────────────────────────────────────────────────────────
from mangaba.core.types import (
    HumanFeedback,
    ImageContent,
    LLMConfig,
    OpenRouterConfig,
    ReasoningOutput,
    TokenUsage,
)

# ── LLM providers ──────────────────────────────────────────────────────
from mangaba.core.llm import (
    HF_OPEN_MODELS,
    OLLAMA_DEFAULT_BASE_URL,
    LLMClient,
    create_llm_client,
    get_supported_providers,
    hf_model_supports_tools,
    list_huggingface_models,
    list_ollama_models,
    ollama_model_supports_tools,
)

# ── Flows ──────────────────────────────────────────────────────────────
from mangaba.flows import (
    Flow,
    FlowState,
    SQLiteFlowStore,
    and_,
    listen,
    or_,
    persist,
    router,
    start,
)

# ── Knowledge ──────────────────────────────────────────────────────────
from mangaba.knowledge import (
    BaseKnowledgeSource,
    CSVKnowledgeSource,
    DirectoryKnowledgeSource,
    DOCXKnowledgeSource,
    ExcelKnowledgeSource,
    JSONKnowledgeSource,
    Knowledge,
    PDFKnowledgeSource,
    StringKnowledgeSource,
    TextFileKnowledgeSource,
    URLKnowledgeSource,
)

# ── Memory ─────────────────────────────────────────────────────────────
from mangaba.memory import (
    EntityMemory,
    InMemoryBackend,
    LongTermMemory,
    Memory,
    MemoryScope,
    MemoryWeights,
    ShortTermMemory,
    SQLiteBackend,
)

# ── Observability ──────────────────────────────────────────────────────
from mangaba.observability import (
    LangfuseCallback,
    MLflowCallback,
    OpenTelemetryCallback,
    PhoenixCallback,
    auto_configure_from_env,
    configure_observability,
)

# ── Training & evaluation ──────────────────────────────────────────────
from mangaba.training import (
    CrewEvaluator,
    CrewTrainer,
    EvaluationResult,
    TrainingResult,
    apply_training_data,
)

# ── Interop: open Agent2Agent protocol ─────────────────────────────────
from mangaba.interop import (
    A2AClient,
    A2AServer,
    AgentCard,
    AgentSkill,
    agent_card_for,
)

# ── Tools ──────────────────────────────────────────────────────────────
from mangaba.tools.base import BaseTool
from mangaba.tools.decorator import tool
from mangaba.tools.mcp_client import MCPClient
from mangaba.tools.registry import REGISTRY, ToolRegistry
from mangaba.tools.web_tools import HTTPRequestTool, ScrapeWebsiteTool
from mangaba.tools.document_tools import DocumentSearchTool, FileSearchTool
from mangaba.tools.code_tools import CodeInterpreterTool
from mangaba.tools.data_tools import SQLQueryTool

__version__ = "4.0.0"

__all__ = [
    # Core
    "Agent",
    "Task",
    "TaskOutput",
    "Crew",
    "CrewOutput",
    "Process",
    # Workflow
    "Pipeline",
    "Stage",
    "ParallelStage",
    "ConditionalStage",
    # Flows
    "Flow",
    "FlowState",
    "SQLiteFlowStore",
    "start",
    "listen",
    "router",
    "persist",
    "and_",
    "or_",
    # LLM
    "LLMClient",
    "create_llm_client",
    "get_supported_providers",
    "list_huggingface_models",
    "hf_model_supports_tools",
    "HF_OPEN_MODELS",
    "list_ollama_models",
    "ollama_model_supports_tools",
    "OLLAMA_DEFAULT_BASE_URL",
    # Events & tracing
    "EventBus",
    "Event",
    "EventType",
    "start_trace",
    "current_trace_id",
    # Reasoning & planning
    "ReActEngine",
    "Deliberator",
    "TaskPlanner",
    "ExecutionPlan",
    # Guardrails & parsers
    "GuardrailChain",
    "LLMGuardrail",
    "FunctionGuardrail",
    "LengthGuardrail",
    "ContentFilterGuardrail",
    "SchemaGuardrail",
    "GuardrailValidationError",
    "JSONOutputParser",
    "PydanticOutputParser",
    # Human-in-the-loop
    "BaseHumanInput",
    "ConsoleHumanInput",
    "AutoApproveHumanInput",
    "CallbackHumanInput",
    "HumanFeedback",
    # Knowledge
    "Knowledge",
    "BaseKnowledgeSource",
    "StringKnowledgeSource",
    "TextFileKnowledgeSource",
    "PDFKnowledgeSource",
    "DOCXKnowledgeSource",
    "CSVKnowledgeSource",
    "ExcelKnowledgeSource",
    "JSONKnowledgeSource",
    "URLKnowledgeSource",
    "DirectoryKnowledgeSource",
    # Memory
    "Memory",
    "MemoryScope",
    "MemoryWeights",
    "InMemoryBackend",
    "SQLiteBackend",
    "ShortTermMemory",
    "LongTermMemory",
    "EntityMemory",
    # Observability
    "OpenTelemetryCallback",
    "LangfuseCallback",
    "MLflowCallback",
    "PhoenixCallback",
    "configure_observability",
    "auto_configure_from_env",
    # Training
    "CrewEvaluator",
    "CrewTrainer",
    "EvaluationResult",
    "TrainingResult",
    "apply_training_data",
    # Interop (open A2A)
    "A2AServer",
    "A2AClient",
    "AgentCard",
    "AgentSkill",
    "agent_card_for",
    # Tools
    "BaseTool",
    "tool",
    "MCPClient",
    "ToolRegistry",
    "REGISTRY",
    "ScrapeWebsiteTool",
    "HTTPRequestTool",
    "DocumentSearchTool",
    "FileSearchTool",
    "CodeInterpreterTool",
    "SQLQueryTool",
    # Types
    "LLMConfig",
    "OpenRouterConfig",
    "TokenUsage",
    "ImageContent",
    "ReasoningOutput",
    # Exceptions
    "MangabaError",
    "FlowError",
]
