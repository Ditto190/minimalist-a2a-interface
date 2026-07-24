"""
Core type definitions for Mangaba AI v3.0

Pydantic v2 models used across the entire framework for validation,
serialization, and JSON schema generation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    COMPLETED = "completed"
    ERROR = "error"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FinishReason(str, Enum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    ERROR = "error"
    CONTENT_FILTER = "content_filter"


# ---------------------------------------------------------------------------
# LLM related
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider: str = "google"
    model: Union[str, List[str]] = "gemini-2.5-flash"
    api_key: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = None
    timeout: int = Field(default=60, ge=1)
    base_url: Optional[str] = None

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, v: str) -> str:
        aliases = {
            "gemini": "google", "google-ai": "google", "googleai": "google",
            "gpt": "openai", "chatgpt": "openai",
            "claude": "anthropic",
            "hf": "huggingface", "hugging-face": "huggingface",
            "openrouter": "openrouter", "open-router":"openrouter", "or":"openrouter",
            # Local / self-hosted
            "ollama": "ollama", "local": "ollama",
            "openai-compatible": "openai_compatible",
            "openai_compatible": "openai_compatible",
            "vllm": "openai_compatible", "lmstudio": "openai_compatible",
            "lm-studio": "openai_compatible", "llamacpp": "openai_compatible",
            "llama-cpp": "openai_compatible", "localai": "openai_compatible",
        }
        return aliases.get(v.lower(), v.lower())


class OpenRouterConfig(LLMConfig):
    """Specialized configuration for OpenRouter with fallback support."""
    provider: str = "openrouter"
    
    # We can define a default fallback list here
    model: List[str] = Field(
        default_factory=lambda: ["google/gemini-2.5-flash", "anthropic/claude-3.5-sonnet"]
    )
    
    # Specific OpenRouter headers
    site_name: str = "Mangaba AI"
    site_url: str = "https://www.mangaba.ia.br/"
    
    # OpenRouter routing preferences (cheap, fast, etc)
    route: Optional[str] = None

class TokenUsage(BaseModel):
    """Token usage statistics from an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from executing a tool."""

    call_id: str
    tool_name: str = ""
    output: Any = None
    error: Optional[str] = None
    success: bool = True


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: Optional[str] = None, tool_calls: Optional[List[ToolCall]] = None) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, results: List[ToolResult]) -> Message:
        return cls(role=Role.TOOL, tool_results=results)


class LLMResponse(BaseModel):
    """Standardised response from any LLM provider."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: FinishReason = FinishReason.STOP
    raw: Any = Field(default=None, exclude=True)

    @property
    def text(self) -> str:
        return self.content or ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Agent / Task / Crew configuration
# ---------------------------------------------------------------------------

class MemoryConfig(BaseModel):
    """Memory configuration for an agent."""

    short_term: bool = True
    long_term: bool = False
    entity: bool = False
    max_short_term_items: int = Field(default=50, ge=1)
    storage_path: Optional[str] = None


class AgentConfig(BaseModel):
    """Full configuration for an Agent."""

    role: str
    goal: str
    backstory: str
    llm_config: Optional[LLMConfig] = None
    tools: List[str] = Field(default_factory=list)
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    max_iterations: int = Field(default=15, ge=1)
    max_retry_on_error: int = Field(default=3, ge=0)
    verbose: bool = False
    allow_delegation: bool = True
    step_callback: Optional[str] = None
    guardrails: List[str] = Field(default_factory=list)
    output_parser: Optional[str] = None

    # Deliberation before acting
    reasoning: bool = False
    max_reasoning_attempts: int = Field(default=3, ge=1)

    # Text + image input
    multimodal: bool = False

    # Execution limits
    max_execution_time: Optional[int] = Field(default=None, ge=1)
    max_rpm: Optional[int] = Field(default=None, ge=1)
    respect_context_window: bool = True

    # Prompt customisation
    system_template: Optional[str] = None
    prompt_template: Optional[str] = None
    inject_date: bool = False
    date_format: str = "%Y-%m-%d"

    @field_validator("role", "goal", "backstory")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class TaskConfig(BaseModel):
    """Full configuration for a Task."""

    description: str
    expected_output: str
    agent_id: Optional[str] = None
    context_ids: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    output_file: Optional[str] = None
    async_execution: bool = False
    human_input: bool = False
    guardrails: List[str] = Field(default_factory=list)
    guardrail_max_retries: int = Field(default=3, ge=0)
    output_parser: Optional[str] = None
    retry_on_failure: int = Field(default=0, ge=0)

    @field_validator("description", "expected_output")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Agent runtime state
# ---------------------------------------------------------------------------

class ReasoningOutput(BaseModel):
    """Result of an agent's pre-execution deliberation.

    Produced when ``Agent(reasoning=True)``: before touching any tool the agent
    drafts a plan and judges whether it is ready to act. If it is not, it
    refines the plan and tries again, up to ``max_reasoning_attempts``.
    """

    plan: str
    ready: bool = False
    attempts: int = 1
    critique: str = ""

    def as_prompt_section(self) -> str:
        """Render the plan for injection into the execution prompt."""
        return f"Your plan for this task:\n{self.plan}"


class HumanFeedback(BaseModel):
    """A human review collected during task execution."""

    approved: bool = True
    feedback: str = ""
    revised_output: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ImageContent(BaseModel):
    """An image supplied to a multimodal agent.

    Exactly one of ``url`` or ``path`` / ``base64_data`` should be set.
    """

    url: Optional[str] = None
    path: Optional[str] = None
    base64_data: Optional[str] = None
    mime_type: str = "image/png"
    detail: str = "auto"


class ReActStep(BaseModel):
    """A single step in the ReAct reasoning loop."""

    step_number: int
    thought: Optional[str] = None
    action: Optional[ToolCall] = None
    observation: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentState(BaseModel):
    """Runtime state of an agent during task execution."""

    agent_id: str
    messages: List[Message] = Field(default_factory=list)
    steps: List[ReActStep] = Field(default_factory=list)
    current_step: int = 0
    iteration_count: int = 0
    status: AgentStatus = AgentStatus.IDLE
    metadata: Dict[str, Any] = Field(default_factory=dict)
