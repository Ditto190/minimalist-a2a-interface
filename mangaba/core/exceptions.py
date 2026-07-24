"""
Exception hierarchy for Mangaba AI v3.0

Typed exceptions for each subsystem, enabling granular error handling
and automatic retry of transient failures.
"""

from __future__ import annotations

from typing import Optional


class MangabaError(Exception):
    """Base exception for all Mangaba errors."""

    def __init__(self, message: str, *, cause: Optional[Exception] = None) -> None:
        self.cause = cause
        super().__init__(message)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigurationError(MangabaError):
    """Invalid or missing configuration."""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class LLMError(MangabaError):
    """Generic LLM provider error."""


class AuthenticationError(LLMError):
    """Invalid or expired API key."""


class RetryableError(LLMError):
    """Transient error that may succeed on retry."""


class RateLimitError(RetryableError):
    """Provider rate-limit hit."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: Optional[float] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, cause=cause)


class TokenLimitError(LLMError):
    """Prompt or response exceeds token limit."""


class ContentFilterError(LLMError):
    """Content blocked by safety filter."""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ToolError(MangabaError):
    """Error during tool execution."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        cause: Optional[Exception] = None,
    ) -> None:
        self.tool_name = tool_name
        super().__init__(message, cause=cause)


class ToolValidationError(ToolError):
    """Tool input validation failed."""


class ToolNotFoundError(ToolError):
    """Requested tool does not exist."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgentError(MangabaError):
    """Error in agent execution."""


class MaxIterationsError(AgentError):
    """Agent hit the maximum iteration limit."""


class DelegationError(AgentError):
    """Failed to delegate task to another agent."""


# ---------------------------------------------------------------------------
# Task / Crew
# ---------------------------------------------------------------------------

class TaskError(MangabaError):
    """Error during task execution."""


class CrewError(MangabaError):
    """Error in crew orchestration."""


class ValidationError(MangabaError):
    """Generic data validation failure."""


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------

class FlowError(MangabaError):
    """Error in flow definition or execution."""


class FlowCycleError(FlowError):
    """Circular dependency detected in the listener graph."""


class FlowPersistenceError(FlowError):
    """Error saving, loading, or forking persisted flow state."""


# ---------------------------------------------------------------------------
# Memory / RAG
# ---------------------------------------------------------------------------

class MemoryError(MangabaError):
    """Error in memory subsystem."""


class EmbeddingError(MangabaError):
    """Error computing embeddings."""


class VectorStoreError(MangabaError):
    """Error in vector storage subsystem."""
