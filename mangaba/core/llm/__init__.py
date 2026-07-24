"""LLM engine for Mangaba AI v3.0 — function calling, streaming, caching."""

from .client import (
    BaseLLMProvider,
    LLMClient,
    OllamaLLMProvider,
    OpenAICompatibleLLMProvider,
    create_llm_client,
    get_supported_providers,
    list_huggingface_models,
    list_ollama_models,
    hf_model_supports_tools,
    ollama_model_supports_tools,
    HF_OPEN_MODELS,
    OLLAMA_DEFAULT_BASE_URL,
)
from .retry import with_retry
from .cache import LLMCache, InMemoryCache, DiskCache
from .token_counter import TokenCounter, UsageTracker
from .prompt_templates import PromptTemplate, ChatPromptTemplate, SystemPromptBuilder

__all__ = [
    "BaseLLMProvider",
    "LLMClient",
    "OllamaLLMProvider",
    "OpenAICompatibleLLMProvider",
    "create_llm_client",
    "get_supported_providers",
    "list_huggingface_models",
    "list_ollama_models",
    "hf_model_supports_tools",
    "ollama_model_supports_tools",
    "HF_OPEN_MODELS",
    "OLLAMA_DEFAULT_BASE_URL",
    "with_retry",
    "LLMCache",
    "InMemoryCache",
    "DiskCache",
    "TokenCounter",
    "UsageTracker",
    "PromptTemplate",
    "ChatPromptTemplate",
    "SystemPromptBuilder",
]
