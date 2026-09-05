"""
Tool system for Mangaba AI v3.0

Professional tool abstraction with Pydantic-based input validation,
automatic JSON schema generation for LLM function calling, and
support for both sync and async execution.
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field

from mangaba.core.events import EventBus, Event, EventType


class EmptyInput(BaseModel):
    """Default input schema when a tool takes no structured input."""
    pass


class BaseTool(ABC):
    """
    Base class for all Mangaba tools.

    Subclasses must:
    - Set ``name`` and ``description`` as class attributes
    - Optionally set ``args_schema`` to a Pydantic model describing inputs
    - Implement ``_run(**kwargs)``

    Example::

        class SearchTool(BaseTool):
            name = "web_search"
            description = "Search the web for current information"
            args_schema = SearchInput  # Pydantic model

            def _run(self, query: str, max_results: int = 5) -> str:
                ...
    """

    name: str = "base_tool"
    description: str = "Base tool"
    args_schema: Optional[Type[BaseModel]] = None
    return_direct: bool = False
    # ── CrewAI-parity knobs (opt-in, backwards compatible) ──
    cache_function: Optional[Any] = None
    max_retries: int = 0
    max_usage_count: Optional[int] = None
    output_schema: Optional[Type[BaseModel]] = None

    # -- public API ----------------------------------------------------------

    def run(self, **kwargs: Any) -> Any:
        """Validate inputs and execute the tool."""
        validated = self._validate_input(kwargs)
        # Usage guard (CrewAI-parity): stop after max_usage_count calls.
        self._usage_count = getattr(self, "_usage_count", 0) + 1
        if self.max_usage_count is not None and self._usage_count > self.max_usage_count:
            raise RuntimeError(f"Tool '{self.name}' exceeded max_usage_count={self.max_usage_count}")
        # Simple opt-in cache: cache_function(validated) -> bool, or True = cache all.
        cache_key = None
        if self.cache_function is not None:
            try:
                should_cache = self.cache_function(validated) if callable(self.cache_function) else bool(self.cache_function)
                if should_cache:
                    import json as _json
                    cache_key = _json.dumps(validated, sort_keys=True, default=str)
                    cached = getattr(self, "_result_cache", {}).get(cache_key)
                    if cached is not None:
                        return cached
            except Exception:
                cache_key = None
        EventBus.emit(Event(
            event_type=EventType.TOOL_START,
            data={"tool": self.name, "args": {k: str(v)[:200] for k, v in validated.items()}},
        ))
        attempts = max(1, (self.max_retries or 0) + 1)
        last_exc: Optional[Exception] = None
        for _ in range(attempts):
            try:
                result = self._run(**validated)
                if self.output_schema is not None:
                    try:
                        if isinstance(result, dict):
                            result = self.output_schema(**result)
                    except Exception:
                        pass
                EventBus.emit(Event(
                    event_type=EventType.TOOL_END,
                    data={"tool": self.name, "result_preview": str(result)[:200]},
                ))
                if cache_key is not None:
                    if not hasattr(self, "_result_cache"):
                        self._result_cache: Dict[str, Any] = {}
                    self._result_cache[cache_key] = result
                return result
            except Exception as exc:
                last_exc = exc
                continue
        EventBus.emit(Event(
            event_type=EventType.TOOL_ERROR,
            data={"tool": self.name, "error": str(last_exc)},
        ))
        assert last_exc is not None
        raise last_exc

    async def arun(self, **kwargs: Any) -> Any:
        """Async entry point — runs sync ``_run`` in a thread by default.

        Subclasses may override ``_arun`` for native async I/O.
        """
        if hasattr(self, "_arun"):
            validated = self._validate_input(kwargs)
            result = await self._arun(**validated)  # type: ignore[attr-defined]
            return result
        import asyncio
        return await asyncio.to_thread(self.run, **kwargs)

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Tool-specific implementation. Override in subclasses."""
        ...

    # -- schema / function calling helpers -----------------------------------

    def get_function_schema(self) -> Dict[str, Any]:
        """Return a JSON-schema representation for LLM function calling."""
        if self.args_schema is not None:
            params = self.args_schema.model_json_schema()
            # Remove the title to keep it concise
            params.pop("title", None)
        else:
            # Auto-detect from _run signature
            params = self._schema_from_signature()
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
        }

    # -- internal ------------------------------------------------------------

    def _validate_input(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate kwargs against args_schema if defined."""
        if self.args_schema is not None:
            validated = self.args_schema(**kwargs)
            return validated.model_dump()
        return kwargs

    def _schema_from_signature(self) -> Dict[str, Any]:
        """Infer a JSON schema from the ``_run`` method signature."""
        sig = inspect.signature(self._run)
        properties: Dict[str, Any] = {}
        required = []

        type_map = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "kwargs", "args"):
                continue
            annotation = param.annotation
            json_type = type_map.get(annotation, "string") if annotation != inspect.Parameter.empty else "string"
            prop: Dict[str, Any] = {"type": json_type}
            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(param_name)
            properties[param_name] = prop

        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
