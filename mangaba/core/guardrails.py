"""
Guardrails for input/output validation in Mangaba AI v3.0
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

from mangaba.core.events import EventBus, Event, EventType

log = logging.getLogger(__name__)


class GuardrailValidationError(ValueError):
    """Raised when output fails a guardrail.

    Carries ``feedback`` so the caller can feed the reason back into a retry
    instead of blindly re-running the same prompt.
    """

    def __init__(self, message: str, feedback: str = "", guardrail: str = "") -> None:
        super().__init__(message)
        self.feedback = feedback or message
        self.guardrail = guardrail


class BaseGuardrail(ABC):
    """Abstract guardrail that validates and optionally transforms text."""

    @abstractmethod
    def validate(self, text: str) -> str:
        """Validate *text*. Return the (possibly modified) text or raise ValueError."""
        ...


class LengthGuardrail(BaseGuardrail):
    """Ensures output length is within bounds."""

    def __init__(self, min_length: int = 0, max_length: int = 50_000) -> None:
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, text: str) -> str:
        if len(text) < self.min_length:
            EventBus.emit(Event(event_type=EventType.GUARDRAIL_FAIL, data={"guardrail": "length", "reason": "too_short"}))
            raise ValueError(f"Output too short ({len(text)} < {self.min_length})")
        if len(text) > self.max_length:
            EventBus.emit(Event(event_type=EventType.GUARDRAIL_FAIL, data={"guardrail": "length", "reason": "too_long"}))
            text = text[: self.max_length]
        EventBus.emit(Event(event_type=EventType.GUARDRAIL_PASS, data={"guardrail": "length"}))
        return text


class ContentFilterGuardrail(BaseGuardrail):
    """Block output containing specific patterns."""

    def __init__(self, blocked_patterns: Optional[List[str]] = None) -> None:
        defaults = [
            r'\b(?:password|secret|api[_-]?key)\s*[:=]\s*\S+',
        ]
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (blocked_patterns or defaults)]

    def validate(self, text: str) -> str:
        for pattern in self.patterns:
            if pattern.search(text):
                EventBus.emit(Event(event_type=EventType.GUARDRAIL_FAIL, data={"guardrail": "content_filter"}))
                # Redact matches
                text = pattern.sub("[REDACTED]", text)
        EventBus.emit(Event(event_type=EventType.GUARDRAIL_PASS, data={"guardrail": "content_filter"}))
        return text


class SchemaGuardrail(BaseGuardrail):
    """Validate that output can be parsed into a Pydantic model."""

    def __init__(self, schema: Type[BaseModel]) -> None:
        self.schema = schema

    def validate(self, text: str) -> str:
        import json as _json
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = _json.loads(text[start:end])
            self.schema(**data)
            EventBus.emit(Event(event_type=EventType.GUARDRAIL_PASS, data={"guardrail": "schema"}))
            return text
        except Exception as exc:
            EventBus.emit(Event(event_type=EventType.GUARDRAIL_FAIL, data={"guardrail": "schema", "error": str(exc)}))
            raise ValueError(f"Output does not match schema {self.schema.__name__}: {exc}") from exc


class LLMGuardrail(BaseGuardrail):
    """Validate output against a plain-language criterion, judged by an LLM.

    Instead of writing a validator function, describe the requirement in
    natural language and let a model decide whether the output satisfies it.
    On failure the raised error carries the judge's reasoning as ``feedback``,
    so a retry can be steered rather than repeated blindly.

    Example::

        guardrail = LLMGuardrail(
            "The answer must cite at least two sources, each with a URL.",
            llm=llm_client,
        )
        task = Task(description="...", expected_output="...", agent=agent,
                    guardrails=[guardrail])
    """

    JUDGE_PROMPT = (
        "You are a strict output validator.\n"
        "Decide whether the OUTPUT satisfies the CRITERION.\n\n"
        "CRITERION:\n{criterion}\n\n"
        "OUTPUT:\n{output}\n\n"
        'Respond ONLY with JSON: {{"valid": true, "feedback": ""}} '
        'or {{"valid": false, "feedback": "<what is missing or wrong>"}}'
    )

    def __init__(
        self,
        criterion: str,
        llm: Any,
        max_output_chars: int = 8000,
        fail_open: bool = False,
    ) -> None:
        if not criterion or not criterion.strip():
            raise ValueError("Guardrail criterion cannot be empty")
        self.criterion = criterion.strip()
        self.llm = llm
        self.max_output_chars = max_output_chars
        self.fail_open = fail_open

    def validate(self, text: str) -> str:
        prompt = self.JUDGE_PROMPT.format(
            criterion=self.criterion,
            output=text[: self.max_output_chars],
        )

        try:
            raw = self.llm.generate_text(prompt)
        except Exception as exc:  # judge unreachable
            log.warning("LLMGuardrail judge failed: %s", exc)
            if self.fail_open:
                return text
            raise GuardrailValidationError(
                f"Guardrail judge unavailable: {exc}",
                feedback=str(exc),
                guardrail="llm",
            ) from exc

        valid, feedback = self._parse_verdict(raw)

        if not valid:
            EventBus.emit(Event(
                event_type=EventType.GUARDRAIL_FAIL,
                data={"guardrail": "llm", "criterion": self.criterion, "feedback": feedback},
            ))
            raise GuardrailValidationError(
                f"Output failed criterion '{self.criterion}': {feedback}",
                feedback=feedback,
                guardrail="llm",
            )

        EventBus.emit(Event(event_type=EventType.GUARDRAIL_PASS, data={"guardrail": "llm"}))
        return text

    @staticmethod
    def _parse_verdict(raw: str) -> Tuple[bool, str]:
        """Extract ``(valid, feedback)`` from the judge's reply."""
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            return bool(data.get("valid", False)), str(data.get("feedback", ""))
        except (ValueError, json.JSONDecodeError):
            # No parseable JSON — fall back to reading the sentiment of the text
            lowered = raw.lower()
            if "true" in lowered or "valid" in lowered and "invalid" not in lowered:
                return True, ""
            return False, raw.strip()[:500]


class FunctionGuardrail(BaseGuardrail):
    """Adapt a plain callable into a guardrail.

    The callable receives the output text and returns either a
    ``(passed, value)`` tuple — where ``value`` is the replacement output when
    passing, or the failure reason when not — or a bare string, which is taken
    as a successful transformation.

    Example::

        def must_be_json(text: str):
            try:
                json.loads(text)
                return True, text
            except ValueError as exc:
                return False, f"not valid JSON: {exc}"

        task = Task(..., guardrails=[FunctionGuardrail(must_be_json)])
    """

    def __init__(self, fn: Callable[[str], Union[str, Tuple[bool, Any]]], name: Optional[str] = None) -> None:
        if not callable(fn):
            raise ValueError("FunctionGuardrail requires a callable")
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "function")

    def validate(self, text: str) -> str:
        result = self.fn(text)

        if isinstance(result, tuple):
            passed, value = result
            if not passed:
                EventBus.emit(Event(
                    event_type=EventType.GUARDRAIL_FAIL,
                    data={"guardrail": self.name, "feedback": str(value)},
                ))
                raise GuardrailValidationError(
                    f"Output failed guardrail '{self.name}': {value}",
                    feedback=str(value),
                    guardrail=self.name,
                )
            text = str(value)
        elif result is not None:
            text = str(result)

        EventBus.emit(Event(event_type=EventType.GUARDRAIL_PASS, data={"guardrail": self.name}))
        return text


class GuardrailChain(BaseGuardrail):
    """Compose multiple guardrails sequentially."""

    def __init__(self, guardrails: List[BaseGuardrail]) -> None:
        self.guardrails = guardrails

    def validate(self, text: str) -> str:
        for g in self.guardrails:
            text = g.validate(text)
        return text
