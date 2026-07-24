"""
Agent v3.0 — ReAct reasoning, memory, planning, guardrails.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from mangaba.core.types import (
    AgentState,
    AgentStatus,
    ImageContent,
    LLMConfig,
    MemoryConfig,
    OpenRouterConfig,
    ReasoningOutput,
)
from mangaba.core.exceptions import AgentError
from mangaba.core.events import EventBus, Event, EventType
from mangaba.core.reasoning import ReActEngine
from mangaba.core.deliberation import Deliberator

if TYPE_CHECKING:
    from mangaba.tools.base import BaseTool
    from mangaba.core.guardrails import BaseGuardrail
    from mangaba.core.output_parsers import BaseOutputParser
    from mangaba.memory.base import BaseMemory
    from mangaba.knowledge.knowledge import Knowledge

log = logging.getLogger(__name__)


class _RateLimiter:
    """Sliding-window limiter shared by every call an agent makes."""

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max_per_minute
        self._calls: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until another call fits inside the window."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < 60.0]
                if len(self._calls) < self.max_per_minute:
                    self._calls.append(now)
                    return
                sleep_for = 60.0 - (now - self._calls[0]) + 0.01
            log.debug("Rate limit reached — sleeping %.2fs", sleep_for)
            time.sleep(max(0.0, sleep_for))


class Agent:
    """Intelligent agent with ReAct reasoning, memory, and tool use.

    Example::

        agent = Agent(
            role="Senior Data Analyst",
            goal="Analyze market trends and provide insights",
            backstory="Expert in financial markets with 15 years of experience",
            tools=[SearchTool(), CalculatorTool()],
            verbose=True,
        )
        result = agent.execute_task("Analyze Q4 revenue trends")

    Deliberate before acting, ground answers in a knowledge base, and cap how
    hard the agent is allowed to work::

        agent = Agent(
            role="Security Analyst",
            goal="Find exploitable flaws",
            backstory="A decade of offensive security work",
            tools=[read_file, grep],
            knowledge=Knowledge(sources=[PDFKnowledgeSource("owasp.pdf")], embedding=emb),
            reasoning=True,            # plan and self-critique before acting
            max_reasoning_attempts=3,
            max_execution_time=120,    # seconds
            max_rpm=20,                # requests per minute
        )

    Read images alongside text::

        agent = Agent(..., multimodal=True)
        agent.execute_task("What is wrong with this diagram?",
                           images=[ImageContent(path="arch.png")])
    """

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[Any] = None,  # LLMClient or provider string
        llm_config: Optional[LLMConfig] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
        memory: Optional[BaseMemory] = None,
        memory_config: Optional[MemoryConfig] = None,
        max_iterations: int = 15,
        max_retry_on_error: int = 3,
        allow_delegation: bool = True,
        step_callback: Optional[Callable] = None,
        guardrails: Optional[List[BaseGuardrail]] = None,
        output_parser: Optional[BaseOutputParser] = None,
        agent_id: Optional[str] = None,
        knowledge: Optional[Knowledge] = None,
        reasoning: bool = False,
        max_reasoning_attempts: int = 3,
        multimodal: bool = False,
        max_execution_time: Optional[int] = None,
        max_rpm: Optional[int] = None,
        respect_context_window: bool = True,
        system_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        inject_date: bool = False,
        date_format: str = "%Y-%m-%d",
        training_context: Optional[str] = None,
    ) -> None:
        if not role or not role.strip():
            raise ValueError("Role cannot be empty")
        if not goal or not goal.strip():
            raise ValueError("Goal cannot be empty")
        if not backstory or not backstory.strip():
            raise ValueError("Backstory cannot be empty")

        self.role = role.strip()
        self.goal = goal.strip()
        self.backstory = backstory.strip()
        self.tools: List[BaseTool] = list(tools or [])
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.max_retry_on_error = max_retry_on_error
        self.allow_delegation = allow_delegation
        self.step_callback = step_callback
        self.guardrails = guardrails or []
        self.output_parser = output_parser
        self.memory = memory
        self.memory_config = memory_config or MemoryConfig()
        self.knowledge = knowledge

        # ── Deliberation / multimodal / execution limits ─────────────
        self.reasoning = reasoning
        self.max_reasoning_attempts = max_reasoning_attempts
        self.multimodal = multimodal
        self.max_execution_time = max_execution_time
        self.max_rpm = max_rpm
        self.respect_context_window = respect_context_window
        self._rate_limiter = _RateLimiter(max_rpm) if max_rpm else None

        # ── Prompt customisation ─────────────────────────────────────
        self.system_template = system_template
        self.prompt_template = prompt_template
        self.inject_date = inject_date
        self.date_format = date_format

        # Lessons distilled by ``mangaba train`` — folded into the system prompt
        self.training_context = training_context or ""
        self._training_hints: List[str] = []

        self.agent_id = agent_id or f"agent_{role.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

        # ── LLM setup ────────────────────────────────────────────────
        if llm is not None and not isinstance(llm, str):
            # Already an LLMClient instance
            self.llm = llm
        else:
            self.llm = self._create_llm(llm, llm_config, api_key)

        # ── Memory (auto-create short-term if nothing provided) ──────
        if self.memory is None and self.memory_config.short_term:
            from mangaba.memory.short_term import ShortTermMemory
            self.memory = ShortTermMemory(max_items=self.memory_config.max_short_term_items)

        # ── ReAct engine ─────────────────────────────────────────────
        self._react = ReActEngine(
            llm=self.llm,
            tools=self.tools,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
        )

        # ── Deliberation (plan before acting) ────────────────────────
        self._deliberator = (
            Deliberator(llm=self.llm, max_attempts=self.max_reasoning_attempts, verbose=self.verbose)
            if self.reasoning
            else None
        )

        # ── State ────────────────────────────────────────────────────
        self.state = AgentState(agent_id=self.agent_id)
        self.last_reasoning: Optional[ReasoningOutput] = None

        # ── Connected agents (for delegation) ────────────────────────
        self._peers: Dict[str, Agent] = {}

        if self.verbose:
            log.info("Agent initialized — role=%s tools=%d", self.role, len(self.tools))

    # ── public API ─────────────────────────────────────────────────────

    def execute_task(
        self,
        task_description: str,
        context: Optional[str] = None,
        images: Optional[List[ImageContent]] = None,
    ) -> str:
        """Execute a task using the ReAct loop with tool/function calling.

        When ``reasoning=True`` the agent first drafts and critiques a plan and
        only then acts. When ``multimodal=True`` any ``images`` supplied are
        described up front and folded into the working context.
        """
        EventBus.emit(Event(
            event_type=EventType.AGENT_START,
            source_id=self.agent_id,
            data={"role": self.role, "task": task_description[:200]},
        ))

        system_prompt = self._build_system_prompt()

        # Gather everything the agent should know before it starts working
        context_parts: List[str] = []
        if context:
            context_parts.append(context)

        memory_context = self._get_memory_context(task_description)
        if memory_context:
            context_parts.append(f"Relevant memories:\n{memory_context}")

        knowledge_context = self._get_knowledge_context(task_description)
        if knowledge_context:
            context_parts.append(f"Relevant knowledge:\n{knowledge_context}")

        if images:
            if not self.multimodal:
                raise AgentError(
                    f"Agent '{self.role}' received images but was not created with multimodal=True"
                )
            described = self._describe_images(images, task_description)
            if described:
                context_parts.append(f"Images provided:\n{described}")

        # Plan before acting
        if self._deliberator is not None:
            self.last_reasoning = self._deliberator.deliberate(
                role=self.role, goal=self.goal, task=task_description, tools=self.tools,
            )
            if self.last_reasoning.plan:
                context_parts.append(self.last_reasoning.as_prompt_section())
            if not self.last_reasoning.ready:
                log.warning(
                    "Agent '%s' proceeding with an unvalidated plan after %d attempts",
                    self.role, self.last_reasoning.attempts,
                )

        full_context = "\n\n".join(context_parts)
        user_prompt = self._build_user_prompt(task_description, None)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retry_on_error + 1):
            try:
                if self._rate_limiter is not None:
                    self._rate_limiter.acquire()

                response = self._run_react(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    context=full_context or None,
                )

                result_text = response.text

                # Guardrails
                result_text = self._apply_guardrails(result_text)

                # Output parser
                if self.output_parser:
                    result_text = self.output_parser.parse(result_text)

                # Store in memory
                if self.memory:
                    self.memory.add(
                        f"Task: {task_description}\nResult: {result_text[:500]}",
                        metadata={"agent": self.role, "type": "task_result"},
                    )

                EventBus.emit(Event(
                    event_type=EventType.AGENT_END,
                    source_id=self.agent_id,
                    data={"result_preview": str(result_text)[:200]},
                ))
                return str(result_text)

            except Exception as exc:
                last_error = exc
                if attempt < self.max_retry_on_error:
                    log.warning("Agent retry %d/%d: %s", attempt, self.max_retry_on_error, exc)
                    continue
                break

        EventBus.emit(Event(
            event_type=EventType.AGENT_ERROR,
            source_id=self.agent_id,
            data={"error": str(last_error)},
        ))
        raise AgentError(f"Task failed after {self.max_retry_on_error} attempts: {last_error}", cause=last_error)

    def connect_to(self, other: Agent) -> None:
        """Register another agent as a peer for delegation."""
        self._peers[other.agent_id] = other
        other._peers[self.agent_id] = self

    def delegate(self, peer_id: str, task_description: str, context: Optional[str] = None) -> str:
        """Delegate a task to a connected peer agent."""
        peer = self._peers.get(peer_id)
        if peer is None:
            raise AgentError(f"No peer agent with id '{peer_id}'")
        return peer.execute_task(task_description, context)

    # ── prompt building ────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        if self.system_template:
            rendered = self.system_template.format(
                role=self.role, goal=self.goal, backstory=self.backstory,
            )
        else:
            parts = [
                f"You are: {self.role}",
                f"Your goal is: {self.goal}",
                f"Background: {self.backstory}",
            ]
            if self.allow_delegation and self._peers:
                peers_desc = ", ".join(f"{p.role}" for p in self._peers.values())
                parts.append(f"\nYou can delegate to these agents: {peers_desc}")
            rendered = "\n\n".join(parts)

        extras = []
        if self.inject_date:
            extras.append(f"Today's date is {datetime.now().strftime(self.date_format)}.")
        if self.training_context:
            extras.append(self.training_context)
        if self._training_hints:
            hints = "\n".join(f"- {h}" for h in self._training_hints)
            extras.append(f"Lessons from previous training runs:\n{hints}")

        return "\n\n".join([rendered, *extras]) if extras else rendered

    def _build_user_prompt(self, task_description: str, context: Optional[str]) -> str:
        if self.prompt_template:
            return self.prompt_template.format(
                task=task_description, context=context or "", role=self.role, goal=self.goal,
            )
        parts = []
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Task:\n{task_description}")
        parts.append("Complete this task according to your role and goal.")
        return "\n\n".join(parts)

    def _get_memory_context(self, query: str) -> str:
        if self.memory is None:
            return ""
        return self.memory.get_relevant(query, max_results=5)

    def _get_knowledge_context(self, query: str) -> str:
        """Retrieve grounding passages from the agent's knowledge sources."""
        if self.knowledge is None:
            return ""
        try:
            return self.knowledge.to_context_string(query)
        except Exception as exc:
            log.warning("Knowledge lookup failed: %s", exc)
            return ""

    def _apply_guardrails(self, text: str) -> str:
        for g in self.guardrails:
            text = g.validate(text)
        return text

    def apply_training_hints(self, hints: List[str]) -> None:
        """Attach distilled feedback from ``mangaba train`` to this agent.

        The hints are appended to the system prompt on every later run.
        """
        self._training_hints = list(hints)

    # ── execution helpers ──────────────────────────────────────────────

    def _run_react(self, system_prompt: str, user_prompt: str, context: Optional[str]) -> Any:
        """Run the ReAct loop, honouring ``max_execution_time`` when set."""
        if context and self.respect_context_window:
            context = self._trim_context(context)

        if not self.max_execution_time:
            return self._react.run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context=context,
                state=self.state,
            )

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self._react.run,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            state=self.state,
        )
        try:
            return future.result(timeout=self.max_execution_time)
        except FuturesTimeout as exc:
            raise AgentError(
                f"Agent '{self.role}' exceeded max_execution_time of {self.max_execution_time}s"
            ) from exc
        finally:
            # Don't block on a run that already timed out
            executor.shutdown(wait=False)

    def _trim_context(self, context: str) -> str:
        """Keep injected context from crowding out the task itself.

        Budgets context at roughly half the model's input allowance, estimated
        from the configured ``max_tokens`` when no explicit window is known.
        """
        max_chars = 24_000
        limit = getattr(self.llm, "max_context_chars", None)
        if isinstance(limit, int) and limit > 0:
            max_chars = limit

        if len(context) <= max_chars:
            return context

        head = context[: int(max_chars * 0.6)]
        tail = context[-int(max_chars * 0.4):]
        log.debug("Trimmed context from %d to ~%d chars", len(context), max_chars)
        return f"{head}\n\n[... context trimmed ...]\n\n{tail}"

    def _describe_images(self, images: List[ImageContent], task_description: str) -> str:
        """Turn images into text the ReAct loop can work with.

        Vision models answer about pixels; the rest of the pipeline reasons over
        text. Describing up front lets a multimodal agent use every existing
        tool without each of them needing to be image-aware.
        """
        prompt = (
            "Describe these images in detail. Focus on anything relevant to "
            f"the following task:\n{task_description}"
        )
        payload = [img.model_dump(exclude_none=True) for img in images]

        try:
            response = self.llm.generate(prompt, images=payload)
            return response.text
        except TypeError as exc:
            raise AgentError(
                "The configured LLM provider does not accept images. Use a "
                "vision-capable provider (Gemini, GPT-4o, Claude, or a "
                "vision model via Ollama) to run a multimodal agent."
            ) from exc
        except Exception as exc:
            log.warning("Image description failed: %s", exc)
            return ""

    # ── LLM factory ────────────────────────────────────────────────────
    @staticmethod
    def _create_llm(provider_str: Optional[str], llm_config: Optional[LLMConfig], api_key: Optional[str]) -> Any:
        from mangaba.core.llm import create_llm_client
    
        # Use the provided config or create a default one
        cfg = llm_config or LLMConfig()
    
        # Basic parameters
        prov = provider_str or cfg.provider
        key = api_key or cfg.api_key
        model = cfg.model
    
        # Initialize options dictionary
        options = {
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_tokens,
        }

        # If it's an OpenRouterConfig, we extract the extra fields
        if isinstance(cfg, OpenRouterConfig):
            options["site_name"] = cfg.site_name
            options["site_url"] = cfg.site_url
            if cfg.route:
                options["route"] = cfg.route

        return create_llm_client(
            provider=prov,
            api_key=key or "",
            model=model,
            **options
        )
