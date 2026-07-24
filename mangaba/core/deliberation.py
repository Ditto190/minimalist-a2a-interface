"""
Pre-execution deliberation for Mangaba AI.

Where the ReAct engine reasons *while* acting, the Deliberator reasons
*before* acting: it drafts a plan for the task, judges whether that plan is
actually good enough to execute, and refines it until it is — or until the
attempt budget runs out. Enabled per agent with ``Agent(reasoning=True)``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Tuple

from mangaba.core.events import Event, EventBus, EventType
from mangaba.core.types import ReasoningOutput

log = logging.getLogger(__name__)


class Deliberator:
    """Draft-and-critique loop run before task execution.

    Example::

        deliberator = Deliberator(llm=llm_client, max_attempts=3)
        outcome = deliberator.deliberate(
            role="Security Analyst",
            goal="Find exploitable flaws",
            task="Audit this authentication module",
            tools=[read_file, grep],
        )
        print(outcome.plan, outcome.ready)
    """

    PLAN_PROMPT = (
        "You are {role}. Your goal: {goal}\n\n"
        "Before acting, draft a short plan for the task below.\n"
        "Available tools: {tools}\n\n"
        "TASK:\n{task}\n\n"
        "Write the plan as numbered steps. Be concrete about which tool each "
        "step uses and what it should produce. Do not execute anything yet."
    )

    REFINE_PROMPT = (
        "You are {role}. Your goal: {goal}\n\n"
        "Your previous plan was judged not ready to execute.\n\n"
        "TASK:\n{task}\n\n"
        "PREVIOUS PLAN:\n{plan}\n\n"
        "PROBLEMS FOUND:\n{critique}\n\n"
        "Available tools: {tools}\n\n"
        "Write an improved plan that fixes those problems."
    )

    CRITIQUE_PROMPT = (
        "Judge whether this plan is ready to execute.\n\n"
        "TASK:\n{task}\n\n"
        "PLAN:\n{plan}\n\n"
        "A plan is ready when its steps are concrete, ordered, achievable with "
        "the available tools ({tools}), and together they actually satisfy the "
        "task. It is not ready if it is vague, skips a required step, or "
        "depends on information nobody has.\n\n"
        'Respond ONLY with JSON: {{"ready": true, "critique": ""}} '
        'or {{"ready": false, "critique": "<what is wrong>"}}'
    )

    def __init__(self, llm: Any, max_attempts: int = 3, verbose: bool = False) -> None:
        self.llm = llm
        self.max_attempts = max(1, max_attempts)
        self.verbose = verbose

    # ── public API ─────────────────────────────────────────────────────

    def deliberate(
        self,
        role: str,
        goal: str,
        task: str,
        tools: Optional[List[Any]] = None,
    ) -> ReasoningOutput:
        """Produce a plan the agent considers ready to execute.

        Returns the last plan drafted even when the agent never declared
        itself ready — a mediocre plan still beats no plan, and the caller can
        inspect ``ready`` to decide how much to trust it.
        """
        tools_str = self._describe_tools(tools)
        plan = ""
        critique = ""

        for attempt in range(1, self.max_attempts + 1):
            prompt = (
                self.PLAN_PROMPT.format(role=role, goal=goal, task=task, tools=tools_str)
                if attempt == 1
                else self.REFINE_PROMPT.format(
                    role=role, goal=goal, task=task, plan=plan, critique=critique, tools=tools_str
                )
            )

            try:
                plan = self.llm.generate_text(prompt).strip()
            except Exception as exc:
                log.warning("Deliberation failed on attempt %d: %s", attempt, exc)
                return ReasoningOutput(plan=plan, ready=False, attempts=attempt, critique=str(exc))

            ready, critique = self._critique(task, plan, tools_str)

            EventBus.emit(Event(
                event_type=EventType.REACT_THOUGHT,
                data={"phase": "deliberation", "attempt": attempt, "ready": ready},
            ))

            if self.verbose:
                log.info("Deliberation attempt %d/%d — ready=%s", attempt, self.max_attempts, ready)

            if ready:
                return ReasoningOutput(plan=plan, ready=True, attempts=attempt)

        return ReasoningOutput(plan=plan, ready=False, attempts=self.max_attempts, critique=critique)

    # ── internal ───────────────────────────────────────────────────────

    def _critique(self, task: str, plan: str, tools_str: str) -> Tuple[bool, str]:
        prompt = self.CRITIQUE_PROMPT.format(task=task, plan=plan, tools=tools_str)
        try:
            raw = self.llm.generate_text(prompt)
        except Exception as exc:
            log.warning("Plan critique failed: %s", exc)
            # Can't judge it — accept the plan rather than burning the budget
            return True, ""

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            return bool(data.get("ready", False)), str(data.get("critique", ""))
        except (ValueError, json.JSONDecodeError):
            lowered = raw.lower()
            if "not ready" in lowered or '"ready": false' in lowered or "false" in lowered:
                return False, raw.strip()[:500]
            return True, ""

    @staticmethod
    def _describe_tools(tools: Optional[List[Any]]) -> str:
        if not tools:
            return "none"
        names = []
        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", None) or str(t)
            description = getattr(t, "description", "")
            names.append(f"{name} ({description})" if description else str(name))
        return ", ".join(names)
