"""
Human-in-the-loop review for Mangaba AI.

A task marked ``human_input=True`` pauses after the agent answers and waits for
a person to approve the result or send it back with notes. Reviewers are
pluggable: the console reviewer suits local runs, while a server or queue
worker can supply its own without the framework assuming a terminal is
attached.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from mangaba.core.events import Event, EventBus, EventType
from mangaba.core.types import HumanFeedback

log = logging.getLogger(__name__)


class BaseHumanInput(ABC):
    """A reviewer that can approve or reject an agent's output."""

    @abstractmethod
    def review(self, task_description: str, output: str, agent_role: str) -> HumanFeedback:
        """Return the human's verdict on *output*."""
        ...


class ConsoleHumanInput(BaseHumanInput):
    """Ask for review on stdin.

    Falls back to approving automatically when stdin is not a terminal, so an
    unattended run never hangs waiting for input that will never arrive.

    Example::

        task = Task(..., human_input=True)   # uses this reviewer by default
    """

    def __init__(self, auto_approve_when_headless: bool = True) -> None:
        self.auto_approve_when_headless = auto_approve_when_headless

    def review(self, task_description: str, output: str, agent_role: str) -> HumanFeedback:
        if not sys.stdin.isatty():
            if self.auto_approve_when_headless:
                log.info("No terminal attached — auto-approving output from %s", agent_role)
                return HumanFeedback(approved=True)
            raise RuntimeError(
                "human_input=True requires a terminal. Pass a custom "
                "human_input_handler for non-interactive runs."
            )

        print("\n" + "=" * 68)
        print(f"REVIEW NEEDED — {agent_role}")
        print("=" * 68)
        print(f"\nTask:\n{task_description}\n")
        print(f"Output:\n{output}\n")
        print("-" * 68)
        print("Press Enter to approve, or type notes to send it back.")
        print("Prefix with 'rewrite:' to replace the output outright.")

        try:
            answer = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            log.info("Review interrupted — approving as-is")
            return HumanFeedback(approved=True)

        if not answer:
            return HumanFeedback(approved=True)

        if answer.lower().startswith("rewrite:"):
            return HumanFeedback(approved=True, revised_output=answer[len("rewrite:"):].strip())

        return HumanFeedback(approved=False, feedback=answer)


class AutoApproveHumanInput(BaseHumanInput):
    """Approve everything. Useful in tests and CI."""

    def review(self, task_description: str, output: str, agent_role: str) -> HumanFeedback:
        return HumanFeedback(approved=True)


class CallbackHumanInput(BaseHumanInput):
    """Delegate review to any callable.

    The callable receives ``(task_description, output, agent_role)`` and returns
    a :class:`HumanFeedback`, a plain string (treated as revision notes), or
    ``None`` / ``True`` to approve.

    Example::

        def review_via_slack(task, output, role):
            reply = slack.ask(f"{role} produced:\\n{output}")
            return HumanFeedback(approved=reply == "ok", feedback=reply)

        task = Task(..., human_input=True,
                    human_input_handler=CallbackHumanInput(review_via_slack))
    """

    def __init__(self, fn: Callable[[str, str, str], Any]) -> None:
        if not callable(fn):
            raise ValueError("CallbackHumanInput requires a callable")
        self.fn = fn

    def review(self, task_description: str, output: str, agent_role: str) -> HumanFeedback:
        result = self.fn(task_description, output, agent_role)

        if isinstance(result, HumanFeedback):
            return result
        if result is None or result is True:
            return HumanFeedback(approved=True)
        if result is False:
            return HumanFeedback(approved=False)
        if isinstance(result, str):
            return HumanFeedback(approved=not result.strip(), feedback=result)

        raise TypeError(f"Human review callback returned unsupported type {type(result).__name__}")


def request_review(
    handler: Optional[BaseHumanInput],
    task_description: str,
    output: str,
    agent_role: str,
) -> HumanFeedback:
    """Run a review, emitting the surrounding events."""
    reviewer = handler or ConsoleHumanInput()

    EventBus.emit(Event(
        event_type=EventType.HUMAN_INPUT_REQUEST,
        data={"agent": agent_role, "task": task_description[:200]},
    ))

    feedback = reviewer.review(task_description, output, agent_role)

    EventBus.emit(Event(
        event_type=EventType.HUMAN_INPUT_RECEIVED,
        data={"agent": agent_role, "approved": feedback.approved, "has_notes": bool(feedback.feedback)},
    ))
    return feedback
