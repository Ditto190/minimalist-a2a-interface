"""
Human-in-the-loop crew training for Mangaba AI v3.0

Runs a crew iteratively, collects human feedback on every agent output,
feeds that feedback back into the agent's prompt context on the next
iteration, and finally distils the accumulated feedback into per-agent
suggestions persisted to ``trained_agents_data.pkl``.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from mangaba.core.crew import Crew
from mangaba.core.exceptions import ConfigurationError
from mangaba.training.evaluator import llm_text, render_table

log = logging.getLogger(__name__)


DEFAULT_ITERATIONS = 3
DEFAULT_TRAINING_FILE = "trained_agents_data.pkl"

#: Delimiter used to inject (and later strip) training context in a backstory.
TRAINING_MARKER = "\n\n--- Training feedback (apply these corrections) ---\n"

DISTILL_PROMPT_TEMPLATE = """You are coaching an AI agent based on human feedback.

## Agent role
{role}

## Human feedback collected over {count} training iteration(s)
{feedback}

Write 3 to 6 short, imperative suggestions this agent must follow in future
runs so the feedback is never needed again. One suggestion per line, no
numbering, no preamble, no commentary.
"""


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class AgentTrainingData(BaseModel):
    """Everything learned about one agent during a training session."""

    role: str
    feedback: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    iterations: int = 0
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def as_context(self) -> str:
        """Render the saved knowledge as a prompt fragment."""
        parts: List[str] = []
        if self.suggestions:
            parts.append("Follow these rules learned from previous training:")
            parts.extend(f"- {s}" for s in self.suggestions)
        elif self.feedback:
            parts.append("Previous human feedback on your work:")
            parts.extend(f"- {f}" for f in self.feedback)
        return "\n".join(parts)


class TrainingResult(BaseModel):
    """Outcome of a full training session.

    Example::

        result = CrewTrainer(crew, iterations=2).train()
        print(result.agents["Senior Researcher"].suggestions)
    """

    crew_id: str = ""
    iterations: int = 0
    completed_iterations: int = 0
    duration: float = 0.0
    filename: str = DEFAULT_TRAINING_FILE
    agents: Dict[str, AgentTrainingData] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_training_data(agents: Dict[str, AgentTrainingData], filename: str = DEFAULT_TRAINING_FILE) -> str:
    """Persist training data as a pickle keyed by agent role.

    Plain dicts are pickled (not pydantic objects) so the file can be read
    back without importing Mangaba classes.

    Example::

        save_training_data({"Writer": AgentTrainingData(role="Writer")})
    """
    payload = {role: data.model_dump() for role, data in agents.items()}
    directory = os.path.dirname(os.path.abspath(filename))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filename, "wb") as fh:
        pickle.dump(payload, fh)
    log.info("Training data saved to %s (%d agents)", filename, len(payload))
    return os.path.abspath(filename)


def load_training_data(filename: str = DEFAULT_TRAINING_FILE) -> Dict[str, AgentTrainingData]:
    """Load training data previously saved by :func:`save_training_data`.

    Returns an empty mapping when the file does not exist.

    Example::

        data = load_training_data()
    """
    if not os.path.isfile(filename):
        return {}
    try:
        with open(filename, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:
        raise ConfigurationError(f"Could not read training data '{filename}': {exc}", cause=exc)

    if not isinstance(payload, dict):
        raise ConfigurationError(f"Training file '{filename}' does not contain a mapping of agent roles.")

    agents: Dict[str, AgentTrainingData] = {}
    for role, raw in payload.items():
        if isinstance(raw, AgentTrainingData):
            agents[role] = raw
        elif isinstance(raw, dict):
            agents[role] = AgentTrainingData(**raw)
        else:
            log.warning("Skipping malformed training entry for role '%s'", role)
    return agents


def set_training_context(agent: Any, context: str) -> None:
    """Attach *context* to an agent's prompt, replacing any previous block.

    ``Agent`` folds ``training_context`` into its system prompt natively, so
    the backstory is left alone — any block written by an older version is
    stripped on the way through.
    """
    agent.training_context = context or ""
    agent.backstory = agent.backstory.split(TRAINING_MARKER)[0]


def apply_training_data(target: Any, filename: str = DEFAULT_TRAINING_FILE) -> int:
    """Apply saved training data to a Crew, an Agent, or a list of agents.

    Returns the number of agents that received training context.

    Example::

        crew = load_crew(".")
        apply_training_data(crew)
        crew.kickoff()
    """
    data = load_training_data(filename)
    if not data:
        return 0

    if isinstance(target, Crew):
        agents = list(target.agents)
    elif isinstance(target, (list, tuple)):
        agents = list(target)
    else:
        agents = [target]

    applied = 0
    for agent in agents:
        entry = data.get(getattr(agent, "role", ""))
        if entry is None:
            continue
        context = entry.as_context()
        if not context:
            continue
        set_training_context(agent, context)
        applied += 1

    log.info("Applied training data to %d/%d agents from %s", applied, len(agents), filename)
    return applied


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CrewTrainer:
    """Iteratively run a crew and improve it with human feedback.

    Example::

        from mangaba.training import CrewTrainer

        trainer = CrewTrainer(crew, iterations=3)
        result = trainer.train()
        print(result.filename)
    """

    def __init__(
        self,
        crew: Crew,
        iterations: int = DEFAULT_ITERATIONS,
        filename: str = DEFAULT_TRAINING_FILE,
        inputs: Optional[Dict[str, Any]] = None,
        input_fn: Optional[Callable[[str], str]] = None,
        output_fn: Optional[Callable[..., Any]] = None,
        verbose: bool = False,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        self.crew = crew
        self.iterations = iterations
        self.filename = filename
        self.inputs = dict(inputs or {})
        self.verbose = verbose
        self._input = input_fn or input
        self._print = output_fn or print

        self._feedback: Dict[str, List[str]] = {}
        self._stopped = False

    # ── public API ─────────────────────────────────────────────────────

    def train(self) -> TrainingResult:
        """Run the full training loop and persist the distilled result."""
        result = TrainingResult(
            crew_id=self.crew.crew_id,
            iterations=self.iterations,
            filename=self.filename,
        )
        started = time.monotonic()

        for iteration in range(1, self.iterations + 1):
            self._print("")
            self._print(f"=== Training iteration {iteration}/{self.iterations} ===")
            self._inject_feedback()

            try:
                output = self.crew.kickoff(inputs=self.inputs or None)
            except Exception as exc:
                self._print(f"Iteration {iteration} failed: {exc}")
                log.warning("Training iteration %d failed: %s", iteration, exc)
                break

            result.completed_iterations = iteration
            self._collect_feedback(output.tasks_outputs, iteration)
            if self._stopped:
                self._print("Training interrupted by user — distilling what was collected so far.")
                break

        result.agents = self._distill()
        result.duration = round(time.monotonic() - started, 2)

        self._restore_agents()
        save_training_data(result.agents, self.filename)
        return result

    def print_report(self, result: TrainingResult) -> None:
        """Render the distilled suggestions as a formatted table."""
        self._print("")
        self._print(f"Training complete — {result.completed_iterations}/{result.iterations} iteration(s) "
                    f"in {result.duration}s")
        rows = [
            [role, str(len(data.feedback)), "; ".join(data.suggestions)[:80] or "(none)"]
            for role, data in result.agents.items()
        ]
        self._print("")
        self._print(render_table(["Agent", "Feedback items", "Suggestions"], rows))
        self._print("")
        self._print(f"Saved to: {os.path.abspath(result.filename)}")

    # ── internals ──────────────────────────────────────────────────────

    def _inject_feedback(self) -> None:
        """Append accumulated feedback to each agent's prompt context."""
        for agent in self.crew.agents:
            items = self._feedback.get(agent.role)
            if not items:
                continue
            context = "\n".join(f"- {item}" for item in items)
            set_training_context(agent, "Human feedback from previous iterations:\n" + context)
            if self.verbose:
                self._print(f"[train] injected {len(items)} feedback item(s) into '{agent.role}'")

    def _collect_feedback(self, task_outputs: List[Any], iteration: int) -> None:
        for task_output in task_outputs:
            role = task_output.agent
            self._print("")
            self._print(f"--- Agent: {role} (iteration {iteration}) ---")
            self._print(str(task_output.result)[:2000])
            self._print("")
            try:
                answer = self._input(
                    "Your feedback for this agent (Enter to skip, 'q' to stop training): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                self._print("")
                self._stopped = True
                return

            if answer.lower() in {"q", "quit", "exit"}:
                self._stopped = True
                return
            if answer:
                self._feedback.setdefault(role, []).append(answer)

    def _distill(self) -> Dict[str, AgentTrainingData]:
        """Turn raw feedback into short, actionable per-agent suggestions."""
        distilled: Dict[str, AgentTrainingData] = {}

        for agent in self.crew.agents:
            items = self._feedback.get(agent.role, [])
            entry = AgentTrainingData(
                role=agent.role,
                feedback=list(items),
                iterations=self.iterations,
            )
            if items:
                entry.suggestions = self._ask_llm_for_suggestions(agent, items)
            distilled[agent.role] = entry

        return distilled

    def _ask_llm_for_suggestions(self, agent: Any, items: List[str]) -> List[str]:
        prompt = DISTILL_PROMPT_TEMPLATE.format(
            role=agent.role,
            count=len(items),
            feedback="\n".join(f"- {item}" for item in items),
        )
        client = getattr(agent, "llm", None)
        if client is None:
            log.warning("Agent '%s' has no LLM client — keeping raw feedback.", agent.role)
            return list(items)

        try:
            raw = llm_text(client.generate(prompt))
        except Exception as exc:
            log.warning("Could not distil feedback for '%s': %s", agent.role, exc)
            return list(items)

        lines = [ln.strip(" -*\t") for ln in raw.splitlines()]
        suggestions = [ln for ln in lines if ln]
        return suggestions[:6] or list(items)

    def _restore_agents(self) -> None:
        """Remove the injected feedback block from every agent's backstory."""
        for agent in self.crew.agents:
            set_training_context(agent, "")


__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_TRAINING_FILE",
    "TRAINING_MARKER",
    "AgentTrainingData",
    "CrewTrainer",
    "TrainingResult",
    "apply_training_data",
    "load_training_data",
    "save_training_data",
    "set_training_context",
]
