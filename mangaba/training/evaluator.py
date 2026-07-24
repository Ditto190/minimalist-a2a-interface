"""
Crew evaluation (LLM-as-judge) for Mangaba AI v3.0

Runs a crew N times, scores every task output from 1 to 10 with an
LLM judge, aggregates per-task / per-agent scores plus wall-clock time
and renders a formatted table.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from mangaba.config_loader import get_api_key, missing_key_message, resolve_llm_settings
from mangaba.core.crew import Crew
from mangaba.core.exceptions import ConfigurationError

log = logging.getLogger(__name__)


DEFAULT_ITERATIONS = 2

JUDGE_SYSTEM_PROMPT = (
    "You are a strict quality evaluator for AI agent outputs. "
    "You grade how well an output fulfils the task and the expected output. "
    "You always answer with a single JSON object and nothing else."
)

JUDGE_PROMPT_TEMPLATE = """{system}

## Task given to the agent
{description}

## Expected output
{expected_output}

## Agent role
{agent}

## Actual output
{output}

Grade the actual output from 1 (useless) to 10 (perfect), considering
completeness, accuracy, and adherence to the expected output.

Respond with JSON only:
{{"score": <integer 1-10>, "reasoning": "<one short sentence>"}}
"""


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class TaskScore(BaseModel):
    """Score given by the judge to a single task output."""

    task_index: int
    description: str
    agent: str
    score: float = Field(ge=0.0, le=10.0)
    reasoning: str = ""
    duration: float = 0.0


class RunScore(BaseModel):
    """All task scores collected during one crew execution."""

    iteration: int
    duration: float = 0.0
    success: bool = True
    error: Optional[str] = None
    tasks: List[TaskScore] = Field(default_factory=list)

    @property
    def average(self) -> float:
        if not self.tasks:
            return 0.0
        return round(statistics.fmean([t.score for t in self.tasks]), 2)


class EvaluationResult(BaseModel):
    """Aggregated evaluation over every iteration.

    Example::

        result = CrewEvaluator(crew, iterations=2).evaluate()
        print(result.overall_score, result.agent_averages)
    """

    crew_id: str = ""
    iterations: int = 0
    total_duration: float = 0.0
    runs: List[RunScore] = Field(default_factory=list)
    task_averages: Dict[str, float] = Field(default_factory=dict)
    agent_averages: Dict[str, float] = Field(default_factory=dict)
    overall_score: float = 0.0

    @property
    def successful_runs(self) -> int:
        return sum(1 for r in self.runs if r.success)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CrewEvaluator:
    """Run a crew repeatedly and score each task output with an LLM judge.

    Example::

        from mangaba.training import CrewEvaluator

        evaluator = CrewEvaluator(crew, iterations=2)
        result = evaluator.evaluate()
        evaluator.print_report(result)
    """

    def __init__(
        self,
        crew: Crew,
        iterations: int = DEFAULT_ITERATIONS,
        judge_llm: Optional[Any] = None,
        inputs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        output_fn: Optional[Any] = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        self.crew = crew
        self.iterations = iterations
        self.inputs = dict(inputs or {})
        self.verbose = verbose
        self._print = output_fn or print
        self.judge_llm = judge_llm or self._resolve_judge_llm()

    # ── public API ─────────────────────────────────────────────────────

    def evaluate(self) -> EvaluationResult:
        """Execute the crew ``iterations`` times and return the aggregated scores."""
        result = EvaluationResult(crew_id=self.crew.crew_id, iterations=self.iterations)
        started = time.monotonic()

        for iteration in range(1, self.iterations + 1):
            if self.verbose:
                self._print(f"[evaluate] iteration {iteration}/{self.iterations}")
            result.runs.append(self._run_once(iteration))

        result.total_duration = round(time.monotonic() - started, 2)
        self._aggregate(result)
        return result

    def print_report(self, result: EvaluationResult) -> None:
        """Render the evaluation as formatted tables on stdout."""
        self._print("")
        self._print(f"Crew evaluation — {result.iterations} iteration(s), crew {result.crew_id}")
        self._print(f"Successful runs: {result.successful_runs}/{len(result.runs)}   "
                    f"Total time: {result.total_duration}s")

        run_headers = ["Iteration", "Avg score", "Duration (s)", "Status"]
        run_rows = [
            [
                str(r.iteration),
                f"{r.average:.2f}",
                f"{r.duration:.2f}",
                "ok" if r.success else f"failed: {(r.error or '')[:40]}",
            ]
            for r in result.runs
        ]
        self._print("")
        self._print(render_table(run_headers, run_rows))

        if result.task_averages:
            self._print("")
            self._print(render_table(
                ["Task", "Avg score"],
                [[name, f"{score:.2f}"] for name, score in result.task_averages.items()],
            ))

        if result.agent_averages:
            self._print("")
            self._print(render_table(
                ["Agent", "Avg score"],
                [[name, f"{score:.2f}"] for name, score in result.agent_averages.items()],
            ))

        self._print("")
        self._print(f"Overall score: {result.overall_score:.2f}/10")

    # ── internals ──────────────────────────────────────────────────────

    def _run_once(self, iteration: int) -> RunScore:
        run = RunScore(iteration=iteration)
        started = time.monotonic()
        try:
            output = self.crew.kickoff(inputs=self.inputs or None)
        except Exception as exc:
            run.duration = round(time.monotonic() - started, 2)
            run.success = False
            run.error = str(exc)
            log.warning("Evaluation iteration %d failed: %s", iteration, exc)
            return run

        run.duration = round(time.monotonic() - started, 2)

        for index, task_output in enumerate(output.tasks_outputs):
            task = self.crew.tasks[index] if index < len(self.crew.tasks) else None
            expected = task.expected_output if task is not None else "N/A"
            score, reasoning = self._judge(
                description=task_output.description,
                expected_output=expected,
                agent=task_output.agent,
                output=task_output.result,
            )
            run.tasks.append(TaskScore(
                task_index=index,
                description=task_output.description,
                agent=task_output.agent,
                score=score,
                reasoning=reasoning,
            ))
        return run

    def _judge(self, description: str, expected_output: str, agent: str, output: str) -> Tuple[float, str]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            system=JUDGE_SYSTEM_PROMPT,
            description=description[:2000],
            expected_output=expected_output[:1000],
            agent=agent,
            output=str(output)[:4000],
        )
        try:
            raw = llm_text(self.judge_llm.generate(prompt))
        except Exception as exc:
            log.warning("Judge LLM call failed: %s", exc)
            return 0.0, f"judge error: {exc}"
        return parse_score(raw)

    def _aggregate(self, result: EvaluationResult) -> None:
        by_task: Dict[str, List[float]] = {}
        by_agent: Dict[str, List[float]] = {}

        for run in result.runs:
            for task_score in run.tasks:
                label = f"{task_score.task_index + 1}. {task_score.description[:48]}"
                by_task.setdefault(label, []).append(task_score.score)
                by_agent.setdefault(task_score.agent, []).append(task_score.score)

        result.task_averages = {k: round(statistics.fmean(v), 2) for k, v in by_task.items()}
        result.agent_averages = {k: round(statistics.fmean(v), 2) for k, v in by_agent.items()}

        all_scores = [s for scores in by_task.values() for s in scores]
        result.overall_score = round(statistics.fmean(all_scores), 2) if all_scores else 0.0

    def _resolve_judge_llm(self) -> Any:
        """Reuse the first agent's LLM client, or build one from the environment."""
        for agent in self.crew.agents:
            client = getattr(agent, "llm", None)
            if client is not None:
                return client

        settings = resolve_llm_settings()
        key = get_api_key(settings)
        if not key:
            raise ConfigurationError(missing_key_message(settings))

        from mangaba.core.llm import create_llm_client

        return create_llm_client(
            provider=settings.provider,
            api_key=key,
            model=settings.model,
            temperature=0.0,
            max_output_tokens=settings.max_tokens,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def llm_text(response: Any) -> str:
    """Extract plain text from anything an LLM client may return."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    for attr in ("text", "content"):
        value = getattr(response, attr, None)
        if isinstance(value, str):
            return value
    return str(response)


def parse_score(raw: str) -> Tuple[float, str]:
    """Parse ``{"score": n, "reasoning": "..."}`` out of a judge response.

    Falls back to the first number between 1 and 10 found in the text.

    Example::

        score, reasoning = parse_score('{"score": 8, "reasoning": "solid"}')
    """
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if 0 <= start < end:
        try:
            data = json.loads(text[start:end])
            score = float(data.get("score", 0))
            return max(0.0, min(10.0, score)), str(data.get("reasoning", ""))[:300]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    match = re.search(r"\b(10(?:\.0+)?|\d(?:\.\d+)?)\b", text)
    if match:
        try:
            return max(0.0, min(10.0, float(match.group(1)))), text[:300]
        except ValueError:
            pass
    return 0.0, text[:300] or "unparseable judge response"


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a simple ASCII table (stdlib only)."""
    if not rows:
        return "(no data)"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(char: str = "-") -> str:
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"

    def fmt(cells: List[str]) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [line("="), fmt(headers), line("=")]
    out.extend(fmt(r) for r in rows)
    out.append(line("-"))
    return "\n".join(out)


__all__ = [
    "DEFAULT_ITERATIONS",
    "CrewEvaluator",
    "EvaluationResult",
    "RunScore",
    "TaskScore",
    "llm_text",
    "parse_score",
    "render_table",
]
