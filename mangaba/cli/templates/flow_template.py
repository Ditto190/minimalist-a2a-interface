"""File templates for ``mangaba create flow <name>``.

A flow project declares the same ``agents.yaml``/``tasks.yaml`` as a crew,
but ``main.py`` wires the tasks into a :class:`~mangaba.core.workflow.Pipeline`
of named stages instead of a single crew kickoff.
"""

from __future__ import annotations

from typing import Dict

from mangaba.cli.templates.crew_template import ENV_EXAMPLE

MAIN_PY = '''"""Entry point for the ${title} flow.

Usage::

    python main.py "renewable energy in Brazil"
    mangaba run --input topic="renewable energy in Brazil"
"""

from __future__ import annotations

import sys
from pathlib import Path

from mangaba.config_loader import load_project
from mangaba.core.workflow import Pipeline, Stage

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TOPIC = "${topic}"


def load_env() -> None:
    """Load a local .env file when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_DIR / ".env")


def build_pipeline(project) -> Pipeline:
    """One stage per task, in declaration order.

    Swap any ``Stage`` for ``ParallelStage`` to run its tasks concurrently,
    or wrap it in ``ConditionalStage`` to make it optional.
    """
    stages = [Stage(name, [task]) for name, task in project.tasks.items()]
    return Pipeline(stages=stages, name="${name}")


def run(topic: str = DEFAULT_TOPIC) -> int:
    """Run the flow for *topic* and print the final output."""
    load_env()

    project = load_project(str(PROJECT_DIR), verbose=True)
    pipeline = build_pipeline(project)
    result = pipeline.run({"topic": topic})

    print("")
    print("===== Final output =====")
    print(result.final_output)
    print("")
    print("Finished %d stage(s) in %.2fs" % (len(result.stages), result.duration))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC))
'''


AGENTS_YAML = '''# Agents for the ${title} flow.
# Every key becomes a mangaba.Agent — see `mangaba config` for the LLM in use.

planner:
  role: Planning Specialist
  goal: Break any request into a short, ordered list of concrete steps
  backstory: >
    A pragmatic planner who turns vague requests into steps that can actually
    be executed, without inventing work that nobody asked for.
  verbose: true
  allow_delegation: false

executor:
  role: Execution Specialist
  goal: Carry out a plan step by step and report exactly what was produced
  backstory: >
    A hands-on operator who follows a plan literally, flags anything that
    cannot be done, and never quietly skips a step.
  verbose: true
  allow_delegation: false

reviewer:
  role: Quality Reviewer
  goal: Check that the delivered work answers the original request
  backstory: >
    A reviewer with an eye for gaps: unanswered questions, unsupported claims
    and missing steps never make it past you.
  verbose: true
  allow_delegation: false
'''


TASKS_YAML = '''# Stages of the ${title} flow — one Stage per task, in this order.
# {topic} is replaced at runtime by the inputs passed to the pipeline.

plan:
  description: >
    Produce a plan to answer this request: "{topic}".
  expected_output: >
    A numbered list of 3 to 5 concrete steps.
  agent: planner

execute:
  description: >
    Execute the plan for "{topic}" and produce the actual deliverable.
  expected_output: >
    The finished deliverable, plus a one-line note per step that was skipped.
  agent: executor
  context: [plan]

review:
  description: >
    Review the deliverable for "{topic}" against the original plan.
  expected_output: >
    A short verdict (approved / needs work) followed by up to 5 specific notes.
  agent: reviewer
  context: [plan, execute]
  output_file: review.md
'''


CREW_YAML = '''# Orchestration settings for the ${title} flow.
name: ${title}
kind: flow
process: sequential
verbose: true

# Default inputs used by `mangaba run` when none are given on the CLI.
inputs:
  topic: ${topic}
'''


README_MD = '''# ${title}

A [Mangaba AI](https://github.com/mangaba-ai/mangaba-ai) flow scaffolded with
`mangaba create flow ${name}`.

A flow is a `Pipeline` of named stages: each stage runs its tasks and feeds
the next one. Stages can be sequential (`Stage`), concurrent
(`ParallelStage`) or optional (`ConditionalStage`).

## Setup

```bash
pip install "mangaba[yaml]"
cp .env.example .env      # then paste your API key
```

## Run

```bash
mangaba run                                  # uses inputs from crew.yaml
mangaba run --input topic="quantum computing"
python main.py "quantum computing"           # same thing, without the CLI
```

## Layout

| File | Purpose |
| --- | --- |
| `agents.yaml` | Declares each agent (role, goal, backstory, tools) |
| `tasks.yaml` | Declares each stage's task and which agent owns it |
| `crew.yaml` | `kind: flow`, process type and default inputs |
| `main.py` | Builds the `Pipeline` and runs it |

## Evaluate and train

```bash
mangaba test -n 2     # run twice, score every task output 1-10
mangaba train -n 3    # human-in-the-loop refinement
```
'''


PYPROJECT_TOML = '''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "${package}"
version = "0.1.0"
description = "${title} — a Mangaba AI flow"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "mangaba>=${mangaba_version}",
    "pyyaml>=6.0",
    "python-dotenv>=0.19.0",
]

[tool.hatch.build.targets.wheel]
include = ["main.py", "agents.yaml", "tasks.yaml", "crew.yaml"]
'''


#: Relative path → template body.
FILES: Dict[str, str] = {
    "main.py": MAIN_PY,
    "agents.yaml": AGENTS_YAML,
    "tasks.yaml": TASKS_YAML,
    "crew.yaml": CREW_YAML,
    ".env.example": ENV_EXAMPLE,
    "README.md": README_MD,
    "pyproject.toml": PYPROJECT_TOML,
}

__all__ = ["FILES"]
