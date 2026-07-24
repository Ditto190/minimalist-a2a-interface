"""File templates for ``mangaba create crew <name>``.

Placeholders use :class:`string.Template` syntax (``${name}``) so the
generated YAML/Python may freely contain ``{}`` characters.
"""

from __future__ import annotations

from typing import Dict

MAIN_PY = '''"""Entry point for the ${title} crew.

Usage::

    python main.py "renewable energy in Brazil"
    mangaba run --input topic="renewable energy in Brazil"
"""

from __future__ import annotations

import sys
from pathlib import Path

from mangaba.config_loader import load_project

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TOPIC = "${topic}"


def load_env() -> None:
    """Load a local .env file when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_DIR / ".env")


def run(topic: str = DEFAULT_TOPIC) -> int:
    """Kick off the crew for *topic* and print the final output."""
    load_env()

    project = load_project(str(PROJECT_DIR), verbose=True)
    result = project.crew.kickoff(inputs={"topic": topic})

    print("")
    print("===== Final output =====")
    print(result.final_output)
    print("")
    print("Finished in %.2fs" % result.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC))
'''


AGENTS_YAML = '''# Agents for the ${title} crew.
# Every key becomes a mangaba.Agent — see `mangaba config` for the LLM in use.

researcher:
  role: Senior Research Analyst
  goal: Gather accurate, current and well-sourced information on any topic
  backstory: >
    A meticulous analyst who has spent a decade turning messy sources into
    clear, verifiable findings. You never invent facts and you always flag
    uncertainty.
  verbose: true
  allow_delegation: false
  # tools: [calculator, duckduckgo_search]

writer:
  role: Technical Writer
  goal: Turn research findings into a concise, well-structured report
  backstory: >
    An editor who writes for busy decision makers: short paragraphs, plain
    language, no filler, and never a claim that the research does not support.
  verbose: true
  allow_delegation: false
'''


TASKS_YAML = '''# Tasks for the ${title} crew.
# {topic} is replaced at runtime by the inputs passed to kickoff().

research_task:
  description: >
    Research the topic "{topic}". Identify the most relevant facts, current
    figures and notable viewpoints. Note anything that looks uncertain or
    disputed.
  expected_output: >
    A bullet list with 5 to 8 key findings, each one sentence long.
  agent: researcher

report_task:
  description: >
    Using the research findings, write a short report about "{topic}" for a
    reader who knows nothing about it.
  expected_output: >
    A markdown report of roughly 300 words with a title, three sections and a
    one-line conclusion.
  agent: writer
  context: [research_task]
  output_file: report.md
'''


CREW_YAML = '''# Orchestration settings for the ${title} crew.
name: ${title}
kind: crew
process: sequential   # sequential | hierarchical | parallel | consensual
verbose: true

# Default inputs used by `mangaba run` when none are given on the CLI.
inputs:
  topic: ${topic}
'''


ENV_EXAMPLE = '''# Copy to .env and fill in the key for the provider you want to use.
# Check what got resolved with: mangaba config

LLM_PROVIDER=google
MODEL_NAME=gemini-2.5-flash

GOOGLE_API_KEY=
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
# HF_TOKEN=
# OPENROUTER_API_KEY=

# Optional generation settings
# MODEL_TEMPERATURE=0.7
# MAX_OUTPUT_TOKENS=1024
'''


README_MD = '''# ${title}

A [Mangaba AI](https://github.com/mangaba-ai/mangaba-ai) crew scaffolded with
`mangaba create crew ${name}`.

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
| `tasks.yaml` | Declares each task and which agent owns it |
| `crew.yaml` | Process type, verbosity and default inputs |
| `main.py` | Loads the YAML into real objects and kicks the crew off |

## Evaluate and train

```bash
mangaba test -n 2     # run twice, score every task output 1-10
mangaba train -n 3    # human-in-the-loop refinement
```

`mangaba train` writes `trained_agents_data.pkl`; re-apply it with:

```python
from mangaba.training import apply_training_data
apply_training_data(project.crew)
```
'''


PYPROJECT_TOML = '''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "${package}"
version = "0.1.0"
description = "${title} — a Mangaba AI crew"
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
