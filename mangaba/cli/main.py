"""
Command-line interface for Mangaba AI v3.0

Implemented with stdlib :mod:`argparse` only. Every subcommand returns an
exit code; failures print an actionable message on stderr and exit non-zero.

Example::

    mangaba create crew market_research
    cd market_research && mangaba run --input topic="solar power"
    mangaba test -n 2
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


PROG = "mangaba"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

#: Local stores wiped by ``mangaba reset-memories``, per category.
MEMORY_TARGETS: Dict[str, List[str]] = {
    "short": [os.path.join(".mangaba", "short_term.json")],
    "long": ["mangaba_memory.db", os.path.join(".mangaba", "long_term.db")],
    "entity": [os.path.join(".mangaba", "entity_memory.json")],
    "knowledge": ["chroma_db", os.path.join(".mangaba", "knowledge")],
}

CHAT_HELP = """Commands:
  /help     show this help
  /reset    forget the conversation so far
  /exit     leave the chat (Ctrl-D works too)
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser, including every subcommand."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Mangaba AI — multi-agent orchestration from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  mangaba create crew market_research\n"
            "  mangaba run --input topic=\"solar power\"\n"
            "  mangaba test -n 3\n"
            "  mangaba reset-memories --all\n"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # ── create ─────────────────────────────────────────────────────────
    create = subparsers.add_parser(
        "create", help="scaffold a new project", description="Scaffold a runnable Mangaba project."
    )
    create.add_argument("kind", choices=["crew", "flow"], help="what to scaffold")
    create.add_argument("name", help="project name (becomes the directory name)")
    create.add_argument("--path", default=".", help="parent directory (default: current directory)")
    create.add_argument("--force", action="store_true", help="overwrite existing files")
    create.set_defaults(func=cmd_create)

    # ── run ────────────────────────────────────────────────────────────
    run = subparsers.add_parser(
        "run", help="run the project in the current directory",
        description="Load agents.yaml/tasks.yaml from the current directory and kick it off.",
    )
    run.add_argument("--path", default=".", help="project directory (default: current directory)")
    run.add_argument(
        "-i", "--input", action="append", default=[], metavar="KEY=VALUE",
        help="input passed to the crew (repeatable); overrides crew.yaml inputs",
    )
    run.add_argument("--no-training", action="store_true", help="ignore trained_agents_data.pkl")
    run.set_defaults(func=cmd_run)

    # ── version ────────────────────────────────────────────────────────
    version = subparsers.add_parser("version", help="print the Mangaba and Python versions")
    version.set_defaults(func=cmd_version)

    # ── test ───────────────────────────────────────────────────────────
    test = subparsers.add_parser(
        "test", help="run the crew N times and score every task output",
        description="LLM-as-judge evaluation of the project in the current directory.",
    )
    test.add_argument("--path", default=".", help="project directory (default: current directory)")
    test.add_argument("-n", "--iterations", type=int, default=2, help="number of runs (default: 2)")
    test.add_argument(
        "-i", "--input", action="append", default=[], metavar="KEY=VALUE",
        help="input passed to the crew (repeatable)",
    )
    test.set_defaults(func=cmd_test)

    # ── train ──────────────────────────────────────────────────────────
    train = subparsers.add_parser(
        "train", help="train the crew with human feedback",
        description="Run the crew iteratively, collecting your feedback after each agent output.",
    )
    train.add_argument("--path", default=".", help="project directory (default: current directory)")
    train.add_argument("-n", "--iterations", type=int, default=3, help="number of runs (default: 3)")
    train.add_argument("-f", "--filename", default=None, help="output pickle (default: trained_agents_data.pkl)")
    train.add_argument(
        "-i", "--input", action="append", default=[], metavar="KEY=VALUE",
        help="input passed to the crew (repeatable)",
    )
    train.set_defaults(func=cmd_train)

    # ── chat ───────────────────────────────────────────────────────────
    chat = subparsers.add_parser(
        "chat", help="interactive REPL against a single agent",
        description="Talk to one agent from agents.yaml, or to a general-purpose assistant.",
    )
    chat.add_argument("--path", default=".", help="project directory (default: current directory)")
    chat.add_argument("-a", "--agent", default=None, help="agent key from agents.yaml (default: the first one)")
    chat.set_defaults(func=cmd_chat)

    # ── reset-memories ─────────────────────────────────────────────────
    reset = subparsers.add_parser(
        "reset-memories", help="clear local memory stores",
        description="Delete the local files backing Mangaba's memory subsystems.",
    )
    reset.add_argument("--path", default=".", help="project directory (default: current directory)")
    reset.add_argument("--short", action="store_true", help="short-term memory")
    reset.add_argument("--long", action="store_true", help="long-term memory (SQLite)")
    reset.add_argument("--entity", action="store_true", help="entity memory")
    reset.add_argument("--knowledge", action="store_true", help="knowledge / vector stores")
    reset.add_argument("--all", action="store_true", help="everything above plus training data")
    reset.set_defaults(func=cmd_reset_memories)

    # ── replay ─────────────────────────────────────────────────────────
    replay = subparsers.add_parser(
        "replay", help="re-run the crew from a specific task id",
        description="Replay execution starting at the given task id, reusing prior outputs.",
    )
    replay.add_argument("--path", default=".", help="project directory (default: current directory)")
    replay.add_argument("-t", "--task-id", required=True, help="task id to replay from")
    replay.add_argument(
        "-i", "--input", action="append", default=[], metavar="KEY=VALUE",
        help="input passed to the crew (repeatable)",
    )
    replay.set_defaults(func=cmd_replay)

    # ── install ────────────────────────────────────────────────────────
    install = subparsers.add_parser(
        "install", help="install project dependencies",
        description="Install the extras the project template needs (yaml, documents, etc).",
    )
    install.add_argument("--path", default=".", help="project directory (default: current directory)")
    install.add_argument("--all", action="store_true", help="install mangaba[all] instead of the base extras")
    install.set_defaults(func=cmd_install)

    # ── config ─────────────────────────────────────────────────────────
    config = subparsers.add_parser(
        "config", help="show the resolved provider, model and key presence",
        description="Print which LLM would be used. API key values are never printed.",
    )
    config.set_defaults(func=cmd_config)

    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_create(args: argparse.Namespace) -> int:
    """Scaffold a runnable crew or flow project."""
    from mangaba.cli.templates import normalize_name, scaffold

    name = normalize_name(args.name)
    directory = os.path.join(args.path, name)
    created = scaffold(args.kind, name, directory, overwrite=args.force)

    print(f"Created {args.kind} '{name}' in {os.path.abspath(directory)}")
    for path in created:
        print(f"  {os.path.relpath(path, args.path)}")
    print("")
    print("Next steps:")
    print(f"  cd {os.path.relpath(directory, '.')}")
    print("  cp .env.example .env   # then paste your API key")
    print("  mangaba run")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """Load the project in the given directory and execute it."""
    from mangaba.config_loader import load_project

    project = load_project(args.path)
    inputs = _merge_inputs(project.inputs, args.input)

    if not args.no_training:
        _maybe_apply_training(project, args.path)

    if project.spec.kind == "flow":
        from mangaba.core.workflow import Pipeline, Stage

        pipeline = Pipeline(
            stages=[Stage(name, [task]) for name, task in project.tasks.items()],
            name=project.spec.name,
        )
        result = pipeline.run(inputs)
        print("")
        print("===== Final output =====")
        print(result.final_output)
        print("")
        print("Finished %d stage(s) in %.2fs" % (len(result.stages), result.duration))
        return EXIT_OK

    output = project.crew.kickoff(inputs=inputs or None)
    print("")
    print("===== Final output =====")
    print(output.final_output)
    print("")
    print("Finished %d task(s) in %.2fs" % (len(output.tasks_outputs), output.duration))
    return EXIT_OK


def cmd_version(args: argparse.Namespace) -> int:
    """Print the framework and interpreter versions."""
    from mangaba import __version__

    py = "%d.%d.%d" % sys.version_info[:3]
    print(f"mangaba {__version__} (Python {py})")
    return EXIT_OK


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay the crew from a given task id."""
    from mangaba.config_loader import load_project

    project = load_project(args.path)
    inputs = _merge_inputs(project.inputs, args.input)
    output = project.crew.replay(task_id=args.task_id, inputs=inputs or None)
    print("")
    print("===== Final output =====")
    print(output.final_output)
    print("")
    print("Replayed from task %s (%d task output(s))" % (args.task_id, len(output.tasks_outputs)))
    return EXIT_OK


def cmd_install(args: argparse.Namespace) -> int:
    """Install project dependencies via pip."""
    import subprocess

    package = "mangaba[all]" if args.all else "mangaba[yaml,documents]"
    print(f"Installing {package} ...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", package])
    return EXIT_OK if result.returncode == 0 else EXIT_ERROR


def cmd_test(args: argparse.Namespace) -> int:
    """Evaluate the project with the LLM-as-judge evaluator."""
    from mangaba.config_loader import load_project
    from mangaba.training import CrewEvaluator

    if args.iterations < 1:
        raise ValueError("-n/--iterations must be >= 1")

    project = load_project(args.path)
    inputs = _merge_inputs(project.inputs, args.input)

    evaluator = CrewEvaluator(project.crew, iterations=args.iterations, inputs=inputs, verbose=True)
    result = evaluator.evaluate()
    evaluator.print_report(result)
    return EXIT_OK if result.successful_runs else EXIT_ERROR


def cmd_train(args: argparse.Namespace) -> int:
    """Train the project's crew with human feedback."""
    from mangaba.config_loader import load_project
    from mangaba.training import DEFAULT_TRAINING_FILE, CrewTrainer

    if args.iterations < 1:
        raise ValueError("-n/--iterations must be >= 1")

    project = load_project(args.path)
    inputs = _merge_inputs(project.inputs, args.input)
    filename = args.filename or os.path.join(args.path, DEFAULT_TRAINING_FILE)

    trainer = CrewTrainer(
        project.crew,
        iterations=args.iterations,
        filename=filename,
        inputs=inputs,
        verbose=True,
    )
    result = trainer.train()
    trainer.print_report(result)
    return EXIT_OK if result.completed_iterations else EXIT_ERROR


def cmd_chat(args: argparse.Namespace) -> int:
    """Start an interactive REPL against a single agent."""
    agent = _build_chat_agent(args.path, args.agent)

    print(f"Chatting with: {agent.role}")
    print(f"Goal: {agent.goal}")
    print("")
    print(CHAT_HELP)

    history: List[str] = []
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            break

        if not message:
            continue
        if message in ("/exit", "/quit"):
            break
        if message == "/help":
            print(CHAT_HELP)
            continue
        if message == "/reset":
            history.clear()
            if agent.memory is not None:
                agent.memory.clear()
            print("(conversation reset)")
            continue

        context = "\n".join(history[-6:]) or None
        try:
            answer = agent.execute_task(message, context)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue

        print(f"{agent.role}> {answer}")
        print("")
        history.append(f"You: {message}")
        history.append(f"{agent.role}: {answer}")

    print("Bye.")
    return EXIT_OK


def cmd_reset_memories(args: argparse.Namespace) -> int:
    """Delete the local files backing the memory subsystems."""
    categories = _selected_memory_categories(args)
    if not categories:
        raise ValueError(
            "Choose what to reset: --short, --long, --entity, --knowledge or --all"
        )

    targets: List[str] = []
    for category in categories:
        targets.extend(MEMORY_TARGETS[category])
    if args.all:
        from mangaba.training import DEFAULT_TRAINING_FILE

        targets.append(DEFAULT_TRAINING_FILE)

    removed, missing = _remove_paths(args.path, targets)

    for path in removed:
        print(f"removed {path}")
    for path in missing:
        print(f"nothing to remove at {path}")

    if "short" in categories or "entity" in categories:
        print("")
        print("Note: ShortTermMemory and EntityMemory live in the process only — "
              "they are already empty in a new run.")
    return EXIT_OK


def cmd_config(args: argparse.Namespace) -> int:
    """Print the resolved LLM configuration without ever revealing a key."""
    from mangaba.config_loader import PROVIDER_KEY_ENVS, resolve_llm_settings
    from mangaba.training.evaluator import render_table

    settings = resolve_llm_settings()

    print(render_table(
        ["Setting", "Value"],
        [
            ["provider", settings.provider],
            ["model", str(settings.model)],
            ["temperature", str(settings.temperature)],
            ["max tokens", str(settings.max_tokens)],
            ["api key", f"set via {settings.api_key_env}" if settings.api_key_set else "NOT SET"],
        ],
    ))

    rows = []
    for provider, envs in sorted(PROVIDER_KEY_ENVS.items()):
        present = [name for name in envs if os.getenv(name)]
        rows.append([provider, ", ".join(envs), "set" if present else "-"])
    print("")
    print(render_table(["Provider", "Environment variables", "Status"], rows))

    if not settings.api_key_set:
        print("")
        print("No API key found for the selected provider — agents will not be able to run.")
        return EXIT_ERROR
    return EXIT_OK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_inputs(defaults: Dict[str, Any], pairs: Sequence[str]) -> Dict[str, Any]:
    """Merge ``KEY=VALUE`` CLI pairs over the project's default inputs."""
    merged: Dict[str, Any] = dict(defaults or {})
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Invalid --input '{pair}'. Use KEY=VALUE.")
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --input '{pair}'. The key cannot be empty.")
        merged[key] = value
    return merged


def _maybe_apply_training(project: Any, directory: str) -> None:
    """Apply ``trained_agents_data.pkl`` when the project ships one."""
    from mangaba.training import DEFAULT_TRAINING_FILE, apply_training_data

    filename = os.path.join(directory, DEFAULT_TRAINING_FILE)
    if not os.path.isfile(filename):
        return
    applied = apply_training_data(project.crew, filename)
    if applied:
        print(f"Applied training data to {applied} agent(s) from {DEFAULT_TRAINING_FILE}")


def _build_chat_agent(directory: str, agent_key: Optional[str]) -> Any:
    """Return the agent to chat with: from agents.yaml, or a default assistant."""
    from mangaba.config_loader import (
        AGENTS_FILE,
        AgentSpec,
        build_agent,
        build_agents,
        read_yaml,
    )
    from mangaba.core.exceptions import ConfigurationError

    agents_path = os.path.join(directory, AGENTS_FILE)
    if os.path.isfile(agents_path):
        agents = build_agents(read_yaml(agents_path))
        if agent_key is None:
            return next(iter(agents.values()))
        if agent_key not in agents:
            raise ConfigurationError(
                f"No agent '{agent_key}' in {AGENTS_FILE}. Available: {', '.join(agents)}"
            )
        return agents[agent_key]

    if agent_key is not None:
        raise ConfigurationError(
            f"--agent {agent_key} was given but no {AGENTS_FILE} exists in {os.path.abspath(directory)}."
        )

    return build_agent(AgentSpec(
        role="Mangaba Assistant",
        goal="Answer the user's questions accurately and concisely",
        backstory=(
            "A helpful general-purpose assistant. You answer directly, admit when "
            "you do not know something, and never invent facts."
        ),
        allow_delegation=False,
    ))


def _selected_memory_categories(args: argparse.Namespace) -> List[str]:
    if args.all:
        return list(MEMORY_TARGETS)
    return [name for name in MEMORY_TARGETS if getattr(args, name, False)]


def _remove_paths(directory: str, targets: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Delete files/directories under *directory*; return (removed, missing)."""
    removed: List[str] = []
    missing: List[str] = []
    for relative in targets:
        path = os.path.join(directory, relative)
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(path)
        elif os.path.exists(path):
            os.remove(path)
            removed.append(path)
        else:
            missing.append(path)
    return removed, missing


def _load_dotenv(directory: str) -> None:
    """Best-effort load of a project-local .env file."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
        return
    env_path = os.path.join(directory, ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns the process exit code.

    Example::

        from mangaba.cli.main import main

        raise SystemExit(main(["version"]))
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if getattr(args, "func", None) is None:
        parser.print_help()
        return EXIT_ERROR

    _load_dotenv(getattr(args, "path", "."))

    from mangaba.core.exceptions import MangabaError

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        print(f"{PROG}: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (MangabaError, ImportError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        log.debug("Command failed", exc_info=True)
        return EXIT_ERROR
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"{PROG}: unexpected error: {exc}", file=sys.stderr)
        log.debug("Command crashed", exc_info=True)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "build_parser", "MEMORY_TARGETS"]
