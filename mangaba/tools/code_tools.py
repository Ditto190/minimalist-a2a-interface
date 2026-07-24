"""
Code execution tool for Mangaba AI.

:class:`CodeInterpreterTool` runs Python written by an LLM and returns its
stdout, stderr and the value of the final expression.

SECURITY — read this before enabling it
=======================================
Executing model-authored code is the single most dangerous thing an agent can
do. Prompt injection in a web page, a PDF or a database row can become code
running with your credentials. This tool therefore:

* is **off by default** — it refuses every call until it is constructed with
  ``enabled=True``;
* runs in a **separate process**, never in the agent's interpreter, so it
  cannot reach your objects, and is killed on a wall-clock timeout;
* prefers a **Docker sandbox** (no network, read-only filesystem, memory and
  process caps) whenever a working Docker daemon is reachable;
* falls back to a plain host subprocess **only** when the caller explicitly
  passes ``unsafe_mode=True``, acknowledging that the code then runs with the
  same privileges as the agent process.

There is no safe way to run untrusted code without isolation. If neither
Docker nor an explicit ``unsafe_mode=True`` is available, the tool refuses.

Example::

    from mangaba.tools.code_tools import CodeInterpreterTool

    # Sandboxed: requires a running Docker daemon
    sandboxed = CodeInterpreterTool(enabled=True)

    # Host execution — only for code you trust, on a machine you can lose
    local = CodeInterpreterTool(enabled=True, unsafe_mode=True)
    result = local.run(code="print(2 + 2)")
    result["stdout"]        # '4\\n'
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from mangaba.tools.base import BaseTool

log = logging.getLogger(__name__)


#: Container image used by the Docker-backed mode.
DEFAULT_DOCKER_IMAGE = "python:3.11-slim"

#: Wall-clock seconds a snippet may run before it is killed.
DEFAULT_TIMEOUT = 30.0

#: Execution modes. ``auto`` picks Docker when available.
MODES = ("auto", "docker", "subprocess")

#: Runner executed in the child process: it isolates the snippet's stdout and
#: stderr and reports everything back as a single JSON object.
_RUNNER = r'''
import ast, contextlib, io, json, sys, traceback

source = sys.stdin.read()
out, err = io.StringIO(), io.StringIO()
result = None
error = None
namespace = {"__name__": "__main__"}

try:
    tree = ast.parse(source)
    tail = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        exec(compile(tree, "<mangaba>", "exec"), namespace)
        if tail is not None:
            result = eval(compile(ast.Expression(tail.value), "<mangaba>", "eval"), namespace)
except BaseException:
    error = traceback.format_exc()

sys.__stdout__.write(json.dumps({
    "stdout": out.getvalue(),
    "stderr": err.getvalue(),
    "result": None if result is None else repr(result),
    "error": error,
}))
'''


class CodeInterpreterInput(BaseModel):
    """Arguments accepted by :class:`CodeInterpreterTool`."""

    code: str = Field(..., description="Python source to execute")
    timeout: Optional[float] = Field(default=None, description="Wall-clock seconds before the run is killed")


class CodeInterpreterTool(BaseTool):
    """Execute Python and return ``stdout``, ``stderr`` and the final value.

    Disabled unless ``enabled=True``; sandboxed in Docker when a daemon is
    reachable; runs as a host subprocess only when ``unsafe_mode=True`` is
    passed explicitly. See the module docstring for the full threat model.

    The result is a dict with ``stdout``, ``stderr``, ``result`` (``repr`` of
    the last expression, or ``None``), ``error`` (traceback text or ``None``),
    ``exit_code``, ``timed_out`` and ``mode``.

    Example::

        tool = CodeInterpreterTool(enabled=True, unsafe_mode=True, timeout=10)
        out = tool.run(code="xs = [1, 2, 3]\\nprint(sum(xs))\\nmax(xs)")
        out["stdout"]   # '6\\n'
        out["result"]   # '3'
    """

    name = "code_interpreter"
    description = (
        "Execute a Python snippet in an isolated process and return its stdout, "
        "stderr and the value of the last expression"
    )
    args_schema = CodeInterpreterInput

    def __init__(
        self,
        enabled: bool = False,
        unsafe_mode: bool = False,
        mode: str = "auto",
        timeout: float = DEFAULT_TIMEOUT,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        memory_limit: str = "512m",
        allow_network: bool = False,
        python_executable: Optional[str] = None,
        max_output_chars: int = 20_000,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.enabled = enabled
        self.unsafe_mode = unsafe_mode
        self.mode = mode
        self.timeout = timeout
        self.docker_image = docker_image
        self.memory_limit = memory_limit
        self.allow_network = allow_network
        self.python_executable = python_executable or sys.executable
        self.max_output_chars = max_output_chars

    # -- mode selection ------------------------------------------------------

    @staticmethod
    def docker_available() -> bool:
        """True when a Docker CLI *and* a responding daemon are present."""
        if shutil.which("docker") is None:
            return False
        try:
            proc = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("Docker probe failed: %s", exc)
            return False
        return proc.returncode == 0

    def resolve_mode(self) -> str:
        """Decide how a run would execute, or explain why it cannot.

        Raises:
            PermissionError: If the tool is disabled or no isolation was chosen.
            RuntimeError: If the requested mode is unavailable.
        """
        if not self.enabled:
            raise PermissionError(
                "CodeInterpreterTool is disabled. Executing model-authored code is "
                "dangerous, so it must be turned on deliberately: "
                "CodeInterpreterTool(enabled=True) for the Docker sandbox, or "
                "CodeInterpreterTool(enabled=True, unsafe_mode=True) to accept "
                "running it directly on this host."
            )

        if self.mode == "docker":
            if not self.docker_available():
                raise RuntimeError(
                    "mode='docker' was requested but no Docker daemon is reachable. "
                    "Start Docker, or pass unsafe_mode=True with mode='subprocess' to "
                    "run on the host instead."
                )
            return "docker"

        if self.mode == "subprocess":
            if not self.unsafe_mode:
                raise PermissionError(
                    "mode='subprocess' runs the code on this host with the agent's own "
                    "privileges. Pass unsafe_mode=True to acknowledge that."
                )
            return "subprocess"

        # auto: an explicit unsafe_mode is the caller choosing the host on purpose.
        if self.unsafe_mode:
            return "subprocess"
        if self.docker_available():
            return "docker"
        raise PermissionError(
            "No isolated runtime is available: Docker is not running, and host "
            "execution was not authorised. Start Docker, or construct the tool with "
            "unsafe_mode=True to accept running the code on this machine."
        )

    # -- execution -----------------------------------------------------------

    def _run(self, code: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        try:
            mode = self.resolve_mode()
        except (PermissionError, RuntimeError) as exc:
            return self._failure(str(exc), mode="refused")

        if not (code or "").strip():
            return self._failure("No code was provided.", mode=mode)

        limit = float(timeout or self.timeout)
        command = self._docker_command() if mode == "docker" else self._subprocess_command()

        try:
            proc = subprocess.run(
                command,
                input=code.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=limit,
            )
        except subprocess.TimeoutExpired:
            return dict(
                self._failure(f"Execution exceeded the {limit:g}s timeout and was killed.", mode=mode),
                timed_out=True,
            )
        except OSError as exc:
            return self._failure(f"Could not start the {mode} runner: {exc}", mode=mode)

        return self._parse(proc, mode)

    def _docker_command(self) -> List[str]:
        command = [
            "docker", "run", "--rm", "--interactive",
            "--network", "bridge" if self.allow_network else "none",
            "--memory", self.memory_limit,
            "--pids-limit", "128",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m",
            "--workdir", "/tmp",
            self.docker_image,
            "python", "-c", _RUNNER,
        ]
        return command

    def _subprocess_command(self) -> List[str]:
        return [self.python_executable, "-I", "-c", _RUNNER]

    def _parse(self, proc: "subprocess.CompletedProcess[bytes]", mode: str) -> Dict[str, Any]:
        raw = proc.stdout.decode("utf-8", errors="replace")
        runner_stderr = proc.stderr.decode("utf-8", errors="replace")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # The runner itself died (image missing, interpreter crash, OOM kill).
            return self._failure(
                runner_stderr.strip() or f"The {mode} runner exited without a result.",
                mode=mode,
                exit_code=proc.returncode,
                stdout=raw,
            )

        return {
            "stdout": self._clip(payload.get("stdout") or ""),
            "stderr": self._clip((payload.get("stderr") or "") + runner_stderr),
            "result": payload.get("result"),
            "error": payload.get("error"),
            "exit_code": proc.returncode,
            "timed_out": False,
            "mode": mode,
        }

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + f"\n[... truncated at {self.max_output_chars} characters ...]"

    @staticmethod
    def _failure(
        message: str,
        mode: str,
        exit_code: Optional[int] = None,
        stdout: str = "",
    ) -> Dict[str, Any]:
        log.debug("CodeInterpreterTool refused or failed (%s): %s", mode, message)
        return {
            "stdout": stdout,
            "stderr": "",
            "result": None,
            "error": message,
            "exit_code": exit_code,
            "timed_out": False,
            "mode": mode,
        }


__all__ = [
    "DEFAULT_DOCKER_IMAGE",
    "DEFAULT_TIMEOUT",
    "MODES",
    "CodeInterpreterInput",
    "CodeInterpreterTool",
]
