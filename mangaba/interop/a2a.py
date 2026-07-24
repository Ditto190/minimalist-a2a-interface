"""
Open Agent2Agent (A2A) protocol for Mangaba AI.

A2A is the open standard for agents built on *different* frameworks to find
each other and work together over HTTP. This module makes a Mangaba
:class:`~mangaba.core.agent.Agent` (or :class:`~mangaba.core.crew.Crew`) both
sides of that conversation:

* :class:`A2AServer` publishes an agent — it serves the Agent Card at
  ``/.well-known/agent.json`` and answers the JSON-RPC methods
  ``message/send``, ``tasks/get`` and ``tasks/cancel``.
* :class:`A2AClient` consumes a remote agent — it fetches the card, sends
  messages, polls tasks, and via :meth:`A2AClient.as_tool` hands you a native
  :class:`~mangaba.tools.base.BaseTool` so an agent written in *any* framework
  drops straight into ``Agent(tools=[...])``.

Serving::

    from mangaba import Agent
    from mangaba.interop import A2AServer

    agent = Agent(role="Researcher", goal="Answer research questions",
                  backstory="A decade of desk research")

    with A2AServer(agent, port=9000) as server:
        print(server.url)          # http://127.0.0.1:9000/
        server.wait_forever()

Consuming::

    from mangaba.interop import A2AClient

    client = A2AClient("http://127.0.0.1:9000/")
    print(client.get_card().name)
    print(client.ask("Summarise the state of solid-state batteries"))

    # Any remote A2A agent, as a local Mangaba tool
    local = Agent(role="Editor", goal="Write the brief", backstory="...",
                  tools=[client.as_tool()])

The server is built on stdlib :mod:`http.server` and the client falls back to
stdlib :mod:`urllib`, so nothing new has to be installed; ``httpx`` or
``requests`` is used for the client when importable.

NOTE: this module implements the *open* A2A standard and is unrelated to
``protocols/a2a.py``, the in-house in-process "Agent-to-Agent" message bus
that only passes messages between agents inside a single Python process. The
two are deliberately kept separate — do not mix them.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, ConfigDict, Field

from mangaba.core.exceptions import MangabaError
from mangaba.interop.agent_card import (
    AgentCapabilities,
    AgentCard,
    agent_card_for,
    slugify,
)
from mangaba.tools.base import BaseTool

log = logging.getLogger(__name__)


#: Well-known locations a compliant client may probe for the Agent Card.
AGENT_CARD_PATHS: Tuple[str, ...] = (
    "/.well-known/agent.json",
    "/.well-known/agent-card.json",
)

#: JSON-RPC 2.0 error codes, plus the A2A-specific extensions.
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603
A2A_TASK_NOT_FOUND = -32001
A2A_TASK_NOT_CANCELABLE = -32002

#: Default seconds a client waits for a task to reach a terminal state.
DEFAULT_POLL_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class A2AError(MangabaError):
    """Generic failure while speaking the A2A protocol."""

    code: int = JSONRPC_INTERNAL_ERROR

    def __init__(self, message: str, *, code: Optional[int] = None, cause: Optional[Exception] = None) -> None:
        if code is not None:
            self.code = code
        super().__init__(message, cause=cause)


class A2ATransportError(A2AError):
    """The remote agent could not be reached or answered malformed JSON."""


class A2ARemoteError(A2AError):
    """The remote agent answered with a JSON-RPC error object."""


class A2ATaskNotFoundError(A2AError):
    """The requested task id is unknown to the server."""

    code = A2A_TASK_NOT_FOUND


class A2AInvalidParamsError(A2AError):
    """The request was well-formed JSON-RPC but its ``params`` were not usable."""

    code = JSONRPC_INVALID_PARAMS


# ---------------------------------------------------------------------------
# Wire models
# ---------------------------------------------------------------------------

def _now() -> str:
    """UTC timestamp in the ISO-8601 form the spec uses."""
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class A2ATaskState(str, Enum):
    """Lifecycle of an A2A task."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        """True once no further transition is expected."""
        return self in (
            A2ATaskState.COMPLETED,
            A2ATaskState.CANCELED,
            A2ATaskState.FAILED,
            A2ATaskState.REJECTED,
        )


#: States from which a task can still be canceled.
CANCELABLE_STATES = (A2ATaskState.SUBMITTED, A2ATaskState.WORKING, A2ATaskState.INPUT_REQUIRED)


class A2APart(BaseModel):
    """A single chunk of a message or artifact.

    Only ``text`` and ``data`` parts are produced by this implementation;
    foreign ``file`` parts are preserved on parse.

    Example::

        A2APart.text_part("hello").model_dump(by_alias=True, exclude_none=True)
        # {'kind': 'text', 'text': 'hello'}
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    kind: str = Field(default="text", description="'text', 'data' or 'file'")
    text: Optional[str] = Field(default=None)
    data: Optional[Dict[str, Any]] = Field(default=None)

    @classmethod
    def text_part(cls, text: str) -> "A2APart":
        return cls(kind="text", text=text)


class A2AMessage(BaseModel):
    """A turn in the conversation with an A2A agent.

    Example::

        A2AMessage.user("What is the capital of Brazil?")
    """

    model_config = ConfigDict(populate_by_name=True)

    role: str = Field(default="user", description="'user' or 'agent'")
    parts: List[A2APart] = Field(default_factory=list)
    message_id: str = Field(
        default_factory=lambda: _new_id("msg"),
        alias="messageId",
        serialization_alias="messageId",
    )
    task_id: Optional[str] = Field(default=None, alias="taskId", serialization_alias="taskId")
    context_id: Optional[str] = Field(default=None, alias="contextId", serialization_alias="contextId")
    kind: str = Field(default="message")

    @classmethod
    def user(cls, text: str, **kwargs: Any) -> "A2AMessage":
        return cls(role="user", parts=[A2APart.text_part(text)], **kwargs)

    @classmethod
    def agent(cls, text: str, **kwargs: Any) -> "A2AMessage":
        return cls(role="agent", parts=[A2APart.text_part(text)], **kwargs)

    @property
    def text(self) -> str:
        """Concatenate every text part of the message."""
        return "\n".join(p.text for p in self.parts if p.kind == "text" and p.text)


class A2AArtifact(BaseModel):
    """A durable output produced by a task."""

    model_config = ConfigDict(populate_by_name=True)

    artifact_id: str = Field(
        default_factory=lambda: _new_id("artifact"),
        alias="artifactId",
        serialization_alias="artifactId",
    )
    name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    parts: List[A2APart] = Field(default_factory=list)


class A2ATaskStatus(BaseModel):
    """Current state of a task, with the timestamp of the transition."""

    model_config = ConfigDict(populate_by_name=True)

    state: A2ATaskState = Field(default=A2ATaskState.SUBMITTED)
    timestamp: str = Field(default_factory=_now)
    message: Optional[A2AMessage] = Field(default=None)


class A2ATask(BaseModel):
    """A unit of work tracked by an A2A server.

    Example::

        task = A2ATask(status=A2ATaskStatus(state=A2ATaskState.WORKING))
        task.to_dict()["status"]["state"]     # 'working'
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: _new_id("task"))
    context_id: str = Field(
        default_factory=lambda: _new_id("ctx"),
        alias="contextId",
        serialization_alias="contextId",
    )
    status: A2ATaskStatus = Field(default_factory=A2ATaskStatus)
    history: List[A2AMessage] = Field(default_factory=list)
    artifacts: List[A2AArtifact] = Field(default_factory=list)
    kind: str = Field(default="task")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with the spec's camelCase field names."""
        return self.model_dump(by_alias=True, exclude_none=True, mode="json")

    @property
    def text(self) -> str:
        """Best-effort answer text: artifacts first, then the status message."""
        for artifact in self.artifacts:
            joined = "\n".join(p.text for p in artifact.parts if p.kind == "text" and p.text)
            if joined.strip():
                return joined
        if self.status.message is not None:
            return self.status.message.text
        for message in reversed(self.history):
            if message.role == "agent":
                return message.text
        return ""


# ---------------------------------------------------------------------------
# Task store
# ---------------------------------------------------------------------------

class InMemoryTaskStore:
    """Thread-safe in-memory store of tasks and their state transitions.

    Good enough for a single process. Swap it for a persistent implementation
    (same three methods) when tasks must survive a restart.

    Example::

        store = InMemoryTaskStore()
        task = store.create(A2ATask())
        store.set_state(task.id, A2ATaskState.WORKING)
        store.state_history(task.id)     # ['submitted', 'working']
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, A2ATask] = {}
        self._history: Dict[str, List[str]] = {}
        self._lock = threading.RLock()

    def create(self, task: A2ATask) -> A2ATask:
        with self._lock:
            self._tasks[task.id] = task
            self._history[task.id] = [task.status.state.value]
            return task

    def get(self, task_id: str) -> A2ATask:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise A2ATaskNotFoundError(f"Unknown task id '{task_id}'")
        return task

    def set_state(
        self,
        task_id: str,
        state: A2ATaskState,
        message: Optional[A2AMessage] = None,
        artifacts: Optional[List[A2AArtifact]] = None,
    ) -> A2ATask:
        """Move a task to *state*, recording the transition."""
        with self._lock:
            task = self.get(task_id)
            task.status = A2ATaskStatus(state=state, message=message)
            if message is not None:
                task.history.append(message)
            if artifacts:
                task.artifacts.extend(artifacts)
            self._history.setdefault(task_id, []).append(state.value)
            return task

    def state_history(self, task_id: str) -> List[str]:
        """Every state the task has been in, oldest first."""
        with self._lock:
            return list(self._history.get(task_id, []))

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._tasks)

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def _rpc_result(rpc_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _rpc_error(rpc_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": rpc_id, "error": error}


class _A2AHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that never blocks shutdown on a keep-alive client."""

    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True


class _A2ARequestHandler(BaseHTTPRequestHandler):
    """Minimal HTTP surface: the card over GET, JSON-RPC over POST."""

    server_version = "MangabaA2A/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def _a2a(self) -> "A2AServer":
        return self.server.a2a_server  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = urlparse(self.path).path
        if path in AGENT_CARD_PATHS:
            self._send_json(200, self._a2a.card.to_dict())
            return
        self._send_json(404, {"error": f"Not found: {path}"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path = urlparse(self.path).path
        if path not in (self._a2a.rpc_path, "/"):
            self._send_json(404, {"error": f"Not found: {path}"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""

        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(200, _rpc_error(None, JSONRPC_PARSE_ERROR, f"Invalid JSON: {exc}"))
            return

        self._send_json(200, self._a2a.handle_rpc(payload))

    def _send_json(self, status: int, body: Dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        log.debug("A2A %s - %s", self.address_string(), format % args)


class A2AServer:
    """Expose a Mangaba agent or crew over the open A2A protocol.

    Serves the Agent Card at ``/.well-known/agent.json`` and answers JSON-RPC
    ``message/send``, ``tasks/get`` and ``tasks/cancel`` on the root path.
    Work runs on a background worker thread, so a client sees the task move
    ``submitted → working → completed`` (or ``failed``); pass ``blocking=True``
    to run the agent inside the request and return an already-completed task.

    Built on stdlib :mod:`http.server` — no extra dependency, and no TLS or
    authentication. Put it behind a reverse proxy before exposing it publicly.

    Example::

        agent = Agent(role="Researcher", goal="Answer questions", backstory="...")

        with A2AServer(agent, port=0) as server:      # port=0 → pick a free port
            print(server.url)
            client = A2AClient(server.url)
            print(client.ask("What is A2A?"))
    """

    def __init__(
        self,
        agent: Any,
        host: str = "127.0.0.1",
        port: int = 8000,
        card: Optional[AgentCard] = None,
        rpc_path: str = "/",
        blocking: bool = False,
        task_store: Optional[InMemoryTaskStore] = None,
        agent_name: Optional[str] = None,
        agent_version: str = "1.0.0",
        public_url: Optional[str] = None,
    ) -> None:
        if not (hasattr(agent, "execute_task") or hasattr(agent, "kickoff")):
            raise TypeError(
                "A2AServer needs a Mangaba Agent (execute_task) or Crew (kickoff), "
                f"got {type(agent).__name__}"
            )
        self.agent = agent
        self.host = host
        self.port = port
        self.rpc_path = rpc_path if rpc_path.startswith("/") else f"/{rpc_path}"
        self.blocking = blocking
        self.tasks = task_store or InMemoryTaskStore()
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.public_url = public_url

        self._explicit_card = card
        self._card: Optional[AgentCard] = card
        self._httpd: Optional[_A2AHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._workers: List[threading.Thread] = []
        self._lock = threading.RLock()

    # -- lifecycle -----------------------------------------------------------

    @property
    def url(self) -> str:
        """Base URL the server is reachable at (valid once started)."""
        if self.public_url:
            return self.public_url if self.public_url.endswith("/") else self.public_url + "/"
        host = self.host if self.host not in ("0.0.0.0", "::") else "127.0.0.1"
        return f"http://{host}:{self.port}/"

    @property
    def card(self) -> AgentCard:
        """The Agent Card served at ``/.well-known/agent.json``."""
        with self._lock:
            if self._card is None:
                self._card = agent_card_for(
                    self.agent,
                    url=self.url,
                    version=self.agent_version,
                    name=self.agent_name,
                    capabilities=AgentCapabilities(state_transition_history=True),
                )
            return self._card

    def start(self) -> str:
        """Bind, start serving in a background thread and return the base URL.

        ``port=0`` asks the OS for a free port; :attr:`url` reports the real
        one afterwards.
        """
        if self._httpd is not None:
            return self.url

        httpd = _A2AHTTPServer((self.host, self.port), _A2ARequestHandler)
        httpd.a2a_server = self  # type: ignore[attr-defined]
        self.port = httpd.server_address[1]
        self._httpd = httpd

        # The card embeds the URL, so only build it once the port is known.
        with self._lock:
            self._card = self._explicit_card

        self._thread = threading.Thread(
            target=httpd.serve_forever,
            name=f"a2a-server-{self.port}",
            daemon=True,
        )
        self._thread.start()
        log.info("A2A server listening on %s", self.url)
        return self.url

    def stop(self, timeout: float = 5.0) -> None:
        """Stop serving and release the socket."""
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        for worker in list(self._workers):
            worker.join(timeout=timeout)
        self._workers.clear()
        log.info("A2A server stopped")

    def wait_forever(self) -> None:
        """Block the calling thread until interrupted (for ``python -m`` usage)."""
        self.start()
        try:
            while self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            self.stop()

    def __enter__(self) -> "A2AServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    # -- JSON-RPC ------------------------------------------------------------

    def handle_rpc(self, payload: Any) -> Dict[str, Any]:
        """Dispatch one JSON-RPC request and return the response envelope.

        Exposed separately from the HTTP layer so the protocol can be tested,
        or mounted inside another web framework, without a socket.
        """
        if not isinstance(payload, dict):
            return _rpc_error(None, JSONRPC_INVALID_REQUEST, "Request must be a JSON object")

        rpc_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            return _rpc_error(rpc_id, JSONRPC_INVALID_PARAMS, "'params' must be an object")

        handlers = {
            "message/send": self._on_message_send,
            "tasks/get": self._on_tasks_get,
            "tasks/cancel": self._on_tasks_cancel,
        }
        handler = handlers.get(str(method))
        if handler is None:
            return _rpc_error(
                rpc_id,
                JSONRPC_METHOD_NOT_FOUND,
                f"Unsupported method '{method}'. Supported: {', '.join(sorted(handlers))}",
            )

        try:
            return _rpc_result(rpc_id, handler(params))
        except A2AError as exc:
            return _rpc_error(rpc_id, exc.code, str(exc))
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the wire
            log.exception("A2A method '%s' failed", method)
            return _rpc_error(rpc_id, JSONRPC_INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    def _on_message_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raw_message = params.get("message")
        if not isinstance(raw_message, dict):
            raise A2AInvalidParamsError("'message' object is required")

        try:
            message = A2AMessage.model_validate(raw_message)
        except Exception as exc:  # noqa: BLE001 - malformed foreign payload
            raise A2AInvalidParamsError(f"Invalid message: {exc}", cause=exc)

        text = message.text.strip()
        if not text:
            raise A2AInvalidParamsError("'message.parts' must contain at least one non-empty text part")

        task = A2ATask(context_id=message.context_id or _new_id("ctx"))
        message.task_id = task.id
        message.context_id = task.context_id
        task.history.append(message)
        self.tasks.create(task)

        if self.blocking:
            self._run_task(task.id, text)
            return self.tasks.get(task.id).to_dict()

        # Snapshot before the worker starts, so the acknowledgement always
        # reports the state the task was accepted in rather than racing it.
        acknowledgement = task.to_dict()
        worker = threading.Thread(
            target=self._run_task,
            args=(task.id, text),
            name=f"a2a-task-{task.id[:12]}",
            daemon=True,
        )
        self._workers.append(worker)
        worker.start()
        return acknowledgement

    def _on_tasks_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task_id = params.get("id") or params.get("taskId")
        if not task_id:
            raise A2AInvalidParamsError("'id' is required")
        task = self.tasks.get(str(task_id))
        payload = task.to_dict()
        if self.card.capabilities.state_transition_history:
            payload["metadata"] = dict(payload.get("metadata") or {})
            payload["metadata"]["stateHistory"] = self.tasks.state_history(task.id)
        return payload

    def _on_tasks_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task_id = params.get("id") or params.get("taskId")
        if not task_id:
            raise A2AInvalidParamsError("'id' is required")
        task = self.tasks.get(str(task_id))
        if task.status.state not in CANCELABLE_STATES:
            raise A2AError(
                f"Task '{task.id}' is {task.status.state.value} and can no longer be canceled",
                code=A2A_TASK_NOT_CANCELABLE,
            )
        return self.tasks.set_state(task.id, A2ATaskState.CANCELED).to_dict()

    # -- execution -----------------------------------------------------------

    def _run_task(self, task_id: str, text: str) -> None:
        """Run the wrapped agent and record the resulting state transitions."""
        task = self.tasks.get(task_id)
        self.tasks.set_state(task_id, A2ATaskState.WORKING)
        try:
            answer = self._execute(text)
        except Exception as exc:  # noqa: BLE001 - a failure is a task state, not a crash
            log.warning("A2A task %s failed: %s", task_id, exc)
            self.tasks.set_state(
                task_id,
                A2ATaskState.FAILED,
                message=A2AMessage.agent(
                    f"{type(exc).__name__}: {exc}",
                    task_id=task_id,
                    context_id=task.context_id,
                ),
            )
            return

        answer_text = "" if answer is None else str(answer)
        artifact = A2AArtifact(
            name="response",
            parts=[A2APart.text_part(answer_text)],
        )
        self.tasks.set_state(
            task_id,
            A2ATaskState.COMPLETED,
            message=A2AMessage.agent(answer_text, task_id=task_id, context_id=task.context_id),
            artifacts=[artifact],
        )

    def _execute(self, text: str) -> str:
        """Run the wrapped Agent or Crew over the incoming message text."""
        if hasattr(self.agent, "execute_task"):
            return str(self.agent.execute_task(text))
        result = self.agent.kickoff(inputs={"input": text, "message": text, "topic": text})
        return str(getattr(result, "final_output", result))


# ---------------------------------------------------------------------------
# Client transport
# ---------------------------------------------------------------------------

def _select_backend() -> str:
    """Pick the best available HTTP client, preferring httpx then requests.

    ``requests`` is already a hard dependency of Mangaba, so the stdlib
    ``urllib`` branch only runs in unusually stripped-down environments.
    """
    try:
        import httpx  # type: ignore  # noqa: F401
        return "httpx"
    except ImportError:
        pass
    try:
        import requests  # type: ignore  # noqa: F401
        return "requests"
    except ImportError:
        pass
    return "urllib"


_BACKEND: Optional[str] = None


def _backend() -> str:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _select_backend()
        log.debug("A2A client HTTP backend: %s", _BACKEND)
    return _BACKEND


def _http_json(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Any:
    """Issue one HTTP request and decode the JSON body."""
    all_headers = {"Accept": "application/json"}
    if payload is not None:
        all_headers["Content-Type"] = "application/json"
    all_headers.update(headers or {})

    backend = _backend()
    try:
        if backend in ("httpx", "requests"):
            module = __import__(backend)
            response = module.request(
                method, url, json=payload, headers=all_headers, timeout=timeout
            )
            response.raise_for_status()
            return response.json()

        import urllib.request

        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=body, headers=all_headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - caller-supplied URL
            return json.loads(resp.read().decode("utf-8"))
    except A2AError:
        raise
    except Exception as exc:  # noqa: BLE001 - every backend raises its own type
        raise A2ATransportError(f"A2A request to {url} failed: {exc}", cause=exc)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class A2AClient:
    """Talk to a remote agent that speaks the open A2A protocol.

    The remote agent may be built on any framework — all that matters is that
    it serves an Agent Card and answers ``message/send`` / ``tasks/get``.

    Example::

        client = A2AClient("http://localhost:9000/")
        card = client.get_card()
        print(card.name, card.skill_ids())

        task = client.send_message("Summarise today's filings")
        final = client.wait_for_task(task["id"])
        print(final["status"]["state"])       # 'completed'

        # or in one call
        print(client.ask("Summarise today's filings"))
    """

    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        poll_interval: float = 0.25,
    ) -> None:
        if not url:
            raise ValueError("A2AClient needs the base URL of the remote agent")
        self.base_url = url if url.endswith("/") else url + "/"
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.poll_interval = poll_interval
        self._card: Optional[AgentCard] = None
        self._rpc_id = 0
        self._lock = threading.Lock()

    # -- discovery -----------------------------------------------------------

    @property
    def rpc_url(self) -> str:
        """Endpoint the JSON-RPC calls are POSTed to."""
        if self._card is not None and self._card.url:
            return self._card.url
        return self.base_url

    def get_card(self, refresh: bool = False) -> AgentCard:
        """Fetch (and cache) the remote Agent Card.

        Example::

            client.get_card().capabilities.streaming
        """
        if self._card is not None and not refresh:
            return self._card

        last_error: Optional[Exception] = None
        for path in AGENT_CARD_PATHS:
            card_url = urljoin(self.base_url, path.lstrip("/"))
            try:
                data = _http_json(card_url, "GET", headers=self.headers, timeout=self.timeout)
            except A2ATransportError as exc:
                last_error = exc
                continue
            try:
                self._card = AgentCard.from_dict(data)
            except Exception as exc:  # noqa: BLE001 - foreign card may be malformed
                raise A2ATransportError(f"Malformed Agent Card at {card_url}: {exc}", cause=exc)
            return self._card

        raise A2ATransportError(
            f"No Agent Card found at {self.base_url} (tried {', '.join(AGENT_CARD_PATHS)})",
            cause=last_error,
        )

    # -- JSON-RPC ------------------------------------------------------------

    def _call(self, method: str, params: Dict[str, Any]) -> Any:
        with self._lock:
            self._rpc_id += 1
            rpc_id = self._rpc_id

        response = _http_json(
            self.rpc_url,
            "POST",
            payload={"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params},
            headers=self.headers,
            timeout=self.timeout,
        )
        if not isinstance(response, dict):
            raise A2ATransportError(f"Remote agent returned a non-object response to '{method}'")
        if "error" in response and response["error"] is not None:
            error = response["error"] or {}
            code = int(error.get("code", JSONRPC_INTERNAL_ERROR))
            message = str(error.get("message", "unknown error"))
            if code == A2A_TASK_NOT_FOUND:
                raise A2ATaskNotFoundError(message)
            raise A2ARemoteError(f"Remote agent rejected '{method}': {message}", code=code)
        return response.get("result")

    def send_message(
        self,
        text: str,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send one user message; returns the Task (or Message) the server created."""
        message = A2AMessage.user(text, context_id=context_id, task_id=task_id)
        params: Dict[str, Any] = {"message": message.model_dump(by_alias=True, exclude_none=True)}
        if metadata:
            params["metadata"] = metadata
        result = self._call("message/send", params)
        if not isinstance(result, dict):
            raise A2ATransportError("Remote agent returned an unexpected 'message/send' result")
        return result

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Fetch the current state of a task."""
        result = self._call("tasks/get", {"id": task_id})
        if not isinstance(result, dict):
            raise A2ATransportError("Remote agent returned an unexpected 'tasks/get' result")
        return result

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Ask the remote agent to cancel a task that has not finished."""
        result = self._call("tasks/cancel", {"id": task_id})
        if not isinstance(result, dict):
            raise A2ATransportError("Remote agent returned an unexpected 'tasks/cancel' result")
        return result

    def wait_for_task(
        self,
        task_id: str,
        timeout: float = DEFAULT_POLL_TIMEOUT,
        poll_interval: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Poll ``tasks/get`` until the task reaches a terminal state.

        Raises:
            A2AError: If the task is still running when *timeout* elapses.
        """
        import time

        interval = poll_interval if poll_interval is not None else self.poll_interval
        deadline = time.monotonic() + timeout
        task = self.get_task(task_id)
        while True:
            state = _state_of(task)
            if state.is_terminal:
                return task
            if time.monotonic() >= deadline:
                raise A2AError(
                    f"Task '{task_id}' still {state.value} after {timeout:g}s"
                )
            time.sleep(interval)
            task = self.get_task(task_id)

    def ask(
        self,
        text: str,
        timeout: float = DEFAULT_POLL_TIMEOUT,
        context_id: Optional[str] = None,
    ) -> str:
        """Send a message, wait for the task and return the answer text.

        Raises:
            A2AError: If the remote task ends in a non-``completed`` state.
        """
        result = self.send_message(text, context_id=context_id)

        # A server may answer a trivial request with a Message instead of a Task.
        if result.get("kind") == "message" or "status" not in result:
            return A2AMessage.model_validate(result).text

        task_id = str(result.get("id") or "")
        if not task_id:
            raise A2ATransportError("Remote agent returned a task without an id")

        final = result if _state_of(result).is_terminal else self.wait_for_task(task_id, timeout=timeout)
        state = _state_of(final)
        parsed = A2ATask.model_validate(final)
        if state is not A2ATaskState.COMPLETED:
            raise A2AError(
                f"Remote task '{task_id}' ended as {state.value}: {parsed.text or 'no details'}"
            )
        return parsed.text

    # -- interop -------------------------------------------------------------

    def as_tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        timeout: float = DEFAULT_POLL_TIMEOUT,
        return_direct: bool = False,
    ) -> BaseTool:
        """Wrap the remote agent as a native Mangaba tool.

        The remote agent's card supplies the tool name and description, so a
        local agent can decide when to delegate without any extra plumbing.

        Example::

            remote = A2AClient("https://weather.example.com/").as_tool()
            agent = Agent(role="Planner", goal="Plan the trip",
                          backstory="...", tools=[remote])
        """
        card: Optional[AgentCard] = None
        try:
            card = self.get_card()
        except A2AError as exc:
            log.warning("Could not fetch the Agent Card at %s: %s", self.base_url, exc)

        if name is None:
            name = f"a2a_{slugify(card.name)}" if card is not None else "a2a_remote_agent"
        if description is None:
            if card is not None:
                skills = ", ".join(s.name for s in card.skills) or "general assistance"
                description = (
                    f"Delegate a request to the remote A2A agent '{card.name}'. "
                    f"{card.description} Skills: {skills}."
                ).strip()
            else:
                description = f"Delegate a request to the remote A2A agent at {self.base_url}"

        return A2ARemoteAgentTool(
            client=self,
            name=name,
            description=description,
            timeout=timeout,
            return_direct=return_direct,
        )

    def __repr__(self) -> str:
        return f"A2AClient(url={self.base_url!r})"


def _state_of(task: Union[Dict[str, Any], A2ATask]) -> A2ATaskState:
    """Read the task state out of a raw payload, tolerating unknown values."""
    if isinstance(task, A2ATask):
        return task.status.state
    raw = ((task.get("status") or {}) if isinstance(task, dict) else {}).get("state")
    try:
        return A2ATaskState(str(raw))
    except ValueError:
        return A2ATaskState.UNKNOWN


# ---------------------------------------------------------------------------
# Remote agent as a local tool
# ---------------------------------------------------------------------------

class RemoteAgentInput(BaseModel):
    """Arguments accepted by :class:`A2ARemoteAgentTool`."""

    message: str = Field(..., description="The request to send to the remote agent, in plain language")


class A2ARemoteAgentTool(BaseTool):
    """A remote A2A agent exposed as a native Mangaba tool.

    Built by :meth:`A2AClient.as_tool`; you rarely construct it directly.

    Example::

        tool = A2AClient("http://localhost:9000/").as_tool()
        tool.run(message="What is the weather in Maceió?")
    """

    args_schema = RemoteAgentInput

    def __init__(
        self,
        client: A2AClient,
        name: str = "a2a_remote_agent",
        description: str = "Delegate a request to a remote A2A agent",
        timeout: float = DEFAULT_POLL_TIMEOUT,
        return_direct: bool = False,
    ) -> None:
        self._client = client
        self.name = name
        self.description = description
        self.timeout = timeout
        self.return_direct = return_direct

    @property
    def client(self) -> A2AClient:
        return self._client

    def _run(self, message: str) -> str:
        return self._client.ask(message, timeout=self.timeout)


__all__ = [
    "A2A_TASK_NOT_FOUND",
    "AGENT_CARD_PATHS",
    "DEFAULT_POLL_TIMEOUT",
    "A2AArtifact",
    "A2AClient",
    "A2AError",
    "A2AInvalidParamsError",
    "A2AMessage",
    "A2APart",
    "A2ARemoteAgentTool",
    "A2ARemoteError",
    "A2AServer",
    "A2ATask",
    "A2ATaskNotFoundError",
    "A2ATaskState",
    "A2ATaskStatus",
    "A2ATransportError",
    "InMemoryTaskStore",
    "RemoteAgentInput",
]
