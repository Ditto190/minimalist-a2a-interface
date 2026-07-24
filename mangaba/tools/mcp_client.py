from __future__ import annotations

"""
Model Context Protocol (MCP) client for Mangaba AI v3.0

Connects to external MCP servers — Anthropic's open protocol for exposing
tools, resources and prompts to LLM applications — and wraps every remote
tool as a native :class:`~mangaba.tools.base.BaseTool`, so it drops straight
into ``Agent(tools=[...])``::

    from mangaba.tools.mcp_client import MCPClient

    with MCPClient(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as mcp:
        agent = Agent(role="Analyst", goal="...", backstory="...", tools=mcp.get_tools())

Two transports are supported:

* **stdio** – the server runs as a child process and speaks JSON-RPC 2.0
  over stdin/stdout (the usual way MCP servers are distributed).
* **HTTP** – JSON-RPC 2.0 over HTTP POST, with ``text/event-stream``
  (SSE) responses handled transparently.

If the official ``mcp`` package is installed it is used for the stdio
transport; otherwise a dependency-free implementation built on stdlib
``subprocess`` + ``json`` takes over, so the feature always works.

NOTE: this module is unrelated to ``protocols/mcp.py``, which implements an
in-house "Multi-Context Protocol" for context management.
"""

import json
import logging
import os
import subprocess
import threading
from abc import ABC, abstractmethod
from collections import deque
from itertools import count
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from mangaba.core.exceptions import MangabaError, ToolError, ToolValidationError
from mangaba.tools.base import BaseTool

log = logging.getLogger(__name__)


#: Protocol revision advertised during the handshake.
MCP_PROTOCOL_VERSION = "2024-11-05"

#: Identity sent to the server in ``initialize``.
MCP_CLIENT_NAME = "mangaba-ai"
MCP_CLIENT_VERSION = "3.0"

#: Default per-request timeout, in seconds.
DEFAULT_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MCPError(MangabaError):
    """Generic failure while talking to an MCP server."""


class MCPConnectionError(MCPError):
    """The MCP server could not be started, reached or kept alive."""


class MCPTimeoutError(MCPError):
    """The MCP server did not answer within the configured timeout."""


class MCPProtocolError(MCPError):
    """The MCP server answered with a malformed or error response."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_command_list(
    command: Union[str, Sequence[str], None],
    args: Optional[Sequence[str]],
) -> List[str]:
    """Normalise ``command``/``args`` into a single argv list."""
    if command is None:
        raise ValueError("MCPClient requires either 'command' (stdio) or 'url' (HTTP)")
    if isinstance(command, str):
        argv = [command]
    else:
        argv = [str(part) for part in command]
    if args:
        argv.extend(str(a) for a in args)
    return argv


def _model_to_dict(obj: Any) -> Any:
    """Best-effort conversion of a pydantic model (official SDK) to plain data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _model_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_model_to_dict(v) for v in obj]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(by_alias=True, mode="json", exclude_none=True)
        except TypeError:  # pragma: no cover - pydantic v1 style
            return dump()
    if hasattr(obj, "__dict__"):
        return {k: _model_to_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _official_mcp_available() -> bool:
    """True when the optional official ``mcp`` package can be imported."""
    try:
        import mcp  # type: ignore  # noqa: F401
        from mcp.client.stdio import stdio_client  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Session interface
# ---------------------------------------------------------------------------

class _MCPSession(ABC):
    """Transport-agnostic view of a connected MCP server."""

    @abstractmethod
    def start(self) -> None:
        """Open the transport (spawn the process / probe the endpoint)."""

    @abstractmethod
    def initialize(self) -> Dict[str, Any]:
        """Perform the MCP handshake and return the server's ``initialize`` result."""

    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the raw tool descriptors advertised by the server."""

    @abstractmethod
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke a remote tool and return the raw ``tools/call`` result."""

    @abstractmethod
    def close(self) -> None:
        """Tear the transport down. Must be idempotent."""


class _JsonRpcSession(_MCPSession):
    """Shared MCP semantics for transports that expose raw JSON-RPC 2.0."""

    transport_name = "json-rpc"

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout
        self._ids = count(1)

    # -- low level (implemented by concrete transports) ----------------------

    @abstractmethod
    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

    @abstractmethod
    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        ...

    def _next_id(self) -> int:
        return next(self._ids)

    @staticmethod
    def _unwrap(response: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Return the ``result`` payload or raise on a JSON-RPC error object."""
        if not isinstance(response, dict):
            raise MCPProtocolError(f"MCP server returned a non-object response to '{method}'")
        error = response.get("error")
        if error:
            code = error.get("code") if isinstance(error, dict) else ""
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise MCPProtocolError(f"MCP server error on '{method}' (code {code}): {message}")
        result = response.get("result")
        if result is None:
            return {}
        if not isinstance(result, dict):
            return {"value": result}
        return result

    # -- MCP operations ------------------------------------------------------

    def initialize(self) -> Dict[str, Any]:
        params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"roots": {"listChanged": False}},
            "clientInfo": {"name": MCP_CLIENT_NAME, "version": MCP_CLIENT_VERSION},
        }
        result = self._unwrap(self._send_request("initialize", params), "initialize")
        # Required by the spec: tell the server the handshake is complete.
        try:
            self._send_notification("notifications/initialized")
        except Exception as exc:  # pragma: no cover - best effort
            log.debug("MCP: could not send notifications/initialized: %s", exc)
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            result = self._unwrap(self._send_request("tools/list", params or None), "tools/list")
            page = result.get("tools") or []
            tools.extend(t for t in page if isinstance(t, dict))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        params = {"name": name, "arguments": arguments or {}}
        return self._unwrap(self._send_request("tools/call", params), "tools/call")


# ---------------------------------------------------------------------------
# stdio transport (dependency-free)
# ---------------------------------------------------------------------------

class _StdioSession(_JsonRpcSession):
    """Spawn an MCP server and speak newline-delimited JSON-RPC 2.0 over stdio."""

    transport_name = "stdio"

    def __init__(
        self,
        argv: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        inherit_env: bool = True,
    ) -> None:
        super().__init__(timeout=timeout)
        self._argv = argv
        self._cwd = cwd
        self._inherit_env = inherit_env
        self._env_overrides = dict(env or {})
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._stderr_reader: Optional[threading.Thread] = None
        self._stderr_tail: deque = deque(maxlen=20)
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._waiters: Dict[int, threading.Event] = {}
        self._responses: Dict[int, Dict[str, Any]] = {}
        self._eof_reason: Optional[str] = None
        self._closed = False

    # -- lifecycle -----------------------------------------------------------

    def _build_env(self) -> Dict[str, str]:
        env = dict(os.environ) if self._inherit_env else {}
        env.update(self._env_overrides)
        return env

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                self._argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._cwd,
                env=self._build_env(),
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise MCPConnectionError(
                f"MCP server command not found: {self._argv[0]!r}. "
                "Check that it is installed and on PATH.",
                cause=exc,
            ) from exc
        except OSError as exc:
            raise MCPConnectionError(
                f"Could not start MCP server {' '.join(self._argv)!r}: {exc}", cause=exc
            ) from exc

        self._reader = threading.Thread(target=self._reader_loop, name="mangaba-mcp-stdout", daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._stderr_loop, name="mangaba-mcp-stderr", daemon=True)
        self._stderr_reader.start()
        log.debug("MCP: started stdio server %s (pid=%s)", self._argv, self._proc.pid)

    def close(self) -> None:
        proc = self._proc
        if proc is None or self._closed:
            return
        self._closed = True
        try:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except (OSError, ValueError):
                    pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.debug("MCP: server did not exit on stdin close, terminating")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
                    log.warning("MCP: server ignored SIGTERM, killing pid %s", proc.pid)
                    proc.kill()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            for thread in (self._reader, self._stderr_reader):
                if thread is not None and thread.is_alive():
                    thread.join(timeout=2)
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream is not None and not stream.closed:
                        stream.close()
                except (OSError, ValueError):  # pragma: no cover
                    pass
            self._fail_pending("MCP server connection closed")
            log.debug("MCP: stdio server reaped (returncode=%s)", proc.returncode)

    # -- reader threads ------------------------------------------------------

    def _reader_loop(self) -> None:
        stream = self._proc.stdout if self._proc else None
        if stream is None:  # pragma: no cover
            return
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("MCP: ignoring non-JSON line from server: %s", line[:200])
                    continue
                self._dispatch(message)
        except (OSError, ValueError) as exc:  # stream closed under us
            log.debug("MCP: stdout reader stopped: %s", exc)
        finally:
            self._fail_pending("MCP server closed its stdout")

    def _stderr_loop(self) -> None:
        stream = self._proc.stderr if self._proc else None
        if stream is None:  # pragma: no cover
            return
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_tail.append(line)
                    log.debug("MCP[stderr]: %s", line)
        except (OSError, ValueError):  # pragma: no cover
            pass

    def _dispatch(self, message: Dict[str, Any]) -> None:
        msg_id = message.get("id")
        if msg_id is None:
            log.debug("MCP: notification from server: %s", message.get("method"))
            return
        try:
            key = int(msg_id)
        except (TypeError, ValueError):  # pragma: no cover - non-numeric ids
            log.debug("MCP: unexpected response id %r", msg_id)
            return
        with self._state_lock:
            event = self._waiters.get(key)
            if event is None:
                log.debug("MCP: response for unknown id %s discarded", key)
                return
            self._responses[key] = message
            event.set()

    def _fail_pending(self, reason: str) -> None:
        with self._state_lock:
            self._eof_reason = reason
            pending = list(self._waiters.items())
            for key, event in pending:
                self._responses.setdefault(key, {})
                event.set()

    # -- JSON-RPC ------------------------------------------------------------

    def _write(self, payload: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            tail = "; ".join(self._stderr_tail)
            raise MCPConnectionError(
                "MCP server process is not running" + (f" (stderr: {tail})" if tail else "")
            )
        data = (json.dumps(payload) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise MCPConnectionError(f"Lost connection to MCP server: {exc}", cause=exc) from exc

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        request_id = self._next_id()
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params

        event = threading.Event()
        with self._state_lock:
            self._waiters[request_id] = event

        try:
            self._write(payload)
            if not event.wait(self._timeout):
                raise MCPTimeoutError(
                    f"MCP server did not answer '{method}' within {self._timeout:g}s"
                )
            with self._state_lock:
                response = self._responses.pop(request_id, None)
            if not response:
                tail = "; ".join(self._stderr_tail)
                reason = self._eof_reason or "MCP server closed the connection"
                raise MCPConnectionError(reason + (f" (stderr: {tail})" if tail else ""))
            return response
        finally:
            with self._state_lock:
                self._waiters.pop(request_id, None)
                self._responses.pop(request_id, None)

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._write(payload)


# ---------------------------------------------------------------------------
# HTTP / SSE transport (dependency-free)
# ---------------------------------------------------------------------------

class _HttpSession(_JsonRpcSession):
    """JSON-RPC 2.0 over HTTP POST, accepting ``application/json`` or SSE replies."""

    transport_name = "http"

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(timeout=timeout)
        self.url = url
        self._headers = dict(headers or {})
        self._session_id: Optional[str] = None

    def start(self) -> None:
        # HTTP is connectionless: nothing to open, the handshake proves reachability.
        return None

    def close(self) -> None:
        self._session_id = None

    def _post(self, payload: Dict[str, Any], expect_response: bool = True) -> Optional[Dict[str, Any]]:
        import urllib.error
        import urllib.request

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        headers.update(self._headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # nosec B310
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self._session_id = session_id
                content_type = (response.headers.get("Content-Type") or "").lower()
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # pragma: no cover
                pass
            raise MCPConnectionError(
                f"MCP server at {self.url} answered HTTP {exc.code}"
                + (f": {detail}" if detail else ""),
                cause=exc,
            ) from exc
        except Exception as exc:
            raise MCPConnectionError(
                f"Could not reach the MCP server at {self.url}: {exc}", cause=exc
            ) from exc

        if not expect_response or not body.strip():
            return None
        if "text/event-stream" in content_type:
            return self._parse_sse(body, payload.get("id"))
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise MCPProtocolError(
                f"MCP server at {self.url} returned invalid JSON: {body[:200]}", cause=exc
            ) from exc

    @staticmethod
    def _parse_sse(body: str, request_id: Any) -> Dict[str, Any]:
        """Pull the JSON-RPC message matching ``request_id`` out of an SSE stream."""
        last: Optional[Dict[str, Any]] = None
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if not chunk or chunk == "[DONE]":
                continue
            try:
                message = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            if message.get("id") == request_id:
                return message
            if "result" in message or "error" in message:
                last = message
        if last is not None:
            return last
        raise MCPProtocolError("MCP server sent an SSE stream without a JSON-RPC response")

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        request_id = self._next_id()
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        response = self._post(payload, expect_response=True)
        if response is None:
            raise MCPProtocolError(f"MCP server returned an empty body for '{method}'")
        return response

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._post(payload, expect_response=False)


# ---------------------------------------------------------------------------
# stdio transport backed by the official ``mcp`` package (optional)
# ---------------------------------------------------------------------------

class _OfficialStdioSession(_MCPSession):
    """Drive the official async ``mcp`` SDK from a private event-loop thread."""

    transport_name = "stdio (official mcp SDK)"

    def __init__(
        self,
        argv: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        inherit_env: bool = True,
    ) -> None:
        self._argv = argv
        self._env_overrides = dict(env or {})
        self._cwd = cwd
        self._timeout = timeout
        self._inherit_env = inherit_env
        self._loop: Any = None
        self._thread: Optional[threading.Thread] = None
        self._session: Any = None
        self._closing: Any = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._server_info: Dict[str, Any] = {}
        self._closed = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        import asyncio

        try:
            import mcp  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover - guarded by caller
            raise ImportError(
                "Package 'mcp' not found. Install with: pip install mcp"
            ) from exc

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="mangaba-mcp-loop", daemon=True)
        self._thread.start()
        if not self._ready.wait(self._timeout):
            self.close()
            raise MCPTimeoutError(
                f"MCP server {' '.join(self._argv)!r} did not complete the handshake "
                f"within {self._timeout:g}s"
            )
        if self._error is not None:
            error = self._error
            self.close()
            raise MCPConnectionError(f"Could not start MCP server: {error}", cause=error if isinstance(error, Exception) else None)

    def _run_loop(self) -> None:
        import asyncio

        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except BaseException as exc:  # pragma: no cover - defensive
            self._error = exc
            self._ready.set()
        finally:
            try:
                self._loop.close()
            except Exception:  # pragma: no cover
                pass

    async def _main(self) -> None:
        import asyncio

        from mcp import ClientSession, StdioServerParameters  # type: ignore
        from mcp.client.stdio import stdio_client  # type: ignore

        self._closing = asyncio.Event()
        env = dict(os.environ) if self._inherit_env else {}
        env.update(self._env_overrides)
        params = StdioServerParameters(
            command=self._argv[0],
            args=list(self._argv[1:]),
            env=env,
            cwd=self._cwd,
        )
        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    init_result = await session.initialize()
                    self._session = session
                    info = _model_to_dict(init_result)
                    self._server_info = info if isinstance(info, dict) else {"result": info}
                    self._ready.set()
                    await self._closing.wait()
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._session = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop, closing = self._loop, self._closing
        if loop is not None and closing is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(closing.set)
            except RuntimeError:  # pragma: no cover - loop already gone
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._session = None

    # -- MCP operations ------------------------------------------------------

    def _await(self, coro: Any) -> Any:
        import asyncio

        if self._session is None or self._loop is None:
            raise MCPConnectionError("MCP session is not connected")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=self._timeout)
        except TimeoutError as exc:
            future.cancel()
            raise MCPTimeoutError(f"MCP request timed out after {self._timeout:g}s", cause=exc) from exc
        except Exception as exc:
            raise MCPError(f"MCP request failed: {exc}", cause=exc) from exc

    def initialize(self) -> Dict[str, Any]:
        # The handshake already happened inside ``start``.
        return self._server_info

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self._await(self._session.list_tools())
        tools = []
        for remote in getattr(result, "tools", None) or []:
            data = _model_to_dict(remote)
            if isinstance(data, dict):
                tools.append(data)
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        result = self._await(self._session.call_tool(name, arguments or {}))
        return _model_to_dict(result)


# ---------------------------------------------------------------------------
# Remote tool wrapper
# ---------------------------------------------------------------------------

class MCPTool(BaseTool):
    """A tool hosted by an MCP server, exposed as a native Mangaba tool.

    The remote JSON Schema is forwarded verbatim to the LLM (via
    :meth:`get_function_schema`) rather than being re-derived from a Pydantic
    model, so nothing is lost in translation. Validation of the payload is left
    to the server, which owns the schema; only ``required`` keys are checked
    locally to fail fast with a clear message.
    """

    def __init__(
        self,
        client: "MCPClient",
        tool_name: str,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
        display_name: Optional[str] = None,
    ) -> None:
        self._client = client
        self.remote_name = tool_name
        self.name = display_name or tool_name
        self.description = description or f"MCP tool '{tool_name}'"
        self.input_schema: Dict[str, Any] = dict(input_schema or {}) or {
            "type": "object",
            "properties": {},
        }
        self.args_schema = None
        self.return_direct = False

    # -- schema --------------------------------------------------------------

    def get_function_schema(self) -> Dict[str, Any]:
        parameters = dict(self.input_schema)
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }

    # -- execution -----------------------------------------------------------

    def _run(self, **kwargs: Any) -> Any:
        required = self.input_schema.get("required") or []
        missing = [key for key in required if key not in kwargs]
        if missing:
            raise ToolValidationError(
                f"Missing required argument(s) for MCP tool '{self.name}': {', '.join(missing)}",
                tool_name=self.name,
            )
        result = self._client.call_tool(self.remote_name, kwargs)
        return self._flatten_result(result)

    def _flatten_result(self, result: Any) -> Any:
        """Turn an MCP ``tools/call`` result into a plain Python value."""
        if not isinstance(result, dict):
            return result

        blocks = result.get("content") or []
        texts: List[str] = []
        others: List[Any] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            else:
                others.append(block)
        joined = "\n".join(texts)

        if result.get("isError"):
            raise ToolError(
                f"MCP tool '{self.name}' failed: {joined or result}",
                tool_name=self.name,
            )

        structured = result.get("structuredContent")
        if structured is not None:
            return structured
        if "content" not in result:
            return result
        if others and not texts:
            return others
        if others:
            return {"text": joined, "content": others}
        return joined

    def __repr__(self) -> str:
        return f"MCPTool(name='{self.name}', server='{self._client.name}')"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class MCPClient:
    """Client for an external MCP (Model Context Protocol) server.

    stdio transport (server runs as a child process)::

        with MCPClient(command="python", args=["my_server.py"]) as mcp:
            tools = mcp.get_tools()

    HTTP/SSE transport::

        with MCPClient(url="http://localhost:3000/mcp") as mcp:
            tools = mcp.get_tools()

    Args:
        command: Executable (or full argv list) of a stdio MCP server.
        args: Extra arguments appended to ``command``.
        url: Endpoint of an HTTP MCP server. Mutually exclusive with ``command``.
        headers: Extra HTTP headers (e.g. ``{"Authorization": "Bearer ..."}``).
        env: Environment overrides for the spawned server.
        cwd: Working directory for the spawned server.
        inherit_env: Whether the child process inherits ``os.environ`` (default True).
        timeout: Per-request timeout in seconds.
        name: Friendly label used in logs and tool reprs.
        tool_prefix: Prefix prepended to every tool name, to avoid collisions
            when several servers are attached to the same agent.
        prefer_official: Use the official ``mcp`` package when it is importable;
            otherwise (or on failure) the built-in stdio transport is used.
    """

    def __init__(
        self,
        command: Union[str, Sequence[str], None] = None,
        args: Optional[Sequence[str]] = None,
        *,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        inherit_env: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        name: Optional[str] = None,
        tool_prefix: Optional[str] = None,
        prefer_official: bool = True,
    ) -> None:
        if command is None and not url:
            raise ValueError("MCPClient requires either 'command' (stdio) or 'url' (HTTP)")
        if command is not None and url:
            raise ValueError("MCPClient accepts 'command' or 'url', not both")

        self.url = url
        self.argv: List[str] = [] if url else _as_command_list(command, args)
        self.name = name or (url or (self.argv[0] if self.argv else "mcp-server"))
        self.timeout = timeout
        self.tool_prefix = tool_prefix
        self.server_info: Dict[str, Any] = {}

        self._headers = dict(headers or {})
        self._env = dict(env or {})
        self._cwd = cwd
        self._inherit_env = inherit_env
        self._prefer_official = prefer_official
        self._session: Optional[_MCPSession] = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    # -- connection ----------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def transport(self) -> str:
        return self._session.transport_name if self._session is not None else "disconnected"

    @property
    def server_name(self) -> str:
        info = self.server_info.get("serverInfo") if isinstance(self.server_info, dict) else None
        if isinstance(info, dict):
            return str(info.get("name") or self.name)
        return self.name

    def connect(self) -> "MCPClient":
        """Open the transport and run the MCP handshake. Idempotent."""
        if self._session is not None:
            return self

        session = self._open_session()
        try:
            self.server_info = session.initialize() or {}
        except MCPError:
            session.close()
            raise
        except Exception as exc:
            session.close()
            raise MCPError(
                f"MCP handshake with '{self.name}' failed: {exc}", cause=exc
            ) from exc

        self._session = session
        self._tools_cache = None
        log.info(
            "MCP: connected to '%s' via %s (protocol %s)",
            self.server_name, session.transport_name,
            self.server_info.get("protocolVersion", "?"),
        )
        return self

    def _open_session(self) -> _MCPSession:
        if self.url:
            session: _MCPSession = _HttpSession(self.url, headers=self._headers, timeout=self.timeout)
            session.start()
            return session

        if self._prefer_official and _official_mcp_available():
            official = _OfficialStdioSession(
                self.argv, env=self._env, cwd=self._cwd,
                timeout=self.timeout, inherit_env=self._inherit_env,
            )
            try:
                official.start()
                return official
            except Exception as exc:
                log.warning(
                    "MCP: official 'mcp' SDK transport failed (%s); "
                    "falling back to the built-in stdio transport.", exc,
                )
                try:
                    official.close()
                except Exception:  # pragma: no cover
                    pass

        builtin = _StdioSession(
            self.argv, env=self._env, cwd=self._cwd,
            timeout=self.timeout, inherit_env=self._inherit_env,
        )
        builtin.start()
        return builtin

    def close(self) -> None:
        """Shut the transport down and reap the server subprocess. Idempotent."""
        session, self._session = self._session, None
        self._tools_cache = None
        if session is not None:
            try:
                session.close()
            except Exception as exc:  # pragma: no cover - teardown is best effort
                log.warning("MCP: error while closing '%s': %s", self.name, exc)

    def __enter__(self) -> "MCPClient":
        return self.connect()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- MCP operations ------------------------------------------------------

    def _require_session(self) -> _MCPSession:
        if self._session is None:
            self.connect()
        if self._session is None:  # pragma: no cover - connect raises on failure
            raise MCPConnectionError(f"MCP client '{self.name}' is not connected")
        return self._session

    def list_tools(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """Return the raw tool descriptors advertised by the server."""
        session = self._require_session()
        if self._tools_cache is None or refresh:
            self._tools_cache = session.list_tools()
            log.debug("MCP: '%s' advertises %d tool(s)", self.name, len(self._tools_cache))
        return list(self._tools_cache)

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke a remote tool and return the raw ``tools/call`` result."""
        session = self._require_session()
        return session.call_tool(name, arguments or {})

    def get_tools(self, include: Optional[Iterable[str]] = None) -> List[BaseTool]:
        """Return every remote tool wrapped as a Mangaba :class:`BaseTool`.

        Args:
            include: Optional whitelist of remote tool names.
        """
        wanted = set(include) if include is not None else None
        tools: List[BaseTool] = []
        for descriptor in self.list_tools():
            remote_name = descriptor.get("name")
            if not remote_name:
                log.warning("MCP: '%s' advertised a tool without a name; skipped", self.name)
                continue
            if wanted is not None and remote_name not in wanted:
                continue
            schema = descriptor.get("inputSchema") or descriptor.get("input_schema") or {}
            tools.append(MCPTool(
                client=self,
                tool_name=remote_name,
                description=descriptor.get("description") or "",
                input_schema=schema if isinstance(schema, dict) else {},
                display_name=f"{self.tool_prefix}{remote_name}" if self.tool_prefix else None,
            ))
        return tools

    def __repr__(self) -> str:
        target = self.url or " ".join(self.argv)
        return f"MCPClient(name='{self.name}', target='{target}', connected={self.is_connected})"
