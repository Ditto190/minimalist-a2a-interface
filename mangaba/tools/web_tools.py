"""
Web tools for Mangaba AI — page scraping and generic HTTP calls.

Two tools live here:

* :class:`ScrapeWebsiteTool` — fetch a URL and reduce it to readable text.
  ``beautifulsoup4`` is used when installed and a dependency-free regex
  fallback takes over when it is not, so the tool always works.
* :class:`HTTPRequestTool` — call an arbitrary REST endpoint (method, URL,
  headers, JSON body) and return status, headers and decoded body.

Both refuse any scheme other than ``http``/``https``: an agent that can be
talked into fetching ``file:///etc/passwd`` is a file-read primitive, not a
web tool.

Example::

    from mangaba.tools.web_tools import HTTPRequestTool, ScrapeWebsiteTool

    text = ScrapeWebsiteTool().run(url="https://example.com")
    resp = HTTPRequestTool().run(url="https://httpbin.org/get", method="GET")
    print(resp["status_code"])
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

from mangaba.tools.base import BaseTool

log = logging.getLogger(__name__)


#: Schemes an agent-driven HTTP tool is allowed to use.
ALLOWED_SCHEMES = ("http", "https")

#: HTTP verbs :class:`HTTPRequestTool` accepts.
ALLOWED_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")

#: Default browser-ish identity — many sites reject an empty User-Agent.
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; MangabaAI/3.0; +https://mangaba.ia.br/)"

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|template)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_RE = re.compile(
    r"</?(p|div|br|li|tr|h[1-6]|section|article|header|footer|table|ul|ol|blockquote)\b[^>]*>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def validate_url(url: str, allowed_domains: Optional[List[str]] = None) -> str:
    """Validate that *url* is an ``http(s)`` URL, optionally on an allowed host.

    Raises:
        ValueError: If the scheme is not http/https, the host is missing, or
            the host is not in *allowed_domains*.

    Example::

        validate_url("https://example.com/a")        # ok
        validate_url("file:///etc/passwd")           # ValueError
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Refusing URL with scheme '{parsed.scheme or 'none'}': only "
            f"{'/'.join(ALLOWED_SCHEMES)} are allowed."
        )
    if not parsed.hostname:
        raise ValueError(f"Refusing URL without a host: {url!r}")

    if allowed_domains:
        host = parsed.hostname.lower()
        if not any(host == d.lower() or host.endswith("." + d.lower()) for d in allowed_domains):
            raise ValueError(
                f"Refusing host '{host}': not in the allow-list {sorted(allowed_domains)}"
            )
    return url.strip()


def is_private_host(url: str) -> bool:
    """True when the URL resolves to a loopback, link-local or private address.

    Used by the optional SSRF guard. Resolution failures are reported as
    ``False`` — the request will fail on its own anyway.
    """
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:  # pragma: no cover - defensive
            continue
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return True
    return False


def html_to_text(html: str) -> str:
    """Strip HTML down to readable text, using bs4 when available.

    Example::

        html_to_text("<p>Hello <b>world</b></p>")   # 'Hello world'
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        log.debug("beautifulsoup4 not installed — using the regex fallback extractor")
        return _regex_html_to_text(html)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return _tidy(soup.get_text(separator="\n"))


def _regex_html_to_text(html: str) -> str:
    """Dependency-free HTML → text extraction (used when bs4 is missing)."""
    text = _COMMENT_RE.sub(" ", html)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = _unescape(text)
    return _tidy(text)


def _unescape(text: str) -> str:
    import html as html_module

    return html_module.unescape(text)


def _tidy(text: str) -> str:
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    joined = "\n".join(line for line in lines if line)
    return _BLANK_LINES_RE.sub("\n\n", joined).strip()


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

class ScrapeWebsiteInput(BaseModel):
    """Arguments accepted by :class:`ScrapeWebsiteTool`."""

    url: str = Field(..., description="Absolute http(s) URL of the page to read")
    max_chars: Optional[int] = Field(
        default=None, description="Truncate the extracted text to this many characters"
    )
    css_selector: Optional[str] = Field(
        default=None,
        description="Only extract text inside this CSS selector (requires beautifulsoup4)",
    )


class ScrapeWebsiteTool(BaseTool):
    """Fetch a web page and return its readable text.

    Scripts, styles and markup are removed; the result is plain text an LLM
    can reason over. ``beautifulsoup4`` (``pip install mangaba[documents]``)
    gives the cleanest extraction and enables ``css_selector``; without it a
    regex fallback is used automatically.

    Example::

        tool = ScrapeWebsiteTool(max_chars=4000)
        text = tool.run(url="https://example.com")

        # Restrict the whole tool to a set of hosts
        safe = ScrapeWebsiteTool(allowed_domains=["docs.python.org"])
    """

    name = "scrape_website"
    description = (
        "Fetch a web page over http(s) and return its readable text content, "
        "with scripts, styles and markup removed"
    )
    args_schema = ScrapeWebsiteInput

    def __init__(
        self,
        max_chars: int = 20_000,
        timeout: float = 20.0,
        headers: Optional[Dict[str, str]] = None,
        allowed_domains: Optional[List[str]] = None,
        verify_tls: bool = True,
    ) -> None:
        self.max_chars = max_chars
        self.timeout = timeout
        self.headers = {"User-Agent": DEFAULT_USER_AGENT}
        self.headers.update(headers or {})
        self.allowed_domains = list(allowed_domains) if allowed_domains else None
        self.verify_tls = verify_tls

    def _run(
        self,
        url: str,
        max_chars: Optional[int] = None,
        css_selector: Optional[str] = None,
    ) -> str:
        try:
            target = validate_url(url, self.allowed_domains)
        except ValueError as exc:
            return f"Error: {exc}"

        try:
            response = requests.get(
                target,
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_tls,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            return f"Error fetching '{target}': {exc}"

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "html" in content_type or not content_type:
            text = (
                self._select(response.text, css_selector)
                if css_selector
                else html_to_text(response.text)
            )
        elif "json" in content_type:
            try:
                text = json.dumps(response.json(), ensure_ascii=False, indent=2)
            except ValueError:
                text = response.text
        else:
            text = response.text

        limit = max_chars or self.max_chars
        if limit and len(text) > limit:
            text = text[:limit] + f"\n\n[... truncated at {limit} characters ...]"
        return text or "(the page returned no readable text)"

    def _select(self, html: str, css_selector: str) -> str:
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 package is required for css_selector. "
                "Install with: pip install mangaba[documents]"
            ) from exc
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        blocks = [node.get_text(separator="\n") for node in soup.select(css_selector)]
        return _tidy("\n".join(blocks))


# ---------------------------------------------------------------------------
# Generic REST calls
# ---------------------------------------------------------------------------

class HTTPRequestInput(BaseModel):
    """Arguments accepted by :class:`HTTPRequestTool`."""

    url: str = Field(..., description="Absolute http(s) URL to call")
    method: str = Field(default="GET", description="HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Extra request headers")
    json_body: Optional[Any] = Field(default=None, description="Body to send as JSON")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Query-string parameters")
    data: Optional[str] = Field(default=None, description="Raw request body (used when json_body is absent)")


class HTTPRequestTool(BaseTool):
    """Call any REST endpoint and return status, headers and body.

    Only ``http``/``https`` are accepted — ``file://``, ``ftp://`` and friends
    are refused, so this tool can never be turned into a local file reader.
    Constructor options narrow it further: ``allowed_domains`` pins it to a
    set of hosts and ``block_private_hosts=True`` adds an SSRF guard against
    loopback and private-range addresses (leave it off when calling services
    on your own network).

    Returns a dict with ``status_code``, ``ok``, ``headers``, ``json`` (when
    the response decodes) and ``text``.

    Example::

        tool = HTTPRequestTool(allowed_domains=["api.github.com"])
        result = tool.run(url="https://api.github.com/repos/mangaba-ai/mangaba-ai")
        print(result["status_code"], result["json"]["stargazers_count"])
    """

    name = "http_request"
    description = (
        "Make an HTTP request to a REST API (method, url, headers, json body) "
        "and return the status code and decoded response body"
    )
    args_schema = HTTPRequestInput

    def __init__(
        self,
        timeout: float = 30.0,
        default_headers: Optional[Dict[str, str]] = None,
        allowed_domains: Optional[List[str]] = None,
        allowed_methods: Optional[List[str]] = None,
        max_response_chars: int = 50_000,
        block_private_hosts: bool = False,
        verify_tls: bool = True,
    ) -> None:
        self.timeout = timeout
        self.default_headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json, */*"}
        self.default_headers.update(default_headers or {})
        self.allowed_domains = list(allowed_domains) if allowed_domains else None
        self.allowed_methods = tuple(m.upper() for m in (allowed_methods or ALLOWED_METHODS))
        self.max_response_chars = max_response_chars
        self.block_private_hosts = block_private_hosts
        self.verify_tls = verify_tls

    def _run(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[str] = None,
    ) -> Dict[str, Any]:
        verb = (method or "GET").strip().upper()
        if verb not in self.allowed_methods:
            return self._error(f"Refusing method '{verb}': allowed are {', '.join(self.allowed_methods)}")

        try:
            target = validate_url(url, self.allowed_domains)
        except ValueError as exc:
            return self._error(str(exc))

        if self.block_private_hosts and is_private_host(target):
            return self._error(
                f"Refusing '{target}': it resolves to a private or loopback address "
                "(construct the tool with block_private_hosts=False to allow it)."
            )

        request_headers = dict(self.default_headers)
        request_headers.update(headers or {})

        try:
            response = requests.request(
                verb,
                target,
                headers=request_headers,
                json=json_body,
                data=None if json_body is not None else data,
                params=params,
                timeout=self.timeout,
                verify=self.verify_tls,
            )
        except requests.exceptions.RequestException as exc:
            return self._error(f"Request to '{target}' failed: {exc}")

        text = response.text or ""
        truncated = len(text) > self.max_response_chars
        if truncated:
            text = text[: self.max_response_chars]

        payload: Dict[str, Any] = {
            "ok": response.ok,
            "status_code": response.status_code,
            "url": response.url,
            "headers": dict(response.headers),
            "text": text,
            "truncated": truncated,
            "error": None,
        }
        try:
            payload["json"] = response.json()
        except ValueError:
            payload["json"] = None
        return payload

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        log.debug("HTTPRequestTool refused a call: %s", message)
        return {
            "ok": False,
            "status_code": None,
            "url": None,
            "headers": {},
            "json": None,
            "text": "",
            "truncated": False,
            "error": message,
        }


__all__ = [
    "ALLOWED_METHODS",
    "ALLOWED_SCHEMES",
    "DEFAULT_USER_AGENT",
    "HTTPRequestInput",
    "HTTPRequestTool",
    "ScrapeWebsiteInput",
    "ScrapeWebsiteTool",
    "html_to_text",
    "is_private_host",
    "validate_url",
]
