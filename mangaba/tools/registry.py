"""
Tool registry for Mangaba AI — short names for tool classes.

:class:`ToolRegistry` maps a short name (``"scrape_website"``) to the class
that implements it, so tools can be requested by name from YAML, a CLI flag
or a database row instead of being imported by hand.

It deliberately reuses the contract the YAML loader already has: entries are
stored as ``"package.module:ClassName"`` reference strings — exactly what
:data:`mangaba.config_loader.TOOL_REGISTRY` holds and what
:func:`mangaba.config_loader.resolve_tool` knows how to import. Registering a
tool here therefore also makes it resolvable from ``agents.yaml``::

    # agents.yaml
    researcher:
      role: Researcher
      goal: Investigate a topic
      backstory: A decade of desk research
      tools: [scrape_website, http_request, file_search]

Example::

    from mangaba.tools.registry import REGISTRY, register, get, available

    available()                      # ['calculator', 'directory_list', ...]
    tool = get("scrape_website")     # an instantiated ScrapeWebsiteTool

    # Add your own — by class, instance or reference string
    register("weather", "my_pkg.tools:WeatherTool")
    get("weather", api_key="…")

Nothing is imported until a name is actually resolved, so listing the
catalogue never pulls in an optional dependency.
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from mangaba.core.exceptions import ConfigurationError
from mangaba.tools.base import BaseTool

log = logging.getLogger(__name__)


#: Built-in tools addressable by short name. Values are the same
#: ``module:ClassName`` references the YAML loader resolves.
BUILTIN_TOOLS: Dict[str, str] = {
    # Math / text
    "calculator": "mangaba.tools.math_tools:CalculatorTool",
    "text_splitter": "mangaba.tools.text_tools:TextSplitterTool",
    "word_counter": "mangaba.tools.text_tools:WordCounterTool",
    # Files
    "file_reader": "mangaba.tools.file_tools:FileReaderTool",
    "file_writer": "mangaba.tools.file_tools:FileWriterTool",
    "directory_list": "mangaba.tools.file_tools:DirectoryListTool",
    "file_search": "mangaba.tools.document_tools:FileSearchTool",
    "document_search": "mangaba.tools.document_tools:DocumentSearchTool",
    # Web
    "serper_search": "mangaba.tools.web_search:SerperSearchTool",
    "duckduckgo_search": "mangaba.tools.web_search:DuckDuckGoSearchTool",
    "scrape_website": "mangaba.tools.web_tools:ScrapeWebsiteTool",
    "http_request": "mangaba.tools.web_tools:HTTPRequestTool",
    # Data / code
    "sql_query": "mangaba.tools.data_tools:SQLQueryTool",
    "code_interpreter": "mangaba.tools.code_tools:CodeInterpreterTool",
}

#: Tools that cannot be built without constructor arguments, and why. Used to
#: turn "TypeError: missing 1 required positional argument" into advice.
REQUIRES_ARGUMENTS: Dict[str, str] = {
    "sql_query": "connection_string (e.g. 'sqlite:///data.db')",
    "document_search": "embedding (and usually file_path)",
    "code_interpreter": "enabled=True, plus Docker or unsafe_mode=True",
}


def reference_for(target: Union[str, Type[BaseTool], BaseTool]) -> Union[str, BaseTool]:
    """Normalise a registration target to a reference string (or an instance).

    Example::

        reference_for(CalculatorTool)   # 'mangaba.tools.math_tools:CalculatorTool'
    """
    if isinstance(target, str):
        return target
    if isinstance(target, type):
        return f"{target.__module__}:{target.__qualname__}"
    return target


class ToolRegistry:
    """A name → tool-class catalogue shared with the YAML config loader.

    Example::

        registry = ToolRegistry()
        registry.register("grep", "mangaba.tools.document_tools:FileSearchTool")
        registry.available()                 # includes 'grep'
        registry.get("grep", root_dir="./src")

    Set ``sync_config_loader=False`` for a private catalogue that does not
    touch :data:`mangaba.config_loader.TOOL_REGISTRY`.
    """

    def __init__(
        self,
        entries: Optional[Dict[str, str]] = None,
        sync_config_loader: bool = True,
    ) -> None:
        self._entries: Dict[str, str] = dict(entries if entries is not None else BUILTIN_TOOLS)
        self._instances: Dict[str, BaseTool] = {}
        self._sync = sync_config_loader
        self._lock = threading.RLock()
        self.sync_to_config_loader()

    # -- registration --------------------------------------------------------

    def register(
        self,
        name: str,
        target: Union[str, Type[BaseTool], BaseTool],
        overwrite: bool = False,
    ) -> None:
        """Register *target* under *name*.

        *target* may be a tool class, a ready-made tool instance, or a
        ``"package.module:ClassName"`` reference string.

        Raises:
            ValueError: If *name* is already taken and ``overwrite`` is False.

        Example::

            registry.register("calculator", CalculatorTool, overwrite=True)
        """
        if not name or not name.strip():
            raise ValueError("Tool name cannot be empty")
        key = name.strip()

        with self._lock:
            if key in self and not overwrite:
                raise ValueError(
                    f"Tool '{key}' is already registered. Pass overwrite=True to replace it."
                )
            resolved = reference_for(target)
            if isinstance(resolved, str):
                self._entries[key] = resolved
                self._instances.pop(key, None)
                self._push_to_config_loader(key, resolved)
            else:
                # A pre-built instance cannot be expressed as an import path, so
                # it stays local to this registry.
                self._instances[key] = resolved
                self._entries.pop(key, None)
                log.debug("Tool '%s' registered as an instance — not visible to YAML resolution", key)

    def unregister(self, name: str) -> None:
        """Remove a name from the registry (no-op when unknown)."""
        with self._lock:
            self._entries.pop(name, None)
            self._instances.pop(name, None)

    # -- lookup --------------------------------------------------------------

    def get(self, name: str, **kwargs: Any) -> BaseTool:
        """Instantiate the tool registered under *name*.

        Extra keyword arguments are forwarded to the constructor.

        Raises:
            ConfigurationError: If the name is unknown, the module cannot be
                imported, or the tool needs arguments that were not supplied.

        Example::

            get("sql_query", connection_string="sqlite:///shop.db")
        """
        with self._lock:
            instance = self._instances.get(name)
        if instance is not None:
            if kwargs:
                raise ConfigurationError(
                    f"Tool '{name}' was registered as a ready-made instance and cannot take arguments."
                )
            return instance

        obj = self.get_class(name)
        if not isinstance(obj, type):
            return obj  # a module-level instance referenced by path
        try:
            return obj(**kwargs)
        except TypeError as exc:
            hint = REQUIRES_ARGUMENTS.get(name)
            suffix = f" It requires: {hint}." if hint else ""
            raise ConfigurationError(f"Could not build tool '{name}': {exc}.{suffix}", cause=exc)

    def get_class(self, name: str) -> Any:
        """Import and return the class (or module-level instance) behind *name*.

        Raises:
            ConfigurationError: If the name is unknown or the import fails.
        """
        with self._lock:
            reference = self._entries.get(name)
        if reference is None:
            if name in self._instances:
                return self._instances[name]
            raise ConfigurationError(
                f"Unknown tool '{name}'. Registered tools: {', '.join(self.available())}. "
                "Any other tool can be referenced as 'package.module:ClassName'."
            )

        module_name, _, attribute = reference.partition(":")
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attribute)
        except (ImportError, AttributeError) as exc:
            raise ConfigurationError(
                f"Could not import tool '{name}' ({reference}): {exc}", cause=exc
            )

    def available(self) -> List[str]:
        """Every registered short name, sorted."""
        with self._lock:
            return sorted(set(self._entries) | set(self._instances))

    def reference(self, name: str) -> Optional[str]:
        """The ``module:ClassName`` string stored for *name*, if any."""
        with self._lock:
            return self._entries.get(name)

    def entries(self) -> Dict[str, str]:
        """A copy of the name → reference mapping."""
        with self._lock:
            return dict(self._entries)

    # -- config_loader integration -------------------------------------------

    def sync_to_config_loader(self) -> bool:
        """Publish every reference entry into the YAML loader's registry.

        Returns True when the sync happened. Called automatically on creation
        and on every :meth:`register`; call it manually after mutating
        :attr:`BUILTIN_TOOLS` directly.
        """
        if not self._sync:
            return False
        try:
            from mangaba import config_loader
        except Exception as exc:  # noqa: BLE001 - never break tool use over this
            log.debug("Could not sync the tool registry into config_loader: %s", exc)
            return False
        with self._lock:
            config_loader.TOOL_REGISTRY.update(self._entries)
        return True

    def _push_to_config_loader(self, name: str, reference: str) -> None:
        if not self._sync:
            return
        try:
            from mangaba import config_loader

            config_loader.TOOL_REGISTRY[name] = reference
        except Exception as exc:  # noqa: BLE001 - never break tool use over this
            log.debug("Could not publish tool '%s' to config_loader: %s", name, exc)

    # -- dunder --------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._entries or name in self._instances

    def __iter__(self) -> Iterator[str]:
        return iter(self.available())

    def __len__(self) -> int:
        return len(self.available())

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self)} tools)"


#: Process-wide registry used by the YAML loader and the convenience helpers.
REGISTRY = ToolRegistry()


def register(name: str, target: Union[str, Type[BaseTool], BaseTool], overwrite: bool = False) -> None:
    """Register a tool in the process-wide :data:`REGISTRY`."""
    REGISTRY.register(name, target, overwrite=overwrite)


def get(name: str, **kwargs: Any) -> BaseTool:
    """Instantiate a tool from the process-wide :data:`REGISTRY`."""
    return REGISTRY.get(name, **kwargs)


def available() -> List[str]:
    """List every tool name in the process-wide :data:`REGISTRY`."""
    return REGISTRY.available()


__all__ = [
    "BUILTIN_TOOLS",
    "REGISTRY",
    "REQUIRES_ARGUMENTS",
    "ToolRegistry",
    "available",
    "get",
    "reference_for",
    "register",
]
