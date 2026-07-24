"""Built-in event callbacks for Mangaba AI v3.0.

The observability integrations live in :mod:`mangaba.observability` and are
re-exported here lazily, so importing this package never pulls in an optional
SDK::

    from mangaba.callbacks import OpenTelemetryCallback   # resolved on access
"""

from __future__ import annotations

from typing import Any

from mangaba.callbacks.console import ConsoleCallback
from mangaba.callbacks.file import FileCallback

_OBSERVABILITY_EXPORTS = {
    "OpenTelemetryCallback",
    "LangfuseCallback",
    "MLflowCallback",
    "PhoenixCallback",
    "configure_observability",
    "auto_configure_from_env",
}

__all__ = ["ConsoleCallback", "FileCallback"] + sorted(_OBSERVABILITY_EXPORTS)


def __getattr__(name: str) -> Any:  # PEP 562 — lazy re-export
    if name in _OBSERVABILITY_EXPORTS:
        import mangaba.observability as _obs

        return getattr(_obs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
