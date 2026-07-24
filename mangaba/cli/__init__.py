"""Command-line interface for Mangaba AI v3.0.

Provides the ``mangaba`` executable (entry point
``mangaba.cli.main:main``). Nothing is re-exported here on purpose, so the
``mangaba.cli.main`` submodule is never shadowed by a same-named function.

Example::

    from mangaba.cli.main import main

    exit_code = main(["create", "crew", "market_research"])
"""

from __future__ import annotations

__all__: list = []
