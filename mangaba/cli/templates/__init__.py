"""Project scaffolding templates for the Mangaba CLI.

Templates are plain Python strings (no package data to ship) rendered with
:class:`string.Template`, so the generated YAML and Python can contain ``{}``
placeholders of their own.

Example::

    from mangaba.cli.templates import scaffold

    created = scaffold("crew", "market_research", "./market_research")
"""

from __future__ import annotations

import logging
import os
import re
from string import Template
from typing import Dict, List

from mangaba.cli.templates import crew_template, flow_template

log = logging.getLogger(__name__)


#: Supported project kinds → the module holding their file templates.
KINDS: Dict[str, Dict[str, str]] = {
    "crew": crew_template.FILES,
    "flow": flow_template.FILES,
}

DEFAULT_TOPIC = "the state of renewable energy in Brazil"


def normalize_name(name: str) -> str:
    """Turn any user-supplied name into a safe snake_case identifier.

    Example::

        normalize_name("Market Research!")  # -> 'market_research'
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", (name or "").strip()).strip("_").lower()
    if not slug:
        raise ValueError("Project name must contain at least one letter or digit")
    if slug[0].isdigit():
        slug = f"p_{slug}"
    return slug


def titleize(name: str) -> str:
    """``market_research`` → ``Market Research``."""
    return " ".join(part.capitalize() for part in normalize_name(name).split("_"))


def build_context(name: str) -> Dict[str, str]:
    """Build the substitution context shared by every template."""
    from mangaba import __version__

    slug = normalize_name(name)
    return {
        "name": slug,
        "package": slug.replace("_", "-"),
        "title": titleize(slug),
        "topic": DEFAULT_TOPIC,
        "mangaba_version": __version__,
    }


def render(kind: str, name: str) -> Dict[str, str]:
    """Render every file of *kind* for a project called *name*."""
    if kind not in KINDS:
        raise ValueError(f"Unknown project kind '{kind}'. Valid: {', '.join(sorted(KINDS))}")
    context = build_context(name)
    return {path: Template(body).safe_substitute(context) for path, body in KINDS[kind].items()}


def scaffold(kind: str, name: str, directory: str, overwrite: bool = False) -> List[str]:
    """Write a rendered project to *directory* and return the paths created.

    Raises ``FileExistsError`` when a file already exists and *overwrite*
    is False, so an existing project is never silently clobbered.

    Example::

        scaffold("crew", "market_research", "./market_research")
    """
    files = render(kind, name)
    os.makedirs(directory, exist_ok=True)

    if not overwrite:
        clashes = [p for p in files if os.path.exists(os.path.join(directory, p))]
        if clashes:
            raise FileExistsError(
                f"{directory} already contains {', '.join(sorted(clashes))} — "
                "choose another name or pass --force."
            )

    created: List[str] = []
    for relative_path, body in sorted(files.items()):
        full_path = os.path.join(directory, relative_path)
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as fh:
            fh.write(body)
        created.append(full_path)

    log.info("Scaffolded %s project '%s' in %s", kind, name, directory)
    return created


__all__ = [
    "KINDS",
    "DEFAULT_TOPIC",
    "normalize_name",
    "titleize",
    "build_context",
    "render",
    "scaffold",
]
