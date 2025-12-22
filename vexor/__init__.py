"""Vexor package initialization."""

from __future__ import annotations

from .api import VexorError, clear_index, index, search

__all__ = [
    "__version__",
    "VexorError",
    "clear_index",
    "get_version",
    "index",
    "search",
]

__version__ = "0.18.0"


def get_version() -> str:
    """Return the current package version."""
    return __version__
