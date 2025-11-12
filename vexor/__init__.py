"""Vexor package initialization."""

from __future__ import annotations

__all__ = ["__version__", "get_version"]

__version__ = "0.6.2"


def get_version() -> str:
    """Return the current package version."""
    return __version__
