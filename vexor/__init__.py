"""Vexor package initialization."""

from __future__ import annotations

from .api import VexorError, clear_index, index, search, set_config_json, set_data_dir

__all__ = [
    "__version__",
    "VexorError",
    "clear_index",
    "get_version",
    "index",
    "search",
    "set_config_json",
    "set_data_dir",
]

__version__ = "0.20.0"


def get_version() -> str:
    """Return the current package version."""
    return __version__
