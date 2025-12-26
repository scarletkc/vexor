"""Vexor package initialization."""

from __future__ import annotations

from .api import (
    InMemoryIndex,
    VexorClient,
    VexorError,
    clear_index,
    config_context,
    index,
    index_in_memory,
    search,
    set_config_json,
    set_data_dir,
)

__all__ = [
    "__version__",
    "InMemoryIndex",
    "VexorClient",
    "VexorError",
    "clear_index",
    "config_context",
    "get_version",
    "index",
    "index_in_memory",
    "search",
    "set_config_json",
    "set_data_dir",
]

__version__ = "0.22.0"


def get_version() -> str:
    """Return the current package version."""
    return __version__
