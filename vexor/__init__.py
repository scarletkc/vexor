"""Vexor package initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import InMemoryIndex, VexorClient, VexorError

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

__version__ = "0.24.3"

_API_EXPORTS = frozenset(
    {
        "InMemoryIndex",
        "VexorClient",
        "VexorError",
        "clear_index",
        "config_context",
        "index",
        "index_in_memory",
        "search",
        "set_config_json",
        "set_data_dir",
    }
)


def __getattr__(name: str) -> Any:
    """Load the public Python API only when an exported object is requested."""

    if name not in _API_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import api

    value = getattr(api, name)
    globals()[name] = value
    return value


def get_version() -> str:
    """Return the current package version."""
    return __version__
