"""Shared helpers for interacting with cached index metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

def is_cache_current(
    root: Path,
    include_hidden: bool,
    respect_gitignore: bool,
    cached_files: Sequence[dict],
    *,
    recursive: bool,
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
    current_files=None,
) -> bool:
    """Return True if cached metadata matches the current directory snapshot."""

    if not cached_files:
        return False
    from ..cache import compare_snapshot  # local import avoids eager heavy deps

    return compare_snapshot(
        root,
        include_hidden,
        cached_files,
        recursive=recursive,
        exclude_patterns=exclude_patterns,
        extensions=extensions,
        current_files=current_files,
        respect_gitignore=respect_gitignore,
    )


def load_index_metadata_safe(
    root: Path,
    model: str,
    include_hidden: bool,
    respect_gitignore: bool,
    mode: str,
    recursive: bool,
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
):
    """Load index metadata when present, returning None if missing."""

    from ..cache import load_index  # local import avoids eager heavy deps

    try:
        return load_index(
            root,
            model,
            include_hidden,
            mode,
            recursive,
            exclude_patterns,
            extensions,
            respect_gitignore=respect_gitignore,
        )
    except FileNotFoundError:
        return None
