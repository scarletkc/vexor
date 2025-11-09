"""Utility helpers for filesystem access and path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import os


def resolve_directory(path: Path | str) -> Path:
    """Resolve and validate a user supplied directory path."""
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    return dir_path


def collect_files(root: Path | str, include_hidden: bool = False) -> List[Path]:
    """Recursively collect files under *root*, optionally keeping hidden entries."""
    directory = resolve_directory(root)
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(directory):
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames = [f for f in filenames if not f.startswith(".")]
        current_dir = Path(dirpath)
        for filename in filenames:
            files.append(current_dir / filename)
    files.sort()
    return files


def format_path(path: Path, base: Path | None = None) -> str:
    """Return a user friendly representation of *path* relative to *base* when possible."""
    if base:
        try:
            relative = path.relative_to(base)
            return f"./{relative.as_posix()}"
        except ValueError:
            return str(path)
    return str(path)


def ensure_positive(value: int, name: str) -> int:
    """Validate that *value* is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value
