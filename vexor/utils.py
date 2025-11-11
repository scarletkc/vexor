"""Utility helpers for filesystem access and path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import os


def resolve_directory(path: Path | str) -> Path:
    """Resolve and validate a user supplied directory path."""
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    return dir_path


def normalize_extensions(values: Iterable[str] | None) -> tuple[str, ...]:
    """Return a sorted, deduplicated tuple of normalized file extensions."""

    if not values:
        return ()

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        token = raw.strip().lower()
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        if token == ".":
            continue
        if token not in seen:
            seen.add(token)
            normalized.append(token)
    if not normalized:
        return ()
    return tuple(sorted(normalized))


def collect_files(
    root: Path | str,
    include_hidden: bool = False,
    recursive: bool = True,
    extensions: Sequence[str] | None = None,
) -> List[Path]:
    """Collect files under *root*; optionally keep hidden entries and recurse."""

    directory = resolve_directory(root)
    files: List[Path] = []
    normalized_exts: Tuple[str, ...] = tuple(extensions or ())

    if recursive:
        for dirpath, dirnames, filenames in os.walk(directory):
            if not include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                filenames = [f for f in filenames if not f.startswith(".")]
            current_dir = Path(dirpath)
            for filename in filenames:
                candidate = current_dir / filename
                if normalized_exts and not _matches_extension(candidate, normalized_exts):
                    continue
                files.append(candidate)
    else:
        for entry in directory.iterdir():
            if entry.is_dir():
                continue
            if not include_hidden and entry.name.startswith("."):
                continue
            if normalized_exts and not _matches_extension(entry, normalized_exts):
                continue
            files.append(entry)

    files.sort()
    return files


def _matches_extension(path: Path, extensions: Sequence[str]) -> bool:
    """Return True if *path* ends with any of the provided *extensions*."""

    filename = path.name.lower()
    return any(filename.endswith(ext) for ext in extensions)


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
