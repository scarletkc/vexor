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


def _relative_posix(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if rel == Path("."):
        return ""
    return rel.as_posix()


def _find_git_root(path: Path) -> Path | None:
    for candidate in (path,) + tuple(path.parents):
        git_entry = candidate / ".git"
        if git_entry.exists():
            return candidate
    return None


def _resolve_git_dir(git_root: Path) -> Path | None:
    git_entry = git_root / ".git"
    if git_entry.is_dir():
        return git_entry
    if not git_entry.is_file():
        return None
    try:
        content = git_entry.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not content.lower().startswith(prefix):
        return None
    target = content[len(prefix) :].strip()
    if not target:
        return None
    git_dir = Path(target)
    if not git_dir.is_absolute():
        git_dir = (git_root / git_dir).resolve()
    return git_dir


def _read_gitignore_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _scope_gitignore_line(line: str, base_dir: str) -> str | None:
    if line == "":
        return None
    if line.startswith("#") and not line.startswith(r"\#"):
        return None
    if not base_dir:
        return line

    negated = line.startswith("!") and not line.startswith(r"\!")
    prefix = "!" if negated else ""
    body = line[1:] if negated else line

    anchored = body.startswith("/") and not body.startswith(r"\/")
    if anchored:
        body = body[1:]
        scoped = f"{base_dir}/{body}" if body else f"{base_dir}/"
        return f"{prefix}{scoped}"

    directory_only = body.endswith("/") and not body.endswith(r"\/")
    body_check = body[:-1] if directory_only else body
    has_slash = "/" in body_check
    if has_slash:
        scoped = f"{base_dir}/{body}"
    else:
        scoped = f"{base_dir}/**/{body}"
    return f"{prefix}{scoped}"


def _gitignore_spec_from_lines(lines: Iterable[str], base_dir: str):
    from pathspec.gitignore import GitIgnoreSpec

    scoped: list[str] = []
    for line in lines:
        scoped_line = _scope_gitignore_line(line, base_dir)
        if scoped_line is not None:
            scoped.append(scoped_line)
    return GitIgnoreSpec.from_lines(scoped)


def _is_ignored(spec, rel_path: str, *, is_dir: bool) -> bool:
    if not rel_path:
        return False
    candidate = f"{rel_path}/" if is_dir and not rel_path.endswith("/") else rel_path
    return spec.check_file(candidate).include is True


def _build_gitignore_base_spec(ignore_root: Path, scan_root: Path):
    from pathspec.gitignore import GitIgnoreSpec

    spec = GitIgnoreSpec.from_lines([])

    git_dir = _resolve_git_dir(ignore_root)
    if git_dir is not None:
        exclude_file = git_dir / "info" / "exclude"
        if exclude_file.is_file():
            spec += _gitignore_spec_from_lines(_read_gitignore_lines(exclude_file), "")

    try:
        parts = scan_root.relative_to(ignore_root).parts
    except ValueError:
        return spec, False

    if not parts:
        return spec, False

    for depth in range(len(parts)):
        ancestor = ignore_root if depth == 0 else ignore_root.joinpath(*parts[:depth])
        rel_ancestor = _relative_posix(ancestor, ignore_root)
        if depth and _is_ignored(spec, rel_ancestor, is_dir=True):
            return spec, True
        gitignore_file = ancestor / ".gitignore"
        if gitignore_file.is_file():
            spec += _gitignore_spec_from_lines(_read_gitignore_lines(gitignore_file), rel_ancestor)

    rel_scan = _relative_posix(scan_root, ignore_root)
    if _is_ignored(spec, rel_scan, is_dir=True):
        return spec, True
    return spec, False


def collect_files(
    root: Path | str,
    include_hidden: bool = False,
    recursive: bool = True,
    extensions: Sequence[str] | None = None,
    respect_gitignore: bool = True,
) -> List[Path]:
    """Collect files under *root*; optionally keep hidden entries and recurse."""

    directory = resolve_directory(root)
    files: List[Path] = []
    normalized_exts: Tuple[str, ...] = tuple(extensions or ())

    ignore_root: Path | None = None
    ignore_spec = None
    if respect_gitignore:
        ignore_root = _find_git_root(directory) or directory
        ignore_spec, ignored = _build_gitignore_base_spec(ignore_root, directory)
        if ignored:
            return []

    if recursive:
        spec_by_dir: dict[Path, object] = {}
        if respect_gitignore and ignore_root is not None and ignore_spec is not None:
            spec_by_dir[directory] = ignore_spec

        for dirpath, dirnames, filenames in os.walk(directory, topdown=True):
            if not include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                filenames = [f for f in filenames if not f.startswith(".")]
            current_dir = Path(dirpath)
            if respect_gitignore:
                dirnames[:] = [d for d in dirnames if d != ".git"]
                filenames = [f for f in filenames if f != ".git"]
            spec = ignore_spec
            if respect_gitignore and ignore_root is not None and ignore_spec is not None:
                spec = spec_by_dir.get(current_dir, ignore_spec)
                gitignore_file = current_dir / ".gitignore"
                if gitignore_file.is_file():
                    rel_dir = _relative_posix(current_dir, ignore_root)
                    spec = spec + _gitignore_spec_from_lines(
                        _read_gitignore_lines(gitignore_file),
                        rel_dir,
                    )
                    spec_by_dir[current_dir] = spec

                kept: list[str] = []
                for dirname in dirnames:
                    child = current_dir / dirname
                    rel_child = _relative_posix(child, ignore_root)
                    if _is_ignored(spec, rel_child, is_dir=True):
                        continue
                    kept.append(dirname)
                    spec_by_dir[child] = spec
                dirnames[:] = kept

            for filename in filenames:
                candidate = current_dir / filename
                if normalized_exts and not _matches_extension(candidate, normalized_exts):
                    continue
                if respect_gitignore and ignore_root is not None and spec is not None:
                    rel_file = _relative_posix(candidate, ignore_root)
                    if _is_ignored(spec, rel_file, is_dir=False):
                        continue
                files.append(candidate)
    else:
        spec = ignore_spec
        if respect_gitignore and ignore_root is not None and ignore_spec is not None:
            gitignore_file = directory / ".gitignore"
            if gitignore_file.is_file():
                rel_dir = _relative_posix(directory, ignore_root)
                spec = ignore_spec + _gitignore_spec_from_lines(
                    _read_gitignore_lines(gitignore_file),
                    rel_dir,
                )
        for entry in directory.iterdir():
            if entry.is_dir():
                continue
            if respect_gitignore and entry.name == ".git":
                continue
            if not include_hidden and entry.name.startswith("."):
                continue
            if normalized_exts and not _matches_extension(entry, normalized_exts):
                continue
            if respect_gitignore and ignore_root is not None and spec is not None:
                rel_file = _relative_posix(entry, ignore_root)
                if _is_ignored(spec, rel_file, is_dir=False):
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
