"""Logic helpers for the `vexor index` command."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from .cache_service import load_index_metadata_safe
from ..cache import IndexedChunk
from ..modes import get_strategy, ModePayload

INCREMENTAL_CHANGE_THRESHOLD = 0.5
MTIME_TOLERANCE = 5e-1


class IndexStatus(str, Enum):
    EMPTY = "empty"
    UP_TO_DATE = "up_to_date"
    STORED = "stored"


@dataclass(slots=True)
class IndexResult:
    status: IndexStatus
    cache_path: Path | None = None
    files_indexed: int = 0


def build_index(
    directory: Path,
    *,
    include_hidden: bool,
    mode: str,
    recursive: bool,
    model_name: str,
    batch_size: int,
    provider: str,
    base_url: str | None,
    api_key: str | None,
) -> IndexResult:
    """Create or refresh the cached index for *directory*."""

    from ..search import VexorSearcher  # local import
    from ..utils import collect_files  # local import
    from ..cache import apply_index_updates, store_index  # local import

    files = collect_files(directory, include_hidden=include_hidden, recursive=recursive)
    if not files:
        return IndexResult(status=IndexStatus.EMPTY)

    existing_meta = load_index_metadata_safe(directory, model_name, include_hidden, mode, recursive)
    cached_files = existing_meta.get("files", []) if existing_meta else []

    strategy = get_strategy(mode)
    payloads = strategy.payloads_for_files(files)
    searcher = VexorSearcher(
        model_name=model_name,
        batch_size=batch_size,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
    )

    if cached_files:
        snapshot = _snapshot_current_files(files, directory)
        diff = _diff_cached_files(snapshot, cached_files)
        if diff.is_noop:
            return IndexResult(status=IndexStatus.UP_TO_DATE, files_indexed=len(files))

        change_ratio = diff.change_ratio(len(snapshot), len(cached_files))
        if change_ratio <= INCREMENTAL_CHANGE_THRESHOLD:
            cache_path = _apply_incremental_update(
                directory=directory,
                include_hidden=include_hidden,
                recursive=recursive,
                mode=mode,
                model_name=model_name,
                payloads=payloads,
                diff=diff,
                searcher=searcher,
                apply_fn=apply_index_updates,
            )
            return IndexResult(
                status=IndexStatus.STORED,
                cache_path=cache_path,
                files_indexed=len(files),
            )

    file_labels = [payload.label for payload in payloads]
    embeddings = searcher.embed_texts(file_labels)
    entries = _build_index_entries(payloads, embeddings, directory)

    cache_path = store_index(
        root=directory,
        model=model_name,
        include_hidden=include_hidden,
        mode=mode,
        recursive=recursive,
        entries=entries,
    )
    return IndexResult(
        status=IndexStatus.STORED,
        cache_path=cache_path,
        files_indexed=len(files),
    )


def clear_index_entries(
    directory: Path,
    *,
    include_hidden: bool,
    mode: str,
    recursive: bool,
    model: str | None = None,
) -> int:
    """Remove cached entries for *directory* and return number removed."""

    from ..cache import clear_index as clear_index_cache  # local import

    return clear_index_cache(
        root=directory,
        include_hidden=include_hidden,
        mode=mode,
        recursive=recursive,
        model=model,
    )


@dataclass(slots=True)
class SnapshotEntry:
    path: Path
    rel_path: str
    mtime: float
    size: int


@dataclass(slots=True)
class FileDiff:
    added: list[Path] = field(default_factory=list)
    modified: list[Path] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return not (self.added or self.modified or self.removed)

    def change_ratio(self, current_count: int, cached_count: int) -> float:
        denom = max(current_count, cached_count, 1)
        change_count = len(self.added) + len(self.modified) + len(self.removed)
        return change_count / denom

    def changed_paths(self) -> list[Path]:
        return self.added + self.modified


def _snapshot_current_files(files: list[Path], root: Path) -> dict[str, SnapshotEntry]:
    snapshot: dict[str, SnapshotEntry] = {}
    for path in files:
        rel = _relative_to_root(path, root)
        stat = path.stat()
        snapshot[rel] = SnapshotEntry(
            path=path,
            rel_path=rel,
            mtime=stat.st_mtime,
            size=stat.st_size,
        )
    return snapshot


def _diff_cached_files(
    current: dict[str, SnapshotEntry],
    cached_files: list[dict],
) -> FileDiff:
    cached_map = {entry["path"]: entry for entry in cached_files}
    diff = FileDiff()

    for rel_path, entry in current.items():
        cached_entry = cached_map.get(rel_path)
        if cached_entry is None:
            diff.added.append(entry.path)
        elif _has_entry_changed(entry, cached_entry):
            diff.modified.append(entry.path)

    for rel_path in cached_map.keys():
        if rel_path not in current:
            diff.removed.append(rel_path)

    return diff


def _has_entry_changed(entry: SnapshotEntry, cached_entry: dict) -> bool:
    cached_mtime = cached_entry.get("mtime")
    cached_size = cached_entry.get("size")
    if cached_mtime is None:
        return True
    if abs(entry.mtime - cached_mtime) > MTIME_TOLERANCE:
        if cached_size is not None and cached_size == entry.size:
            return False
        return True
    if cached_size is not None and cached_size != entry.size:
        return True
    return False


def _apply_incremental_update(
    *,
    directory: Path,
    include_hidden: bool,
    mode: str,
    recursive: bool,
    model_name: str,
    payloads: list[ModePayload],
    diff: FileDiff,
    searcher,
    apply_fn,
) -> Path:
    ordered_entries = [
        (_relative_to_root(payload.file, directory), payload.chunk_index)
        for payload in payloads
    ]
    changed_set = set(diff.changed_paths())
    if changed_set:
        targets = [payload for payload in payloads if payload.file in changed_set]
        if targets:
            labels = [payload.label for payload in targets]
            embeddings = searcher.embed_texts(labels)
            changed_entries = _build_index_entries(targets, embeddings, directory)
        else:
            changed_entries = []
    else:
        changed_entries = []

    cache_path = apply_fn(
        root=directory,
        model=model_name,
        include_hidden=include_hidden,
        mode=mode,
        recursive=recursive,
        ordered_entries=ordered_entries,
        changed_entries=changed_entries,
        removed_rel_paths=diff.removed,
    )
    return cache_path


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return str(rel)


def _build_index_entries(
    payloads: Sequence[ModePayload],
    embeddings: Sequence[Sequence[float]],
    root: Path,
) -> list[IndexedChunk]:
    entries: list[IndexedChunk] = []
    for idx, payload in enumerate(payloads):
        entries.append(
            IndexedChunk(
                path=payload.file,
                rel_path=_relative_to_root(payload.file, root),
                chunk_index=payload.chunk_index,
                preview=payload.preview or "",
                embedding=embeddings[idx],
            )
        )
    return entries
