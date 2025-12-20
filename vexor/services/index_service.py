"""Logic helpers for the `vexor index` command."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import MutableMapping, Sequence

import numpy as np

from .cache_service import load_index_metadata_safe
from .content_extract_service import TEXT_EXTENSIONS
from .js_parser import JSTS_EXTENSIONS
from ..cache import CACHE_VERSION, IndexedChunk, backfill_chunk_lines
from ..config import DEFAULT_EMBED_CONCURRENCY
from ..modes import get_strategy, ModePayload

INCREMENTAL_CHANGE_THRESHOLD = 0.5
MTIME_TOLERANCE = 5e-1
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx"}


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
    respect_gitignore: bool = True,
    mode: str,
    recursive: bool,
    model_name: str,
    batch_size: int,
    embed_concurrency: int = DEFAULT_EMBED_CONCURRENCY,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    local_cuda: bool = False,
    extensions: Sequence[str] | None = None,
) -> IndexResult:
    """Create or refresh the cached index for *directory*."""

    from ..search import VexorSearcher  # local import
    from ..utils import collect_files  # local import
    from ..cache import apply_index_updates, store_index  # local import

    files = collect_files(
        directory,
        include_hidden=include_hidden,
        recursive=recursive,
        extensions=extensions,
        respect_gitignore=respect_gitignore,
    )
    if not files:
        return IndexResult(status=IndexStatus.EMPTY)
    stat_cache: dict[Path, os.stat_result] = {}

    existing_meta = load_index_metadata_safe(
        directory,
        model_name,
        include_hidden,
        respect_gitignore,
        mode,
        recursive,
        extensions=extensions,
    )
    cached_files = existing_meta.get("files", []) if existing_meta else []

    strategy = get_strategy(mode)
    searcher = VexorSearcher(
        model_name=model_name,
        batch_size=batch_size,
        embed_concurrency=embed_concurrency,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        local_cuda=local_cuda,
    )

    if cached_files:
        cached_version = int(existing_meta.get("version", 0) or 0) if existing_meta else 0
        full_max_bytes = (
            getattr(strategy, "full_max_bytes", 10_000) if mode == "auto" else None
        )
        missing_line_files = _missing_line_files(existing_meta, mode, full_max_bytes)

        snapshot = _snapshot_current_files(files, directory, stat_cache=stat_cache)
        diff = _diff_cached_files(snapshot, cached_files)
        if diff.is_noop:
            if missing_line_files:
                updates = _build_line_backfill_updates(
                    strategy=strategy,
                    files=files,
                    missing_rel_paths=missing_line_files,
                    root=directory,
                )
                cache_path = backfill_chunk_lines(
                    root=directory,
                    model=model_name,
                    include_hidden=include_hidden,
                    respect_gitignore=respect_gitignore,
                    mode=mode,
                    recursive=recursive,
                    updates=updates,
                    extensions=extensions,
                )
                return IndexResult(
                    status=IndexStatus.STORED,
                    cache_path=cache_path,
                    files_indexed=len(files),
                )
            if cached_version < CACHE_VERSION:
                cache_path = backfill_chunk_lines(
                    root=directory,
                    model=model_name,
                    include_hidden=include_hidden,
                    respect_gitignore=respect_gitignore,
                    mode=mode,
                    recursive=recursive,
                    updates=[],
                    extensions=extensions,
                )
                return IndexResult(
                    status=IndexStatus.STORED,
                    cache_path=cache_path,
                    files_indexed=len(files),
                )
            return IndexResult(status=IndexStatus.UP_TO_DATE, files_indexed=len(files))

        change_ratio = diff.change_ratio(len(snapshot), len(cached_files))
        if change_ratio <= INCREMENTAL_CHANGE_THRESHOLD:
            cached_chunks = existing_meta.get("chunks", []) if existing_meta else []
            cached_chunk_map = _cached_chunk_map(cached_chunks)
            removed_rel_paths = set(diff.removed)
            changed_rel_paths = {
                _relative_to_root(path, directory) for path in diff.changed_paths()
            }
            files_with_rel = [
                (_relative_to_root(path, directory), path) for path in files
            ]
            for rel, _path in files_with_rel:
                if rel in removed_rel_paths or rel in changed_rel_paths:
                    continue
                if rel not in cached_chunk_map:
                    changed_rel_paths.add(rel)

            changed_files = [
                path for rel, path in files_with_rel if rel in changed_rel_paths
            ]
            changed_payloads = (
                strategy.payloads_for_files(changed_files) if changed_files else []
            )

            cache_path = _apply_incremental_update(
                directory=directory,
                include_hidden=include_hidden,
                respect_gitignore=respect_gitignore,
                recursive=recursive,
                mode=mode,
                model_name=model_name,
                files=files,
                changed_payloads=changed_payloads,
                removed_rel_paths=removed_rel_paths,
                cached_chunk_map=cached_chunk_map,
                searcher=searcher,
                apply_fn=apply_index_updates,
                extensions=extensions,
                stat_cache=stat_cache,
            )

            line_backfill_targets = missing_line_files - changed_rel_paths - removed_rel_paths
            if line_backfill_targets:
                updates = _build_line_backfill_updates(
                    strategy=strategy,
                    files=files,
                    missing_rel_paths=line_backfill_targets,
                    root=directory,
                )
                cache_path = backfill_chunk_lines(
                    root=directory,
                    model=model_name,
                    include_hidden=include_hidden,
                    respect_gitignore=respect_gitignore,
                    mode=mode,
                    recursive=recursive,
                    updates=updates,
                    extensions=extensions,
                )
            return IndexResult(
                status=IndexStatus.STORED,
                cache_path=cache_path,
                files_indexed=len(files),
            )

    payloads = strategy.payloads_for_files(files)
    file_labels = [payload.label for payload in payloads]
    embeddings = _embed_labels_with_cache(
        searcher=searcher,
        model_name=model_name,
        labels=file_labels,
    )
    entries = _build_index_entries(payloads, embeddings, directory, stat_cache=stat_cache)

    cache_path = store_index(
        root=directory,
        model=model_name,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode,
        recursive=recursive,
        entries=entries,
        extensions=extensions,
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
    respect_gitignore: bool = True,
    mode: str,
    recursive: bool,
    model: str | None = None,
    extensions: Sequence[str] | None = None,
) -> int:
    """Remove cached entries for *directory* and return number removed."""

    from ..cache import clear_index as clear_index_cache  # local import

    return clear_index_cache(
        root=directory,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode,
        recursive=recursive,
        model=model,
        extensions=extensions,
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


def _snapshot_current_files(
    files: list[Path],
    root: Path,
    *,
    stat_cache: MutableMapping[Path, os.stat_result] | None = None,
) -> dict[str, SnapshotEntry]:
    snapshot: dict[str, SnapshotEntry] = {}
    for path in files:
        rel = _relative_to_root(path, root)
        stat = _stat_for_path(path, stat_cache)
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
    respect_gitignore: bool,
    mode: str,
    recursive: bool,
    model_name: str,
    files: Sequence[Path],
    changed_payloads: Sequence[ModePayload],
    removed_rel_paths: set[str],
    cached_chunk_map: dict[str, list[int]],
    searcher,
    apply_fn,
    extensions: Sequence[str] | None,
    stat_cache: MutableMapping[Path, os.stat_result] | None = None,
) -> Path:
    changed_payloads_by_rel = _payloads_by_rel_path(changed_payloads, directory)
    ordered_entries = _build_ordered_entries(
        files=files,
        root=directory,
        cached_chunk_map=cached_chunk_map,
        changed_payloads_by_rel=changed_payloads_by_rel,
        removed_rel_paths=removed_rel_paths,
    )
    if changed_payloads:
        labels = [payload.label for payload in changed_payloads]
        embeddings = _embed_labels_with_cache(
            searcher=searcher,
            model_name=model_name,
            labels=labels,
        )
        changed_entries = _build_index_entries(
            changed_payloads,
            embeddings,
            directory,
            stat_cache=stat_cache,
        )
    else:
        changed_entries = []

    cache_path = apply_fn(
        root=directory,
        model=model_name,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode,
        recursive=recursive,
        ordered_entries=ordered_entries,
        changed_entries=changed_entries,
        removed_rel_paths=sorted(removed_rel_paths),
        extensions=extensions,
    )
    return cache_path


def _embed_labels_with_cache(
    *,
    searcher,
    model_name: str,
    labels: Sequence[str],
) -> np.ndarray:
    if not labels:
        return np.empty((0, 0), dtype=np.float32)
    from ..cache import embedding_cache_key, load_embedding_cache, store_embedding_cache

    hashes = [embedding_cache_key(label) for label in labels]
    cached = load_embedding_cache(model_name, hashes)
    missing: dict[str, str] = {}
    for label, text_hash in zip(labels, hashes):
        vector = cached.get(text_hash)
        if vector is None or vector.size == 0:
            if text_hash not in missing:
                missing[text_hash] = label

    if missing:
        missing_items = list(missing.items())
        missing_labels = [label for _, label in missing_items]
        new_vectors = searcher.embed_texts(missing_labels)
        stored: dict[str, np.ndarray] = {}
        for idx, (text_hash, _) in enumerate(missing_items):
            vector = np.asarray(new_vectors[idx], dtype=np.float32)
            cached[text_hash] = vector
            stored[text_hash] = vector
        store_embedding_cache(model=model_name, embeddings=stored)

    vectors = [cached[text_hash] for text_hash in hashes]
    return np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return str(rel)


def _cached_chunk_map(chunk_entries: Sequence[dict]) -> dict[str, list[int]]:
    chunk_map: dict[str, list[int]] = {}
    for entry in chunk_entries:
        rel_path = entry.get("path")
        if not isinstance(rel_path, str):
            continue
        try:
            chunk_index = int(entry.get("chunk_index", 0))
        except (TypeError, ValueError):
            chunk_index = 0
        chunk_map.setdefault(rel_path, []).append(chunk_index)
    return chunk_map


def _payloads_by_rel_path(
    payloads: Sequence[ModePayload],
    root: Path,
) -> dict[str, list[ModePayload]]:
    payload_map: dict[str, list[ModePayload]] = {}
    for payload in payloads:
        rel_path = _relative_to_root(payload.file, root)
        payload_map.setdefault(rel_path, []).append(payload)
    return payload_map


def _build_ordered_entries(
    *,
    files: Sequence[Path],
    root: Path,
    cached_chunk_map: dict[str, list[int]],
    changed_payloads_by_rel: dict[str, list[ModePayload]],
    removed_rel_paths: set[str],
) -> list[tuple[str, int]]:
    ordered_entries: list[tuple[str, int]] = []
    for path in files:
        rel_path = _relative_to_root(path, root)
        if rel_path in removed_rel_paths:
            continue
        payloads = changed_payloads_by_rel.get(rel_path)
        if payloads is not None:
            ordered_entries.extend(
                (rel_path, payload.chunk_index) for payload in payloads
            )
            continue
        chunk_indices = cached_chunk_map.get(rel_path)
        if not chunk_indices:
            continue
        ordered_entries.extend((rel_path, chunk_index) for chunk_index in chunk_indices)
    return ordered_entries


def _line_metadata_expected(
    mode: str,
    rel_path: str,
    size_bytes: int | None,
    full_max_bytes: int | None,
) -> bool:
    suffix = Path(rel_path).suffix.lower()
    if mode in {"name", "head", "brief"}:
        return False
    if mode == "outline":
        if suffix in MARKDOWN_EXTENSIONS:
            return True
        return suffix in TEXT_EXTENSIONS
    if mode == "code":
        if suffix == ".py" or suffix in JSTS_EXTENSIONS:
            return True
        return suffix in TEXT_EXTENSIONS
    if mode == "full":
        return suffix in TEXT_EXTENSIONS
    if mode == "auto":
        if suffix == ".py" or suffix in JSTS_EXTENSIONS:
            return True
        if suffix in MARKDOWN_EXTENSIONS:
            return True
        if size_bytes is None:
            return suffix in TEXT_EXTENSIONS
        if full_max_bytes is not None and size_bytes <= full_max_bytes:
            return suffix in TEXT_EXTENSIONS
        return False
    return False


def _missing_line_files(
    metadata: dict | None,
    mode: str,
    full_max_bytes: int | None,
) -> set[str]:
    if not metadata:
        return set()
    chunk_entries = metadata.get("chunks") or []
    if not chunk_entries:
        return set()
    missing: set[str] = set()
    for entry in chunk_entries:
        rel_path = entry.get("path")
        if not isinstance(rel_path, str):
            continue
        if entry.get("start_line") is not None or entry.get("end_line") is not None:
            continue
        size = entry.get("size")
        size_bytes = int(size) if isinstance(size, (int, float)) else None
        if _line_metadata_expected(mode, rel_path, size_bytes, full_max_bytes):
            missing.add(rel_path)
    return missing


def _build_line_backfill_updates(
    *,
    strategy,
    files: Sequence[Path],
    missing_rel_paths: set[str],
    root: Path,
) -> list[tuple[str, int, int | None, int | None]]:
    if not missing_rel_paths:
        return []
    files_by_rel = {_relative_to_root(path, root): path for path in files}
    targets = [files_by_rel[rel] for rel in missing_rel_paths if rel in files_by_rel]
    if not targets:
        return []
    payloads = strategy.payloads_for_files(targets)
    return [
        (
            _relative_to_root(payload.file, root),
            payload.chunk_index,
            payload.start_line,
            payload.end_line,
        )
        for payload in payloads
    ]


def _build_index_entries(
    payloads: Sequence[ModePayload],
    embeddings: Sequence[Sequence[float]],
    root: Path,
    *,
    stat_cache: MutableMapping[Path, os.stat_result] | None = None,
) -> list[IndexedChunk]:
    entries: list[IndexedChunk] = []
    for idx, payload in enumerate(payloads):
        stat = _stat_for_path(payload.file, stat_cache)
        entries.append(
            IndexedChunk(
                path=payload.file,
                rel_path=_relative_to_root(payload.file, root),
                chunk_index=payload.chunk_index,
                preview=payload.preview or "",
                embedding=embeddings[idx],
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                start_line=payload.start_line,
                end_line=payload.end_line,
            )
        )
    return entries


def _stat_for_path(
    path: Path,
    cache: MutableMapping[Path, os.stat_result] | None = None,
) -> os.stat_result:
    if cache is None:
        return path.stat()
    stat = cache.get(path)
    if stat is None:
        stat = path.stat()
        cache[path] = stat
    return stat
