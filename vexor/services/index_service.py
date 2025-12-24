"""Logic helpers for the `vexor index` command."""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import MutableMapping, Sequence

import numpy as np

from .cache_service import load_index_metadata_safe
from .content_extract_service import TEXT_EXTENSIONS
from .js_parser import JSTS_EXTENSIONS
from ..cache import CACHE_VERSION, IndexedChunk, backfill_chunk_lines
from ..config import (
    DEFAULT_EMBED_CONCURRENCY,
    DEFAULT_EXTRACT_BACKEND,
    DEFAULT_EXTRACT_CONCURRENCY,
)
from ..modes import get_strategy, ModePayload

INCREMENTAL_CHANGE_THRESHOLD = 0.5
MTIME_TOLERANCE = 5e-1
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx"}
_EXTRACT_PROCESS_MIN_FILES = 16
_CPU_HEAVY_MODES = {"auto", "code", "outline", "full"}


class IndexStatus(str, Enum):
    EMPTY = "empty"
    UP_TO_DATE = "up_to_date"
    STORED = "stored"


@dataclass(slots=True)
class IndexResult:
    status: IndexStatus
    cache_path: Path | None = None
    files_indexed: int = 0


def _resolve_extract_concurrency(value: int) -> int:
    return max(int(value or 1), 1)


def _resolve_extract_backend(
    value: str | None,
    *,
    mode: str,
    file_count: int,
    concurrency: int,
) -> str:
    normalized = (value or DEFAULT_EXTRACT_BACKEND).strip().lower()
    if normalized not in {"auto", "thread", "process"}:
        normalized = DEFAULT_EXTRACT_BACKEND
    if normalized == "auto":
        if (
            concurrency > 1
            and file_count >= _EXTRACT_PROCESS_MIN_FILES
            and mode in _CPU_HEAVY_MODES
        ):
            return "process"
        return "thread"
    return normalized


def _extract_payloads_for_mode(path: Path, mode: str) -> list[ModePayload]:
    strategy = get_strategy(mode)
    return strategy.payloads_for_files([path])


def _payloads_for_files(
    strategy,
    files: Sequence[Path],
    *,
    mode: str,
    extract_concurrency: int,
    extract_backend: str,
) -> list[ModePayload]:
    if not files:
        return []
    concurrency = _resolve_extract_concurrency(extract_concurrency)
    if concurrency <= 1 or len(files) <= 1:
        return strategy.payloads_for_files(files)
    max_workers = min(concurrency, len(files))

    def _extract_with_thread_pool() -> list[ModePayload]:
        def _extract_one(path: Path) -> list[ModePayload]:
            return strategy.payloads_for_files([path])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(_extract_one, files)
            payloads: list[ModePayload] = []
            for batch in results:
                payloads.extend(batch)
            return payloads

    effective_backend = _resolve_extract_backend(
        extract_backend,
        mode=mode,
        file_count=len(files),
        concurrency=concurrency,
    )
    if effective_backend == "process":
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(
                    _extract_payloads_for_mode,
                    files,
                    itertools.repeat(mode),
                )
                payloads: list[ModePayload] = []
                for batch in results:
                    payloads.extend(batch)
                return payloads
        except Exception:
            return _extract_with_thread_pool()
    return _extract_with_thread_pool()


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
    extract_concurrency: int = DEFAULT_EXTRACT_CONCURRENCY,
    extract_backend: str = DEFAULT_EXTRACT_BACKEND,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    local_cuda: bool = False,
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
    no_cache: bool = False,
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
        exclude_patterns=exclude_patterns,
        respect_gitignore=respect_gitignore,
    )
    if not files:
        return IndexResult(status=IndexStatus.EMPTY)
    stat_cache: dict[Path, os.stat_result] = {}
    extract_concurrency = _resolve_extract_concurrency(extract_concurrency)

    existing_meta = load_index_metadata_safe(
        directory,
        model_name,
        include_hidden,
        respect_gitignore,
        mode,
        recursive,
        exclude_patterns=exclude_patterns,
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
                    extract_concurrency=extract_concurrency,
                    extract_backend=extract_backend,
                    mode=mode,
                )
                cache_path = backfill_chunk_lines(
                    root=directory,
                    model=model_name,
                    include_hidden=include_hidden,
                    respect_gitignore=respect_gitignore,
                    mode=mode,
                    recursive=recursive,
                    updates=updates,
                    exclude_patterns=exclude_patterns,
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
                    exclude_patterns=exclude_patterns,
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
            cached_label_map = _cached_label_map(cached_chunks)
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
                _payloads_for_files(
                    strategy,
                    changed_files,
                    mode=mode,
                    extract_concurrency=extract_concurrency,
                    extract_backend=extract_backend,
                )
                if changed_files
                else []
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
                cached_label_map=cached_label_map,
                searcher=searcher,
                apply_fn=apply_index_updates,
                exclude_patterns=exclude_patterns,
                extensions=extensions,
                stat_cache=stat_cache,
                no_cache=no_cache,
            )

            line_backfill_targets = missing_line_files - changed_rel_paths - removed_rel_paths
            if line_backfill_targets:
                updates = _build_line_backfill_updates(
                    strategy=strategy,
                    files=files,
                    missing_rel_paths=line_backfill_targets,
                    root=directory,
                    extract_concurrency=extract_concurrency,
                    extract_backend=extract_backend,
                    mode=mode,
                )
                cache_path = backfill_chunk_lines(
                    root=directory,
                    model=model_name,
                    include_hidden=include_hidden,
                    respect_gitignore=respect_gitignore,
                    mode=mode,
                    recursive=recursive,
                    updates=updates,
                    exclude_patterns=exclude_patterns,
                    extensions=extensions,
                )
            return IndexResult(
                status=IndexStatus.STORED,
                cache_path=cache_path,
                files_indexed=len(files),
            )

    payloads = _payloads_for_files(
        strategy,
        files,
        mode=mode,
        extract_concurrency=extract_concurrency,
        extract_backend=extract_backend,
    )
    file_labels = [payload.label for payload in payloads]
    embeddings = _embed_labels_with_cache(
        searcher=searcher,
        model_name=model_name,
        labels=file_labels,
        no_cache=no_cache,
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
        exclude_patterns=exclude_patterns,
        extensions=extensions,
    )
    return IndexResult(
        status=IndexStatus.STORED,
        cache_path=cache_path,
        files_indexed=len(files),
    )


def build_index_in_memory(
    directory: Path,
    *,
    include_hidden: bool,
    respect_gitignore: bool = True,
    mode: str,
    recursive: bool,
    model_name: str,
    batch_size: int,
    embed_concurrency: int = DEFAULT_EMBED_CONCURRENCY,
    extract_concurrency: int = DEFAULT_EXTRACT_CONCURRENCY,
    extract_backend: str = DEFAULT_EXTRACT_BACKEND,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    local_cuda: bool = False,
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
    no_cache: bool = False,
) -> tuple[list[Path], np.ndarray, dict]:
    """Build an index in memory without writing to disk."""

    from ..search import VexorSearcher  # local import
    from ..utils import collect_files  # local import

    files = collect_files(
        directory,
        include_hidden=include_hidden,
        recursive=recursive,
        extensions=extensions,
        exclude_patterns=exclude_patterns,
        respect_gitignore=respect_gitignore,
    )
    if not files:
        empty = np.empty((0, 0), dtype=np.float32)
        metadata = {
            "index_id": None,
            "version": CACHE_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "root": str(directory),
            "model": model_name,
            "include_hidden": include_hidden,
            "respect_gitignore": respect_gitignore,
            "recursive": recursive,
            "mode": mode,
            "dimension": 0,
            "exclude_patterns": tuple(exclude_patterns or ()),
            "extensions": tuple(extensions or ()),
            "files": [],
            "chunks": [],
        }
        return [], empty, metadata

    stat_cache: dict[Path, os.stat_result] = {}
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
    payloads = _payloads_for_files(
        strategy,
        files,
        mode=mode,
        extract_concurrency=extract_concurrency,
        extract_backend=extract_backend,
    )
    if not payloads:
        empty = np.empty((0, 0), dtype=np.float32)
        metadata = {
            "index_id": None,
            "version": CACHE_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "root": str(directory),
            "model": model_name,
            "include_hidden": include_hidden,
            "respect_gitignore": respect_gitignore,
            "recursive": recursive,
            "mode": mode,
            "dimension": 0,
            "exclude_patterns": tuple(exclude_patterns or ()),
            "extensions": tuple(extensions or ()),
            "files": [],
            "chunks": [],
        }
        return [], empty, metadata

    labels = [payload.label for payload in payloads]
    if no_cache:
        embeddings = searcher.embed_texts(labels)
        vectors = np.asarray(embeddings, dtype=np.float32)
    else:
        vectors = _embed_labels_with_cache(
            searcher=searcher,
            model_name=model_name,
            labels=labels,
        )
    entries = _build_index_entries(
        payloads,
        vectors,
        directory,
        stat_cache=stat_cache,
    )
    paths = [entry.path for entry in entries]
    file_snapshot: dict[str, dict] = {}
    chunk_entries: list[dict] = []
    for entry in entries:
        rel_path = entry.rel_path
        chunk_entries.append(
            {
                "path": rel_path,
                "absolute": str(entry.path),
                "mtime": entry.mtime,
                "size": entry.size_bytes,
                "preview": entry.preview,
                "label_hash": entry.label_hash,
                "chunk_index": entry.chunk_index,
                "start_line": entry.start_line,
                "end_line": entry.end_line,
            }
        )
        if rel_path not in file_snapshot:
            file_snapshot[rel_path] = {
                "path": rel_path,
                "absolute": str(entry.path),
                "mtime": entry.mtime,
                "size": entry.size_bytes,
            }

    metadata = {
        "index_id": None,
        "version": CACHE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(directory),
        "model": model_name,
        "include_hidden": include_hidden,
        "respect_gitignore": respect_gitignore,
        "recursive": recursive,
        "mode": mode,
        "dimension": int(vectors.shape[1]) if vectors.size else 0,
        "exclude_patterns": tuple(exclude_patterns or ()),
        "extensions": tuple(extensions or ()),
        "files": list(file_snapshot.values()),
        "chunks": chunk_entries,
    }
    return paths, vectors, metadata


def clear_index_entries(
    directory: Path,
    *,
    include_hidden: bool,
    respect_gitignore: bool = True,
    mode: str,
    recursive: bool,
    model: str | None = None,
    exclude_patterns: Sequence[str] | None = None,
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
        exclude_patterns=exclude_patterns,
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
    cached_label_map: dict[str, dict[int, str]] | None,
    searcher,
    apply_fn,
    exclude_patterns: Sequence[str] | None,
    extensions: Sequence[str] | None,
    stat_cache: MutableMapping[Path, os.stat_result] | None = None,
    no_cache: bool = False,
) -> Path:
    payloads_to_embed, payloads_to_touch = _split_payloads_by_label(
        changed_payloads,
        cached_label_map or {},
        directory,
    )
    changed_payloads_by_rel = _payloads_by_rel_path(payloads_to_embed, directory)
    ordered_entries = _build_ordered_entries(
        files=files,
        root=directory,
        cached_chunk_map=cached_chunk_map,
        changed_payloads_by_rel=changed_payloads_by_rel,
        removed_rel_paths=removed_rel_paths,
    )
    if payloads_to_embed:
        labels = [payload.label for payload in payloads_to_embed]
        embeddings = _embed_labels_with_cache(
            searcher=searcher,
            model_name=model_name,
            labels=labels,
            no_cache=no_cache,
        )
        changed_entries = _build_index_entries(
            payloads_to_embed,
            embeddings,
            directory,
            stat_cache=stat_cache,
        )
    else:
        changed_entries = []
    touched_entries = _build_touched_entries(
        payloads_to_touch,
        directory,
        stat_cache=stat_cache,
    )

    cache_path = apply_fn(
        root=directory,
        model=model_name,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode,
        recursive=recursive,
        ordered_entries=ordered_entries,
        changed_entries=changed_entries,
        touched_entries=touched_entries,
        removed_rel_paths=sorted(removed_rel_paths),
        exclude_patterns=exclude_patterns,
        extensions=extensions,
    )
    return cache_path


def _embed_labels_with_cache(
    *,
    searcher,
    model_name: str,
    labels: Sequence[str],
    no_cache: bool = False,
) -> np.ndarray:
    if not labels:
        return np.empty((0, 0), dtype=np.float32)
    if no_cache:
        vectors = searcher.embed_texts(labels)
        return np.asarray(vectors, dtype=np.float32)
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


def _cached_label_map(chunk_entries: Sequence[dict]) -> dict[str, dict[int, str]]:
    label_map: dict[str, dict[int, str]] = {}
    for entry in chunk_entries:
        rel_path = entry.get("path")
        if not isinstance(rel_path, str):
            continue
        label_hash = entry.get("label_hash")
        if not isinstance(label_hash, str) or not label_hash:
            continue
        try:
            chunk_index = int(entry.get("chunk_index", 0))
        except (TypeError, ValueError):
            chunk_index = 0
        label_map.setdefault(rel_path, {})[chunk_index] = label_hash
    return label_map


def _payloads_by_rel_path(
    payloads: Sequence[ModePayload],
    root: Path,
) -> dict[str, list[ModePayload]]:
    payload_map: dict[str, list[ModePayload]] = {}
    for payload in payloads:
        rel_path = _relative_to_root(payload.file, root)
        payload_map.setdefault(rel_path, []).append(payload)
    return payload_map


def _split_payloads_by_label(
    payloads: Sequence[ModePayload],
    cached_label_map: dict[str, dict[int, str]],
    root: Path,
) -> tuple[list[ModePayload], list[ModePayload]]:
    if not payloads:
        return [], []
    from ..cache import embedding_cache_key  # local import

    payloads_by_rel = _payloads_by_rel_path(payloads, root)
    to_embed: list[ModePayload] = []
    to_touch: list[ModePayload] = []
    for rel_path, file_payloads in payloads_by_rel.items():
        cached_chunks = cached_label_map.get(rel_path)
        if not cached_chunks:
            to_embed.extend(file_payloads)
            continue
        if len(file_payloads) != len(cached_chunks):
            to_embed.extend(file_payloads)
            continue
        matched = True
        for payload in file_payloads:
            cached_hash = cached_chunks.get(payload.chunk_index)
            if not cached_hash:
                matched = False
                break
            if cached_hash != embedding_cache_key(payload.label):
                matched = False
                break
        if matched:
            to_touch.extend(file_payloads)
        else:
            to_embed.extend(file_payloads)
    return to_embed, to_touch


def _build_touched_entries(
    payloads: Sequence[ModePayload],
    root: Path,
    *,
    stat_cache: MutableMapping[Path, os.stat_result] | None = None,
) -> list[tuple[str, int, int, float, str | None, int | None, int | None, str]]:
    if not payloads:
        return []
    from ..cache import embedding_cache_key  # local import

    entries: list[tuple[str, int, int, float, str | None, int | None, int | None, str]] = []
    for payload in payloads:
        stat = _stat_for_path(payload.file, stat_cache)
        entries.append(
            (
                _relative_to_root(payload.file, root),
                payload.chunk_index,
                stat.st_size,
                stat.st_mtime,
                payload.preview,
                payload.start_line,
                payload.end_line,
                embedding_cache_key(payload.label),
            )
        )
    return entries


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
    extract_concurrency: int,
    extract_backend: str,
    mode: str,
) -> list[tuple[str, int, int | None, int | None]]:
    if not missing_rel_paths:
        return []
    files_by_rel = {_relative_to_root(path, root): path for path in files}
    targets = [files_by_rel[rel] for rel in missing_rel_paths if rel in files_by_rel]
    if not targets:
        return []
    payloads = _payloads_for_files(
        strategy,
        targets,
        mode=mode,
        extract_concurrency=extract_concurrency,
        extract_backend=extract_backend,
    )
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
    from ..cache import embedding_cache_key  # local import
    for idx, payload in enumerate(payloads):
        stat = _stat_for_path(payload.file, stat_cache)
        entries.append(
            IndexedChunk(
                path=payload.file,
                rel_path=_relative_to_root(payload.file, root),
                chunk_index=payload.chunk_index,
                preview=payload.preview or "",
                embedding=embeddings[idx],
                label_hash=embedding_cache_key(payload.label),
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
