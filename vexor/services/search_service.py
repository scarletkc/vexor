"""Logic helpers for the `vexor search` command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import DEFAULT_EMBED_CONCURRENCY
from .cache_service import is_cache_current


@dataclass(slots=True)
class SearchRequest:
    query: str
    directory: Path
    include_hidden: bool
    respect_gitignore: bool
    mode: str
    recursive: bool
    top_k: int
    model_name: str
    batch_size: int
    provider: str
    base_url: str | None
    api_key: str | None
    local_cuda: bool
    extensions: tuple[str, ...]
    auto_index: bool = True
    embed_concurrency: int = DEFAULT_EMBED_CONCURRENCY


@dataclass(slots=True)
class SearchResponse:
    base_path: Path
    backend: str | None
    results: Sequence[SearchResult]
    is_stale: bool
    index_empty: bool


def perform_search(request: SearchRequest) -> SearchResponse:
    """Execute the semantic search flow and return ranked results."""

    from ..cache import (  # local import
        embedding_cache_key,
        list_cache_entries,
        load_embedding_cache,
        load_index_vectors,
        load_query_vector,
        query_cache_key,
        store_embedding_cache,
        store_query_vector,
    )
    from .index_service import IndexStatus, build_index  # local import

    try:
        (
            paths,
            file_vectors,
            metadata,
            ext_filter,
            index_extensions,
            index_root,
            index_recursive,
        ) = _load_index_vectors_for_request(
            request,
            load_index_vectors=load_index_vectors,
            list_cache_entries=list_cache_entries,
        )
    except FileNotFoundError:
        if not request.auto_index:
            raise
        result = build_index(
            request.directory,
            include_hidden=request.include_hidden,
            respect_gitignore=request.respect_gitignore,
            mode=request.mode,
            recursive=request.recursive,
            model_name=request.model_name,
            batch_size=request.batch_size,
            embed_concurrency=request.embed_concurrency,
            provider=request.provider,
            base_url=request.base_url,
            api_key=request.api_key,
            local_cuda=request.local_cuda,
            extensions=request.extensions,
        )
        if result.status == IndexStatus.EMPTY:
            return SearchResponse(
                base_path=request.directory,
                backend=None,
                results=[],
                is_stale=False,
                index_empty=True,
            )
        (
            paths,
            file_vectors,
            metadata,
            ext_filter,
            index_extensions,
            index_root,
            index_recursive,
        ) = _load_index_vectors_for_request(
            request,
            load_index_vectors=load_index_vectors,
            list_cache_entries=list_cache_entries,
        )

    if index_root != request.directory:
        paths, file_vectors, metadata = _filter_index_by_directory(
            paths,
            file_vectors,
            metadata,
            request.directory,
            index_root,
            recursive=request.recursive,
        )

    if ext_filter:
        paths, file_vectors, metadata = _filter_index_by_extensions(
            paths,
            file_vectors,
            metadata,
            ext_filter,
        )

    file_snapshot = metadata.get("files", [])
    chunk_entries = metadata.get("chunks", [])
    stale = bool(file_snapshot) and not is_cache_current(
        request.directory,
        request.include_hidden,
        request.respect_gitignore,
        file_snapshot,
        recursive=request.recursive,
        extensions=request.extensions,
    )

    if stale and request.auto_index:
        result = build_index(
            index_root,
            include_hidden=request.include_hidden,
            respect_gitignore=request.respect_gitignore,
            mode=request.mode,
            recursive=index_recursive,
            model_name=request.model_name,
            batch_size=request.batch_size,
            embed_concurrency=request.embed_concurrency,
            provider=request.provider,
            base_url=request.base_url,
            api_key=request.api_key,
            local_cuda=request.local_cuda,
            extensions=index_extensions,
        )
        if result.status == IndexStatus.EMPTY:
            return SearchResponse(
                base_path=request.directory,
                backend=None,
                results=[],
                is_stale=False,
                index_empty=True,
            )
        (
            paths,
            file_vectors,
            metadata,
            ext_filter,
            index_extensions,
            index_root,
            index_recursive,
        ) = _load_index_vectors_for_request(
            request,
            load_index_vectors=load_index_vectors,
            list_cache_entries=list_cache_entries,
        )
        if index_root != request.directory:
            paths, file_vectors, metadata = _filter_index_by_directory(
                paths,
                file_vectors,
                metadata,
                request.directory,
                index_root,
                recursive=request.recursive,
            )
        if ext_filter:
            paths, file_vectors, metadata = _filter_index_by_extensions(
                paths,
                file_vectors,
                metadata,
                ext_filter,
            )
        file_snapshot = metadata.get("files", [])
        chunk_entries = metadata.get("chunks", [])
        stale = bool(file_snapshot) and not is_cache_current(
            request.directory,
            request.include_hidden,
            request.respect_gitignore,
            file_snapshot,
            recursive=request.recursive,
            extensions=request.extensions,
        )

    if not len(paths):
        return SearchResponse(
            base_path=request.directory,
            backend=None,
            results=[],
            is_stale=stale,
            index_empty=True,
        )

    from sklearn.metrics.pairwise import cosine_similarity  # local import
    from ..search import SearchResult, VexorSearcher  # local import
    searcher = VexorSearcher(
        model_name=request.model_name,
        batch_size=request.batch_size,
        embed_concurrency=request.embed_concurrency,
        provider=request.provider,
        base_url=request.base_url,
        api_key=request.api_key,
        local_cuda=request.local_cuda,
    )
    query_vector = None
    query_hash = None
    query_text_hash = embedding_cache_key(request.query)
    index_id = metadata.get("index_id")
    if index_id is not None:
        query_hash = query_cache_key(request.query, request.model_name)
        try:
            query_vector = load_query_vector(int(index_id), query_hash)
        except Exception:  # pragma: no cover - best-effort cache lookup
            query_vector = None

        if query_vector is not None and query_vector.size != file_vectors.shape[1]:
            query_vector = None

    if query_vector is None:
        cached = load_embedding_cache(request.model_name, [query_text_hash])
        query_vector = cached.get(query_text_hash)
        if query_vector is not None and query_vector.size != file_vectors.shape[1]:
            query_vector = None

    if query_vector is None:
        query_vector = searcher.embed_texts([request.query])[0]
        try:
            store_embedding_cache(
                model=request.model_name,
                embeddings={query_text_hash: query_vector},
            )
        except Exception:  # pragma: no cover - best-effort cache storage
            pass
    if query_vector is not None and index_id is not None and query_hash is not None:
        try:
            store_query_vector(int(index_id), query_hash, request.query, query_vector)
        except Exception:  # pragma: no cover - best-effort cache storage
            pass
    similarities = cosine_similarity(
        query_vector.reshape(1, -1),
        file_vectors,
    )[0]
    scored = []
    for idx, (path, score) in enumerate(zip(paths, similarities)):
        chunk_meta = chunk_entries[idx] if idx < len(chunk_entries) else {}
        start_line = chunk_meta.get("start_line")
        end_line = chunk_meta.get("end_line")
        scored.append(
            SearchResult(
                path=path,
                score=float(score),
                preview=chunk_meta.get("preview"),
                chunk_index=int(chunk_meta.get("chunk_index", 0)),
                start_line=int(start_line) if start_line is not None else None,
                end_line=int(end_line) if end_line is not None else None,
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    results = scored[: request.top_k]
    return SearchResponse(
        base_path=request.directory,
        backend=searcher.device,
        results=results,
        is_stale=stale,
        index_empty=False,
    )


def _load_index_vectors_for_request(
    request: SearchRequest,
    *,
    load_index_vectors,
    list_cache_entries,
) -> tuple[
    Sequence[Path],
    Sequence[Sequence[float]],
    dict,
    tuple[str, ...] | None,
    tuple[str, ...],
    Path,
    bool,
]:
    try:
        paths, file_vectors, metadata = load_index_vectors(
            request.directory,
            request.model_name,
            request.include_hidden,
            request.mode,
            request.recursive,
            request.extensions,
            respect_gitignore=request.respect_gitignore,
        )
        return (
            paths,
            file_vectors,
            metadata,
            None,
            request.extensions,
            request.directory,
            request.recursive,
        )
    except FileNotFoundError as exc:
        missing_exc = exc
    superset_entry = _select_cache_superset(request, list_cache_entries)
    if superset_entry is None:
        raise missing_exc
    superset_root = Path(superset_entry.get("root_path", "")).expanduser().resolve()
    superset_recursive = bool(superset_entry.get("recursive"))
    superset_extensions = tuple(superset_entry.get("extensions") or ())
    paths, file_vectors, metadata = load_index_vectors(
        superset_root,
        request.model_name,
        request.include_hidden,
        request.mode,
        superset_recursive,
        superset_extensions,
        respect_gitignore=request.respect_gitignore,
    )
    ext_filter = None
    if request.extensions and request.extensions != superset_extensions:
        ext_filter = request.extensions
    return (
        paths,
        file_vectors,
        metadata,
        ext_filter,
        superset_extensions,
        superset_root,
        superset_recursive,
    )


def _select_cache_superset(
    request: SearchRequest,
    list_cache_entries,
) -> dict | None:
    requested = set(request.extensions or ())
    root = request.directory.resolve()
    candidates: list[tuple[int, int, int, dict]] = []
    for entry in list_cache_entries():
        entry_root = Path(entry.get("root_path", "")).expanduser().resolve()
        try:
            relative = root.relative_to(entry_root)
        except ValueError:
            continue
        if entry.get("model") != request.model_name:
            continue
        if entry.get("include_hidden") != request.include_hidden:
            continue
        if entry.get("respect_gitignore") != request.respect_gitignore:
            continue
        entry_recursive = bool(entry.get("recursive"))
        if request.recursive and not entry_recursive:
            continue
        if entry.get("mode") != request.mode:
            continue
        cached_exts = tuple(entry.get("extensions") or ())
        if not requested:
            if cached_exts:
                continue
        else:
            if cached_exts and not requested.issubset(set(cached_exts)):
                continue
        distance = 0 if relative == Path(".") else len(relative.parts)
        recursive_mismatch = 1 if (entry_recursive and not request.recursive) else 0
        file_count = int(entry.get("file_count") or 0)
        if file_count <= 0:
            file_count = 1_000_000_000
        ext_count = len(cached_exts)
        candidates.append((distance, recursive_mismatch, file_count, ext_count, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates[0][4]


def _filter_index_by_extensions(
    paths: Sequence[Path],
    file_vectors,
    metadata: dict,
    extensions: Sequence[str],
) -> tuple[list[Path], Sequence[Sequence[float]], dict]:
    ext_set = {ext.lower() for ext in extensions if ext}
    if not ext_set:
        return list(paths), file_vectors, metadata
    keep_indices: list[int] = []
    filtered_paths: list[Path] = []
    for idx, path in enumerate(paths):
        if path.suffix.lower() in ext_set:
            keep_indices.append(idx)
            filtered_paths.append(path)
    if not keep_indices:
        filtered_vectors = file_vectors[:0]
        filtered_metadata = dict(metadata)
        filtered_metadata["files"] = _filter_file_snapshot(
            metadata.get("files", []),
            ext_set,
        )
        filtered_metadata["chunks"] = []
        return [], filtered_vectors, filtered_metadata
    filtered_vectors = file_vectors[keep_indices]
    chunk_entries = metadata.get("chunks", [])
    filtered_chunks = [
        chunk_entries[idx] for idx in keep_indices if idx < len(chunk_entries)
    ]
    filtered_metadata = dict(metadata)
    filtered_metadata["files"] = _filter_file_snapshot(
        metadata.get("files", []),
        ext_set,
    )
    filtered_metadata["chunks"] = filtered_chunks
    return filtered_paths, filtered_vectors, filtered_metadata


def _filter_index_by_directory(
    paths: Sequence[Path],
    file_vectors,
    metadata: dict,
    directory: Path,
    index_root: Path,
    *,
    recursive: bool,
) -> tuple[list[Path], Sequence[Sequence[float]], dict]:
    try:
        relative_dir = directory.resolve().relative_to(index_root.resolve())
    except ValueError:
        return list(paths), file_vectors, metadata
    keep_indices: list[int] = []
    filtered_paths: list[Path] = []
    for idx, path in enumerate(paths):
        try:
            rel_to_dir = path.resolve().relative_to(directory.resolve())
        except ValueError:
            continue
        if not recursive and len(rel_to_dir.parts) > 1:
            continue
        keep_indices.append(idx)
        filtered_paths.append(path)
    if not keep_indices:
        filtered_vectors = file_vectors[:0]
        filtered_metadata = dict(metadata)
        filtered_metadata["files"] = _filter_file_snapshot_by_directory(
            metadata.get("files", []),
            relative_dir,
            recursive=recursive,
        )
        filtered_metadata["chunks"] = []
        filtered_metadata["root"] = str(directory)
        return [], filtered_vectors, filtered_metadata
    filtered_vectors = file_vectors[keep_indices]
    chunk_entries = metadata.get("chunks", [])
    filtered_chunks = [
        chunk_entries[idx] for idx in keep_indices if idx < len(chunk_entries)
    ]
    filtered_metadata = dict(metadata)
    filtered_metadata["files"] = _filter_file_snapshot_by_directory(
        metadata.get("files", []),
        relative_dir,
        recursive=recursive,
    )
    filtered_metadata["chunks"] = filtered_chunks
    filtered_metadata["root"] = str(directory)
    return filtered_paths, filtered_vectors, filtered_metadata


def _filter_file_snapshot(
    entries: Sequence[dict],
    extensions: set[str],
) -> list[dict]:
    filtered: list[dict] = []
    for entry in entries:
        rel_path = entry.get("path", "")
        if Path(rel_path).suffix.lower() in extensions:
            filtered.append(entry)
    return filtered


def _filter_file_snapshot_by_directory(
    entries: Sequence[dict],
    relative_dir: Path,
    *,
    recursive: bool,
) -> list[dict]:
    filtered: list[dict] = []
    for entry in entries:
        rel_path = entry.get("path", "")
        try:
            rel_subpath = Path(rel_path).relative_to(relative_dir)
        except ValueError:
            continue
        if not recursive and len(rel_subpath.parts) > 1:
            continue
        updated = dict(entry)
        updated["path"] = rel_subpath.as_posix()
        filtered.append(updated)
    return filtered
