"""Logic helpers for the `vexor search` command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    extensions: tuple[str, ...]
    auto_index: bool = True


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
        load_index_vectors,
        load_query_vector,
        query_cache_key,
        store_query_vector,
    )
    from .index_service import IndexStatus, build_index  # local import

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
            provider=request.provider,
            base_url=request.base_url,
            api_key=request.api_key,
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
        paths, file_vectors, metadata = load_index_vectors(
            request.directory,
            request.model_name,
            request.include_hidden,
            request.mode,
            request.recursive,
            request.extensions,
            respect_gitignore=request.respect_gitignore,
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
            request.directory,
            include_hidden=request.include_hidden,
            respect_gitignore=request.respect_gitignore,
            mode=request.mode,
            recursive=request.recursive,
            model_name=request.model_name,
            batch_size=request.batch_size,
            provider=request.provider,
            base_url=request.base_url,
            api_key=request.api_key,
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
        paths, file_vectors, metadata = load_index_vectors(
            request.directory,
            request.model_name,
            request.include_hidden,
            request.mode,
            request.recursive,
            request.extensions,
            respect_gitignore=request.respect_gitignore,
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
        provider=request.provider,
        base_url=request.base_url,
        api_key=request.api_key,
    )
    query_vector = None
    query_hash = None
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
        query_vector = searcher.embed_texts([request.query])[0]
        if index_id is not None and query_hash is not None:
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
