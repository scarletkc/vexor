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
    mode: str
    recursive: bool
    top_k: int
    model_name: str
    batch_size: int


@dataclass(slots=True)
class SearchResponse:
    base_path: Path
    backend: str | None
    results: Sequence[SearchResult]
    is_stale: bool
    index_empty: bool


def perform_search(request: SearchRequest) -> SearchResponse:
    """Execute the semantic search flow and return ranked results."""

    from sklearn.metrics.pairwise import cosine_similarity  # local import
    from ..cache import load_index_vectors  # local import
    from ..search import SearchResult, VexorSearcher  # local import

    paths, file_vectors, metadata = load_index_vectors(
        request.directory,
        request.model_name,
        request.include_hidden,
        request.mode,
        request.recursive,
    )
    cached_files = metadata.get("files", [])
    stale = bool(cached_files) and not is_cache_current(
        request.directory,
        request.include_hidden,
        cached_files,
        recursive=request.recursive,
    )
    preview_lookup = {
        path: entry.get("preview")
        for path, entry in zip(paths, cached_files)
    }

    if not len(paths):
        return SearchResponse(
            base_path=request.directory,
            backend=None,
            results=[],
            is_stale=stale,
            index_empty=True,
        )

    searcher = VexorSearcher(
        model_name=request.model_name,
        batch_size=request.batch_size,
    )
    query_vector = searcher.embed_texts([request.query])[0]
    similarities = cosine_similarity(
        query_vector.reshape(1, -1),
        file_vectors,
    )[0]
    scored = [
        SearchResult(path=path, score=float(score), preview=preview_lookup.get(path))
        for path, score in zip(paths, similarities)
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    results = scored[: request.top_k]
    return SearchResponse(
        base_path=request.directory,
        backend=searcher.device,
        results=results,
        is_stale=stale,
        index_empty=False,
    )
