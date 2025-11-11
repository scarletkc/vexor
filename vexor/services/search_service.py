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
    provider: str
    base_url: str | None
    api_key: str | None
    extensions: tuple[str, ...]


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
        request.extensions,
    )
    file_snapshot = metadata.get("files", [])
    chunk_entries = metadata.get("chunks", [])
    stale = bool(file_snapshot) and not is_cache_current(
        request.directory,
        request.include_hidden,
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

    searcher = VexorSearcher(
        model_name=request.model_name,
        batch_size=request.batch_size,
        provider=request.provider,
        base_url=request.base_url,
        api_key=request.api_key,
    )
    query_vector = searcher.embed_texts([request.query])[0]
    similarities = cosine_similarity(
        query_vector.reshape(1, -1),
        file_vectors,
    )[0]
    scored = []
    for idx, (path, score) in enumerate(zip(paths, similarities)):
        chunk_meta = chunk_entries[idx] if idx < len(chunk_entries) else {}
        scored.append(
            SearchResult(
                path=path,
                score=float(score),
                preview=chunk_meta.get("preview"),
                chunk_index=int(chunk_meta.get("chunk_index", 0)),
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
