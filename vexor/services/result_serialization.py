"""Structured search payloads shared by CLI JSON and MCP."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import format_path

if TYPE_CHECKING:
    from ..search import SearchResult
    from .search_service import SearchResponse


def search_result_payload(
    result: SearchResult, rank: int, base: Path, *, include_chunk_index: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "rank": rank,
        "score": round(float(result.score), 4),
        "path": format_path(result.path, base),
        "absolute_path": str(result.path),
    }
    if include_chunk_index:
        payload["chunk_index"] = result.chunk_index
    payload.update({
        "start_line": result.start_line,
        "end_line": result.end_line,
        "preview": result.preview,
        "content": result.content,
        "content_start_line": result.content_start_line,
        "content_end_line": result.content_end_line,
        "content_truncated": result.content_truncated,
        "content_unavailable": result.content_unavailable,
    })
    return payload


def search_response_payload(
    response: SearchResponse, base: Path, *, include_chunk_index: bool = False,
) -> dict[str, object]:
    return {
        "path": str(base),
        "backend": response.backend,
        "reranker": response.reranker,
        "stale": response.is_stale,
        "index_empty": response.index_empty,
        "results": [
            search_result_payload(result, rank, base, include_chunk_index=include_chunk_index)
            for rank, result in enumerate(response.results, start=1)
        ],
        "content_budget": (
            {"limit": response.content_budget.limit, "used": response.content_budget.used}
            if response.content_budget is not None else None
        ),
    }
