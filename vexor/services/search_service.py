"""Logic helpers for the `vexor search` command."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import json
import re

import numpy as np
from typing import Callable, Sequence, TYPE_CHECKING
from urllib import error as urlerror
from urllib import request as urlrequest

from .. import bm25
from ..config import (
    DEFAULT_EMBED_CONCURRENCY,
    DEFAULT_EXTRACT_BACKEND,
    DEFAULT_EXTRACT_CONCURRENCY,
    DEFAULT_FLASHRANK_MAX_LENGTH,
    DEFAULT_FLASHRANK_MODEL,
    DEFAULT_RERANK,
    RemoteRerankConfig,
    normalize_remote_rerank_url,
    resolve_remote_rerank_api_key,
)
from ..utils import build_exclude_spec, is_excluded_path, normalize_exclude_patterns
from .cache_service import is_cache_current

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..cache import IndexVectorCache
    from ..search import SearchResult
    from .freshness_service import FreshnessTracker

# Chunk text is returned to callers verbatim, so it has to be capped or a single
# search could bury an agent's context window. Counted in characters rather than
# tokens on purpose: a tokenizer dependency would not earn its weight here.
DEFAULT_CONTENT_CHARS_PER_RESULT = 2000
DEFAULT_CONTENT_CHARS_TOTAL = 8000

# Below this much remaining budget a read would return a sliver too small to be worth
# anything — and too small to survive the preview check, which would then misreport the
# result as stale rather than as out of budget.
MIN_USEFUL_CONTENT_CHARS = 200

# Marks one window of a chunk that was split to fit the embedding window; see
# CodeStrategy in modes.py, which tags them ``display [#N] :: snippet``.
_CHUNK_WINDOW_RE = re.compile(r"\[#\d+\]")

# How much of a stored preview is compared against re-read text, and the shortest
# probe worth comparing at all. Both are heuristics: the check exists to catch text
# read from the wrong place, and it errs toward accepting, since a false negative
# costs a preview fallback while a false positive hands back the wrong lines.
PREVIEW_PROBE_CHARS = 40
PREVIEW_PROBE_MIN_CHARS = 12

# Reasons a result carries no content despite the caller asking for it.
CONTENT_NO_LINE_RANGE = "no_line_range"
CONTENT_STALE_LINE_RANGE = "stale_line_range"
CONTENT_BUDGET_EXHAUSTED = "budget_exhausted"
CONTENT_UNREADABLE = "unreadable"


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
    exclude_patterns: tuple[str, ...]
    extensions: tuple[str, ...]
    auto_index: bool = True
    temporary_index: bool = False
    no_cache: bool = False
    embed_concurrency: int = DEFAULT_EMBED_CONCURRENCY
    extract_concurrency: int = DEFAULT_EXTRACT_CONCURRENCY
    extract_backend: str = DEFAULT_EXTRACT_BACKEND
    rerank: str = DEFAULT_RERANK
    flashrank_model: str | None = None
    remote_rerank: RemoteRerankConfig | None = None
    embedding_dimensions: int | None = None
    include_content: bool = False
    content_chars_per_result: int = DEFAULT_CONTENT_CHARS_PER_RESULT
    content_chars_total: int = DEFAULT_CONTENT_CHARS_TOTAL
    index_vector_cache: IndexVectorCache | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    freshness_tracker: FreshnessTracker | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(slots=True)
class ContentBudget:
    """How much chunk text a single response was allowed to carry, and how much it used."""

    limit: int
    used: int


@dataclass(slots=True)
class SearchResponse:
    base_path: Path
    backend: str | None
    results: Sequence[SearchResult]
    is_stale: bool
    index_empty: bool
    reranker: str | None = None
    content_budget: ContentBudget | None = None


_TOKEN_RE = bm25._TOKEN_RE
_BM25_K1 = bm25.BM25_K1
_BM25_B = bm25.BM25_B
_FUSION_SEMANTIC_WEIGHT = 0.7


_get_bm25_tokenizer = bm25._get_bm25_tokenizer


def _bm25_tokenize(text: str) -> list[str]:
    # Keep the local getter indirection for callers that monkeypatch this
    # long-standing private compatibility surface.
    tokenizer = _get_bm25_tokenizer()
    if tokenizer is None:
        return _TOKEN_RE.findall(text.lower())
    tokens = [token for token, _ in tokenizer.pre_tokenize_str(text)]
    normalized: list[str] = []
    for token in tokens:
        cleaned = token.strip()
        if not cleaned:
            continue
        if any(ch.isalnum() for ch in cleaned):
            normalized.append(cleaned.lower())
    return normalized


def _hybrid_scorer_from_cache(
    index_id: int | None,
    chunk_ids: Sequence[int],
) -> Callable[[Sequence[str]], dict[int, float]] | None:
    """Build a full-corpus lexical scorer backed by persisted postings."""

    if index_id is None or not chunk_ids:
        return None
    from .. import cache

    row_by_chunk_id = {int(chunk_id): row for row, chunk_id in enumerate(chunk_ids)}

    def score(query_terms: Sequence[str]) -> dict[int, float]:
        doc_count, avg_doc_len = cache.load_bm25_stats(int(index_id))
        score.has_data = doc_count > 0
        if not score.has_data:
            return {}
        postings = cache.load_bm25_postings(int(index_id), query_terms)
        scores_by_chunk = bm25.score_postings(
            query_terms, postings, doc_count, avg_doc_len
        )
        return {
            row_by_chunk_id[chunk_id]: value
            for chunk_id, value in scores_by_chunk.items()
            if chunk_id in row_by_chunk_id
        }

    score.has_data = False
    return score


def _hybrid_scorer_from_entries(
    chunk_entries: Sequence[dict],
) -> Callable[[Sequence[str]], dict[int, float]] | None:
    """Build a transient full-corpus lexical scorer from in-memory chunks."""

    postings: dict[str, list[tuple[int, int, int]]] = {}
    doc_lengths: list[int] = []
    for row, entry in enumerate(chunk_entries):
        terms = entry.get("bm25_terms")
        doc_len = entry.get("bm25_doc_len")
        if terms is None or doc_len is None:
            continue
        length = int(doc_len)
        doc_lengths.append(length)
        for term, tf in terms.items():
            postings.setdefault(str(term), []).append((row, int(tf), length))
    if not doc_lengths:
        return None
    doc_count = len(doc_lengths)
    avg_doc_len = sum(doc_lengths) / doc_count

    def score(query_terms: Sequence[str]) -> dict[int, float]:
        return bm25.score_postings(query_terms, postings, doc_count, avg_doc_len)

    score.has_data = True
    return score


def _preview_probe(preview: str | None) -> str:
    """Return the part of a stored preview that should also appear in the chunk text.

    ``code`` and ``outline`` previews are prefixed with a symbol path or heading
    breadcrumb (``display :: snippet``); only the trailing snippet came from the file.
    ``_trim_preview`` may also have appended an ellipsis.
    """

    if not preview:
        return ""
    return preview.rsplit(" :: ", 1)[-1].rstrip("…").strip()


def _chunk_window_anchor(preview: str | None) -> str | None:
    """Return the text that locates a split chunk's window inside its symbol.

    A ``code`` chunk too long to embed in one piece is split into overlapping
    windows tagged ``[#N]``, and every window carries its whole symbol's line
    range, so reading from the first line hands back window 1 whatever matched.
    The preview snippet says where the window actually starts.

    Only these windows are anchored. The other modes chunk with per-chunk ranges
    that are already right, and anchoring them would move content that is correct
    where it is: an ``outline`` chunk's snippet starts one line below its own
    ``start_line``, so the heading would drop out of the content.
    """

    label, separator, _ = (preview or "").rpartition(" :: ")
    if not separator or not _CHUNK_WINDOW_RE.search(label):
        return None
    probe = _preview_probe(preview)
    if len(probe) < PREVIEW_PROBE_MIN_CHARS:
        return None
    return probe[:PREVIEW_PROBE_CHARS]


def _content_matches_preview(content: str, preview: str | None) -> bool:
    """Check that re-read text still looks like what was indexed at that line range.

    Guards two cases that both produce plausible-looking but wrong text: indexes built
    before the full-mode line offset fix, and files edited since they were indexed.
    Deliberately lenient — a false negative only costs the caller a preview fallback,
    while a false positive hands an agent code from the wrong part of the file.
    """

    probe = _preview_probe(preview)
    if len(probe) < PREVIEW_PROBE_MIN_CHARS:
        # Too short to identify a location; nothing useful to verify against.
        return True
    from ..modes import normalize_preview_chunk

    normalized = normalize_preview_chunk(content) or ""
    return probe[:PREVIEW_PROBE_CHARS] in normalized


def _attach_chunk_content(
    request: SearchRequest, results: Sequence[SearchResult]
) -> ContentBudget | None:
    """Fill in ``content`` for each result, in rank order, until the budget runs out."""

    if not request.include_content:
        return None
    from .content_extract_service import read_chunk_content  # local import

    per_result = max(int(request.content_chars_per_result), 0)
    total = max(int(request.content_chars_total), 0)
    used = 0
    for result in results:
        if result.start_line is None or result.end_line is None:
            result.content_unavailable = CONTENT_NO_LINE_RANGE
            continue
        remaining = total - used
        if remaining < min(MIN_USEFUL_CONTENT_CHARS, per_result):
            result.content_unavailable = CONTENT_BUDGET_EXHAUSTED
            continue
        try:
            chunk = read_chunk_content(
                result.path,
                result.start_line,
                result.end_line,
                max_chars=min(per_result, remaining),
                anchor=_chunk_window_anchor(result.preview),
            )
        except OSError:
            chunk = None
        if chunk is None:
            result.content_unavailable = CONTENT_UNREADABLE
            continue
        if not _content_matches_preview(chunk.text, result.preview):
            result.content_unavailable = CONTENT_STALE_LINE_RANGE
            continue
        result.content = chunk.text
        result.content_start_line = chunk.start_line
        result.content_end_line = chunk.end_line
        result.content_truncated = chunk.truncated
        used += len(chunk.text)
    return ContentBudget(limit=total, used=used)


def _build_rerank_document(result: SearchResult) -> str:
    preview = result.preview or ""
    document = f"{result.path.name} {result.path.as_posix()} {preview}".strip()
    return document or result.path.as_posix()


def _build_rerank_documents(results: Sequence[SearchResult]) -> list[str]:
    return [_build_rerank_document(result) for result in results]


def _apply_ranking(
    results: Sequence[SearchResult],
    ranking: Sequence[tuple[int, float | None]],
) -> list[SearchResult]:
    """Reorder *results* by a reranker's ``(index, score)`` pairs.

    Unknown, duplicate, and out-of-range indices are dropped, and anything the
    reranker left out keeps its original relative order at the tail, so a partial
    response degrades to "reranked head, dense tail" instead of losing results.
    """

    ordered: list[SearchResult] = []
    seen: set[int] = set()
    for index, score in ranking:
        if index < 0 or index >= len(results) or index in seen:
            continue
        result = results[index]
        if score is not None:
            result.score = float(score)
        ordered.append(result)
        seen.add(index)
    if len(ordered) < len(results):
        for index, result in enumerate(results):
            if index not in seen:
                ordered.append(result)
    return ordered


def _normalize_by_max(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    if max_score <= 0:
        return [0.0 for _ in scores]
    return [score / max_score for score in scores]


def _resolve_rerank_candidates(top_k: int) -> int:
    candidate = int(top_k * 2)
    return max(20, min(candidate, 150))


def _top_indices(scores: np.ndarray, limit: int) -> list[int]:
    if limit <= 0:
        return []
    if limit >= scores.size:
        return sorted(range(scores.size), key=lambda idx: (-scores[idx], idx))
    indices = np.argpartition(-scores, limit - 1)[:limit]
    return sorted(indices.tolist(), key=lambda idx: (-scores[idx], idx))


def _bm25_scores(
    query_tokens: Sequence[str],
    documents: Sequence[Sequence[str]],
) -> list[float]:
    if not documents:
        return []
    from rank_bm25 import BM25L

    # BM25L avoids zero-idf scores on tiny candidate sets.
    bm25 = BM25L(documents, k1=_BM25_K1, b=_BM25_B)
    scores = bm25.get_scores(query_tokens)
    return [float(score) for score in scores]


def _rank_documents_bm25(
    query: str,
    documents: Sequence[str],
    base_scores: Sequence[float],
) -> list[tuple[int, float]] | None:
    """Fuse lexical scores over *documents* with the retrieval scores behind them.

    Returns ``None`` when the query carries no usable tokens, which leaves the
    caller's original order untouched.
    """

    query_tokens = _bm25_tokenize(query)
    if not query_tokens:
        return None
    tokenized = [_bm25_tokenize(document) for document in documents]
    lexical_norm = _normalize_by_max(_bm25_scores(query_tokens, tokenized))
    base_norm = _normalize_by_max([max(score, 0.0) for score in base_scores])
    fused = [
        (
            index,
            _FUSION_SEMANTIC_WEIGHT * base
            + (1.0 - _FUSION_SEMANTIC_WEIGHT) * lexical,
        )
        for index, (base, lexical) in enumerate(zip(base_norm, lexical_norm))
    ]
    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


def _apply_bm25_rerank(query: str, results: Sequence[SearchResult]) -> list[SearchResult]:
    if not results:
        return []
    ranking = _rank_documents_bm25(
        query,
        _build_rerank_documents(results),
        [result.score for result in results],
    )
    if ranking is None:
        return list(results)
    return _apply_ranking(results, ranking)


@lru_cache(maxsize=4)
def _get_flashranker(model_name: str | None, max_length: int):
    from flashrank import Ranker
    from ..config import flashrank_cache_dir

    cache_dir = flashrank_cache_dir()
    kwargs = {"max_length": max_length, "cache_dir": str(cache_dir)}
    if model_name:
        kwargs["model_name"] = model_name
    return Ranker(**kwargs)


def _rank_documents_flashrank(
    query: str,
    documents: Sequence[str],
    model_name: str | None,
) -> list[tuple[int, float | None]]:
    try:
        from flashrank import RerankRequest
    except ImportError as exc:
        from ..text import Messages

        raise RuntimeError(Messages.ERROR_FLASHRANK_MISSING) from exc
    try:
        effective_model = model_name or DEFAULT_FLASHRANK_MODEL
        ranker = _get_flashranker(effective_model, DEFAULT_FLASHRANK_MAX_LENGTH)
    except ImportError as exc:
        from ..text import Messages

        raise RuntimeError(Messages.ERROR_FLASHRANK_MISSING) from exc
    passages = [
        {"id": index, "text": document} for index, document in enumerate(documents)
    ]
    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    ranking: list[tuple[int, float | None]] = []
    for item in reranked:
        index = item.get("id")
        if index is None:
            continue
        try:
            position = int(index)
        except (TypeError, ValueError):
            continue
        score = item.get("score")
        ranking.append((position, float(score) if score is not None else None))
    return ranking


def _apply_flashrank_rerank(
    query: str,
    results: Sequence[SearchResult],
    model_name: str | None,
) -> list[SearchResult]:
    if not results:
        return []
    ranking = _rank_documents_flashrank(
        query,
        _build_rerank_documents(results),
        model_name,
    )
    return _apply_ranking(results, ranking)


def _resolve_remote_rerank_config(
    config: RemoteRerankConfig | None,
) -> RemoteRerankConfig:
    if not config:
        from ..text import Messages

        raise RuntimeError(Messages.ERROR_REMOTE_RERANK_INCOMPLETE)
    base_url = normalize_remote_rerank_url(config.base_url)
    api_key = resolve_remote_rerank_api_key(config.api_key)
    if not (base_url and config.model and api_key):
        from ..text import Messages

        raise RuntimeError(Messages.ERROR_REMOTE_RERANK_INCOMPLETE)
    if base_url != config.base_url or api_key != config.api_key:
        return RemoteRerankConfig(
            base_url=base_url,
            api_key=api_key,
            model=config.model,
        )
    return config


def _remote_rerank_request(
    *,
    config: RemoteRerankConfig,
    query: str,
    documents: Sequence[str],
) -> dict:
    from ..text import Messages

    payload = {
        "model": config.model,
        "query": query,
        "documents": list(documents),
    }
    data = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(config.base_url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", f"Bearer {config.api_key}")
    try:
        with urlrequest.urlopen(request) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as exc:
        reason = f"HTTP {exc.code}"
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        if detail:
            reason = f"{reason}: {detail[:200]}"
        raise RuntimeError(Messages.ERROR_REMOTE_RERANK_FAILED.format(reason=reason)) from exc
    except urlerror.URLError as exc:
        raise RuntimeError(
            Messages.ERROR_REMOTE_RERANK_FAILED.format(reason=str(exc))
        ) from exc
    except Exception as exc:  # pragma: no cover - network edge cases
        raise RuntimeError(
            Messages.ERROR_REMOTE_RERANK_FAILED.format(reason=str(exc))
        ) from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            Messages.ERROR_REMOTE_RERANK_FAILED.format(reason="Invalid JSON response")
        ) from exc


def _extract_remote_rerank_items(payload: object) -> list[tuple[int, float | None]]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("results")
    if not isinstance(items, list):
        items = payload.get("data")
    if not isinstance(items, list):
        return []
    parsed: list[tuple[int, float | None]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if index is None:
            continue
        try:
            idx = int(index)
        except (TypeError, ValueError):
            continue
        score = item.get("relevance_score")
        if score is None:
            score = item.get("score")
        try:
            parsed_score = float(score) if score is not None else None
        except (TypeError, ValueError):
            parsed_score = None
        parsed.append((idx, parsed_score))
    return parsed


def _rank_documents_remote(
    query: str,
    documents: Sequence[str],
    config: RemoteRerankConfig | None,
) -> list[tuple[int, float | None]]:
    resolved = _resolve_remote_rerank_config(config)
    payload = _remote_rerank_request(
        config=resolved,
        query=query,
        documents=documents,
    )
    return _extract_remote_rerank_items(payload)


def _apply_remote_rerank(
    query: str,
    results: Sequence[SearchResult],
    config: RemoteRerankConfig | None,
) -> list[SearchResult]:
    if not results:
        return []
    ranking = _rank_documents_remote(
        query,
        _build_rerank_documents(results),
        config,
    )
    if not ranking:
        return list(results)
    return _apply_ranking(results, ranking)


def _empty_response(directory: Path, *, is_stale: bool) -> SearchResponse:
    return SearchResponse(
        base_path=directory,
        backend=None,
        results=[],
        is_stale=is_stale,
        index_empty=True,
    )


def _build_searcher(request: SearchRequest):
    from ..search import VexorSearcher  # local import

    return VexorSearcher(
        model_name=request.model_name,
        batch_size=request.batch_size,
        embed_concurrency=request.embed_concurrency,
        provider=request.provider,
        base_url=request.base_url,
        api_key=request.api_key,
        local_cuda=request.local_cuda,
        embedding_dimensions=request.embedding_dimensions,
    )


def _resolve_query_vector(
    request: SearchRequest,
    searcher,
    file_vectors: np.ndarray,
    *,
    index_id: object = None,
) -> np.ndarray:
    """Resolve the query embedding through the cache layers.

    Lookup order: per-index query cache (when *index_id* is known), shared
    embedding cache, then a live embed call whose result is written back to
    the caches (best effort).
    """
    from ..cache import (  # local import
        embedding_cache_key,
        load_embedding_cache,
        load_query_vector,
        query_cache_key,
        store_embedding_cache,
        store_query_vector,
    )

    expected_dim = file_vectors.shape[1] if file_vectors.ndim == 2 else 0
    query_vector = None
    query_hash = None
    query_text_hash = None
    query_cache_hit = False
    if index_id is not None and not request.no_cache:
        query_hash = query_cache_key(request.query, request.model_name)
        try:
            query_vector = load_query_vector(int(index_id), query_hash)
        except Exception:  # pragma: no cover - best-effort cache lookup
            query_vector = None
        if query_vector is not None and query_vector.size != expected_dim:
            query_vector = None
        elif query_vector is not None:
            query_cache_hit = True

    if query_vector is None and not request.no_cache:
        query_text_hash = embedding_cache_key(
            request.query, dimension=request.embedding_dimensions
        )
        cached = load_embedding_cache(
            request.model_name, [query_text_hash], dimension=request.embedding_dimensions
        )
        query_vector = cached.get(query_text_hash)
        if query_vector is not None and query_vector.size != expected_dim:
            query_vector = None

    if query_vector is None:
        query_vector = searcher.embed_texts([request.query])[0]
        if not request.no_cache:
            if query_text_hash is None:
                query_text_hash = embedding_cache_key(
                    request.query, dimension=request.embedding_dimensions
                )
            try:
                store_embedding_cache(
                    model=request.model_name,
                    embeddings={query_text_hash: query_vector},
                    dimension=request.embedding_dimensions,
                )
            except Exception:  # pragma: no cover - best-effort cache storage
                pass
    if (
        not request.no_cache
        and not query_cache_hit
        and query_vector is not None
        and index_id is not None
        and query_hash is not None
    ):
        try:
            store_query_vector(int(index_id), query_hash, request.query, query_vector)
        except Exception:  # pragma: no cover - best-effort cache storage
            pass
    return query_vector


def _chunk_meta_from_entries(chunk_entries: Sequence[dict]):
    """Chunk metadata getter backed by in-memory chunk entries."""

    def prepare(top_indices: Sequence[int]):
        def get(idx: int) -> dict:
            return chunk_entries[idx] if idx < len(chunk_entries) else {}

        return get

    return prepare


def _chunk_meta_from_cache(chunk_ids: Sequence[int], chunk_entries: Sequence[dict]):
    """Chunk metadata getter that batch-loads cached chunk rows by id."""

    def prepare(top_indices: Sequence[int]):
        from ..cache import load_chunk_metadata  # local import

        chunk_meta_by_id: dict[int, dict] = {}
        if chunk_ids:
            candidate_ids = [
                chunk_ids[idx] for idx in top_indices if idx < len(chunk_ids)
            ]
            if candidate_ids:
                try:
                    chunk_meta_by_id = load_chunk_metadata(candidate_ids)
                except Exception:  # pragma: no cover - best-effort metadata lookup
                    chunk_meta_by_id = {}

        def get(idx: int) -> dict:
            if chunk_ids and idx < len(chunk_ids):
                return chunk_meta_by_id.get(chunk_ids[idx], {})
            if idx < len(chunk_entries):
                return chunk_entries[idx]
            return {}

        return get

    return prepare


def _rank_results(
    request: SearchRequest,
    *,
    paths: Sequence[Path],
    file_vectors: np.ndarray,
    query_vector: np.ndarray,
    chunk_meta_getter,
    lexical_scorer: Callable[[Sequence[str]], dict[int, float]] | None = None,
) -> tuple[list, str | None, ContentBudget | None]:
    """Score the index against the query, then rank and optionally rerank.

    Chunk content is attached last, after reranking has settled the final order, so the
    shared budget is spent on the results the caller actually sees first.
    """
    from ..search import SearchResult  # local import

    reranker = None
    rerank = (request.rerank or DEFAULT_RERANK).strip().lower()
    use_rerank = rerank in {"bm25", "flashrank", "remote"}
    candidate_limit = (
        _resolve_rerank_candidates(request.top_k) if use_rerank else request.top_k
    )
    candidate_count = min(len(paths), candidate_limit)

    query_vector = np.asarray(query_vector, dtype=np.float32).ravel()

    # Validate dimension compatibility between query and index
    index_dimension = file_vectors.shape[1] if file_vectors.ndim == 2 else 0
    query_dimension = query_vector.shape[0]
    if index_dimension != query_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: index has {index_dimension}-dim vectors, "
            f"but query embedding is {query_dimension}-dim. "
            f"This typically happens when embedding_dimensions was changed after building the index. "
            f"Rebuild the index with: vexor index {request.directory}"
        )

    similarities = np.asarray(file_vectors @ query_vector, dtype=np.float32)
    if rerank == "hybrid":
        query_terms = list(dict.fromkeys(bm25.tokenize(request.query)))[
            : bm25.MAX_QUERY_TERMS
        ]
        if query_terms and lexical_scorer is not None:
            bm25_scores_by_row = lexical_scorer(query_terms)
            if bm25_scores_by_row or getattr(lexical_scorer, "has_data", False):
                dense_order = np.argsort(-similarities, kind="stable")
                fused = bm25.rrf_fuse(
                    dense_order, bm25_scores_by_row, len(paths)
                )
                top_indices = _top_indices(fused, request.top_k)
                chunk_meta_for = chunk_meta_getter(top_indices)
                scored: list[SearchResult] = []
                for idx in top_indices:
                    chunk_meta = chunk_meta_for(idx) or {}
                    start_line = chunk_meta.get("start_line")
                    end_line = chunk_meta.get("end_line")
                    scored.append(
                        SearchResult(
                            path=paths[idx],
                            score=float(fused[idx]),
                            preview=chunk_meta.get("preview"),
                            chunk_index=int(chunk_meta.get("chunk_index", 0)),
                            start_line=(
                                int(start_line) if start_line is not None else None
                            ),
                            end_line=int(end_line) if end_line is not None else None,
                        )
                    )
                return scored, "hybrid", _attach_chunk_content(request, scored)
    top_indices = _top_indices(similarities, candidate_count)
    chunk_meta_for = chunk_meta_getter(top_indices)
    scored: list[SearchResult] = []
    for idx in top_indices:
        chunk_meta = chunk_meta_for(idx) or {}
        start_line = chunk_meta.get("start_line")
        end_line = chunk_meta.get("end_line")
        scored.append(
            SearchResult(
                path=paths[idx],
                score=float(similarities[idx]),
                preview=chunk_meta.get("preview"),
                chunk_index=int(chunk_meta.get("chunk_index", 0)),
                start_line=int(start_line) if start_line is not None else None,
                end_line=int(end_line) if end_line is not None else None,
            )
        )
    if use_rerank:
        if rerank == "bm25":
            scored = _apply_bm25_rerank(request.query, scored)
            reranker = "bm25"
        elif rerank == "flashrank":
            scored = _apply_flashrank_rerank(
                request.query, scored, request.flashrank_model
            )
            reranker = "flashrank"
        else:
            scored = _apply_remote_rerank(request.query, scored, request.remote_rerank)
            reranker = "remote"
    final = scored[: request.top_k]
    return final, reranker, _attach_chunk_content(request, final)


@dataclass(slots=True)
class _IndexState:
    paths: Sequence[Path]
    file_vectors: np.ndarray
    metadata: dict
    chunk_entries: Sequence[dict]
    chunk_ids: Sequence[int]
    stale: bool
    index_root: Path
    index_recursive: bool
    index_excludes: tuple
    index_extensions: tuple


def _freshness_key(request: SearchRequest, metadata: dict) -> tuple[object, ...]:
    return (
        str(request.directory.resolve()),
        request.include_hidden,
        request.respect_gitignore,
        request.recursive,
        request.exclude_patterns,
        request.extensions,
        metadata.get("index_id"),
        metadata.get("generated_at"),
        metadata.get("vector_file"),
    )


def _load_filtered_index(
    request: SearchRequest,
    exclude_spec,
    *,
    load_index_vectors,
    list_cache_entries,
) -> _IndexState:
    """Load the cached index, apply request filters, and compute staleness."""
    (
        paths,
        file_vectors,
        metadata,
        ext_filter,
        index_extensions,
        index_root,
        index_recursive,
        index_excludes,
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
    if exclude_spec is not None:
        paths, file_vectors, metadata = _filter_index_by_exclude_patterns(
            paths,
            file_vectors,
            metadata,
            request.directory,
            exclude_spec,
        )
    file_snapshot = metadata.get("files", [])
    stale = False
    if file_snapshot:
        freshness_tracker = request.freshness_tracker
        freshness_key = _freshness_key(request, metadata)
        validation_token = None
        if freshness_tracker is not None:
            validation_token = freshness_tracker.begin_validation(request.directory)
            if freshness_tracker.is_fresh(request.directory, freshness_key):
                file_snapshot = []
        if file_snapshot:
            current = is_cache_current(
                request.directory,
                request.include_hidden,
                request.respect_gitignore,
                file_snapshot,
                recursive=request.recursive,
                exclude_patterns=request.exclude_patterns,
                extensions=request.extensions,
            )
            stable = True
            if (
                current
                and freshness_tracker is not None
                and validation_token is not None
            ):
                stable = freshness_tracker.finish_validation(
                    request.directory,
                    freshness_key,
                    validation_token,
                )
            stale = not current or not stable
    return _IndexState(
        paths=paths,
        file_vectors=file_vectors,
        metadata=metadata,
        chunk_entries=metadata.get("chunks", []),
        chunk_ids=metadata.get("chunk_ids", []),
        stale=stale,
        index_root=index_root,
        index_recursive=index_recursive,
        index_excludes=index_excludes,
        index_extensions=index_extensions,
    )


def _build_index_for_request(
    request: SearchRequest,
    build_index,
    *,
    root: Path,
    recursive: bool,
    exclude_patterns,
    extensions,
):
    return build_index(
        root,
        include_hidden=request.include_hidden,
        respect_gitignore=request.respect_gitignore,
        mode=request.mode,
        recursive=recursive,
        model_name=request.model_name,
        batch_size=request.batch_size,
        embed_concurrency=request.embed_concurrency,
        extract_concurrency=request.extract_concurrency,
        extract_backend=request.extract_backend,
        provider=request.provider,
        base_url=request.base_url,
        api_key=request.api_key,
        local_cuda=request.local_cuda,
        exclude_patterns=exclude_patterns,
        extensions=extensions,
        no_cache=request.no_cache,
        embedding_dimensions=request.embedding_dimensions,
    )


def perform_search(request: SearchRequest) -> SearchResponse:
    """Execute the semantic search flow and return ranked results."""

    if request.temporary_index or request.no_cache:
        return _perform_search_with_temporary_index(request)

    from ..cache import list_cache_entries, load_index_vectors  # local import
    from .index_service import IndexStatus, build_index  # local import

    exclude_spec = build_exclude_spec(request.exclude_patterns)

    def load_state() -> _IndexState:
        return _load_filtered_index(
            request,
            exclude_spec,
            load_index_vectors=load_index_vectors,
            list_cache_entries=list_cache_entries,
        )

    try:
        state = load_state()
    except FileNotFoundError:
        if not request.auto_index:
            raise
        result = _build_index_for_request(
            request,
            build_index,
            root=request.directory,
            recursive=request.recursive,
            exclude_patterns=request.exclude_patterns,
            extensions=request.extensions,
        )
        if result.status == IndexStatus.EMPTY:
            return _empty_response(request.directory, is_stale=False)
        state = load_state()

    if state.stale and request.auto_index:
        if request.index_vector_cache is not None:
            request.index_vector_cache.clear()
        result = _build_index_for_request(
            request,
            build_index,
            root=state.index_root,
            recursive=state.index_recursive,
            exclude_patterns=state.index_excludes,
            extensions=state.index_extensions,
        )
        if result.status == IndexStatus.EMPTY:
            del state
            if request.index_vector_cache is not None:
                request.index_vector_cache.prune()
            return _empty_response(request.directory, is_stale=False)
        state = load_state()
        if request.index_vector_cache is not None:
            request.index_vector_cache.prune()

    if not len(state.paths):
        return _empty_response(request.directory, is_stale=state.stale)

    searcher = _build_searcher(request)
    query_vector = _resolve_query_vector(
        request,
        searcher,
        state.file_vectors,
        index_id=state.metadata.get("index_id"),
    )
    results, reranker, content_budget = _rank_results(
        request,
        paths=state.paths,
        file_vectors=state.file_vectors,
        query_vector=query_vector,
        chunk_meta_getter=_chunk_meta_from_cache(state.chunk_ids, state.chunk_entries),
        lexical_scorer=(
            _hybrid_scorer_from_cache(
                state.metadata.get("index_id"), state.chunk_ids
            )
            if (request.rerank or "").strip().lower() == "hybrid"
            else None
        ),
    )
    return SearchResponse(
        base_path=request.directory,
        backend=searcher.device,
        results=results,
        is_stale=state.stale,
        index_empty=False,
        reranker=reranker,
        content_budget=content_budget,
    )


def search_from_vectors(
    request: SearchRequest,
    *,
    paths: Sequence[Path],
    file_vectors: np.ndarray,
    metadata: dict,
    is_stale: bool = False,
) -> SearchResponse:
    """Return ranked results from an in-memory index."""

    if not len(paths):
        return _empty_response(request.directory, is_stale=is_stale)

    searcher = _build_searcher(request)
    query_vector = _resolve_query_vector(request, searcher, file_vectors)
    results, reranker, content_budget = _rank_results(
        request,
        paths=paths,
        file_vectors=file_vectors,
        query_vector=query_vector,
        chunk_meta_getter=_chunk_meta_from_entries(metadata.get("chunks", [])),
        lexical_scorer=(
            _hybrid_scorer_from_entries(metadata.get("chunks", []))
            if (request.rerank or "").strip().lower() == "hybrid"
            else None
        ),
    )
    return SearchResponse(
        base_path=request.directory,
        backend=searcher.device,
        results=results,
        is_stale=is_stale,
        index_empty=False,
        reranker=reranker,
        content_budget=content_budget,
    )


def _perform_search_with_temporary_index(request: SearchRequest) -> SearchResponse:
    from .index_service import build_index_in_memory  # local import

    paths, file_vectors, metadata = build_index_in_memory(
        request.directory,
        include_hidden=request.include_hidden,
        respect_gitignore=request.respect_gitignore,
        mode=request.mode,
        recursive=request.recursive,
        model_name=request.model_name,
        batch_size=request.batch_size,
        embed_concurrency=request.embed_concurrency,
        extract_concurrency=request.extract_concurrency,
        extract_backend=request.extract_backend,
        provider=request.provider,
        base_url=request.base_url,
        api_key=request.api_key,
        local_cuda=request.local_cuda,
        exclude_patterns=request.exclude_patterns,
        extensions=request.extensions,
        no_cache=request.no_cache,
        embedding_dimensions=request.embedding_dimensions,
    )
    return search_from_vectors(
        request,
        paths=paths,
        file_vectors=file_vectors,
        metadata=metadata,
        is_stale=False,
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
    tuple[str, ...],
]:
    def load_vectors(
        root: Path,
        recursive: bool,
        exclude_patterns: Sequence[str],
        extensions: Sequence[str],
    ):
        kwargs = {"respect_gitignore": request.respect_gitignore}
        if request.index_vector_cache is not None:
            kwargs["memory_cache"] = request.index_vector_cache
        return load_index_vectors(
            root,
            request.model_name,
            request.include_hidden,
            request.mode,
            recursive,
            exclude_patterns,
            extensions,
            **kwargs,
        )

    try:
        paths, file_vectors, metadata = load_vectors(
            request.directory,
            request.recursive,
            request.exclude_patterns,
            request.extensions,
        )
        # Check dimension compatibility when user explicitly requests a specific dimension
        cached_dimension = metadata.get("dimension")
        requested_dimension = request.embedding_dimensions
        if (
            cached_dimension is not None
            and requested_dimension is not None
            and cached_dimension != requested_dimension
        ):
            raise FileNotFoundError(
                f"Cached index has dimension {cached_dimension}, "
                f"but requested {requested_dimension}"
            )
        return (
            paths,
            file_vectors,
            metadata,
            None,
            request.extensions,
            request.directory,
            request.recursive,
            request.exclude_patterns,
        )
    except FileNotFoundError as exc:
        missing_exc = exc
    superset_entry = _select_cache_superset(request, list_cache_entries)
    if superset_entry is None:
        raise missing_exc
    superset_root = Path(superset_entry.get("root_path", "")).expanduser().resolve()
    superset_recursive = bool(superset_entry.get("recursive"))
    superset_extensions = tuple(superset_entry.get("extensions") or ())
    superset_excludes = tuple(superset_entry.get("exclude_patterns") or ())
    paths, file_vectors, metadata = load_vectors(
        superset_root,
        superset_recursive,
        superset_excludes,
        superset_extensions,
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
        superset_excludes,
    )


def _select_cache_superset(
    request: SearchRequest,
    list_cache_entries,
) -> dict | None:
    requested = set(request.extensions or ())
    requested_excludes = normalize_exclude_patterns(request.exclude_patterns or ())
    requested_exclude_set = set(requested_excludes)
    root = request.directory.resolve()
    candidates: list[tuple[int, int, int, int, int, dict]] = []
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
        # Check embedding dimension compatibility when user explicitly requests a specific dimension
        cached_dimension = entry.get("dimension")
        requested_dimension = request.embedding_dimensions
        if (
            cached_dimension is not None
            and requested_dimension is not None
            and cached_dimension != requested_dimension
        ):
            continue
        cached_excludes = tuple(entry.get("exclude_patterns") or ())
        cached_exclude_set = set(normalize_exclude_patterns(cached_excludes))
        if requested_exclude_set:
            if cached_exclude_set and not cached_exclude_set.issubset(requested_exclude_set):
                continue
        elif cached_exclude_set:
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
        exclude_gap = len(requested_exclude_set) - len(cached_exclude_set)
        candidates.append(
            (distance, recursive_mismatch, file_count, ext_count, exclude_gap, entry)
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return candidates[0][5]


def _filter_index_by_extensions(
    paths: Sequence[Path],
    file_vectors,
    metadata: dict,
    extensions: Sequence[str],
) -> tuple[list[Path], Sequence[Sequence[float]], dict]:
    ext_set = {ext.lower() for ext in extensions if ext}
    if not ext_set:
        return list(paths), file_vectors, metadata
    chunk_ids = metadata.get("chunk_ids")
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
        if chunk_ids is not None:
            filtered_metadata["chunk_ids"] = []
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
    if chunk_ids is not None:
        filtered_metadata["chunk_ids"] = [
            chunk_ids[idx] for idx in keep_indices if idx < len(chunk_ids)
        ]
    return filtered_paths, filtered_vectors, filtered_metadata


def _filter_index_by_exclude_patterns(
    paths: Sequence[Path],
    file_vectors,
    metadata: dict,
    root: Path,
    exclude_spec,
) -> tuple[list[Path], Sequence[Sequence[float]], dict]:
    if exclude_spec is None:
        return list(paths), file_vectors, metadata
    chunk_ids = metadata.get("chunk_ids")
    keep_indices: list[int] = []
    filtered_paths: list[Path] = []
    root_resolved = root.resolve()
    for idx, path in enumerate(paths):
        try:
            rel = path.resolve().relative_to(root_resolved).as_posix()
        except ValueError:
            rel = path.as_posix()
        if is_excluded_path(exclude_spec, rel, is_dir=False):
            continue
        keep_indices.append(idx)
        filtered_paths.append(path)
    if not keep_indices:
        filtered_vectors = file_vectors[:0]
        filtered_metadata = dict(metadata)
        filtered_metadata["files"] = _filter_file_snapshot_by_exclude_patterns(
            metadata.get("files", []),
            exclude_spec,
        )
        filtered_metadata["chunks"] = []
        if chunk_ids is not None:
            filtered_metadata["chunk_ids"] = []
        return [], filtered_vectors, filtered_metadata
    filtered_vectors = file_vectors[keep_indices]
    chunk_entries = metadata.get("chunks", [])
    filtered_chunks = [
        chunk_entries[idx] for idx in keep_indices if idx < len(chunk_entries)
    ]
    filtered_metadata = dict(metadata)
    filtered_metadata["files"] = _filter_file_snapshot_by_exclude_patterns(
        metadata.get("files", []),
        exclude_spec,
    )
    filtered_metadata["chunks"] = filtered_chunks
    if chunk_ids is not None:
        filtered_metadata["chunk_ids"] = [
            chunk_ids[idx] for idx in keep_indices if idx < len(chunk_ids)
        ]
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
    chunk_ids = metadata.get("chunk_ids")
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
        if chunk_ids is not None:
            filtered_metadata["chunk_ids"] = []
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
    if chunk_ids is not None:
        filtered_metadata["chunk_ids"] = [
            chunk_ids[idx] for idx in keep_indices if idx < len(chunk_ids)
        ]
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


def _filter_file_snapshot_by_exclude_patterns(
    entries: Sequence[dict],
    spec,
) -> list[dict]:
    if spec is None:
        return list(entries)
    filtered: list[dict] = []
    for entry in entries:
        rel_path = entry.get("path", "")
        rel_posix = Path(rel_path).as_posix() if rel_path else ""
        if is_excluded_path(spec, rel_posix, is_dir=False):
            continue
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
