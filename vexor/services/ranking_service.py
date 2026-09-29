"""Document ranking shared by file search, collections, and diagnostics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, TypeVar
from urllib import error as urlerror
from urllib import request as urlrequest

from .. import bm25
from ..config import (
    DEFAULT_FLASHRANK_MAX_LENGTH,
    DEFAULT_FLASHRANK_MODEL,
    RemoteRerankConfig,
    normalize_remote_rerank_url,
    resolve_remote_rerank_api_key,
)
from ..text import Messages

CANDIDATE_RERANKERS = frozenset({"bm25", "flashrank", "remote"})
_TOKEN_RE = bm25._TOKEN_RE
_BM25_K1 = bm25.BM25_K1
_BM25_B = bm25.BM25_B
_FUSION_SEMANTIC_WEIGHT = 0.7
_get_bm25_tokenizer = bm25._get_bm25_tokenizer


def _bm25_tokenize(text: str) -> list[str]:
    # Candidate BM25 preserves its legacy tokens; persisted BM25 also adds
    # whole identifiers, which would change this reranker's scores.
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


class _ScoredResult(Protocol):
    score: float


_ScoredResultT = TypeVar("_ScoredResultT", bound=_ScoredResult)


def apply_ranking(
    results: Sequence[_ScoredResultT],
    ranking: Sequence[tuple[int, float | None]],
) -> list[_ScoredResultT]:
    """Reorder *results* by a reranker's ``(index, score)`` pairs.

    Unknown, duplicate, and out-of-range indices are dropped, and anything the
    reranker left out keeps its original relative order at the tail, so a partial
    response degrades to "reranked head, dense tail" instead of losing results.
    """

    ordered: list[_ScoredResultT] = []
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


def resolve_rerank_candidates(top_k: int) -> int:
    candidate = int(top_k * 2)
    return max(20, min(candidate, 150))


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


def rank_documents_bm25(
    query: str,
    documents: Sequence[str],
    base_scores: Sequence[float],
) -> list[tuple[int, float]] | None:
    """Fuse lexical scores over *documents* with the retrieval scores behind them.

    Returns ``None`` when the query carries no usable tokens, which leaves the
    caller's original order untouched.
    """

    if len(documents) != len(base_scores):
        raise ValueError(
            "rerank documents and base scores must line up: "
            f"got {len(documents)} documents and {len(base_scores)} scores"
        )
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
        for index, (base, lexical) in enumerate(
            zip(base_norm, lexical_norm, strict=True)
        )
    ]
    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


@lru_cache(maxsize=4)
def _get_flashranker(model_name: str | None, max_length: int):
    from flashrank import Ranker

    from ..config import flashrank_cache_dir

    cache_dir = flashrank_cache_dir()
    kwargs = {"max_length": max_length, "cache_dir": str(cache_dir)}
    if model_name:
        kwargs["model_name"] = model_name
    return Ranker(**kwargs)


def rank_documents_flashrank(
    query: str,
    documents: Sequence[str],
    model_name: str | None,
) -> list[tuple[int, float | None]]:
    if not documents:
        # Nothing to rank, and loading a ranker model to say so would be waste.
        return []
    try:
        from flashrank import RerankRequest
    except ImportError as exc:
        raise RuntimeError(Messages.ERROR_FLASHRANK_MISSING) from exc
    try:
        effective_model = model_name or DEFAULT_FLASHRANK_MODEL
        ranker = _get_flashranker(effective_model, DEFAULT_FLASHRANK_MAX_LENGTH)
    except ImportError as exc:
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


def _resolve_remote_rerank_config(
    config: RemoteRerankConfig | None,
) -> RemoteRerankConfig:
    if not config:
        raise RuntimeError(Messages.ERROR_REMOTE_RERANK_INCOMPLETE)
    base_url = normalize_remote_rerank_url(config.base_url)
    api_key = resolve_remote_rerank_api_key(config.api_key)
    if not (base_url and config.model and api_key):
        raise RuntimeError(Messages.ERROR_REMOTE_RERANK_INCOMPLETE)
    if base_url != config.base_url or api_key != config.api_key:
        return RemoteRerankConfig(
            base_url=base_url,
            api_key=api_key,
            model=config.model,
        )
    return config


def remote_rerank_request(
    *,
    config: RemoteRerankConfig,
    query: str,
    documents: Sequence[str],
) -> dict:
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


def rank_documents_remote(
    query: str,
    documents: Sequence[str],
    config: RemoteRerankConfig | None,
) -> list[tuple[int, float | None]]:
    if not documents:
        # Rerank endpoints reject an empty document array, so answer locally
        # rather than turning "nothing to rank" into a provider error.
        return []
    resolved = _resolve_remote_rerank_config(config)
    payload = remote_rerank_request(
        config=resolved,
        query=query,
        documents=documents,
    )
    return _extract_remote_rerank_items(payload)


def rerank_candidates(
    query: str,
    candidates: Sequence[_ScoredResultT],
    documents: Sequence[str],
    *,
    rerank: str,
    flashrank_model: str | None = None,
    remote_rerank: RemoteRerankConfig | None = None,
) -> list[_ScoredResultT]:
    """Rank prepared documents while preserving each source's result objects."""
    if rerank == "bm25":
        ranking = rank_documents_bm25(query, documents, [item.score for item in candidates])
    elif rerank == "flashrank":
        ranking = rank_documents_flashrank(query, documents, flashrank_model)
    elif rerank == "remote":
        ranking = rank_documents_remote(query, documents, remote_rerank)
    elif rerank in {"off", "hybrid"}:
        return list(candidates)
    else:
        raise ValueError(Messages.ERROR_RERANK_INVALID.format(
            value=rerank, allowed=", ".join(sorted(CANDIDATE_RERANKERS))
        ))
    return list(candidates) if ranking is None else apply_ranking(candidates, ranking)
