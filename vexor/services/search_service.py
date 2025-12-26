"""Logic helpers for the `vexor search` command."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import re
import numpy as np
from typing import Sequence, TYPE_CHECKING
from urllib import error as urlerror
from urllib import request as urlrequest

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
    from ..search import SearchResult


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


@dataclass(slots=True)
class SearchResponse:
    base_path: Path
    backend: str | None
    results: Sequence[SearchResult]
    is_stale: bool
    index_empty: bool
    reranker: str | None = None


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_BM25_K1 = 1.5
_BM25_B = 0.75
_FUSION_SEMANTIC_WEIGHT = 0.7


@lru_cache(maxsize=1)
def _get_bm25_tokenizer():
    try:
        from tokenizers.pre_tokenizers import BertPreTokenizer
    except Exception:
        return None
    return BertPreTokenizer()


def _bm25_tokenize(text: str) -> list[str]:
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


def _build_rerank_document(result: SearchResult) -> str:
    preview = result.preview or ""
    return f"{result.path.name} {result.path.as_posix()} {preview}".strip()


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


def _apply_bm25_rerank(query: str, results: Sequence[SearchResult]) -> list[SearchResult]:
    if not results:
        return []
    query_tokens = _bm25_tokenize(query)
    if not query_tokens:
        return list(results)
    documents = [_bm25_tokenize(_build_rerank_document(result)) for result in results]
    bm25_scores = _bm25_scores(query_tokens, documents)
    semantic_scores = [max(result.score, 0.0) for result in results]
    semantic_norm = _normalize_by_max(semantic_scores)
    bm25_norm = _normalize_by_max(bm25_scores)
    fused: list[SearchResult] = []
    for result, sem_score, bm25_score in zip(results, semantic_norm, bm25_norm):
        fused_score = _FUSION_SEMANTIC_WEIGHT * sem_score + (
            (1.0 - _FUSION_SEMANTIC_WEIGHT) * bm25_score
        )
        result.score = float(fused_score)
        fused.append(result)
    fused.sort(key=lambda item: item.score, reverse=True)
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


def _apply_flashrank_rerank(
    query: str,
    results: Sequence[SearchResult],
    model_name: str | None,
) -> list[SearchResult]:
    if not results:
        return []
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
    passages = []
    for idx, result in enumerate(results):
        text = _build_rerank_document(result) or result.path.as_posix()
        passages.append({"id": idx, "text": text})
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)
    id_to_result = {idx: result for idx, result in enumerate(results)}
    ordered: list[SearchResult] = []
    seen: set[int] = set()
    for item in reranked:
        idx = item.get("id")
        if idx is None:
            continue
        result = id_to_result.get(idx)
        if result is None:
            continue
        score = item.get("score")
        if score is not None:
            result.score = float(score)
        ordered.append(result)
        seen.add(idx)
    if len(ordered) < len(results):
        for idx, result in enumerate(results):
            if idx not in seen:
                ordered.append(result)
    return ordered


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


def _apply_remote_rerank(
    query: str,
    results: Sequence[SearchResult],
    config: RemoteRerankConfig | None,
) -> list[SearchResult]:
    if not results:
        return []
    resolved = _resolve_remote_rerank_config(config)
    documents = [
        _build_rerank_document(result) or result.path.as_posix() for result in results
    ]
    payload = _remote_rerank_request(
        config=resolved,
        query=query,
        documents=documents,
    )
    items = _extract_remote_rerank_items(payload)
    if not items:
        return list(results)
    ordered: list[SearchResult] = []
    seen: set[int] = set()
    for idx, score in items:
        if idx < 0 or idx >= len(results) or idx in seen:
            continue
        result = results[idx]
        if score is not None:
            result.score = score
        ordered.append(result)
        seen.add(idx)
    for idx, result in enumerate(results):
        if idx not in seen:
            ordered.append(result)
    return ordered


def perform_search(request: SearchRequest) -> SearchResponse:
    """Execute the semantic search flow and return ranked results."""

    if request.temporary_index or request.no_cache:
        return _perform_search_with_temporary_index(request)

    from ..cache import (  # local import
        embedding_cache_key,
        list_cache_entries,
        load_chunk_metadata,
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
            index_excludes,
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
            extract_concurrency=request.extract_concurrency,
            extract_backend=request.extract_backend,
            provider=request.provider,
            base_url=request.base_url,
            api_key=request.api_key,
            local_cuda=request.local_cuda,
            exclude_patterns=request.exclude_patterns,
            extensions=request.extensions,
            no_cache=request.no_cache,
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
            index_excludes,
        ) = _load_index_vectors_for_request(
            request,
            load_index_vectors=load_index_vectors,
            list_cache_entries=list_cache_entries,
        )

    exclude_spec = build_exclude_spec(request.exclude_patterns)

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
    chunk_entries = metadata.get("chunks", [])
    chunk_ids = metadata.get("chunk_ids", [])
    stale = bool(file_snapshot) and not is_cache_current(
        request.directory,
        request.include_hidden,
        request.respect_gitignore,
        file_snapshot,
        recursive=request.recursive,
        exclude_patterns=request.exclude_patterns,
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
            extract_concurrency=request.extract_concurrency,
            extract_backend=request.extract_backend,
            provider=request.provider,
            base_url=request.base_url,
            api_key=request.api_key,
            local_cuda=request.local_cuda,
            exclude_patterns=index_excludes,
            extensions=index_extensions,
            no_cache=request.no_cache,
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
        chunk_entries = metadata.get("chunks", [])
        stale = bool(file_snapshot) and not is_cache_current(
            request.directory,
            request.include_hidden,
            request.respect_gitignore,
            file_snapshot,
            recursive=request.recursive,
            exclude_patterns=request.exclude_patterns,
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
    query_text_hash = None
    index_id = metadata.get("index_id")
    if index_id is not None and not request.no_cache:
        query_hash = query_cache_key(request.query, request.model_name)
        try:
            query_vector = load_query_vector(int(index_id), query_hash)
        except Exception:  # pragma: no cover - best-effort cache lookup
            query_vector = None

        if query_vector is not None and query_vector.size != file_vectors.shape[1]:
            query_vector = None

    if query_vector is None and not request.no_cache:
        query_text_hash = embedding_cache_key(request.query)
        cached = load_embedding_cache(request.model_name, [query_text_hash])
        query_vector = cached.get(query_text_hash)
        if query_vector is not None and query_vector.size != file_vectors.shape[1]:
            query_vector = None

    if query_vector is None:
        query_vector = searcher.embed_texts([request.query])[0]
        if not request.no_cache:
            if query_text_hash is None:
                query_text_hash = embedding_cache_key(request.query)
            try:
                store_embedding_cache(
                    model=request.model_name,
                    embeddings={query_text_hash: query_vector},
                )
            except Exception:  # pragma: no cover - best-effort cache storage
                pass
    if (
        not request.no_cache
        and query_vector is not None
        and index_id is not None
        and query_hash is not None
    ):
        try:
            store_query_vector(int(index_id), query_hash, request.query, query_vector)
        except Exception:  # pragma: no cover - best-effort cache storage
            pass
    reranker = None
    rerank = (request.rerank or DEFAULT_RERANK).strip().lower()
    use_rerank = rerank in {"bm25", "flashrank", "remote"}
    if use_rerank:
        candidate_limit = _resolve_rerank_candidates(request.top_k)
    else:
        candidate_limit = request.top_k
    candidate_count = min(len(paths), candidate_limit)

    query_vector = np.asarray(query_vector, dtype=np.float32).ravel()
    similarities = np.asarray(file_vectors @ query_vector, dtype=np.float32)
    top_indices = _top_indices(similarities, candidate_count)
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
    scored: list[SearchResult] = []
    for idx in top_indices:
        path = paths[idx]
        score = similarities[idx]
        chunk_meta = {}
        if chunk_ids and idx < len(chunk_ids):
            chunk_meta = chunk_meta_by_id.get(chunk_ids[idx], {})
        elif idx < len(chunk_entries):
            chunk_meta = chunk_entries[idx]
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
    if use_rerank:
        candidates = scored
        if rerank == "bm25":
            candidates = _apply_bm25_rerank(request.query, candidates)
            reranker = "bm25"
        elif rerank == "flashrank":
            candidates = _apply_flashrank_rerank(
                request.query,
                candidates,
                request.flashrank_model,
            )
            reranker = "flashrank"
        else:
            candidates = _apply_remote_rerank(
                request.query,
                candidates,
                request.remote_rerank,
            )
            reranker = "remote"
        results = candidates[: request.top_k]
    else:
        results = scored[: request.top_k]
    return SearchResponse(
        base_path=request.directory,
        backend=searcher.device,
        results=results,
        is_stale=stale,
        index_empty=False,
        reranker=reranker,
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
        return SearchResponse(
            base_path=request.directory,
            backend=None,
            results=[],
            is_stale=is_stale,
            index_empty=True,
        )

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
    query_text_hash = None
    if not request.no_cache:
        from ..cache import embedding_cache_key, load_embedding_cache, store_embedding_cache

        query_text_hash = embedding_cache_key(request.query)
        cached = load_embedding_cache(request.model_name, [query_text_hash])
        query_vector = cached.get(query_text_hash)
        if query_vector is not None and query_vector.size != file_vectors.shape[1]:
            query_vector = None

    if query_vector is None:
        query_vector = searcher.embed_texts([request.query])[0]
        if not request.no_cache:
            if query_text_hash is None:
                from ..cache import embedding_cache_key, store_embedding_cache

                query_text_hash = embedding_cache_key(request.query)
            try:
                store_embedding_cache(
                    model=request.model_name,
                    embeddings={query_text_hash: query_vector},
                )
            except Exception:  # pragma: no cover - best-effort cache storage
                pass
    reranker = None
    rerank = (request.rerank or DEFAULT_RERANK).strip().lower()
    use_rerank = rerank in {"bm25", "flashrank", "remote"}
    if use_rerank:
        candidate_limit = _resolve_rerank_candidates(request.top_k)
    else:
        candidate_limit = request.top_k
    candidate_count = min(len(paths), candidate_limit)

    query_vector = np.asarray(query_vector, dtype=np.float32).ravel()
    similarities = np.asarray(file_vectors @ query_vector, dtype=np.float32)
    top_indices = _top_indices(similarities, candidate_count)
    chunk_entries = metadata.get("chunks", [])
    scored: list[SearchResult] = []
    for idx in top_indices:
        path = paths[idx]
        score = similarities[idx]
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
    if use_rerank:
        candidates = scored
        if rerank == "bm25":
            candidates = _apply_bm25_rerank(request.query, candidates)
            reranker = "bm25"
        elif rerank == "flashrank":
            candidates = _apply_flashrank_rerank(
                request.query,
                candidates,
                request.flashrank_model,
            )
            reranker = "flashrank"
        else:
            candidates = _apply_remote_rerank(
                request.query,
                candidates,
                request.remote_rerank,
            )
            reranker = "remote"
        results = candidates[: request.top_k]
    else:
        results = scored[: request.top_k]
    return SearchResponse(
        base_path=request.directory,
        backend=searcher.device,
        results=results,
        is_stale=is_stale,
        index_empty=False,
        reranker=reranker,
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
    try:
        paths, file_vectors, metadata = load_index_vectors(
            request.directory,
            request.model_name,
            request.include_hidden,
            request.mode,
            request.recursive,
            request.exclude_patterns,
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
    paths, file_vectors, metadata = load_index_vectors(
        superset_root,
        request.model_name,
        request.include_hidden,
        request.mode,
        superset_recursive,
        superset_excludes,
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
