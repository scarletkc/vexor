"""Orchestration for filesystem-independent text collections.

The retrieval core Vexor already had is source-agnostic: ``bm25`` holds pure
functions over ids, the embedding cache is keyed by ``(model, text_hash)``, and
``VexorSearcher.embed_texts`` L2-normalizes any text you hand it. This module is
the second consumer of that core — records arrive from a caller's database
instead of from disk, so results carry a record id and metadata instead of a
path and line range.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass

import numpy as np

from .. import bm25, collection_store
from ..cache import embedding_cache_key
from ..collection_store import (
    CollectionError,
    CollectionInfo,
    PreparedRecord,
    ScalarValue,
    StoredRecord,
)
from ..config import DEFAULT_RERANK, SUPPORTED_RERANKERS, RemoteRerankConfig
from ..text import Messages
from .embedding_service import embed_texts_with_cache
from .search_service import (
    _apply_ranking,
    _rank_documents_bm25,
    _rank_documents_flashrank,
    _rank_documents_remote,
    _resolve_rerank_candidates,
)

SUPPORTED_COLLECTION_RERANKERS = SUPPORTED_RERANKERS
DEFAULT_COLLECTION_RERANK = DEFAULT_RERANK


@dataclass(slots=True)
class RecordResult:
    """One scored record: the caller's id and text, never a path."""

    id: str
    text: str
    metadata: dict[str, ScalarValue]
    score: float


@dataclass(slots=True)
class UpsertReport:
    """What an upsert actually did, so callers can see cache hits."""

    written: int
    embedded: int
    skipped: int


def _require_collection(name: str) -> CollectionInfo:
    info = collection_store.get_collection(name)
    if info is None:
        raise CollectionError(Messages.ERROR_COLLECTION_NOT_FOUND.format(name=name))
    return info


def _verify_contract(
    info: CollectionInfo,
    *,
    name: str,
    provider: str,
    model_name: str,
    dimension: int | None = None,
) -> None:
    """Raise unless the caller's embedding contract matches what is pinned.

    This has to guard reads as well as writes. Two different models can share a
    vector width, so a dimension check alone lets a query embedded by one model
    score vectors produced by another: the arithmetic succeeds and the ranking
    is meaningless. Silently wrong results are worse than an error.
    """

    effective_dimension = info.dimension if dimension is None else dimension
    if (
        info.provider != provider
        or info.model != model_name
        or info.dimension != effective_dimension
    ):
        raise CollectionError(
            Messages.ERROR_COLLECTION_CONTRACT_MISMATCH.format(
                name=name,
                stored=f"{info.provider}/{info.model}/{info.dimension}",
                requested=f"{provider}/{model_name}/{effective_dimension}",
            )
        )


def _normalize_records(
    records: Sequence[Mapping[str, object]],
    *,
    embedding_dimension: int | None,
) -> list[tuple[str, str, str, dict[str, ScalarValue]]]:
    """Validate raw records into ``(key, text, text_hash, metadata)`` tuples."""

    seen: set[str] = set()
    normalized: list[tuple[str, str, str, dict[str, ScalarValue]]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise CollectionError(Messages.ERROR_COLLECTION_RECORD_NOT_MAPPING)
        raw_key = record.get("id")
        key = str(raw_key).strip() if raw_key is not None else ""
        if not key:
            raise CollectionError(Messages.ERROR_COLLECTION_RECORD_KEY_REQUIRED)
        if key in seen:
            raise CollectionError(
                Messages.ERROR_COLLECTION_DUPLICATE_KEY.format(key=key)
            )
        seen.add(key)
        text = record.get("text")
        text_value = str(text) if text is not None else ""
        if not text_value.strip():
            raise CollectionError(
                Messages.ERROR_COLLECTION_TEXT_EMPTY.format(key=key)
            )
        metadata = collection_store.normalize_metadata(
            record.get("metadata")  # type: ignore[arg-type]
        )
        text_hash = embedding_cache_key(text_value, dimension=embedding_dimension)
        normalized.append((key, text_value, text_hash, metadata))
    return normalized


def upsert_records(
    *,
    name: str,
    records: Sequence[Mapping[str, object]],
    searcher,
    model_name: str,
    provider: str,
    embedding_dimension: int | None = None,
    no_cache: bool = False,
) -> UpsertReport:
    """Insert or replace *records*, embedding only the texts that changed."""

    if not records:
        return UpsertReport(written=0, embedded=0, skipped=0)
    normalized = _normalize_records(records, embedding_dimension=embedding_dimension)

    info = collection_store.get_collection(name)
    existed_before = info is not None
    existing_hashes: dict[str, str] = {}
    if info is not None:
        _verify_contract(info, name=name, provider=provider, model_name=model_name)
        existing_hashes = collection_store.load_record_hashes(
            info.id, [key for key, _, _, _ in normalized]
        )

    pending = [
        entry for entry in normalized if existing_hashes.get(entry[0]) != entry[2]
    ]
    unchanged = [
        entry for entry in normalized if existing_hashes.get(entry[0]) == entry[2]
    ]

    vectors: np.ndarray | None = None
    if pending:
        vectors = embed_texts_with_cache(
            searcher=searcher,
            model_name=model_name,
            labels=[text for _, text, _, _ in pending],
            no_cache=no_cache,
            embedding_dimension=embedding_dimension,
        )
        if vectors.size == 0 or vectors.ndim != 2:
            raise CollectionError(Messages.ERROR_COLLECTION_EMBED_FAILED)

    if info is None:
        if vectors is None:
            # Every record was unchanged, which cannot happen before the
            # collection exists; guard anyway so a future caller gets an error
            # rather than a collection pinned to a guessed dimension.
            raise CollectionError(Messages.ERROR_COLLECTION_EMBED_FAILED)
        dimension = int(vectors.shape[1])
    else:
        dimension = info.dimension
        if vectors is not None:
            _verify_contract(
                info,
                name=name,
                provider=provider,
                model_name=model_name,
                dimension=int(vectors.shape[1]),
            )

    info = collection_store.ensure_collection(
        name,
        provider=provider,
        model=model_name,
        dimension=dimension,
    )

    prepared: list[PreparedRecord] = []
    for offset, (key, text, text_hash, metadata) in enumerate(pending):
        vector = np.asarray(vectors[offset], dtype=np.float32)
        tokens = bm25.tokenize(text)
        prepared.append(
            PreparedRecord(
                record_key=key,
                text=text,
                text_hash=text_hash,
                metadata=metadata,
                vector=vector,
                bm25_terms=bm25.term_frequencies(tokens),
                token_count=len(tokens),
                refresh_embedding=True,
            )
        )
    for key, text, text_hash, metadata in unchanged:
        prepared.append(
            PreparedRecord(
                record_key=key,
                text=text,
                text_hash=text_hash,
                metadata=metadata,
                vector=None,
                bm25_terms={},
                token_count=0,
                refresh_embedding=False,
            )
        )

    try:
        written = collection_store.upsert_records(info.id, prepared)
    except BaseException:
        # Creating the collection and writing its first records are separate
        # transactions. If the write fails, a collection this call brought into
        # existence would linger holding a pinned contract the caller never
        # successfully used — and block retrying the name under another one.
        #
        # The drop is conditional on the row still being this exact collection
        # and still being empty, decided inside one transaction, so a concurrent
        # writer that populated it in the meantime keeps its records.
        #
        # Best-effort on purpose: whatever goes wrong while cleaning up matters
        # far less than the error the caller is about to see, and must not
        # replace it.
        if not existed_before:
            with suppress(Exception):
                collection_store.drop_collection_if_empty(name, info.id)
        raise
    return UpsertReport(
        written=written,
        embedded=len(pending),
        skipped=len(unchanged),
    )


def search_records(
    *,
    name: str,
    query: str,
    searcher,
    model_name: str,
    provider: str,
    top_k: int = 10,
    filters: Mapping[str, object] | None = None,
    rerank: str = DEFAULT_COLLECTION_RERANK,
    flashrank_model: str | None = None,
    remote_rerank: RemoteRerankConfig | None = None,
    embedding_dimension: int | None = None,
    no_cache: bool = False,
) -> list[RecordResult]:
    """Search *name*, applying metadata filters before anything is scored."""

    clean_query = (query or "").strip()
    if not clean_query:
        raise CollectionError(Messages.ERROR_COLLECTION_QUERY_EMPTY)
    rerank_value = (rerank or DEFAULT_COLLECTION_RERANK).strip().lower()
    if rerank_value not in SUPPORTED_COLLECTION_RERANKERS:
        raise CollectionError(
            Messages.ERROR_COLLECTION_RERANK_UNSUPPORTED.format(
                value=rerank_value,
                allowed=", ".join(SUPPORTED_COLLECTION_RERANKERS),
            )
        )
    if top_k <= 0:
        return []
    # Resolve the contract and embed the query before opening the snapshot:
    # embedding can be a network round trip, and holding a read transaction open
    # across it would pin the WAL for the whole call.
    info = _require_collection(name)
    _verify_contract(info, name=name, provider=provider, model_name=model_name)

    query_matrix = embed_texts_with_cache(
        searcher=searcher,
        model_name=model_name,
        labels=[clean_query],
        no_cache=no_cache,
        embedding_dimension=embedding_dimension,
    )
    if query_matrix.size == 0:
        raise CollectionError(Messages.ERROR_COLLECTION_EMBED_FAILED)
    query_vector = np.asarray(query_matrix[0], dtype=np.float32).ravel()

    # Every read below shares one snapshot. Filtering, vector loading, posting
    # loading, and the final record fetch must observe the same data: a writer
    # moving a record out of the filtered set between two of those steps would
    # otherwise let the result reach a caller whose filter excludes it.
    with collection_store.read_snapshot() as snapshot:
        if snapshot is None:
            raise CollectionError(Messages.ERROR_COLLECTION_NOT_FOUND.format(name=name))
        info = collection_store.get_collection(name, snapshot)
        if info is None:
            raise CollectionError(Messages.ERROR_COLLECTION_NOT_FOUND.format(name=name))
        # Same provider and model can still yield a different width when the
        # configured embedding_dimensions changed since the collection was
        # pinned, so the width is checked against the snapshot's contract.
        _verify_contract(
            info,
            name=name,
            provider=provider,
            model_name=model_name,
            dimension=int(query_vector.shape[0]),
        )

        candidate_ids = collection_store.resolve_filter_ids(info.id, filters, snapshot)
        if not candidate_ids:
            return []
        record_ids, matrix = collection_store.load_vectors(
            info.id, candidate_ids, info.dimension, snapshot
        )
        if not record_ids:
            return []

        # Both sides are L2-normalized, so the dot product is cosine similarity.
        scores = np.asarray(matrix @ query_vector, dtype=np.float32)
        if rerank_value == "hybrid":
            scores = _fuse_hybrid(
                collection_id=info.id,
                query=clean_query,
                record_ids=record_ids,
                dense_scores=scores,
                conn=snapshot,
            )

        order = sorted(range(len(record_ids)), key=lambda idx: (-scores[idx], idx))
        use_candidate_rerank = rerank_value in {"bm25", "flashrank", "remote"}
        candidate_limit = (
            _resolve_rerank_candidates(top_k) if use_candidate_rerank else top_k
        )
        candidate_rows = order[: int(candidate_limit)]
        selected_ids = [record_ids[row] for row in candidate_rows]
        stored = collection_store.fetch_by_ids(info.id, selected_ids, snapshot)

    candidates: list[RecordResult] = []
    for row in candidate_rows:
        record = stored.get(record_ids[row])
        if record is None:
            continue
        candidates.append(
            RecordResult(
                id=record.id,
                text=record.text,
                metadata=record.metadata,
                score=float(scores[row]),
            )
        )
    # Model loading and remote calls can be slow, so rerank only after the read
    # snapshot closes instead of pinning the collection WAL for their duration.
    if rerank_value == "bm25":
        ranking = _rank_documents_bm25(
            clean_query,
            [result.text for result in candidates],
            [result.score for result in candidates],
        )
        if ranking is not None:
            candidates = _apply_ranking(candidates, ranking)
    elif rerank_value == "flashrank":
        candidates = _apply_ranking(
            candidates,
            _rank_documents_flashrank(
                clean_query,
                [result.text for result in candidates],
                flashrank_model,
            ),
        )
    elif rerank_value == "remote":
        candidates = _apply_ranking(
            candidates,
            _rank_documents_remote(
                clean_query,
                [result.text for result in candidates],
                remote_rerank,
            ),
        )
    return candidates[: int(top_k)]


def _fuse_hybrid(
    *,
    collection_id: int,
    query: str,
    record_ids: Sequence[int],
    dense_scores: np.ndarray,
    conn=None,
) -> np.ndarray:
    """Fuse dense similarity with BM25 over the filtered subset only."""

    query_terms = list(dict.fromkeys(bm25.tokenize(query)))[: bm25.MAX_QUERY_TERMS]
    if not query_terms:
        return dense_scores
    doc_count, avg_doc_len = collection_store.load_bm25_stats(
        collection_id, record_ids, conn
    )
    if doc_count <= 0 or avg_doc_len <= 0:
        return dense_scores
    postings = collection_store.load_bm25_postings(
        collection_id, record_ids, query_terms, conn
    )
    if not postings:
        return dense_scores
    scores_by_record = bm25.score_postings(
        query_terms, postings, doc_count, avg_doc_len
    )
    row_by_record = {record_id: row for row, record_id in enumerate(record_ids)}
    scores_by_row = {
        row_by_record[record_id]: value
        for record_id, value in scores_by_record.items()
        if record_id in row_by_record
    }
    dense_order = np.argsort(-dense_scores, kind="stable")
    return bm25.rrf_fuse(dense_order, scores_by_row, len(record_ids))


def delete_records(*, name: str, record_keys: Sequence[str]) -> int:
    """Delete records by id, returning how many existed."""

    info = collection_store.get_collection(name)
    if info is None:
        return 0
    return collection_store.delete_records(info.id, record_keys)


def get_records(*, name: str, record_keys: Sequence[str]) -> list[StoredRecord]:
    """Return stored records by id, skipping ids that are absent."""

    info = collection_store.get_collection(name)
    if info is None:
        return []
    return collection_store.fetch_records(info.id, record_keys)


def count_records(*, name: str) -> int:
    """Return how many records *name* holds, or 0 when it does not exist."""

    info = collection_store.get_collection(name)
    if info is None:
        return 0
    return collection_store.count_records(info.id)


def drop_collection(*, name: str) -> bool:
    """Delete a collection and everything under it."""

    return collection_store.drop_collection(name)


def collection_info(*, name: str) -> CollectionInfo | None:
    """Return the pinned contract for *name*, or ``None``."""

    return collection_store.get_collection(name)


def list_collections() -> list[CollectionInfo]:
    """Return every stored collection."""

    return collection_store.list_collections()
