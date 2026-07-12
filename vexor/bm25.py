"""Shared BM25 tokenization, scoring, and rank-fusion helpers."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
import re
from typing import Mapping, Sequence

import numpy as np

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
# Intentionally mirrors the legacy reranker's _FUSION_SEMANTIC_WEIGHT split.
RRF_DENSE_WEIGHT = 0.7
RRF_BM25_WEIGHT = 0.3
MAX_QUERY_TERMS = 32

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@lru_cache(maxsize=1)
def _get_bm25_tokenizer():
    try:
        from tokenizers.pre_tokenizers import BertPreTokenizer
    except Exception:
        return None
    return BertPreTokenizer()


def tokenize(text: str) -> list[str]:
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
    sub_tokens = set(normalized)
    normalized.extend(
        whole_token
        for whole_token in _TOKEN_RE.findall(text.lower())
        if whole_token not in sub_tokens
    )
    return normalized


def build_document(rel_path: str, label: str) -> str:
    """Build the canonical lexical document for an indexed chunk."""

    return f"{rel_path} {label}"


def term_frequencies(tokens: Sequence[str]) -> dict[str, int]:
    return dict(Counter(tokens))


def score_postings(
    query_terms: Sequence[str],
    postings: Mapping[str, Sequence[tuple[int, int, int]]],
    doc_count: int,
    avg_doc_len: float,
) -> dict[int, float]:
    """Score matching posting lists with non-negative-idf Okapi BM25."""

    if doc_count <= 0 or avg_doc_len <= 0:
        return {}
    scores: dict[int, float] = {}
    for term in query_terms:
        term_postings = postings.get(term, ())
        if not term_postings:
            continue
        df = len(term_postings)
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
        for chunk_id, tf, doc_len in term_postings:
            denominator = tf + BM25_K1 * (
                1.0 - BM25_B + BM25_B * doc_len / avg_doc_len
            )
            if denominator <= 0:
                continue
            contribution = idf * tf * (BM25_K1 + 1.0) / denominator
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
    return scores


def rrf_fuse(
    dense_order: Sequence[int],
    bm25_scores_by_row: Mapping[int, float],
    total_rows: int,
    *,
    k: int = RRF_K,
) -> np.ndarray:
    """Fuse dense and BM25 rankings with normalized reciprocal rank fusion."""

    fused = np.zeros(total_rows, dtype=np.float32)
    for rank, row in enumerate(dense_order, start=1):
        if 0 <= row < total_rows:
            fused[row] += RRF_DENSE_WEIGHT * (k + 1.0) / (k + rank)
    bm25_order = sorted(
        (
            (row, score)
            for row, score in bm25_scores_by_row.items()
            if score > 0 and 0 <= row < total_rows
        ),
        key=lambda item: (-item[1], item[0]),
    )
    for rank, (row, _score) in enumerate(bm25_order, start=1):
        fused[row] += RRF_BM25_WEIGHT * (k + 1.0) / (k + rank)
    return fused
