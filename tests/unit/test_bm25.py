from __future__ import annotations

import math

import numpy as np

from vexor import bm25


def test_tokenize_identifiers_cjk_and_punctuation() -> None:
    identifier_tokens = bm25.tokenize("_apply_bm25_rerank")
    assert {"apply", "bm25", "rerank"}.issubset(identifier_tokens)
    assert "_apply_bm25_rerank" in identifier_tokens
    assert bm25.tokenize("中文测试")
    assert bm25.tokenize("!!!") == []


def test_tokenize_plain_words_are_not_double_counted() -> None:
    tokens = bm25.tokenize("plain config words")

    assert tokens.count("plain") == 1
    assert tokens.count("config") == 1
    assert tokens.count("words") == 1


def test_term_frequencies() -> None:
    assert bm25.term_frequencies(["alpha", "beta", "alpha"]) == {
        "alpha": 2,
        "beta": 1,
    }


def test_score_postings_matches_hand_computed_values() -> None:
    postings = {
        "alpha": [(0, 2, 3), (1, 1, 2)],
        "beta": [(1, 1, 2)],
    }
    scores = bm25.score_postings(["alpha", "beta"], postings, 3, 2.0)

    alpha_idf = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
    beta_idf = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
    expected_zero = alpha_idf * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / 2))
    expected_one = alpha_idf + beta_idf
    assert math.isclose(scores[0], expected_zero, rel_tol=1e-12)
    assert math.isclose(scores[1], expected_one, rel_tol=1e-12)
    assert bm25.score_postings(["alpha"], postings, 3, 0.0) == {}


def test_rrf_fuse_normalizes_and_keeps_dense_only_rows() -> None:
    fused = bm25.rrf_fuse([0, 1, 2], {0: 3.0, 2: 2.0}, 3)

    expected_dense_only = bm25.RRF_DENSE_WEIGHT * (bm25.RRF_K + 1) / (
        bm25.RRF_K + 2
    )
    expected_row_two = (
        bm25.RRF_DENSE_WEIGHT * (bm25.RRF_K + 1) / (bm25.RRF_K + 3)
        + bm25.RRF_BM25_WEIGHT * (bm25.RRF_K + 1) / (bm25.RRF_K + 2)
    )
    assert fused.dtype == np.float32
    assert fused[0] == 1.0
    assert math.isclose(fused[1], expected_dense_only, rel_tol=1e-6)
    assert math.isclose(fused[2], expected_row_two, rel_tol=1e-6)
    assert fused[2] > fused[1] > 0
    assert list(np.argsort(-fused, kind="stable")) == [0, 2, 1]


def test_rrf_fuse_dense_weight_prevents_bm25_backed_rows_displacing_dense_top() -> None:
    dense_order = list(range(52))
    bm25_rows = range(47, 52)
    bm25_scores = {row: float(52 - row) for row in bm25_rows}

    fused = bm25.rrf_fuse(dense_order, bm25_scores, 52)

    dense_top_expected = bm25.RRF_DENSE_WEIGHT
    backed_expected = {
        row: (
            bm25.RRF_DENSE_WEIGHT
            * (bm25.RRF_K + 1)
            / (bm25.RRF_K + row + 1)
            + bm25.RRF_BM25_WEIGHT
            * (bm25.RRF_K + 1)
            / (bm25.RRF_K + bm25_rank)
        )
        for bm25_rank, row in enumerate(bm25_rows, start=1)
    }
    equal_weight_scores = {
        row: (
            0.5 * (bm25.RRF_K + 1) / (bm25.RRF_K + row + 1)
            + 0.5 * (bm25.RRF_K + 1) / (bm25.RRF_K + bm25_rank)
        )
        for bm25_rank, row in enumerate(bm25_rows, start=1)
    }

    assert math.isclose(fused[0], dense_top_expected, rel_tol=1e-6)
    for row, expected in backed_expected.items():
        assert math.isclose(fused[row], expected, rel_tol=1e-6)
        assert fused[row] < fused[0]
    assert all(score > 0.5 for score in equal_weight_scores.values())
    assert int(np.argmax(fused)) == 0
