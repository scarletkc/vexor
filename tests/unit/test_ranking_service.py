from types import SimpleNamespace

import pytest

from vexor import bm25
from vexor.services import ranking_service


def test_candidate_and_persisted_tokenizers_keep_identifier_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace(pre_tokenize_str=lambda _text: [
        ("snake", (0, 5)), ("_", (5, 6)), ("case", (6, 10)),
    ])
    monkeypatch.setattr(bm25, "_get_bm25_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(ranking_service, "_get_bm25_tokenizer", lambda: tokenizer)
    assert ranking_service._bm25_tokenize("snake_case") == ["snake", "case"]
    assert bm25.tokenize("snake_case") == ["snake", "case", "snake_case"]


def test_unknown_candidate_reranker_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ranking_service.rerank_candidates("query", [], [], rerank="unknown")
