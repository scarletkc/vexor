from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib import error as urlerror

import numpy as np
import pytest

from vexor.config import RemoteRerankConfig
from vexor.search import SearchResult
from vexor.services import search_service


def _result(name: str, score: float, preview: str | None = None) -> SearchResult:
    return SearchResult(path=Path(name), score=score, preview=preview)


def test_bm25_helpers_fallbacks(monkeypatch):
    search_service._get_bm25_tokenizer.cache_clear()
    monkeypatch.setitem(sys.modules, "tokenizers.pre_tokenizers", None)
    monkeypatch.setattr(search_service, "_get_bm25_tokenizer", lambda: None)

    assert search_service._bm25_tokenize("Alpha_beta 中文!") == ["alpha_beta"]
    assert search_service._normalize_by_max([]) == []
    assert search_service._normalize_by_max([0.0, -1.0]) == [0.0, 0.0]
    assert search_service._top_indices(np.array([0.1, 0.9]), 0) == []
    assert search_service._top_indices(np.array([0.1, 0.9]), 5) == [1, 0]
    assert search_service._bm25_scores(["alpha"], []) == []
    assert search_service._apply_bm25_rerank("", [_result("a.txt", 0.1)]) == [
        _result("a.txt", 0.1)
    ]


def test_flashrank_rerank_import_error_and_success(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError):
        search_service._apply_flashrank_rerank("q", [_result("a.txt", 0.1)], None)

    flashrank_module = ModuleType("flashrank")

    class RerankRequest:
        def __init__(self, *, query, passages):
            self.query = query
            self.passages = passages

    class Ranker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def rerank(self, request):
            assert request.query == "alpha"
            return [
                {"id": None, "score": 0.0},
                {"id": 99, "score": 0.0},
                {"id": 1, "score": 0.95},
            ]

    flashrank_module.RerankRequest = RerankRequest
    flashrank_module.Ranker = Ranker
    monkeypatch.setitem(sys.modules, "flashrank", flashrank_module)
    monkeypatch.setattr("vexor.config.flashrank_cache_dir", lambda: tmp_path)
    search_service._get_flashranker.cache_clear()

    results = [_result("a.txt", 0.2), _result("b.txt", 0.1)]
    ordered = search_service._apply_flashrank_rerank("alpha", results, "ranker-model")

    assert [item.path.name for item in ordered] == ["b.txt", "a.txt"]
    assert ordered[0].score == 0.95


def test_remote_rerank_config_validation(monkeypatch):
    with pytest.raises(RuntimeError):
        search_service._resolve_remote_rerank_config(None)

    with pytest.raises(RuntimeError):
        search_service._resolve_remote_rerank_config(
            RemoteRerankConfig(base_url="", api_key=None, model="")
        )

    monkeypatch.setenv("VEXOR_REMOTE_RERANK_API_KEY", "env-key")
    resolved = search_service._resolve_remote_rerank_config(
        RemoteRerankConfig(
            base_url="https://rerank.example.com",
            api_key=None,
            model="model-x",
        )
    )

    assert resolved.base_url == "https://rerank.example.com/rerank"
    assert resolved.api_key == "env-key"


class DummyResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_remote_rerank_request_success_and_errors(monkeypatch):
    config = RemoteRerankConfig(
        base_url="https://rerank.example.com/rerank",
        api_key="secret",
        model="model-x",
    )
    captured = {}

    def ok_urlopen(request):
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse('{"results":[{"index":0,"score":1.0}]}')

    monkeypatch.setattr(search_service.urlrequest, "urlopen", ok_urlopen)
    payload = search_service._remote_rerank_request(
        config=config,
        query="alpha",
        documents=["doc"],
    )
    assert payload["results"][0]["index"] == 0
    assert captured["auth"] == "Bearer secret"
    assert captured["body"]["documents"] == ["doc"]

    def http_error(_request):
        raise urlerror.HTTPError(
            url=config.base_url,
            code=500,
            msg="bad",
            hdrs=None,
            fp=io.BytesIO(b"server exploded"),
        )

    monkeypatch.setattr(search_service.urlrequest, "urlopen", http_error)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        search_service._remote_rerank_request(config=config, query="q", documents=["d"])

    monkeypatch.setattr(
        search_service.urlrequest,
        "urlopen",
        lambda _request: (_ for _ in ()).throw(urlerror.URLError("offline")),
    )
    with pytest.raises(RuntimeError, match="offline"):
        search_service._remote_rerank_request(config=config, query="q", documents=["d"])

    monkeypatch.setattr(search_service.urlrequest, "urlopen", lambda _request: DummyResponse("{"))
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        search_service._remote_rerank_request(config=config, query="q", documents=["d"])


def test_remote_rerank_item_parsing_and_apply(monkeypatch):
    payload = {
        "data": [
            {"index": "1", "score": "0.8"},
            {"index": 0, "relevance_score": "bad"},
            {"index": None, "score": 1.0},
            {"index": "nope", "score": 1.0},
            "bad",
        ]
    }
    assert search_service._extract_remote_rerank_items(payload) == [(1, 0.8), (0, None)]
    assert search_service._extract_remote_rerank_items([]) == []
    assert search_service._extract_remote_rerank_items({"results": "bad"}) == []

    monkeypatch.setattr(
        search_service,
        "_remote_rerank_request",
        lambda **_kwargs: {
            "results": [
                {"index": -1, "score": 9.0},
                {"index": 1, "score": 0.9},
                {"index": 1, "score": 0.1},
            ]
        },
    )
    results = [_result("a.txt", 0.2), _result("b.txt", 0.1)]
    ordered = search_service._apply_remote_rerank(
        "alpha",
        results,
        RemoteRerankConfig(
            base_url="https://rerank.example.com/rerank",
            api_key="secret",
            model="model-x",
        ),
    )
    assert [item.path.name for item in ordered] == ["b.txt", "a.txt"]
    assert ordered[0].score == 0.9

    monkeypatch.setattr(
        search_service,
        "_remote_rerank_request",
        lambda **_kwargs: {"results": []},
    )
    assert search_service._apply_remote_rerank(
        "alpha",
        results,
        RemoteRerankConfig(
            base_url="https://rerank.example.com/rerank",
            api_key="secret",
            model="model-x",
        ),
    ) == results


def test_filter_helpers_cover_empty_and_directory_cases(tmp_path):
    paths = [tmp_path / "pkg" / "a.py", tmp_path / "pkg" / "nested" / "b.py"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metadata = {
        "files": [
            {"path": "pkg/a.py"},
            {"path": "pkg/nested/b.py"},
        ],
        "chunks": [
            {"path": "pkg/a.py"},
            {"path": "pkg/nested/b.py"},
        ],
        "chunk_ids": [10, 11],
    }

    same_paths, same_vectors, same_meta = search_service._filter_index_by_extensions(
        paths,
        vectors,
        metadata,
        (),
    )
    assert same_paths == paths
    assert same_vectors is vectors
    assert same_meta is metadata

    empty_paths, empty_vectors, empty_meta = search_service._filter_index_by_extensions(
        paths,
        vectors,
        metadata,
        (".md",),
    )
    assert empty_paths == []
    assert empty_vectors.shape == (0, 2)
    assert empty_meta["chunk_ids"] == []

    spec = SimpleNamespace(
        check_file=lambda path: SimpleNamespace(include=path.endswith("nested/b.py"))
    )
    monkeypatch_like_spec = spec
    excluded_paths, _, excluded_meta = search_service._filter_index_by_exclude_patterns(
        paths,
        vectors,
        metadata,
        tmp_path,
        monkeypatch_like_spec,
    )
    assert [path.name for path in excluded_paths] == ["a.py"]
    assert excluded_meta["chunk_ids"] == [10]

    out_paths, out_vectors, out_meta = search_service._filter_index_by_directory(
        paths,
        vectors,
        metadata,
        tmp_path.parent / "elsewhere",
        tmp_path,
        recursive=True,
    )
    assert out_paths == paths
    assert out_vectors is vectors
    assert out_meta is metadata

    empty_dir_paths, empty_dir_vectors, empty_dir_meta = search_service._filter_index_by_directory(
        paths,
        vectors,
        metadata,
        tmp_path / "pkg" / "missing",
        tmp_path,
        recursive=True,
    )
    assert empty_dir_paths == []
    assert empty_dir_vectors.shape == (0, 2)
    assert empty_dir_meta["root"] == str(tmp_path / "pkg" / "missing")

    filtered_snapshot = search_service._filter_file_snapshot_by_directory(
        metadata["files"],
        Path("pkg"),
        recursive=False,
    )
    assert filtered_snapshot == [{"path": "a.py"}]
