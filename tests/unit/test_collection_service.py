"""Unit coverage for collection orchestration and the public handle."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from test_collection_contract import CountingBackend

import vexor
import vexor.api as api_module
import vexor.cache as cache
from vexor.collection_store import CollectionError
from vexor.config import RemoteRerankConfig
from vexor.search import VexorSearcher
from vexor.services import collection_service


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    cache._clear_embedding_memory_cache()
    yield tmp_path
    cache._clear_embedding_memory_cache()


def _searcher(backend: CountingBackend) -> VexorSearcher:
    return VexorSearcher(model_name="stub-model", backend=backend, provider="local")


def _upsert(
    name: str,
    records,
    backend: CountingBackend,
):
    return collection_service.upsert_records(
        name=name,
        records=records,
        searcher=_searcher(backend),
        model_name="stub-model",
        provider="local",
    )


def _search(
    name: str,
    query: str,
    backend: CountingBackend,
    **kwargs,
):
    return collection_service.search_records(
        name=name,
        query=query,
        searcher=_searcher(backend),
        model_name=kwargs.pop("model_name", "stub-model"),
        provider=kwargs.pop("provider", "local"),
        **kwargs,
    )


def test_record_that_is_not_a_mapping_raises_collection_error(isolated_cache):
    backend = CountingBackend({})
    with pytest.raises(CollectionError):
        _upsert("records", [object()], backend)


@pytest.mark.parametrize("record", [{"text": "body"}, {"id": "", "text": "body"}])
def test_empty_or_missing_record_id_raises_collection_error(isolated_cache, record):
    backend = CountingBackend({})
    with pytest.raises(CollectionError):
        _upsert("records", [record], backend)


@pytest.mark.parametrize(
    "record",
    [
        {"id": "one"},
        {"id": "one", "text": "   \t"},
    ],
)
def test_empty_or_whitespace_text_raises_collection_error(isolated_cache, record):
    backend = CountingBackend({})
    with pytest.raises(CollectionError):
        _upsert("records", [record], backend)


def test_duplicate_ids_in_one_call_raise_collection_error(isolated_cache):
    backend = CountingBackend({})
    with pytest.raises(CollectionError):
        _upsert(
            "records",
            [
                {"id": "same", "text": "first"},
                {"id": " same ", "text": "second"},
            ],
            backend,
        )


def test_empty_search_query_raises_collection_error(isolated_cache):
    backend = CountingBackend({})
    with pytest.raises(CollectionError):
        _search("missing", " \t ", backend)


def test_unsupported_rerank_value_raises_collection_error(isolated_cache):
    backend = CountingBackend({})
    with pytest.raises(CollectionError, match="bogus"):
        _search("missing", "query", backend, rerank="bogus")


def test_bm25_reranks_dense_candidates_from_record_text(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CountingBackend(
        {
            "first full record": [1.0, 0.0, 0.0],
            "second full record": [0.8, 0.2, 0.0],
            "query": [1.0, 0.0, 0.0],
        }
    )
    _upsert(
        "records",
        [
            {"id": "first", "text": "first full record"},
            {"id": "second", "text": "second full record"},
        ],
        backend,
    )
    captured: dict[str, object] = {}

    def fake_rank(
        query: str,
        documents: Sequence[str],
        base_scores: Sequence[float],
    ) -> list[tuple[int, float]]:
        captured.update(query=query, documents=documents, base_scores=base_scores)
        return [(1, 0.95), (0, 0.25)]

    monkeypatch.setattr(collection_service, "_rank_documents_bm25", fake_rank)

    results = _search("records", "query", backend, top_k=1, rerank="bm25")

    assert [result.id for result in results] == ["second"]
    assert results[0].score == pytest.approx(0.95)
    assert captured["query"] == "query"
    assert captured["documents"] == ["first full record", "second full record"]
    assert captured["base_scores"] == pytest.approx([1.0, 0.9701425])


def test_flashrank_uses_configured_model_and_record_text(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CountingBackend(
        {
            "first full record": [1.0, 0.0, 0.0],
            "second full record": [0.8, 0.2, 0.0],
            "query": [1.0, 0.0, 0.0],
        }
    )
    _upsert(
        "records",
        [
            {"id": "first", "text": "first full record"},
            {"id": "second", "text": "second full record"},
        ],
        backend,
    )
    captured: dict[str, object] = {}

    def fake_rank(
        query: str,
        documents: Sequence[str],
        model_name: str | None,
    ) -> list[tuple[int, float]]:
        captured.update(query=query, documents=documents, model_name=model_name)
        return [(1, 0.9), (0, 0.4)]

    monkeypatch.setattr(collection_service, "_rank_documents_flashrank", fake_rank)

    results = _search(
        "records",
        "query",
        backend,
        top_k=1,
        rerank="flashrank",
        flashrank_model="ranker-model",
    )

    assert [result.id for result in results] == ["second"]
    assert captured == {
        "query": "query",
        "documents": ["first full record", "second full record"],
        "model_name": "ranker-model",
    }


def test_remote_rerank_only_receives_filtered_candidates(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CountingBackend(
        {
            "allowed record": [1.0, 0.0, 0.0],
            "also allowed": [0.8, 0.2, 0.0],
            "excluded record": [0.9, 0.1, 0.0],
            "query": [1.0, 0.0, 0.0],
        }
    )
    _upsert(
        "records",
        [
            {"id": "a", "text": "allowed record", "metadata": {"tenant": "a"}},
            {"id": "b", "text": "also allowed", "metadata": {"tenant": "a"}},
            {
                "id": "excluded",
                "text": "excluded record",
                "metadata": {"tenant": "b"},
            },
        ],
        backend,
    )
    remote = RemoteRerankConfig(
        base_url="https://rerank.example.test/v1/rerank",
        api_key="secret",
        model="rerank-model",
    )
    captured: dict[str, object] = {}

    def fake_rank(
        query: str,
        documents: Sequence[str],
        config: RemoteRerankConfig | None,
    ) -> list[tuple[int, float | None]]:
        captured.update(query=query, documents=documents, config=config)
        return [(1, 0.85), (0, None)]

    monkeypatch.setattr(collection_service, "_rank_documents_remote", fake_rank)

    results = _search(
        "records",
        "query",
        backend,
        top_k=2,
        filters={"tenant": "a"},
        rerank="remote",
        remote_rerank=remote,
    )

    assert [result.id for result in results] == ["b", "a"]
    assert captured == {
        "query": "query",
        "documents": ["allowed record", "also allowed"],
        "config": remote,
    }


def test_non_positive_top_k_returns_empty_without_embedding(isolated_cache):
    backend = CountingBackend({})

    assert _search("missing", "query", backend, top_k=0) == []
    assert _search("missing", "query", backend, top_k=-1) == []
    assert backend.calls == 0


def test_searching_missing_collection_raises_collection_error(isolated_cache):
    backend = CountingBackend({})
    with pytest.raises(CollectionError):
        _search("missing", "query", backend)


def test_missing_collection_read_and_delete_operations_are_empty(isolated_cache):
    assert collection_service.get_records(name="missing", record_keys=["one"]) == []
    assert collection_service.count_records(name="missing") == 0
    assert collection_service.delete_records(name="missing", record_keys=["one"]) == 0


def test_list_collections_and_collection_info(isolated_cache):
    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0], "beta": [0.0, 1.0, 0.0]})
    _upsert("zeta", [{"id": "z", "text": "alpha"}], backend)
    _upsert("alpha", [{"id": "a", "text": "beta"}], backend)

    assert [info.name for info in collection_service.list_collections()] == ["alpha", "zeta"]
    info = collection_service.collection_info(name="zeta")
    assert info is not None
    assert (info.provider, info.model, info.dimension) == ("local", "stub-model", 3)
    assert collection_service.collection_info(name="missing") is None


def _inject_backend(monkeypatch: pytest.MonkeyPatch, backend: CountingBackend) -> None:
    searcher_type = VexorSearcher

    def make_searcher(**_kwargs) -> VexorSearcher:
        return searcher_type(model_name="stub-model", backend=backend, provider="local")

    monkeypatch.setattr(api_module, "VexorSearcher", make_searcher)


def test_collection_handle_exercises_full_lifecycle(tmp_path: Path, monkeypatch):
    backend = CountingBackend(
        {
            "alpha": [1.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0],
            "gamma": [0.0, 0.0, 1.0],
            "find alpha": [1.0, 0.0, 0.0],
        }
    )
    _inject_backend(monkeypatch, backend)
    cache_dir = tmp_path / "api-cache"

    with api_module.VexorClient(cache_dir=cache_dir, use_config=False) as client:
        handle = client.collection(
            "api-records",
            provider="local",
            model="stub-model",
            no_cache=True,
        )
        first = handle.upsert("a", "alpha", {"kind": "first"})
        rest = handle.upsert_many(
            [
                {"id": "b", "text": "beta", "metadata": {"kind": "second"}},
                {"id": "c", "text": "gamma", "metadata": {"kind": "third"}},
            ]
        )
        assert (first.written, first.embedded, first.skipped) == (1, 1, 0)
        assert (rest.written, rest.embedded, rest.skipped) == (2, 2, 0)

        results = handle.search("find alpha", top_k=3)
        assert [result.id for result in results] == ["a", "b", "c"]
        assert handle.get("a") is not None
        assert handle.get("missing") is None
        assert [record.id for record in handle.get_many(["c", "missing", "a"])] == [
            "c",
            "a",
        ]
        assert handle.count() == 3
        info = handle.info()
        assert info is not None
        assert (info.name, info.provider, info.model, info.dimension) == (
            "api-records",
            "local",
            "stub-model",
            3,
        )

        assert handle.delete("b") == 1
        assert handle.delete_many(["a", "missing"]) == 1
        assert handle.count() == 1
        assert handle.drop() is True
        assert handle.info() is None

    assert (cache_dir / "collections.db").is_file()


def test_collection_handle_translates_contract_errors_to_vexor_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = CountingBackend({})
    _inject_backend(monkeypatch, backend)

    with api_module.VexorClient(cache_dir=tmp_path / "api-cache", use_config=False) as client:
        handle = client.collection(
            "api-records",
            provider="local",
            model="stub-model",
            no_cache=True,
        )
        with pytest.raises(vexor.VexorError) as excinfo:
            handle.upsert("", "body")

    assert not isinstance(excinfo.value, CollectionError)


def test_collection_handle_uses_configured_reranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_search_records(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(collection_service, "search_records", fake_search_records)
    config = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "embedding-key",
        "rerank": "remote",
        "remote_rerank": {
            "base_url": "https://rerank.example.test/v1",
            "api_key": "remote-key",
            "model": "rerank-model",
        },
    }

    with api_module.VexorClient(cache_dir=tmp_path / "api-cache") as client:
        client.collection("api-records", config=config).search("query")

    assert captured["rerank"] == "remote"
    remote = captured["remote_rerank"]
    assert isinstance(remote, RemoteRerankConfig)
    assert remote.base_url == "https://rerank.example.test/v1/rerank"
    assert remote.api_key == "remote-key"
    assert remote.model == "rerank-model"


def test_collection_handle_per_call_reranker_overrides_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_search_records(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(collection_service, "search_records", fake_search_records)
    config = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "embedding-key",
        "rerank": "remote",
        "remote_rerank": {
            "base_url": "https://configured.example.test/v1",
            "api_key": "configured-key",
            "model": "configured-model",
        },
    }

    with api_module.VexorClient(cache_dir=tmp_path / "api-cache") as client:
        handle = client.collection("api-records", config=config)
        handle.search(
            "query",
            rerank="flashrank",
            flashrank_model="ranker-model",
        )

        assert captured["rerank"] == "flashrank"
        assert captured["flashrank_model"] == "ranker-model"

        per_call_remote = RemoteRerankConfig(
            base_url="https://override.example.test/v1/rerank",
            api_key="override-key",
            model="override-model",
        )
        handle.search(
            "query",
            rerank="remote",
            remote_rerank=per_call_remote,
        )

    assert captured["rerank"] == "remote"
    assert captured["remote_rerank"] is per_call_remote
