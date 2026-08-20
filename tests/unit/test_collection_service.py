"""Unit coverage for collection orchestration and the public handle."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_collection_contract import CountingBackend

import vexor
import vexor.api as api_module
import vexor.cache as cache
from vexor.collection_store import CollectionError
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
    with pytest.raises(CollectionError):
        _search("missing", "query", backend, rerank="flashrank")


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
