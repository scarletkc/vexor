"""Batch retrieval contracts across the public API and real local stores."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Connection
from typing import TypedDict
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray

import vexor
from vexor import api, cache, collection_store
from vexor.config import RemoteRerankConfig
from vexor.search import VexorSearcher
from vexor.services import ranking_service, search_service
from vexor.services.query_service import validate_embedding_vectors
from vexor.services.search_service import SearchResponse


class BatchBackend:
    """Return deterministic vectors and retain every embedding request for assertions."""

    def __init__(self) -> None:
        """Start with an empty embedding-call history."""
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Map each query or document to one of three known unit vectors."""
        self.calls.append(list(texts))
        return np.array([
            [1.0, 0.0, 0.0] if "alpha" in text.lower()
            else [0.0, 1.0, 0.0] if "beta" in text.lower()
            else [0.0, 0.0, 1.0]
            for text in texts
        ], dtype=np.float32)


Corpus = tuple[Path, BatchBackend]


class FileOptions(TypedDict, total=False):
    """Keyword arguments shared by the file API test cases."""

    path: Path
    mode: str
    provider: str
    model: str
    use_config: bool
    config: dict[str, object]
    temporary_index: bool
    no_cache: bool


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Corpus]:
    """Provide a real temporary corpus and a deterministic recording backend."""
    backend = BatchBackend()

    def create_backend(searcher: VexorSearcher) -> BatchBackend:
        """Reuse the recording backend across all searcher instances."""
        searcher._device = "batch-test"
        return backend

    monkeypatch.setattr(VexorSearcher, "_create_backend", create_backend)
    monkeypatch.setattr(api, "_RUNTIME_CONFIG", None)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    cache._clear_embedding_memory_cache()
    root = tmp_path / "source"
    root.mkdir()
    for name in ("alpha", "beta", "gamma"):
        (root / f"{name}.py").write_text(
            f"# {name} module for testing retrieval\ndef {name}():\n    return '{name}'\n",
            encoding="utf-8",
        )
    yield root, backend
    cache._clear_embedding_memory_cache()


def file_options(root: Path, rerank: str = "off") -> FileOptions:
    """Build explicit file options without consulting user configuration."""
    return {"path": root, "mode": "full", "provider": "local", "model": "batch-model",
            "use_config": False, "config": {"rerank": rerank}}


def make_collection(client: api.VexorClient) -> api.CollectionHandle:
    """Populate records in two tenants for filtered retrieval assertions."""
    handle = client.collection("records", provider="local", model="batch-model")
    handle.upsert_many([
        {"id": "a", "text": "alpha document", "metadata": {"tenant": "allowed"}},
        {"id": "b", "text": "beta document", "metadata": {"tenant": "allowed"}},
        {"id": "hidden", "text": "alpha secret", "metadata": {"tenant": "other"}},
    ])
    return handle


@pytest.mark.parametrize("entry", ["module", "client", "temporary", "no_cache", "memory"])
@pytest.mark.parametrize("rerank", ["off", "hybrid", "bm25", "flashrank", "remote"])
def test_batch_matches_single_queries_and_embeds_once(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch, entry: str, rerank: str
) -> None:
    """Preserve each ranking strategy while embedding unique queries together."""
    root, backend = corpus

    def rank_documents(
        query: str, documents: Sequence[str], config: str | RemoteRerankConfig | None
    ) -> list[tuple[int, float]]:
        """Keep candidate order fixed without loading or calling an external reranker."""
        return [(i, float(len(documents) - i)) for i in range(len(documents))]

    monkeypatch.setattr(ranking_service, "rank_documents_flashrank", rank_documents)
    monkeypatch.setattr(ranking_service, "rank_documents_remote", rank_documents)
    batch: Callable[..., list[SearchResponse]]
    single: Callable[..., SearchResponse]
    options = file_options(root, rerank)
    queries = [" alpha query ", "beta query", "alpha query"]
    with api.VexorClient(use_config=False) as client:
        if entry == "memory":
            index = client.index_in_memory(**options)
            batch, single = index.search_many, index.search
            options = {}
        else:
            batch = client.search_many if entry == "client" else api.search_many
            single = client.search if entry == "client" else api.search
            if entry in {"temporary", "no_cache"}:
                options["temporary_index" if entry == "temporary" else "no_cache"] = True
            else:
                client.index(**options)
        backend.calls.clear()
        responses = batch(queries, include_content=True, **options)
        query_calls = [call for call in backend.calls if any("query" in t for t in call)]
        assert query_calls == [["alpha query", "beta query"]]
        assert len(responses) == len(queries)
        assert responses[0].results[0].path.name == "alpha.py"
        assert responses[1].results[0].path.name == "beta.py"
        for query, response in zip(queries, responses, strict=True):
            expected = single(query, include_content=True, **options)
            assert response == expected
            assert response.content_budget.used > 0
        assert responses[0] is not responses[2]
        assert responses[0].results[0] is not responses[2].results[0]
        if entry in {"no_cache", "memory"}:
            assert not (cache.CACHE_DIR / "index.db").exists()


def test_persisted_batch_prepares_index_only_once(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load and validate the persisted file corpus only once per batch."""
    root, _ = corpus
    api.index(**file_options(root))
    load = Mock(wraps=cache.load_index_vectors)
    freshness = Mock(wraps=search_service.is_cache_current)
    monkeypatch.setattr(cache, "load_index_vectors", load)
    monkeypatch.setattr(search_service, "is_cache_current", freshness)
    api.search_many(["alpha query", "beta query", "gamma query"], **file_options(root))
    assert load.call_count == 1
    assert freshness.call_count == 1


def test_mixed_query_and_shared_cache_hits_only_embed_misses(corpus: Corpus) -> None:
    """Combine both cache layers and embed only the remaining query."""
    root, backend = corpus
    opts = file_options(root)
    api.index(**opts)
    api.search("alpha query", **opts)
    cache.store_embedding_cache(model="batch-model", embeddings={
        cache.embedding_cache_key("beta query"): np.array([0, 1, 0], dtype=np.float32),
    })
    backend.calls.clear()
    results = api.search_many(["alpha query", "beta query", "gamma query", "beta query"],
                              **opts)
    assert backend.calls == [["gamma query"]]
    assert [r.results[0].path.stem for r in results] == ["alpha", "beta", "gamma", "beta"]
    backend.calls.clear()
    api.search_many(["alpha query", "beta query", "gamma query"], **opts)
    assert backend.calls == []


def test_stale_batch_rebuilds_once_and_filters(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refresh a changed corpus once while retaining file filters."""
    from vexor.services import index_service

    root, _ = corpus
    opts = file_options(root)
    api.index(**opts)
    (root / "beta.py").unlink()
    (root / "alpha.txt").write_text("alpha ignored", encoding="utf-8")
    build = Mock(wraps=index_service.build_index)
    monkeypatch.setattr(index_service, "build_index", build)
    results = api.search_many(["alpha query", "beta query"], **opts)
    assert build.call_count == 1
    assert all(hit.path.name != "beta.py" for r in results for hit in r.results)
    filtered = api.search_many(["alpha query", "beta query"], extensions=".py",
                               exclude_patterns="gamma.py", **opts)
    assert all([hit.path.name for hit in r.results] == ["alpha.py"] for r in filtered)


@pytest.mark.parametrize("entry", ["module", "client", "memory", "collection"])
@pytest.mark.parametrize("queries", ["alpha", b"alpha", ["alpha", " "], ["alpha", None],
                                     ["alpha", 1], {"alpha"}, iter(["alpha"])])
def test_invalid_batch_fails_before_any_provider_call(
    corpus: Corpus, entry: str, queries: object
) -> None:
    """Reject malformed query sequences before generating any embeddings."""
    root, backend = corpus
    with api.VexorClient(use_config=False) as client:
        if entry == "memory":
            target = client.index_in_memory(**file_options(root)).search_many
            opts = {}
        elif entry == "collection":
            target = client.collection("absent").search_many
            opts = {}
        else:
            target = api.search_many if entry == "module" else client.search_many
            opts = file_options(root)
        backend.calls.clear()
        with pytest.raises(api.VexorError, match=r"sequence|position 1"):
            target(queries, **opts)
        assert not backend.calls


def test_empty_batch_does_not_resolve_paths_config_or_models(
    corpus: Corpus, tmp_path: Path
) -> None:
    """Return an empty batch without filesystem or provider side effects."""
    _, backend = corpus
    absent = tmp_path / "absent"
    assert vexor.search_many([], path=absent, data_dir=absent) == []
    with api.VexorClient(data_dir=absent) as client:
        assert client.search_many([], path=absent) == []
        assert client.collection("missing").search_many([]) == []
    assert not absent.exists()
    assert not backend.calls


def test_empty_corpus_returns_one_distinct_empty_response_per_query(corpus: Corpus) -> None:
    """Preserve query positions when no documents can be indexed."""
    root, backend = corpus
    empty = root / "empty"
    empty.mkdir()
    for no_cache in (False, True):
        responses = api.search_many(["alpha", "beta"], no_cache=no_cache,
                                    **file_options(empty))
        assert len(responses) == 2
        assert all(r.index_empty and not r.results for r in responses)
        assert responses[0] is not responses[1]
    assert not backend.calls


@pytest.mark.parametrize("rerank", ["off", "hybrid", "bm25", "flashrank", "remote"])
def test_collection_batch_shares_snapshot_and_reranks_after_close(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch, rerank: str
) -> None:
    """Retrieve all queries in one filtered snapshot and rerank after it closes."""
    _, backend = corpus
    active = False
    snapshots = 0
    original = collection_store.read_snapshot

    @contextmanager
    def snapshot() -> Iterator[Connection | None]:
        """Track the lifetime of the actual SQLite read transaction."""
        nonlocal active, snapshots
        with original() as conn:
            snapshots += 1
            active = True
            try:
                yield conn
            finally:
                active = False

    def rank(
        query: str, documents: Sequence[str], config: str | RemoteRerankConfig | None
    ) -> list[tuple[int, float]]:
        """Reject reranking while a read snapshot remains open."""
        assert not active
        return [(i, float(len(documents) - i)) for i in range(len(documents))]

    monkeypatch.setattr(ranking_service, "rank_documents_flashrank", rank)
    monkeypatch.setattr(ranking_service, "rank_documents_remote", rank)
    with api.VexorClient(use_config=False) as client:
        handle = make_collection(client)
        monkeypatch.setattr(collection_store, "read_snapshot", snapshot)
        load = Mock(wraps=collection_store.load_vectors)
        monkeypatch.setattr(collection_store, "load_vectors", load)
        backend.calls.clear()
        queries = ["alpha query", "beta query", "alpha query"]
        opts = {"filters": {"tenant": "allowed"}, "rerank": rerank, "top_k": 2}
        results = handle.search_many(queries, **opts)
        assert snapshots == 1
        assert load.call_count == 1
        assert backend.calls == [["alpha query", "beta query"]]
        assert all(r.metadata["tenant"] == "allowed" for group in results for r in group)
        assert results[0][0] is not results[2][0]
        assert results == [handle.search(q, **opts) for q in queries]


def test_collection_batch_filters_use_one_snapshot_during_concurrent_write(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep tenant filters consistent when a concurrent writer changes metadata."""
    with api.VexorClient(use_config=False) as client:
        handle = make_collection(client)
        original = collection_store.load_vectors

        def racing_load(
            collection_id: int,
            record_ids: Sequence[int],
            dimension: int,
            conn: Connection | None = None,
        ) -> tuple[list[int], NDArray[np.float32]]:
            """Move a record across tenants after filtering but before vector loading."""
            handle.upsert("a", "alpha document", {"tenant": "other"})
            return original(collection_id, record_ids, dimension, conn)

        monkeypatch.setattr(collection_store, "load_vectors", racing_load)
        results = handle.search_many(["alpha query", "beta query"],
                                     filters={"tenant": "allowed"}, rerank="hybrid")
        assert all(r.metadata["tenant"] == "allowed" for group in results for r in group)
        assert all({r.id for r in group} == {"a", "b"} for group in results)
        assert handle.get("a").metadata["tenant"] == "other"


@pytest.mark.parametrize("bad", [np.empty((0, 3)), np.ones((1, 3)), np.ones((3, 3)),
                                 np.ones(3), np.ones((2, 0)),
                                 [[1, 0, 0], [np.nan, 0, 0]],
                                 [[1, 0, 0], [np.inf, 0, 0]],
                                 [[1, 0], [1]], [["bad"], ["bad"]]])
@pytest.mark.parametrize("entry", ["file", "collection"])
def test_invalid_provider_batch_is_never_retried_or_cached(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch, bad: object, entry: str
) -> None:
    """Reject malformed embedding responses without retrying or caching them."""
    root, backend = corpus
    with api.VexorClient(use_config=False) as client:
        if entry == "file":
            api.index(**file_options(root))
            target = api.search_many
            opts = file_options(root)
        else:
            target = make_collection(client).search_many
            opts = {"rerank": "off"}
        bad_embed = Mock(return_value=bad)
        monkeypatch.setattr(backend, "embed", bad_embed)
        with pytest.raises(ValueError, match="Invalid embeddings"):
            target(["alpha query", "beta query"], **opts)
        assert bad_embed.call_count == 1
        hashes = [cache.embedding_cache_key(q) for q in ["alpha query", "beta query"]]
        assert cache.load_embedding_cache("batch-model", hashes) == {}


def test_batch_propagates_provider_failure_without_partial_results(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surface a provider failure instead of returning partial results."""
    root, backend = corpus
    api.index(**file_options(root))
    monkeypatch.setattr(backend, "embed", Mock(side_effect=RuntimeError("provider unavailable")))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        api.search_many(["alpha query", "beta query"], **file_options(root))


def test_batch_content_budgets_are_independent(corpus: Corpus) -> None:
    """Apply the content limit separately to each query, including duplicates."""
    root, _ = corpus
    responses = api.search_many(["alpha", "beta", "alpha"], include_content=True,
                                content_chars_total=220, **file_options(root))
    for response in responses:
        assert 0 < response.content_budget.used <= 220
        assert response.content_budget.used == sum(len(hit.content or "")
                                                   for hit in response.results)
        assert any(hit.content_unavailable == "budget_exhausted" for hit in response.results)
    assert responses[0].content_budget == responses[2].content_budget
    assert responses[0].content_budget is not responses[2].content_budget


@pytest.mark.parametrize("entry", ["module", "client"])
def test_batch_resolves_configuration_once(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """Resolve configuration once and preserve client cache resources."""
    root, _ = corpus
    load = Mock(return_value=api.Config(provider="local", model="batch-model", rerank="hybrid"))
    monkeypatch.setattr(api, "load_config", load)
    perform = Mock(return_value=[])
    monkeypatch.setattr(api, "perform_search_many", perform)
    with api.VexorClient() as client:
        batch = api.search_many if entry == "module" else client.search_many
        batch([" alpha ", "beta"], path=root, mode="full", top=3,
              extensions=".py", exclude_patterns="gamma.py", include_content=True)
        assert load.call_count == 1
        request, queries = perform.call_args.args
        assert queries == ["alpha", "beta"]
        assert request.provider == "local"
        assert request.model_name == "batch-model"
        assert request.rerank == "hybrid"
        assert request.top_k == 3
        assert request.include_content
        assert request.extensions == (".py",)
        if entry == "client":
            assert request.index_vector_cache is client._index_vector_cache
            assert request.freshness_tracker is client._freshness_tracker


def test_empty_memory_batch_and_empty_collection_filter(corpus: Corpus) -> None:
    """Keep empty batches and empty filtered subsets distinct."""
    root, backend = corpus
    with api.VexorClient(use_config=False) as client:
        index = client.index_in_memory(**file_options(root))
        handle = make_collection(client)
        backend.calls.clear()
        assert index.search_many([]) == []
        assert not backend.calls
        results = handle.search_many(["alpha", "beta"], filters={"tenant": "missing"})
        assert results == [[], []]
        assert results[0] is not results[1]


def test_embedding_normalization_does_not_overflow() -> None:
    """Normalize large finite vectors without overflowing float32 norms."""
    backend = Mock()
    backend.embed.return_value = np.array([[3e38, 3e38]], dtype=np.float32)
    matrix = VexorSearcher(backend=backend).embed_texts(["query"])
    assert np.isfinite(matrix).all()
    assert np.linalg.norm(matrix[0]) == pytest.approx(1)
    assert validate_embedding_vectors(matrix, 1).shape == (1, 2)
