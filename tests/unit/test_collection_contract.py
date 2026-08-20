"""Contract tests for the collection API.

These lock the four guarantees the design rests on: filtering happens before
scoring, unchanged text is never re-embedded, deleting a record takes its
derived rows with it, and the pinned embedding contract cannot drift. A fifth
covers the negative-filter recall rule, which is the failure mode the metadata
contract exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

import vexor.cache as cache
import vexor.collection_store as store
from vexor.collection_store import CollectionError
from vexor.search import VexorSearcher
from vexor.services import collection_service

DIM = 3


class CountingBackend:
    """Deterministic embedding stub that records every call."""

    def __init__(self, vectors: dict[str, list[float]], dim: int = DIM) -> None:
        self._vectors = vectors
        self._dim = dim
        self.calls = 0
        self.embedded: list[str] = []

    def embed(self, texts):
        self.calls += 1
        self.embedded.extend(texts)
        rows = []
        for text in texts:
            vector = self._vectors.get(text)
            if vector is None:
                vector = [1.0] + [0.0] * (self._dim - 1)
            rows.append(vector)
        return np.array(rows, dtype=np.float32)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    cache._clear_embedding_memory_cache()
    yield tmp_path
    cache._clear_embedding_memory_cache()


def _searcher(backend: CountingBackend) -> VexorSearcher:
    return VexorSearcher(model_name="stub-model", backend=backend, provider="local")


def _upsert(name, records, backend, *, model="stub-model", provider="local"):
    return collection_service.upsert_records(
        name=name,
        records=records,
        searcher=_searcher(backend),
        model_name=model,
        provider=provider,
    )


def _search(name, query, backend, **kwargs):
    return collection_service.search_records(
        name=name,
        query=query,
        searcher=_searcher(backend),
        model_name="stub-model",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Strict pre-filtering
# ---------------------------------------------------------------------------


def test_filter_excludes_the_globally_top_ranked_record(isolated_cache):
    """A filtered-out record must not appear even when it wins globally.

    This is the whole reason filtering resolves to a candidate id set before
    scoring. Post-filtering a global top-k would return nothing for a chat whose
    records never reach the global head.
    """

    backend = CountingBackend(
        {
            "deploy broke again": [1.0, 0.0, 0.0],
            "lunch plans": [0.0, 1.0, 0.0],
            "dinner plans": [0.0, 0.9, 0.1],
            "deploy": [1.0, 0.0, 0.0],
        }
    )
    _upsert(
        "chat",
        [
            {"id": "a", "text": "deploy broke again", "metadata": {"chat_id": 1}},
            {"id": "b", "text": "lunch plans", "metadata": {"chat_id": 2}},
            {"id": "c", "text": "dinner plans", "metadata": {"chat_id": 2}},
        ],
        backend,
    )

    unfiltered = _search("chat", "deploy", backend, top_k=3)
    assert unfiltered[0].id == "a", "record a should win without a filter"

    filtered = _search("chat", "deploy", backend, top_k=3, filters={"chat_id": 2})
    assert [result.id for result in filtered] == ["b", "c"] or [
        result.id for result in filtered
    ] == ["c", "b"]
    assert "a" not in {result.id for result in filtered}


def test_filter_on_unknown_key_returns_nothing(isolated_cache):
    backend = CountingBackend({"hello": [1.0, 0.0, 0.0]})
    _upsert("chat", [{"id": "a", "text": "hello", "metadata": {"chat_id": 1}}], backend)
    assert _search("chat", "hello", backend, filters={"nope": 1}) == []


# ---------------------------------------------------------------------------
# 2. Upsert idempotency
# ---------------------------------------------------------------------------


def test_unchanged_text_is_not_re_embedded(isolated_cache):
    backend = CountingBackend({"first": [1.0, 0.0, 0.0], "second": [0.0, 1.0, 0.0]})
    records = [
        {"id": "a", "text": "first", "metadata": {"n": 1}},
        {"id": "b", "text": "second", "metadata": {"n": 2}},
    ]
    first = _upsert("c", records, backend)
    assert first.embedded == 2
    assert first.skipped == 0
    embedded_after_first = list(backend.embedded)

    second = _upsert("c", records, backend)
    assert second.embedded == 0
    assert second.skipped == 2
    assert backend.embedded == embedded_after_first, "no text should be embedded twice"


def test_metadata_only_edit_skips_embedding_but_updates_the_filter(isolated_cache):
    """Changing metadata alone must not cost an embedding, and must take effect.

    The stored vector and postings belong to the text, not the metadata, so a
    pure metadata edit keeps them; dropping them would silently remove the
    record from lexical search.
    """

    backend = CountingBackend({"stable text": [1.0, 0.0, 0.0], "stable": [1.0, 0.0, 0.0]})
    _upsert("c", [{"id": "a", "text": "stable text", "metadata": {"tag": "old"}}], backend)
    calls_after_first = backend.calls

    report = _upsert(
        "c", [{"id": "a", "text": "stable text", "metadata": {"tag": "new"}}], backend
    )
    assert report.embedded == 0
    assert backend.calls == calls_after_first

    assert _search("c", "stable", backend, filters={"tag": "old"}) == []
    refreshed = _search("c", "stable", backend, filters={"tag": "new"})
    assert [result.id for result in refreshed] == ["a"]

    # The vector survived the metadata-only rewrite, so hybrid still ranks it.
    hybrid = _search("c", "stable text", backend, rerank="hybrid")
    assert [result.id for result in hybrid] == ["a"]


def test_changed_text_re_embeds_only_that_record(isolated_cache):
    backend = CountingBackend(
        {
            "first": [1.0, 0.0, 0.0],
            "second": [0.0, 1.0, 0.0],
            "second edited": [0.0, 0.0, 1.0],
        }
    )
    _upsert(
        "c",
        [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}],
        backend,
    )
    backend.embedded.clear()

    report = _upsert(
        "c",
        [{"id": "a", "text": "first"}, {"id": "b", "text": "second edited"}],
        backend,
    )
    assert report.embedded == 1
    assert report.skipped == 1
    assert backend.embedded == ["second edited"]


# ---------------------------------------------------------------------------
# 3. Cascade delete
# ---------------------------------------------------------------------------


def _row_counts(db_path) -> dict[str, int]:
    from vexor.sqlite_util import connect

    conn = connect(db_path, readonly=True)
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            for table in (
                "collection_record",
                "collection_embedding",
                "collection_bm25_doc",
                "collection_bm25_posting",
                "collection_meta",
            )
        }
    finally:
        conn.close()


def test_deleting_a_record_removes_its_derived_rows(isolated_cache):
    backend = CountingBackend({"alpha beta": [1.0, 0.0, 0.0], "gamma": [0.0, 1.0, 0.0]})
    _upsert(
        "c",
        [
            {"id": "a", "text": "alpha beta", "metadata": {"k": "v"}},
            {"id": "b", "text": "gamma", "metadata": {"k": "w"}},
        ],
        backend,
    )
    db_path = store.collections_db_path()
    before = _row_counts(db_path)
    assert before["collection_record"] == 2
    assert before["collection_embedding"] == 2
    assert before["collection_meta"] == 2
    assert before["collection_bm25_posting"] > 0

    assert collection_service.delete_records(name="c", record_keys=["a"]) == 1
    after = _row_counts(db_path)
    assert after["collection_record"] == 1
    assert after["collection_embedding"] == 1
    assert after["collection_bm25_doc"] == 1
    assert after["collection_meta"] == 1
    assert 0 < after["collection_bm25_posting"] < before["collection_bm25_posting"]


def test_dropping_a_collection_removes_every_record(isolated_cache):
    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0]})
    _upsert("c", [{"id": "a", "text": "alpha", "metadata": {"k": "v"}}], backend)
    assert collection_service.drop_collection(name="c") is True
    counts = _row_counts(store.collections_db_path())
    assert all(value == 0 for value in counts.values())


# ---------------------------------------------------------------------------
# 4. The pinned embedding contract cannot drift
# ---------------------------------------------------------------------------


def test_dimension_change_raises_instead_of_mixing_vector_widths(isolated_cache):
    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0]})
    _upsert("c", [{"id": "a", "text": "alpha"}], backend)

    wider = CountingBackend({"beta": [1.0, 0.0, 0.0, 0.0]}, dim=4)
    with pytest.raises(CollectionError) as excinfo:
        _upsert("c", [{"id": "b", "text": "beta"}], wider)
    assert "recreate" in str(excinfo.value).lower()


def test_model_change_raises(isolated_cache):
    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0], "beta": [0.0, 1.0, 0.0]})
    _upsert("c", [{"id": "a", "text": "alpha"}], backend)
    with pytest.raises(CollectionError):
        _upsert("c", [{"id": "b", "text": "beta"}], backend, model="other-model")


def test_provider_change_raises(isolated_cache):
    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0], "beta": [0.0, 1.0, 0.0]})
    _upsert("c", [{"id": "a", "text": "alpha"}], backend)
    with pytest.raises(CollectionError):
        _upsert("c", [{"id": "b", "text": "beta"}], backend, provider="openai")


# ---------------------------------------------------------------------------
# 5. Negative filters must not silently drop unlabeled records
# ---------------------------------------------------------------------------


def test_ne_matches_records_that_lack_the_key(isolated_cache):
    """``ne`` must include records without the key at all.

    Requiring the key to exist would quietly drop every unlabeled record from a
    negative filter — the caller believes they excluded closed tickets and
    actually lost all untagged ones too.
    """

    backend = CountingBackend(
        {
            "closed ticket": [1.0, 0.0, 0.0],
            "open ticket": [0.9, 0.1, 0.0],
            "untagged note": [0.8, 0.2, 0.0],
            "ticket": [1.0, 0.0, 0.0],
        }
    )
    _upsert(
        "c",
        [
            {"id": "closed", "text": "closed ticket", "metadata": {"status": "closed"}},
            {"id": "open", "text": "open ticket", "metadata": {"status": "open"}},
            {"id": "untagged", "text": "untagged note", "metadata": {"other": 1}},
        ],
        backend,
    )

    found = {
        result.id
        for result in _search(
            "c", "ticket", backend, top_k=10, filters={"status": {"ne": "closed"}}
        )
    }
    assert found == {"open", "untagged"}


def test_nin_matches_records_that_lack_the_key(isolated_cache):
    backend = CountingBackend(
        {
            "a text": [1.0, 0.0, 0.0],
            "b text": [0.9, 0.1, 0.0],
            "c text": [0.8, 0.2, 0.0],
            "text": [1.0, 0.0, 0.0],
        }
    )
    _upsert(
        "c",
        [
            {"id": "a", "text": "a text", "metadata": {"kind": "spam"}},
            {"id": "b", "text": "b text", "metadata": {"kind": "ham"}},
            {"id": "c", "text": "c text", "metadata": {}},
        ],
        backend,
    )
    found = {
        result.id
        for result in _search(
            "c", "text", backend, top_k=10, filters={"kind": {"nin": ["spam"]}}
        )
    }
    assert found == {"b", "c"}


def test_exists_is_the_way_to_require_a_key(isolated_cache):
    backend = CountingBackend(
        {"tagged": [1.0, 0.0, 0.0], "plain": [0.9, 0.1, 0.0], "q": [1.0, 0.0, 0.0]}
    )
    _upsert(
        "c",
        [
            {"id": "tagged", "text": "tagged", "metadata": {"status": "open"}},
            {"id": "plain", "text": "plain", "metadata": {}},
        ],
        backend,
    )
    present = {
        result.id
        for result in _search("c", "q", backend, filters={"status": {"exists": True}})
    }
    assert present == {"tagged"}
    absent = {
        result.id
        for result in _search("c", "q", backend, filters={"status": {"exists": False}})
    }
    assert absent == {"plain"}
