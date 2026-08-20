"""Contract tests for the collection API.

These lock the four guarantees the design rests on: filtering happens before
scoring, unchanged text is never re-embedded, deleting a record takes its
derived rows with it, and the pinned embedding contract cannot drift. A fifth
covers the negative-filter recall rule, which is the failure mode the metadata
contract exists to prevent.
"""

from __future__ import annotations

import sqlite3

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
        model_name=kwargs.pop("model_name", "stub-model"),
        provider=kwargs.pop("provider", "local"),
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


# ---------------------------------------------------------------------------
# 6. Teardown must not create state, and must not fail on a held file handle
# ---------------------------------------------------------------------------


def test_dropping_a_missing_collection_does_not_create_the_database(isolated_cache):
    assert collection_service.drop_collection(name="never-existed") is False
    assert not store.collections_db_path().exists()


def test_clear_all_collections_succeeds_while_a_connection_is_open(isolated_cache):
    """Clearing must work even when a reader still holds the file.

    On Windows an open handle makes ``unlink`` raise, so emptying the tables has
    to happen first: the caller asked for the records to be gone, and a lock
    somewhere else in the process must not turn that into an error.
    """

    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0]})
    _upsert("c", [{"id": "a", "text": "alpha"}], backend)
    held = store._open_readonly()
    try:
        store.clear_all_collections()
        assert store.list_collections() == []
    finally:
        if held is not None:
            held.close()

    # The store stays usable after a clear, whether or not the file survived.
    _upsert("c2", [{"id": "a", "text": "alpha"}], backend)
    assert [info.name for info in store.list_collections()] == ["c2"]


# ---------------------------------------------------------------------------
# 7. The pinned contract guards reads, not just writes
# ---------------------------------------------------------------------------


def test_searching_with_a_different_model_raises_even_at_the_same_dimension(
    isolated_cache,
):
    """A same-width model must not be allowed to query someone else's vectors.

    Two models can share a vector width while embedding into unrelated spaces.
    A dimension check alone lets the dot product succeed and return a ranking
    that means nothing, which is worse than an error because nothing surfaces.
    """

    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0], "query": [1.0, 0.0, 0.0]})
    _upsert("c", [{"id": "a", "text": "alpha"}], backend)

    # Same dimension, different model: must raise rather than score.
    with pytest.raises(CollectionError) as excinfo:
        _search("c", "query", backend, model_name="other-model")
    assert "recreate" in str(excinfo.value).lower()

    # Same dimension, different provider: same rule.
    with pytest.raises(CollectionError):
        _search("c", "query", backend, provider="openai")

    # The matching contract still works.
    assert [hit.id for hit in _search("c", "query", backend)] == ["a"]


# ---------------------------------------------------------------------------
# 8. Concurrent writers must queue, not fail
# ---------------------------------------------------------------------------


def test_concurrent_first_writes_do_not_deadlock(isolated_cache):
    """Racing writers must serialize instead of raising "database is locked".

    A DEFERRED transaction takes a read lock on its first SELECT and only tries
    to upgrade on the first write. Two writers holding read locks can never both
    upgrade, so SQLite returns BUSY immediately and busy_timeout does not help --
    waiting cannot resolve a deadlock. Writes therefore open with BEGIN
    IMMEDIATE, which busy_timeout does cover.
    """

    import threading

    errors: list[str] = []
    created: list[int] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait()
            info = store.ensure_collection(
                "race", provider="local", model="m", dimension=3
            )
            created.append(info.id)
        # Broad on purpose: the failure mode under test is any raise at all.
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(set(created)) == 1, "all writers must converge on one collection"


def test_concurrent_upserts_all_land(isolated_cache):
    import threading

    backend = CountingBackend({})
    errors: list[str] = []
    barrier = threading.Barrier(8)

    def worker(index: int) -> None:
        try:
            barrier.wait()
            _upsert("race", [{"id": f"r{index}", "text": f"text {index}"}], backend)
        # Broad on purpose: the failure mode under test is any raise at all.
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert collection_service.count_records(name="race") == 8


# ---------------------------------------------------------------------------
# 9. Findings from the review pass, each locked against regression
# ---------------------------------------------------------------------------


def test_a_falsey_non_mapping_filter_is_an_error_not_no_filter(isolated_cache):
    """`filters=[]` must not silently widen the search to the whole collection.

    Only ``None`` means "no filter". A caller who decoded a tenant filter into
    the wrong shape has to hear about it, not receive every tenant's records.
    """

    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0], "q": [1.0, 0.0, 0.0]})
    _upsert(
        "c",
        [
            {"id": "a", "text": "alpha", "metadata": {"tenant": "A"}},
            {"id": "b", "text": "beta", "metadata": {"tenant": "B"}},
        ],
        backend,
    )
    for malformed in ([], False, 0, ""):
        with pytest.raises(CollectionError):
            _search("c", "q", backend, filters=malformed)
    # None still means "search everything".
    assert len(_search("c", "q", backend, filters=None, top_k=10)) == 2


def test_text_losing_all_tokens_clears_its_bm25_statistics(isolated_cache):
    """Stale token counts would skew the corpus average for every hybrid query."""

    backend = CountingBackend({"alpha beta": [1.0, 0.0, 0.0], "!!!": [0.0, 0.0, 1.0]})
    _upsert("c", [{"id": "a", "text": "alpha beta"}], backend)

    def doc_rows() -> list[tuple[int, int]]:
        conn = store._open_readonly()
        try:
            return [
                (int(row["record_id"]), int(row["token_count"]))
                for row in conn.execute(
                    "SELECT record_id, token_count FROM collection_bm25_doc"
                )
            ]
        finally:
            conn.close()

    assert doc_rows() == [(1, 2)]
    _upsert("c", [{"id": "a", "text": "!!!"}], backend)
    assert doc_rows() == [], "a text with no tokens must leave no length statistic"


def test_metadata_only_upsert_loses_to_a_concurrent_text_change(isolated_cache):
    """The record's text and its vector must never disagree.

    "Unchanged" is decided before the write lock is taken. If another writer
    changes the text in between, the metadata-only update is writing about a
    version that no longer exists, so it is skipped rather than committing a row
    whose text and embedding describe different strings.
    """

    backend = CountingBackend(
        {"old text": [1.0, 0.0, 0.0], "new text": [0.0, 1.0, 0.0]}
    )
    _upsert("c", [{"id": "r", "text": "old text"}], backend)

    real_upsert = store.upsert_records
    fired = False

    def interleave(collection_id, rows):
        nonlocal fired
        if not fired and rows and rows[0].refresh_embedding is False:
            fired = True
            store.upsert_records = real_upsert
            _upsert("c", [{"id": "r", "text": "new text"}], backend)
            store.upsert_records = interleave
        return real_upsert(collection_id, rows)

    store.upsert_records = interleave
    try:
        _upsert("c", [{"id": "r", "text": "old text", "metadata": {"w": "late"}}], backend)
    finally:
        store.upsert_records = real_upsert

    info = store.get_collection("c")
    ids = store.resolve_filter_ids(info.id, None)
    record = store.fetch_by_ids(info.id, ids)[ids[0]]
    _, matrix = store.load_vectors(info.id, ids, info.dimension)
    expected = {"old text": [1.0, 0.0, 0.0], "new text": [0.0, 1.0, 0.0]}[record.text]
    assert np.allclose(matrix[0], expected), "text and vector disagree"


def test_search_holds_one_snapshot_so_a_filter_cannot_leak(isolated_cache):
    """A record moved out of the filtered set mid-search must not be returned.

    Filtering, vector loading, and the final fetch are separate statements. On
    an idle connection WAL lets the snapshot advance between them, so without an
    explicit read transaction a writer could hand tenant B's record to a caller
    who filtered for tenant A.
    """

    backend = CountingBackend(
        {"alpha": [1.0, 0.0, 0.0], "beta": [0.0, 1.0, 0.0], "q": [1.0, 0.0, 0.0]}
    )
    _upsert(
        "c",
        [
            {"id": "a", "text": "alpha", "metadata": {"tenant": "A"}},
            {"id": "b", "text": "beta", "metadata": {"tenant": "B"}},
        ],
        backend,
    )

    real_load = store.load_vectors
    fired = False

    def racing_load(collection_id, record_ids, dimension, conn=None):
        nonlocal fired
        if not fired:
            fired = True
            # The filter already picked 'a' for tenant A; move it to B.
            _upsert(
                "c",
                [{"id": "a", "text": "alpha", "metadata": {"tenant": "B"}}],
                backend,
            )
        return real_load(collection_id, record_ids, dimension, conn)

    store.load_vectors = racing_load
    try:
        results = _search("c", "q", backend, filters={"tenant": "A"}, top_k=10)
    finally:
        store.load_vectors = real_load

    assert [r for r in results if r.metadata.get("tenant") != "A"] == []


def test_a_damaged_database_raises_instead_of_returning_nothing(isolated_cache):
    """Collection records cannot be rebuilt, so silence is not an acceptable answer.

    The file index degrades to empty on a read error because it can always be
    rebuilt from disk. These records can only come back if the caller re-upserts
    everything and pays for embeddings again, so damage has to surface.
    """

    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0], "q": [1.0, 0.0, 0.0]})
    _upsert("c", [{"id": "a", "text": "alpha"}], backend)

    conn = store._open()
    try:
        conn.execute("DROP TABLE collection_embedding")
    finally:
        conn.close()

    with pytest.raises(sqlite3.OperationalError):
        _search("c", "q", backend)


def test_a_failed_first_write_leaves_no_pinned_collection(isolated_cache):
    """Creation and the first write are separate transactions.

    If the write fails, a collection created by that same call must not survive
    holding a contract the caller never successfully used, since that would also
    block retrying the name under a different provider or model.
    """

    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0]})
    real_upsert = store.upsert_records

    def explode(collection_id, rows):
        raise RuntimeError("write failed")

    store.upsert_records = explode
    try:
        with pytest.raises(RuntimeError):
            _upsert("fresh", [{"id": "a", "text": "alpha"}], backend)
    finally:
        store.upsert_records = real_upsert

    assert store.get_collection("fresh") is None
    # The name is free again, including under a different contract.
    _upsert("fresh", [{"id": "a", "text": "alpha"}], backend, model="other-model")
    assert store.get_collection("fresh").model == "other-model"


# ---------------------------------------------------------------------------
# 10. Second review pass: the fixes themselves
# ---------------------------------------------------------------------------


def test_cleanup_never_deletes_a_concurrent_callers_records(isolated_cache):
    """Failed-creation cleanup must not take someone else's collection with it.

    Checking "is it empty" and deleting it are one transaction, conditional on
    the row still being the same collection. Done as separate statements, a
    caller cleaning up its own failed creation could delete records another
    caller wrote in between.
    """

    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0], "beta": [0.0, 1.0, 0.0]})
    real_upsert = store.upsert_records

    def fail_after_letting_b_win(collection_id, rows):
        store.upsert_records = real_upsert
        # Writer B succeeds against the collection A just created.
        _upsert("shared", [{"id": "b-record", "text": "beta"}], backend)
        raise RuntimeError("A's write failed")

    store.upsert_records = fail_after_letting_b_win
    try:
        with pytest.raises(RuntimeError):
            _upsert("shared", [{"id": "a-record", "text": "alpha"}], backend)
    finally:
        store.upsert_records = real_upsert

    assert store.get_collection("shared") is not None
    assert collection_service.count_records(name="shared") == 1
    assert collection_service.get_records(name="shared", record_keys=["b-record"])


def test_cleanup_failure_does_not_replace_the_original_error(isolated_cache):
    backend = CountingBackend({"alpha": [1.0, 0.0, 0.0]})
    real_upsert = store.upsert_records
    real_drop = store.drop_collection_if_empty

    def explode(collection_id, rows):
        raise RuntimeError("ORIGINAL")

    def cleanup_explodes(name, collection_id):
        raise OSError("cleanup blew up")

    store.upsert_records = explode
    store.drop_collection_if_empty = cleanup_explodes
    try:
        with pytest.raises(RuntimeError, match="ORIGINAL"):
            _upsert("fresh", [{"id": "a", "text": "alpha"}], backend)
    finally:
        store.upsert_records = real_upsert
        store.drop_collection_if_empty = real_drop


def test_ensure_wal_reports_the_mode_actually_in_force(tmp_path):
    """An unsuccessful journal_mode switch does not raise, it reports the old mode.

    So the result has to be inspected rather than assumed, or a database left on
    the rollback journal would be treated as if it were in WAL.
    """

    from vexor import sqlite_util

    conn = sqlite_util.connect(tmp_path / "fresh.db")
    try:
        assert str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower() == "wal"
        assert sqlite_util._ensure_wal(conn) is True
    finally:
        conn.close()


def test_ensure_wal_propagates_errors_that_are_not_contention(tmp_path):
    """Lock contention is tolerated; a disk or corruption error is not.

    Swallowing everything would defer a real failure to some later, unrelated
    query where it is far harder to attribute.
    """

    from vexor import sqlite_util

    class Failing:
        def execute(self, sql, *args):
            raise sqlite3.OperationalError("disk I/O error")

    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        sqlite_util._ensure_wal(Failing())

    class Contended:
        def execute(self, sql, *args):
            raise sqlite3.OperationalError("database is locked")

    assert sqlite_util._ensure_wal(Contended()) is False
