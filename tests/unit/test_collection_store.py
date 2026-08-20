"""Unit coverage for the SQLite collection store."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import vexor.cache as cache
import vexor.collection_store as store
from vexor.collection_store import CollectionError, PreparedRecord
from vexor.sqlite_util import connect


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    cache._clear_embedding_memory_cache()
    yield tmp_path
    cache._clear_embedding_memory_cache()


def _create_collection(name: str = "records") -> store.CollectionInfo:
    return store.ensure_collection(
        name,
        provider="local",
        model="stub-model",
        dimension=3,
    )


def _write_records(
    info: store.CollectionInfo,
    records: list[tuple[str, dict[str, store.ScalarValue]]],
) -> None:
    prepared = [
        PreparedRecord(
            record_key=record_id,
            text=f"text for {record_id}",
            text_hash=f"hash-{record_id}",
            metadata=metadata,
            vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            bm25_terms={"text": 1, record_id: 1},
            token_count=2,
        )
        for record_id, metadata in records
    ]
    assert store.upsert_records(info.id, prepared) == len(records)


def _filtered_keys(
    info: store.CollectionInfo,
    filters: dict[str, object],
) -> list[str]:
    internal_ids = store.resolve_filter_ids(info.id, filters)
    found = store.fetch_by_ids(info.id, internal_ids)
    return [found[record_id].id for record_id in internal_ids]


def test_metadata_scalars_round_trip_filter_and_use_expected_columns(isolated_cache):
    timestamp = datetime(2026, 8, 20, 12, 30, 45, tzinfo=timezone.utc)
    metadata: dict[str, store.ScalarValue] = {
        "string": "alpha",
        "integer": 7,
        "float": 2.5,
        "boolean": True,
        "timestamp": timestamp,
        "nullable": None,
    }
    info = _create_collection()
    _write_records(info, [("one", metadata)])

    [record] = store.fetch_records(info.id, ["one"])
    assert record.metadata == {
        "string": "alpha",
        "integer": 7,
        "float": 2.5,
        "boolean": True,
        "timestamp": timestamp.isoformat(),
        "nullable": None,
    }

    for key, value in metadata.items():
        assert _filtered_keys(info, {key: value}) == ["one"]

    conn = connect(store.collections_db_path(), readonly=True)
    try:
        rows = conn.execute("SELECT key, value_text, value_num FROM collection_meta").fetchall()
    finally:
        conn.close()
    encoded = {str(row["key"]): (row["value_text"], row["value_num"]) for row in rows}
    assert encoded == {
        "string": ("alpha", None),
        "integer": (None, 7.0),
        "float": (None, 2.5),
        "boolean": (None, 1.0),
        "timestamp": (timestamp.isoformat(), timestamp.timestamp()),
        "nullable": (None, None),
    }


def test_eq_explicit_and_bare_shorthand(isolated_cache):
    info = _create_collection()
    _write_records(
        info,
        [
            ("a", {"group": "x"}),
            ("b", {"group": "x"}),
            ("c", {"group": "y"}),
        ],
    )

    assert _filtered_keys(info, {"group": {"eq": "x"}}) == ["a", "b"]
    assert _filtered_keys(info, {"group": "x"}) == ["a", "b"]


def test_in_filter_accepts_multiple_scalar_types(isolated_cache):
    info = _create_collection()
    _write_records(
        info,
        [
            ("a", {"value": "one"}),
            ("b", {"value": 2}),
            ("c", {"value": "three"}),
        ],
    )

    assert _filtered_keys(info, {"value": {"in": ["one", 2]}}) == ["a", "b"]


@pytest.mark.parametrize(
    ("operator", "threshold", "expected"),
    [
        ("gt", 1, ["b", "c"]),
        ("gte", 2, ["b", "c"]),
        ("lt", 3, ["a", "b"]),
        ("lte", 2, ["a", "b"]),
    ],
)
def test_range_filters(
    isolated_cache,
    operator: str,
    threshold: int,
    expected: list[str],
):
    info = _create_collection()
    _write_records(
        info,
        [
            ("a", {"score": 1}),
            ("b", {"score": 2}),
            ("c", {"score": 3}),
        ],
    )

    assert _filtered_keys(info, {"score": {operator: threshold}}) == expected


def test_multiple_filter_keys_are_anded(isolated_cache):
    info = _create_collection()
    _write_records(
        info,
        [
            ("a", {"group": "x", "score": 1}),
            ("b", {"group": "x", "score": 2}),
            ("c", {"group": "y", "score": 3}),
        ],
    )

    assert _filtered_keys(
        info,
        {"group": "x", "score": {"gte": 2}},
    ) == ["b"]


def test_nested_metadata_value_raises_collection_error():
    with pytest.raises(CollectionError):
        store.normalize_metadata({"nested": {"child": "value"}})


def test_metadata_that_is_not_a_mapping_raises_collection_error():
    with pytest.raises(CollectionError):
        store.normalize_metadata([("key", "value")])  # type: ignore[arg-type]


def test_unknown_filter_operator_raises_collection_error():
    with pytest.raises(CollectionError):
        store.resolve_filter_ids(1, {"key": {"contains": "value"}})


def test_exists_with_non_bool_raises_collection_error():
    with pytest.raises(CollectionError):
        store.resolve_filter_ids(1, {"key": {"exists": 1}})


@pytest.mark.parametrize("operator", ["in", "nin"])
def test_set_operator_with_non_sequence_raises_collection_error(operator: str):
    with pytest.raises(CollectionError):
        store.resolve_filter_ids(1, {"key": {operator: 1}})


@pytest.mark.parametrize("operator", ["in", "nin"])
def test_set_operator_with_empty_sequence_raises_collection_error(operator: str):
    with pytest.raises(CollectionError):
        store.resolve_filter_ids(1, {"key": {operator: []}})


@pytest.mark.parametrize("operator", ["gt", "gte", "lt", "lte"])
def test_range_operator_with_string_raises_collection_error(operator: str):
    with pytest.raises(CollectionError):
        store.resolve_filter_ids(1, {"key": {operator: "value"}})


def test_empty_filter_condition_mapping_raises_collection_error():
    with pytest.raises(CollectionError):
        store.resolve_filter_ids(1, {"key": {}})


def test_fetch_records_preserves_order_and_skips_absent_ids(isolated_cache):
    info = _create_collection()
    _write_records(
        info,
        [
            ("a", {"position": 1}),
            ("b", {"position": 2}),
            ("c", {"position": 3}),
        ],
    )

    records = store.fetch_records(info.id, ["c", "missing", "a"])
    assert [record.id for record in records] == ["c", "a"]


def test_empty_and_missing_store_operations_are_noops(isolated_cache):
    assert store.count_records(1) == 0
    assert store.load_record_hashes(1, ["missing"]) == {}
    assert store.fetch_records(1, ["missing"]) == []
    assert store.resolve_filter_ids(1, None) == []
    assert store.load_bm25_stats(1, [1]) == (0, 0.0)
    assert store.load_bm25_postings(1, [1], ["term"]) == {}
    assert store.fetch_by_ids(1, [1]) == {}
    vector_ids, vectors = store.load_vectors(1, [1], 3)
    assert vector_ids == []
    assert vectors.shape == (0, 3)

    assert store.load_record_hashes(1, []) == {}
    assert store.upsert_records(1, []) == 0
    assert store.delete_records(1, []) == 0
    assert store.fetch_records(1, []) == []
    assert store.load_bm25_stats(1, []) == (0, 0.0)
    assert store.load_bm25_postings(1, [], ["term"]) == {}
    assert store.fetch_by_ids(1, []) == {}
    vector_ids, vectors = store.load_vectors(1, [], 3)
    assert vector_ids == []
    assert vectors.shape == (0, 3)


def test_clear_all_collections_removes_database_file(isolated_cache):
    _create_collection()
    db_path = store.collections_db_path()
    assert db_path.is_file()

    store.clear_all_collections()

    assert not db_path.exists()
    assert store.list_collections() == []
