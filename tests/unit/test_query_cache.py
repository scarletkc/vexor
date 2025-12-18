from __future__ import annotations

from pathlib import Path

import numpy as np

import vexor.cache as cache


MODE = "name"


def _store_minimal_index(root: Path, model: str) -> int:
    file_path = root / "a.txt"
    file_path.write_text("data")
    entries = [
        cache.IndexedChunk(
            path=file_path,
            rel_path="a.txt",
            chunk_index=0,
            preview="preview-a",
            embedding=[1.0, 0.0],
        )
    ]
    cache.store_index(
        root=root,
        model=model,
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=entries,
    )
    _, _, metadata = cache.load_index_vectors(
        root=root,
        model=model,
        include_hidden=False,
        mode=MODE,
        recursive=True,
    )
    return int(metadata["index_id"])


def test_store_and_load_query_vector(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()

    index_id = _store_minimal_index(root, model="test-model")
    query_hash = cache.query_cache_key("hello", "test-model")
    query_vector = np.array([0.2, 0.8], dtype=np.float32)

    cache.store_query_vector(index_id, query_hash, "hello", query_vector)
    loaded = cache.load_query_vector(index_id, query_hash)

    assert loaded is not None
    assert np.allclose(loaded, query_vector)


def test_cache_miss_returns_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()

    index_id = _store_minimal_index(root, model="test-model")

    assert cache.load_query_vector(index_id, "missing-hash") is None


def test_query_cache_key_includes_model() -> None:
    assert cache.query_cache_key("hello", "model-a") != cache.query_cache_key("hello", "model-b")


def test_query_cache_key_strips_provider_prefix() -> None:
    assert cache.query_cache_key("hello", "openai/text-embedding-3-small") == cache.query_cache_key(
        "hello",
        "text-embedding-3-small",
    )


def test_cascade_delete_on_index_rebuild(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()

    index_id = _store_minimal_index(root, model="test-model")
    query_hash = cache.query_cache_key("hello", "test-model")
    cache.store_query_vector(index_id, query_hash, "hello", np.array([0.2, 0.8], dtype=np.float32))

    assert cache.load_query_vector(index_id, query_hash) is not None

    file_path = root / "a.txt"
    entries = [
        cache.IndexedChunk(
            path=file_path,
            rel_path="a.txt",
            chunk_index=0,
            preview="preview-a",
            embedding=[0.0, 1.0],
        )
    ]
    cache.store_index(
        root=root,
        model="test-model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=entries,
    )

    assert cache.load_query_vector(index_id, query_hash) is None

