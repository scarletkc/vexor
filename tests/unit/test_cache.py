from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

import vexor.cache as cache

MODE = "name"
PROJECT_CACHE_GITIGNORE = "*\n!.gitignore\n!config.json\n"


def _entries_for_files(
    root: Path,
    files: list[Path],
    embeddings: np.ndarray,
    *,
    prefix: str = "preview",
    previews: Sequence[str] | None = None,
):
    result: list[cache.IndexedChunk] = []
    for idx, file in enumerate(files):
        result.append(
            cache.IndexedChunk(
                path=file,
                rel_path=str(file.relative_to(root)),
                chunk_index=0,
                preview=(previews[idx] if previews is not None else f"{prefix}-{file.name}"),
                embedding=embeddings[idx],
            )
        )
    return result


def test_store_and_load_index(tmp_path, monkeypatch):
    # Redirect cache directory to temporary path
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    root = tmp_path / "project"
    root.mkdir()
    files = []
    for name in ["a.txt", "b.txt", "c.txt"]:
        file_path = root / name
        file_path.write_text("data")
        files.append(file_path)

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )

    entries = _entries_for_files(root, files, embeddings)
    cache_path = cache.store_index(
        root=root,
        model="test-model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=entries,
    )

    assert cache_path.exists()

    loaded_paths, loaded_vectors, meta = cache.load_index_vectors(
        root=root,
        model="test-model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
    )

    assert [p.name for p in loaded_paths] == ["a.txt", "b.txt", "c.txt"]
    assert np.allclose(loaded_vectors, embeddings)
    assert meta["model"] == "test-model"
    chunk_ids = meta.get("chunk_ids", [])
    assert len(chunk_ids) == 3
    chunk_meta = cache.load_chunk_metadata(chunk_ids)
    assert chunk_meta[chunk_ids[0]]["preview"] == "preview-a.txt"
    assert meta["exclude_patterns"] == ()
    assert meta["extensions"] == ()


def test_store_and_load_bm25_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    root = tmp_path / "project"
    root.mkdir()
    file_path = root / "alpha.py"
    file_path.write_text("alpha")
    entry = cache.IndexedChunk(
        path=file_path,
        rel_path="alpha.py",
        chunk_index=0,
        preview="alpha",
        embedding=[1.0, 0.0],
        bm25_terms={"alpha": 2, "python": 1},
        bm25_doc_len=3,
    )
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=[entry],
    )
    _, _, metadata = cache.load_index_vectors(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
    )
    index_id = metadata["index_id"]
    chunk_id = metadata["chunk_ids"][0]

    assert cache.load_bm25_stats(index_id) == (1, 3.0)
    assert cache.load_bm25_postings(index_id, ["alpha", "missing"]) == {
        "alpha": [(chunk_id, 2, 3)]
    }


def test_bm25_readers_fall_back_when_tables_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    connection = cache._connect(cache.cache_db_path())
    try:
        cache._ensure_schema(connection)
        connection.execute("DROP TABLE bm25_posting")
        connection.execute("DROP TABLE bm25_doc")
        assert cache.load_bm25_stats(1, conn=connection) == (0, 0.0)
        assert cache.load_bm25_postings(1, ["alpha"], conn=connection) == {}
    finally:
        connection.close()


def test_cache_dir_context_overrides_cache_db_path(tmp_path):
    original_cache_dir = cache.CACHE_DIR
    with cache.cache_dir_context(tmp_path):
        db_path = cache.cache_db_path()
        assert db_path.parent == tmp_path
    assert cache.CACHE_DIR == original_cache_dir


def test_embedding_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")

    text_hash = cache.embedding_cache_key("hello")
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.store_embedding_cache(model="model", embeddings={text_hash: vector})

    loaded = cache.load_embedding_cache("model", [text_hash])
    assert text_hash in loaded
    assert np.allclose(loaded[text_hash], vector)
    assert cache.load_embedding_cache("other-model", [text_hash]) == {}


def test_embedding_cache_memory_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    cache._clear_embedding_memory_cache()

    text_hash = cache.embedding_cache_key("memory-only")
    vector = np.array([4.0, 5.0], dtype=np.float32)
    cache.store_embedding_cache(model="model", embeddings={text_hash: vector})

    db_path = cache.cache_db_path()
    if db_path.exists():
        db_path.unlink()

    loaded = cache.load_embedding_cache("model", [text_hash])
    assert text_hash in loaded
    assert np.allclose(loaded[text_hash], vector)


def test_embedding_cache_prunes_by_ttl_and_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache, "EMBED_CACHE_TTL_DAYS", 1)
    monkeypatch.setattr(cache, "EMBED_CACHE_MAX_ENTRIES", 2)

    db_path = cache.cache_db_path()
    conn = cache._connect(db_path)
    try:
        cache._ensure_schema(conn)
        now = datetime.now(timezone.utc)
        entries = [
            (
                "model",
                cache.embedding_cache_key("old"),
                np.array([1.0], dtype=np.float32).tobytes(),
                (now - timedelta(days=2)).isoformat(),
            ),
            (
                "model",
                cache.embedding_cache_key("mid"),
                np.array([2.0], dtype=np.float32).tobytes(),
                (now - timedelta(hours=1)).isoformat(),
            ),
            (
                "model",
                cache.embedding_cache_key("new"),
                np.array([3.0], dtype=np.float32).tobytes(),
                (now - timedelta(minutes=1)).isoformat(),
            ),
        ]
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (
                    model,
                    text_hash,
                    vector_blob,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                entries,
            )
    finally:
        conn.close()

    extra_hash = cache.embedding_cache_key("extra")
    cache.store_embedding_cache(
        model="model",
        embeddings={extra_hash: np.array([4.0], dtype=np.float32)},
    )

    hashes = [
        cache.embedding_cache_key("old"),
        cache.embedding_cache_key("mid"),
        cache.embedding_cache_key("new"),
        extra_hash,
    ]
    loaded = cache.load_embedding_cache("model", hashes)
    assert cache.embedding_cache_key("old") not in loaded
    assert cache.embedding_cache_key("mid") not in loaded
    assert cache.embedding_cache_key("new") in loaded
    assert extra_hash in loaded


# --- Embedding Dimension Cache Tests ---


def test_embedding_cache_key_includes_dimension():
    """Test that dimension is included in the cache key hash."""
    key_no_dim = cache.embedding_cache_key("hello")
    key_dim_512 = cache.embedding_cache_key("hello", dimension=512)
    key_dim_1024 = cache.embedding_cache_key("hello", dimension=1024)

    # All keys should be different
    assert key_no_dim != key_dim_512
    assert key_no_dim != key_dim_1024
    assert key_dim_512 != key_dim_1024


def test_embedding_cache_key_none_dimension_same_as_no_dimension():
    """Test that explicit None dimension produces same key as no dimension."""
    key_no_dim = cache.embedding_cache_key("hello")
    key_none_dim = cache.embedding_cache_key("hello", dimension=None)

    assert key_no_dim == key_none_dim


def test_embedding_cache_dimension_isolation(tmp_path, monkeypatch):
    """Test that embeddings with different dimensions don't pollute each other."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    cache._clear_embedding_memory_cache()

    text = "test text"
    vector_512 = np.array([1.0] * 512, dtype=np.float32)
    vector_1024 = np.array([2.0] * 1024, dtype=np.float32)

    hash_512 = cache.embedding_cache_key(text, dimension=512)
    hash_1024 = cache.embedding_cache_key(text, dimension=1024)

    # Store both vectors
    cache.store_embedding_cache(model="model", embeddings={hash_512: vector_512}, dimension=512)
    cache.store_embedding_cache(model="model", embeddings={hash_1024: vector_1024}, dimension=1024)

    # Load with correct dimensions should work
    loaded_512 = cache.load_embedding_cache("model", [hash_512], dimension=512)
    loaded_1024 = cache.load_embedding_cache("model", [hash_1024], dimension=1024)

    assert hash_512 in loaded_512
    assert hash_1024 in loaded_1024
    assert np.allclose(loaded_512[hash_512], vector_512)
    assert np.allclose(loaded_1024[hash_1024], vector_1024)


def test_embedding_cache_wrong_dimension_not_found(tmp_path, monkeypatch):
    """Test that querying with wrong dimension returns empty."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    cache._clear_embedding_memory_cache()

    text = "test text"
    vector_512 = np.array([1.0] * 512, dtype=np.float32)
    hash_512 = cache.embedding_cache_key(text, dimension=512)

    cache.store_embedding_cache(model="model", embeddings={hash_512: vector_512}, dimension=512)

    # Query with different dimension should not find it
    # (because the hash is different and the key includes dimension)
    hash_1024 = cache.embedding_cache_key(text, dimension=1024)
    loaded = cache.load_embedding_cache("model", [hash_1024], dimension=1024)

    assert hash_1024 not in loaded


def test_cache_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        cache.load_index(root, "model", False, MODE, True)


def test_compare_snapshot_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    file_a = root / "a.txt"
    file_a.write_text("data")
    file_b = root / "b.txt"
    file_b.write_text("data")

    cached = [
        {"path": "a.txt", "mtime": file_a.stat().st_mtime, "size": file_a.stat().st_size},
        {"path": "b.txt", "mtime": file_b.stat().st_mtime, "size": file_b.stat().st_size},
    ]

    assert cache.compare_snapshot(root, False, cached, recursive=True) is True


def test_compare_snapshot_detects_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    file_a = root / "a.txt"
    file_a.write_text("data")

    cached = [{"path": "a.txt", "mtime": file_a.stat().st_mtime, "size": file_a.stat().st_size}]

    assert cache.compare_snapshot(root, False, cached, recursive=True) is True

    # modify file
    file_a.write_text("new data")
    assert cache.compare_snapshot(root, False, cached, recursive=True) is False

    # missing file
    file_a.unlink()
    assert cache.compare_snapshot(root, False, cached, recursive=True) is False


def test_clear_index_removes_cached_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    files = []
    for name in ["a.txt", "b.txt"]:
        file_path = root / name
        file_path.write_text("data")
        files.append(file_path)

    embeddings = np.eye(2, dtype=np.float32)

    entries = _entries_for_files(root, files, embeddings, previews=[file.name for file in files])
    cache.store_index(
        root=root,
        model="model-a",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=entries,
    )
    cache.store_index(
        root=root,
        model="model-b",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=entries,
    )

    removed = cache.clear_index(root=root, include_hidden=False, mode=MODE, recursive=True)
    assert removed == 2

    with pytest.raises(FileNotFoundError):
        cache.load_index(root=root, model="model-a", include_hidden=False, mode=MODE, recursive=True)

    # unrelated include_hidden flag should remain untouched
    cache.store_index(
        root=root,
        model="model-c",
        include_hidden=True,
        mode=MODE,
        recursive=True,
        entries=_entries_for_files(root, files, embeddings, previews=[file.name for file in files]),
    )
    removed_hidden = cache.clear_index(
        root=root,
        include_hidden=True,
        mode=MODE,
        recursive=True,
        model="model-c",
    )
    assert removed_hidden == 1


def test_recursive_and_non_recursive_caches_are_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    file_path = root / "a.txt"
    file_path.write_text("data")

    embeddings = np.array([[1.0]], dtype=np.float32)

    entries = _entries_for_files(root, [file_path], embeddings, previews=[file_path.name])
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=entries,
    )

    with pytest.raises(FileNotFoundError):
        cache.load_index(root=root, model="model", include_hidden=False, mode=MODE, recursive=False)

    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=False,
        entries=_entries_for_files(root, [file_path], embeddings, previews=[file_path.name]),
    )

    data = cache.load_index(root=root, model="model", include_hidden=False, mode=MODE, recursive=False)
    assert data["recursive"] is False


def test_apply_index_updates_handles_add_modify_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    root = tmp_path / "project"
    root.mkdir()
    file_a = root / "a.txt"
    file_b = root / "b.txt"
    file_a.write_text("a")
    file_b.write_text("b")

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=_entries_for_files(root, [file_a, file_b], embeddings, previews=[file_a.name, file_b.name]),
    )

    file_a.write_text("updated")
    file_b.unlink()
    file_c = root / "c.txt"
    file_c.write_text("c")

    ordered_entries = [
        (str(file_a.relative_to(root)), 0),
        (str(file_c.relative_to(root)), 0),
    ]
    changed_entries = [
        cache.IndexedChunk(
            path=file_a,
            rel_path=str(file_a.relative_to(root)),
            chunk_index=0,
            preview="updated",
            embedding=np.array([0.2, 0.8], dtype=np.float32),
        ),
        cache.IndexedChunk(
            path=file_c,
            rel_path=str(file_c.relative_to(root)),
            chunk_index=0,
            preview="c",
            embedding=np.array([0.3, 0.7], dtype=np.float32),
        ),
    ]

    cache.apply_index_updates(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        ordered_entries=ordered_entries,
        changed_entries=changed_entries,
        removed_rel_paths=["b.txt"],
    )

    paths, vectors, meta = cache.load_index_vectors(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
    )

    assert [p.name for p in paths] == ["a.txt", "c.txt"]
    expected = np.stack([changed_entries[0].embedding, changed_entries[1].embedding], dtype=np.float32)
    assert np.allclose(vectors, expected)
    assert meta["dimension"] == 2


def test_apply_index_updates_rewrites_cascades_and_preserves_bm25(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    root = tmp_path / "project"
    root.mkdir()
    files = [root / name for name in ("a.txt", "b.txt", "c.txt")]
    for file_path in files:
        file_path.write_text(file_path.stem)
    entries = []
    for idx, (file_path, term) in enumerate(
        zip(files, ("alpha", "beta", "gamma"))
    ):
        entries.append(
            cache.IndexedChunk(
                path=file_path,
                rel_path=file_path.name,
                chunk_index=0,
                preview=term,
                embedding=np.array([1.0, float(idx)], dtype=np.float32),
                bm25_terms={term: 1},
                bm25_doc_len=1,
            )
        )
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=entries,
    )
    _, _, before_meta = cache.load_index_vectors(
        root, "model", False, MODE, True
    )
    index_id = before_meta["index_id"]
    gamma_before = cache.load_bm25_postings(index_id, ["gamma"])

    files[0].write_text("delta")
    stat_c = files[2].stat()
    changed = cache.IndexedChunk(
        path=files[0],
        rel_path="a.txt",
        chunk_index=0,
        preview="delta",
        embedding=np.array([0.0, 1.0], dtype=np.float32),
        bm25_terms={"delta": 1},
        bm25_doc_len=1,
    )
    cache.apply_index_updates(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        ordered_entries=[("a.txt", 0), ("c.txt", 0)],
        changed_entries=[changed],
        touched_entries=[
            (
                "c.txt",
                0,
                stat_c.st_size,
                stat_c.st_mtime,
                "gamma",
                None,
                None,
                "",
            )
        ],
        removed_rel_paths=["b.txt"],
    )

    postings = cache.load_bm25_postings(
        index_id, ["alpha", "beta", "gamma", "delta"]
    )
    assert "alpha" not in postings
    assert "beta" not in postings
    assert postings["delta"][0][1:] == (1, 1)
    assert postings["gamma"] == gamma_before["gamma"]


def test_apply_index_updates_allows_deletions_without_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    root = tmp_path / "project"
    root.mkdir()
    file_a = root / "a.txt"
    file_b = root / "b.txt"
    file_a.write_text("a")
    file_b.write_text("b")

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=_entries_for_files(root, [file_a, file_b], embeddings, previews=[file_a.name, file_b.name]),
    )

    file_b.unlink()

    cache.apply_index_updates(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        ordered_entries=[(str(file_a.relative_to(root)), 0)],
        changed_entries=[],
        removed_rel_paths=["b.txt"],
    )

    paths, vectors, meta = cache.load_index_vectors(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
    )

    assert [p.name for p in paths] == ["a.txt"]
    assert vectors.shape == (1, 2)
    assert meta["dimension"] == 2


def test_list_cache_entries_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")

    assert cache.list_cache_entries() == []


def test_list_cache_entries_reports_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    root = tmp_path / "project"
    root.mkdir()
    file_path = root / "a.txt"
    file_path.write_text("data")

    embeddings = np.array([[1.0]], dtype=np.float32)
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=_entries_for_files(root, [file_path], embeddings, previews=[file_path.name]),
    )

    entries = cache.list_cache_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["root_path"] == str(root)
    assert entry["file_count"] == 1
    assert entry["include_hidden"] is False
    assert entry["extensions"] == ()


def test_extension_specific_caches_have_distinct_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    root = tmp_path / "project"
    root.mkdir()
    py_file = root / "demo.py"
    md_file = root / "readme.md"
    py_file.write_text("py")
    md_file.write_text("md")

    embeddings = np.array([[1.0], [0.5]], dtype=np.float32)
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=_entries_for_files(root, [py_file], embeddings[:1], previews=[py_file.name]),
        extensions=(".py",),
    )
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=_entries_for_files(root, [md_file], embeddings[1:], previews=[md_file.name]),
        extensions=(".md",),
    )

    py_meta = cache.load_index(root, "model", False, MODE, True, extensions=(".py",))
    assert py_meta["extensions"] == (".py",)
    md_meta = cache.load_index(root, "model", False, MODE, True, extensions=(".md",))
    assert md_meta["extensions"] == (".md",)

    with pytest.raises(FileNotFoundError):
        cache.load_index(root, "model", False, MODE, True, extensions=(".txt",))

    removed = cache.clear_index(
        root=root,
        include_hidden=False,
        mode=MODE,
        recursive=True,
        extensions=(".py",),
    )
    assert removed == 1
    with pytest.raises(FileNotFoundError):
        cache.load_index(root, "model", False, MODE, True, extensions=(".py",))
    assert cache.load_index(root, "model", False, MODE, True, extensions=(".md",))["extensions"] == (".md",)


def test_clear_all_cache_removes_database(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)
    root = tmp_path / "project"
    root.mkdir()
    file_path = root / "a.txt"
    file_path.write_text("data")

    embeddings = np.array([[1.0]], dtype=np.float32)
    entries = _entries_for_files(root, [file_path], embeddings, previews=[file_path.name])
    cache.store_index(
        root=root,
        model="model-a",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=entries,
    )
    cache.store_index(
        root=root,
        model="model-b",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=entries,
    )

    removed = cache.clear_all_cache()
    assert removed == 2
    db_path = cache_dir / cache.DB_FILENAME
    assert not db_path.exists()
    assert cache.list_cache_entries() == []
    assert cache.clear_all_cache() == 0


def test_cache_key_serialization_context_and_memory_cache(tmp_path, monkeypatch):
    key_a = cache._cache_key(
        tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        recursive=True,
        mode="name",
    )
    key_b = cache._cache_key(
        tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        recursive=True,
        mode="name",
        extensions=(".py", ".md"),
        exclude_patterns=("build/", "*.tmp"),
    )
    assert key_a != key_b
    assert cache._deserialize_extensions("a,,b") == ("a", "b")
    assert cache._deserialize_exclude_patterns("a\n\nb") == ("a", "b")
    assert list(cache._chunk_values([1, 2, 3], 2)) == [[1, 2], [3]]
    assert cache.embedding_cache_key("hello") != cache.embedding_cache_key("hello", dimension=3)

    original_dir = cache.CACHE_DIR
    override_dir = tmp_path / "override"
    with cache.cache_dir_context(override_dir):
        assert cache.ensure_cache_dir() == override_dir.resolve()
        assert cache.cache_db_path() == override_dir.resolve() / cache.DB_FILENAME
    assert cache.CACHE_DIR == original_dir

    marker = tmp_path / "not-a-dir"
    marker.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        with cache.cache_dir_context(marker):
            pass
    with pytest.raises(NotADirectoryError):
        cache.set_cache_dir(marker)
    cache.set_cache_dir(tmp_path / "explicit")
    assert cache.CACHE_DIR == (tmp_path / "explicit").resolve()
    cache.set_cache_dir(None)
    assert cache.CACHE_DIR == cache.DEFAULT_CACHE_DIR

    monkeypatch.setattr(cache, "EMBED_MEMORY_CACHE_MAX_ENTRIES", 2)
    cache._clear_embedding_memory_cache()
    cache._store_embedding_memory_cache(
        model="model",
        embeddings={
            "": np.array([9.0], dtype=np.float32),
            "empty": np.array([], dtype=np.float32),
            "a": np.array([1.0], dtype=np.float32),
            "b": np.array([2.0], dtype=np.float32),
            "c": np.array([3.0], dtype=np.float32),
        },
        dimension=2,
    )
    assert cache._load_embedding_memory_cache("model", ["a"], dimension=2) == {}
    loaded = cache._load_embedding_memory_cache("model", ["b", "c", ""], dimension=2)
    assert set(loaded) == {"b", "c"}
    assert cache._load_embedding_memory_cache("model", ["b"], dimension=3) == {}

    monkeypatch.setattr(cache, "EMBED_MEMORY_CACHE_MAX_ENTRIES", 0)
    cache._clear_embedding_memory_cache()
    cache._store_embedding_memory_cache(
        model="model",
        embeddings={"x": np.array([1.0], dtype=np.float32)},
    )
    assert cache._load_embedding_memory_cache("model", ["x"]) == {}


def test_find_project_cache_dir_at_path(tmp_path):
    marker = tmp_path / ".vexor"
    marker.mkdir()

    assert cache.find_project_cache_dir(tmp_path) == marker.resolve()


def test_find_project_cache_dir_at_ancestor(tmp_path):
    marker = tmp_path / ".vexor"
    marker.mkdir()
    child = tmp_path / "src" / "package"
    child.mkdir(parents=True)

    assert cache.find_project_cache_dir(child) == marker.resolve()


def test_find_project_cache_dir_nearest_marker_wins(tmp_path):
    (tmp_path / ".vexor").mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    marker = nested / ".vexor"
    marker.mkdir()
    child = nested / "src"
    child.mkdir()

    assert cache.find_project_cache_dir(child) == marker.resolve()


def test_find_project_cache_dir_returns_none_without_marker(tmp_path):
    assert cache.find_project_cache_dir(tmp_path) is None


def test_find_project_cache_dir_ignores_plain_file(tmp_path):
    (tmp_path / ".vexor").write_text("not a directory", encoding="utf-8")

    assert cache.find_project_cache_dir(tmp_path) is None


def test_find_project_cache_dir_skips_global_data_dir(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    project = fake_home / "projects" / "demo"
    project.mkdir(parents=True)
    (fake_home / ".vexor").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    # The walk-up may continue past the (fake) home into the host filesystem,
    # so only assert that the home-level marker itself is never adopted.
    result = cache.find_project_cache_dir(project)

    assert result != (fake_home / ".vexor").resolve()


def test_project_cache_context_uses_detected_marker(tmp_path, monkeypatch):
    global_cache = tmp_path / "global" / ".vexor"
    monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", global_cache)
    monkeypatch.setattr(cache, "CACHE_DIR", global_cache)
    project = tmp_path / "project"
    marker = project / ".vexor"
    marker.mkdir(parents=True)

    with cache.project_cache_context(project):
        assert cache.cache_db_path() == marker.resolve() / cache.DB_FILENAME


def test_project_cache_context_respects_active_override(tmp_path, monkeypatch):
    global_cache = tmp_path / "global" / ".vexor"
    monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", global_cache)
    monkeypatch.setattr(cache, "CACHE_DIR", global_cache)
    project = tmp_path / "project"
    (project / ".vexor").mkdir(parents=True)
    explicit = tmp_path / "explicit"

    with cache.cache_dir_context(explicit):
        with cache.project_cache_context(project):
            assert cache.cache_db_path() == explicit.resolve() / cache.DB_FILENAME


def test_project_cache_context_respects_relocated_global_cache(tmp_path, monkeypatch):
    default_cache = tmp_path / "default"
    monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", default_cache)
    monkeypatch.setattr(cache, "CACHE_DIR", default_cache)
    project = tmp_path / "project"
    (project / ".vexor").mkdir(parents=True)
    explicit = tmp_path / "explicit"

    cache.set_cache_dir(explicit)
    try:
        with cache.project_cache_context(project):
            assert cache.cache_db_path() == explicit.resolve() / cache.DB_FILENAME
    finally:
        cache.set_cache_dir(None)


def test_create_project_cache_dir_creates_marker_and_gitignore(tmp_path):
    marker = cache.create_project_cache_dir(tmp_path)

    assert marker == tmp_path / ".vexor"
    assert marker.is_dir()
    assert (marker / ".gitignore").read_text(encoding="utf-8") == PROJECT_CACHE_GITIGNORE


def test_create_project_cache_dir_migrates_legacy_gitignore(tmp_path):
    marker = tmp_path / ".vexor"
    marker.mkdir()
    gitignore = marker / ".gitignore"
    gitignore.write_text("*\n", encoding="utf-8")

    assert cache.create_project_cache_dir(tmp_path) == marker
    assert gitignore.read_text(encoding="utf-8") == PROJECT_CACHE_GITIGNORE


def test_create_project_cache_dir_preserves_existing_gitignore(tmp_path):
    marker = tmp_path / ".vexor"
    marker.mkdir()
    gitignore = marker / ".gitignore"
    gitignore.write_text("keep-this\n", encoding="utf-8")

    assert cache.create_project_cache_dir(tmp_path) == marker
    assert gitignore.read_text(encoding="utf-8") == "keep-this\n"


def test_create_project_cache_dir_preserves_custom_legacy_based_gitignore(tmp_path):
    marker = tmp_path / ".vexor"
    marker.mkdir()
    gitignore = marker / ".gitignore"
    custom = "*\n!custom.json\n"
    gitignore.write_text(custom, encoding="utf-8")

    assert cache.create_project_cache_dir(tmp_path) == marker
    assert gitignore.read_text(encoding="utf-8") == custom


def test_project_cache_context_writes_self_ignore_for_manual_marker(
    tmp_path, monkeypatch
):
    global_cache = tmp_path / "global" / ".vexor"
    monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", global_cache)
    monkeypatch.setattr(cache, "CACHE_DIR", global_cache)
    project = tmp_path / "project"
    marker = project / ".vexor"
    marker.mkdir(parents=True)

    with cache.project_cache_context(project):
        pass

    assert (marker / ".gitignore").read_text(encoding="utf-8") == PROJECT_CACHE_GITIGNORE


def test_project_cache_context_preserves_existing_gitignore(tmp_path, monkeypatch):
    global_cache = tmp_path / "global" / ".vexor"
    monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", global_cache)
    monkeypatch.setattr(cache, "CACHE_DIR", global_cache)
    project = tmp_path / "project"
    marker = project / ".vexor"
    marker.mkdir(parents=True)
    (marker / ".gitignore").write_text("custom\n", encoding="utf-8")

    with cache.project_cache_context(project):
        pass

    assert (marker / ".gitignore").read_text(encoding="utf-8") == "custom\n"


def test_compare_snapshot_current_for_nested_paths(tmp_path, monkeypatch):
    """Stored posix rel_paths must match compare_snapshot's rel_path form."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")

    root = tmp_path / "project"
    sub = root / "sub"
    sub.mkdir(parents=True)
    nested = sub / "nested.txt"
    nested.write_text("data", encoding="utf-8")

    entries = [
        cache.IndexedChunk(
            path=nested,
            rel_path="sub/nested.txt",
            chunk_index=0,
            preview="nested",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
    ]
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        entries=entries,
    )
    meta = cache.load_index(root, "model", False, MODE, True)

    assert cache.compare_snapshot(
        root,
        False,
        meta["files"],
        recursive=True,
    ) is True
