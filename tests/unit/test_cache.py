from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

import vexor.cache as cache

MODE = "name"


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
    assert meta["chunks"][0]["preview"] == "preview-a.txt"
    assert meta["extensions"] == ()


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
