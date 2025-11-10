from pathlib import Path

import numpy as np
import pytest

import vexor.cache as cache

MODE = "name"


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

    cache_path = cache.store_index(
        root=root,
        model="test-model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        files=files,
        previews=[f"preview-{file.name}" for file in files],
        embeddings=embeddings,
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
    assert meta["files"][0]["preview"] == "preview-a.txt"


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

    cache.store_index(
        root=root,
        model="model-a",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        files=files,
        previews=[file.name for file in files],
        embeddings=embeddings,
    )
    cache.store_index(
        root=root,
        model="model-b",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        files=files,
        previews=[file.name for file in files],
        embeddings=embeddings,
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
        files=files,
        previews=[file.name for file in files],
        embeddings=embeddings,
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

    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        files=[file_path],
        previews=[file_path.name],
        embeddings=embeddings,
    )

    with pytest.raises(FileNotFoundError):
        cache.load_index(root=root, model="model", include_hidden=False, mode=MODE, recursive=False)

    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=False,
        files=[file_path],
        previews=[file_path.name],
        embeddings=embeddings,
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
        files=[file_a, file_b],
        previews=[file_a.name, file_b.name],
        embeddings=embeddings,
    )

    file_a.write_text("updated")
    file_b.unlink()
    file_c = root / "c.txt"
    file_c.write_text("c")

    current = [file_a, file_c]
    embeddings_map = {
        str(file_a.relative_to(root)): np.array([0.2, 0.8], dtype=np.float32),
        str(file_c.relative_to(root)): np.array([0.3, 0.7], dtype=np.float32),
    }
    preview_map = {
        str(file_a.relative_to(root)): "updated",
        str(file_c.relative_to(root)): "c",
    }

    cache.apply_index_updates(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        current_files=current,
        changed_files=current,
        removed_rel_paths=["b.txt"],
        embeddings=embeddings_map,
        previews=preview_map,
    )

    paths, vectors, meta = cache.load_index_vectors(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
    )

    assert [p.name for p in paths] == ["a.txt", "c.txt"]
    expected = np.stack([embeddings_map["a.txt"], embeddings_map["c.txt"]], dtype=np.float32)
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
        files=[file_a, file_b],
        previews=[file_a.name, file_b.name],
        embeddings=embeddings,
    )

    file_b.unlink()

    cache.apply_index_updates(
        root=root,
        model="model",
        include_hidden=False,
        mode=MODE,
        recursive=True,
        current_files=[file_a],
        changed_files=[],
        removed_rel_paths=["b.txt"],
        embeddings={},
        previews={},
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
