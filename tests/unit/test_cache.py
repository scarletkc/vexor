from pathlib import Path

import numpy as np
import pytest

import vexor.cache as cache


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
        files=files,
        embeddings=embeddings,
    )

    assert cache_path.exists()

    loaded_paths, loaded_vectors, meta = cache.load_index_vectors(
        root=root,
        model="test-model",
        include_hidden=False,
    )

    assert [p.name for p in loaded_paths] == ["a.txt", "b.txt", "c.txt"]
    assert np.allclose(loaded_vectors, embeddings)
    assert meta["model"] == "test-model"


def test_cache_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        cache.load_index(root, "model", False)


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

    assert cache.compare_snapshot(root, False, cached) is True


def test_compare_snapshot_detects_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    file_a = root / "a.txt"
    file_a.write_text("data")

    cached = [{"path": "a.txt", "mtime": file_a.stat().st_mtime, "size": file_a.stat().st_size}]

    assert cache.compare_snapshot(root, False, cached) is True

    # modify file
    file_a.write_text("new data")
    assert cache.compare_snapshot(root, False, cached) is False

    # missing file
    file_a.unlink()
    assert cache.compare_snapshot(root, False, cached) is False
