from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vexor.services.index_service import IndexResult, IndexStatus
from vexor.services.search_service import SearchRequest, perform_search


class DummySearcher:
    device = "dummy-backend"

    def __init__(self, *args, **kwargs) -> None:
        return None

    def embed_texts(self, texts):
        if not texts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_perform_search_auto_indexes_when_missing(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {"load": 0, "indexed": 0, "index_kwargs": None}

    def fake_load_index_vectors(*_args, **_kwargs):
        calls["load"] = int(calls["load"]) + 1
        if calls["load"] == 1:
            raise FileNotFoundError("missing index")
        paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        metadata = {
            "files": [{"path": "a.txt", "absolute": str(paths[0]), "mtime": 0.0, "size": 1}],
            "chunks": [
                {"path": "a.txt", "chunk_index": 0, "preview": "a"},
                {"path": "b.txt", "chunk_index": 0, "preview": "b"},
            ],
        }
        return paths, vectors, metadata

    def fake_build_index(directory: Path, **kwargs):
        calls["indexed"] = int(calls["indexed"]) + 1
        calls["index_kwargs"] = {"directory": directory, **kwargs}
        return IndexResult(status=IndexStatus.STORED, files_indexed=2)

    monkeypatch.setattr("vexor.cache.load_index_vectors", fake_load_index_vectors)
    monkeypatch.setattr("vexor.services.index_service.build_index", fake_build_index)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", lambda *_a, **_k: True)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)

    request = SearchRequest(
        query="alpha",
        directory=tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=2,
        model_name="model",
        batch_size=0,
        provider="gemini",
        base_url=None,
        api_key="k",
        extensions=(),
        auto_index=True,
    )
    response = perform_search(request)

    assert response.index_empty is False
    assert response.is_stale is False
    assert len(response.results) == 2
    assert calls["load"] == 2
    assert calls["indexed"] == 1
    assert isinstance(calls["index_kwargs"], dict)
    assert calls["index_kwargs"]["directory"] == tmp_path
    assert calls["index_kwargs"]["mode"] == "name"


def test_perform_search_missing_index_raises_when_auto_index_disabled(monkeypatch, tmp_path: Path) -> None:
    def fake_load_index_vectors(*_args, **_kwargs):
        raise FileNotFoundError("missing index")

    monkeypatch.setattr("vexor.cache.load_index_vectors", fake_load_index_vectors)

    request = SearchRequest(
        query="alpha",
        directory=tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=2,
        model_name="model",
        batch_size=0,
        provider="gemini",
        base_url=None,
        api_key="k",
        extensions=(),
        auto_index=False,
    )
    with pytest.raises(FileNotFoundError):
        perform_search(request)


def test_perform_search_auto_indexes_when_stale(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {"load": 0, "indexed": 0, "cache_checks": 0}

    def fake_load_index_vectors(*_args, **_kwargs):
        calls["load"] += 1
        paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        if calls["load"] == 1:
            vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        else:
            vectors = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        metadata = {
            "files": [{"path": "a.txt", "absolute": str(paths[0]), "mtime": 0.0, "size": 1}],
            "chunks": [
                {"path": "a.txt", "chunk_index": 0, "preview": "a"},
                {"path": "b.txt", "chunk_index": 0, "preview": "b"},
            ],
        }
        return paths, vectors, metadata

    def fake_is_cache_current(*_args, **_kwargs) -> bool:
        calls["cache_checks"] += 1
        return calls["cache_checks"] > 1

    def fake_build_index(*_args, **_kwargs):
        calls["indexed"] += 1
        return IndexResult(status=IndexStatus.STORED, files_indexed=2)

    monkeypatch.setattr("vexor.cache.load_index_vectors", fake_load_index_vectors)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", fake_is_cache_current)
    monkeypatch.setattr("vexor.services.index_service.build_index", fake_build_index)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)

    request = SearchRequest(
        query="alpha",
        directory=tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=2,
        model_name="model",
        batch_size=0,
        provider="gemini",
        base_url=None,
        api_key="k",
        extensions=(),
        auto_index=True,
    )
    response = perform_search(request)

    assert response.index_empty is False
    assert response.is_stale is False
    assert calls["indexed"] == 1
    assert calls["load"] == 2
    assert response.results[0].path.name == "b.txt"

