from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vexor.services.index_service import IndexResult, IndexStatus
from vexor.config import RemoteRerankConfig
from vexor.services.search_service import SearchRequest, perform_search


class DummySearcher:
    device = "dummy-backend"

    def __init__(self, *args, **kwargs) -> None:
        return None

    def embed_texts(self, texts):
        if not texts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_bm25_tokenizer_handles_cjk() -> None:
    from vexor.services import search_service as search_service_module

    tokens = search_service_module._bm25_tokenize("中文测试")
    assert tokens


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
        local_cuda=False,
        exclude_patterns=(),
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
        local_cuda=False,
        exclude_patterns=(),
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
        local_cuda=False,
        exclude_patterns=(),
        extensions=(),
        auto_index=True,
    )
    response = perform_search(request)

    assert response.index_empty is False
    assert response.is_stale is False
    assert calls["indexed"] == 1
    assert calls["load"] == 2
    assert response.results[0].path.name == "b.txt"


def test_perform_search_filters_exclude_patterns_with_superset(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, list[dict[str, object]]] = {"loads": []}

    def fake_load_index_vectors(
        root: Path,
        model: str,
        include_hidden: bool,
        mode: str,
        recursive: bool,
        exclude_patterns,
        extensions,
        *,
        respect_gitignore: bool,
    ):
        calls["loads"].append(
            {"exclude_patterns": exclude_patterns, "extensions": extensions}
        )
        if len(calls["loads"]) == 1:
            raise FileNotFoundError("missing index")
        paths = [tmp_path / "keep.py", tmp_path / "skip.js"]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        metadata = {
            "files": [
                {"path": "keep.py", "absolute": str(paths[0]), "mtime": 0.0, "size": 1},
                {"path": "skip.js", "absolute": str(paths[1]), "mtime": 0.0, "size": 1},
            ],
            "chunks": [
                {"path": "keep.py", "chunk_index": 0, "preview": "keep"},
                {"path": "skip.js", "chunk_index": 0, "preview": "skip"},
            ],
        }
        return paths, vectors, metadata

    def fake_list_cache_entries():
        return [
            {
                "root_path": str(tmp_path),
                "model": "model",
                "include_hidden": False,
                "respect_gitignore": True,
                "recursive": True,
                "mode": "name",
                "exclude_patterns": (),
                "extensions": (),
                "file_count": 2,
            }
        ]

    monkeypatch.setattr("vexor.cache.load_index_vectors", fake_load_index_vectors)
    monkeypatch.setattr("vexor.cache.list_cache_entries", fake_list_cache_entries)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", lambda *_a, **_k: True)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)

    request = SearchRequest(
        query="alpha",
        directory=tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=5,
        model_name="model",
        batch_size=0,
        provider="gemini",
        base_url=None,
        api_key="k",
        local_cuda=False,
        exclude_patterns=(".js",),
        extensions=(".py",),
        auto_index=False,
    )
    response = perform_search(request)

    assert len(calls["loads"]) == 2
    assert calls["loads"][0]["exclude_patterns"] == (".js",)
    assert calls["loads"][0]["extensions"] == (".py",)
    assert calls["loads"][1]["exclude_patterns"] == ()
    assert calls["loads"][1]["extensions"] == ()
    assert len(response.results) == 1
    assert response.results[0].path.name == "keep.py"


def test_perform_search_uses_cached_query_vector(monkeypatch, tmp_path: Path) -> None:
    import vexor.cache as cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("data")
    (root / "b.txt").write_text("data")

    entries = [
        cache.IndexedChunk(
            path=root / "a.txt",
            rel_path="a.txt",
            chunk_index=0,
            preview="a",
            embedding=[1.0, 0.0],
        ),
        cache.IndexedChunk(
            path=root / "b.txt",
            rel_path="b.txt",
            chunk_index=0,
            preview="b",
            embedding=[0.0, 1.0],
        ),
    ]
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode="name",
        recursive=True,
        entries=entries,
    )

    calls = {"embeds": 0}

    class CountingSearcher:
        device = "dummy-backend"

        def __init__(self, *args, **kwargs) -> None:
            return None

        def embed_texts(self, texts):
            calls["embeds"] += 1
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr("vexor.search.VexorSearcher", CountingSearcher)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", lambda *_a, **_k: True)

    request = SearchRequest(
        query="alpha",
        directory=root,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=2,
        model_name="model",
        batch_size=0,
        provider="openai",
        base_url=None,
        api_key="k",
        local_cuda=False,
        exclude_patterns=(),
        extensions=(),
        auto_index=False,
    )

    response1 = perform_search(request)
    response2 = perform_search(request)

    assert response1.index_empty is False
    assert response2.index_empty is False
    assert calls["embeds"] == 1


def test_perform_search_reuses_superset_index_for_extension_filter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import vexor.cache as cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    file_py = root / "a.py"
    file_md = root / "b.md"
    file_py.write_text("print('a')", encoding="utf-8")
    file_md.write_text("# doc", encoding="utf-8")

    entries = [
        cache.IndexedChunk(
            path=file_py,
            rel_path="a.py",
            chunk_index=0,
            preview="a",
            embedding=[1.0, 0.0],
        ),
        cache.IndexedChunk(
            path=file_md,
            rel_path="b.md",
            chunk_index=0,
            preview="b",
            embedding=[0.0, 1.0],
        ),
    ]
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode="name",
        recursive=True,
        entries=entries,
    )

    calls = {"indexed": 0}

    def fake_build_index(*_args, **_kwargs):
        calls["indexed"] += 1
        return IndexResult(status=IndexStatus.STORED, files_indexed=2)

    class DummySearcher:
        device = "dummy-backend"

        def __init__(self, *args, **kwargs) -> None:
            return None

        def embed_texts(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr("vexor.services.index_service.build_index", fake_build_index)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", lambda *_a, **_k: True)

    request = SearchRequest(
        query="alpha",
        directory=root,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=5,
        model_name="model",
        batch_size=0,
        provider="openai",
        base_url=None,
        api_key="k",
        local_cuda=False,
        exclude_patterns=(),
        extensions=(".py",),
        auto_index=True,
    )

    response = perform_search(request)

    assert calls["indexed"] == 0
    assert [result.path.name for result in response.results] == ["a.py"]


def test_perform_search_reindexes_superset_for_extension_filter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import vexor.cache as cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    file_py = root / "a.py"
    file_md = root / "b.md"
    file_py.write_text("print('a')", encoding="utf-8")
    file_md.write_text("# doc", encoding="utf-8")

    entries = [
        cache.IndexedChunk(
            path=file_py,
            rel_path="a.py",
            chunk_index=0,
            preview="a",
            embedding=[1.0, 0.0],
        ),
        cache.IndexedChunk(
            path=file_md,
            rel_path="b.md",
            chunk_index=0,
            preview="b",
            embedding=[0.0, 1.0],
        ),
    ]
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode="name",
        recursive=True,
        entries=entries,
    )

    calls = {"indexed": 0, "extensions": None}

    def fake_build_index(*_args, **kwargs):
        calls["indexed"] += 1
        calls["extensions"] = kwargs.get("extensions")
        return IndexResult(status=IndexStatus.STORED, files_indexed=2)

    class DummySearcher:
        device = "dummy-backend"

        def __init__(self, *args, **kwargs) -> None:
            return None

        def embed_texts(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    state = {"count": 0}

    def fake_is_cache_current(*_args, **_kwargs) -> bool:
        state["count"] += 1
        return state["count"] > 1

    monkeypatch.setattr("vexor.services.index_service.build_index", fake_build_index)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", fake_is_cache_current)

    request = SearchRequest(
        query="alpha",
        directory=root,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=5,
        model_name="model",
        batch_size=0,
        provider="openai",
        base_url=None,
        api_key="k",
        local_cuda=False,
        exclude_patterns=(),
        extensions=(".py",),
        auto_index=True,
    )

    response = perform_search(request)

    assert calls["indexed"] == 1
    assert calls["extensions"] == ()
    assert [result.path.name for result in response.results] == ["a.py"]


def test_perform_search_reuses_parent_index_for_subdir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import vexor.cache as cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    subdir = root / "pkg"
    subdir.mkdir(parents=True)
    file_py = subdir / "a.py"
    nested_py = subdir / "nested" / "c.py"
    file_md = root / "b.md"
    file_py.write_text("print('a')", encoding="utf-8")
    nested_py.parent.mkdir(parents=True)
    nested_py.write_text("print('c')", encoding="utf-8")
    file_md.write_text("# doc", encoding="utf-8")

    entries = [
        cache.IndexedChunk(
            path=file_py,
            rel_path="pkg/a.py",
            chunk_index=0,
            preview="a",
            embedding=[1.0, 0.0],
        ),
        cache.IndexedChunk(
            path=nested_py,
            rel_path="pkg/nested/c.py",
            chunk_index=0,
            preview="c",
            embedding=[0.5, 0.5],
        ),
        cache.IndexedChunk(
            path=file_md,
            rel_path="b.md",
            chunk_index=0,
            preview="b",
            embedding=[0.0, 1.0],
        ),
    ]
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode="name",
        recursive=True,
        entries=entries,
    )

    calls = {"indexed": 0}

    def fake_build_index(*_args, **_kwargs):
        calls["indexed"] += 1
        return IndexResult(status=IndexStatus.STORED, files_indexed=2)

    class DummySearcher:
        device = "dummy-backend"

        def __init__(self, *args, **kwargs) -> None:
            return None

        def embed_texts(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr("vexor.services.index_service.build_index", fake_build_index)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", lambda *_a, **_k: True)

    request = SearchRequest(
        query="alpha",
        directory=subdir,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=False,
        top_k=5,
        model_name="model",
        batch_size=0,
        provider="openai",
        base_url=None,
        api_key="k",
        local_cuda=False,
        exclude_patterns=(),
        extensions=(),
        auto_index=True,
    )

    response = perform_search(request)

    assert calls["indexed"] == 0
    assert [result.path.name for result in response.results] == ["a.py"]


def test_perform_search_reindexes_parent_for_subdir_stale(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import vexor.cache as cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    subdir = root / "pkg"
    subdir.mkdir(parents=True)
    file_py = subdir / "a.py"
    file_md = root / "b.md"
    file_py.write_text("print('a')", encoding="utf-8")
    file_md.write_text("# doc", encoding="utf-8")

    entries = [
        cache.IndexedChunk(
            path=file_py,
            rel_path="pkg/a.py",
            chunk_index=0,
            preview="a",
            embedding=[1.0, 0.0],
        ),
        cache.IndexedChunk(
            path=file_md,
            rel_path="b.md",
            chunk_index=0,
            preview="b",
            embedding=[0.0, 1.0],
        ),
    ]
    cache.store_index(
        root=root,
        model="model",
        include_hidden=False,
        mode="name",
        recursive=True,
        entries=entries,
    )

    calls = {"indexed": 0, "directory": None, "recursive": None}

    def fake_build_index(directory: Path, **kwargs):
        calls["indexed"] += 1
        calls["directory"] = directory
        calls["recursive"] = kwargs.get("recursive")
        return IndexResult(status=IndexStatus.STORED, files_indexed=2)

    class DummySearcher:
        device = "dummy-backend"

        def __init__(self, *args, **kwargs) -> None:
            return None

        def embed_texts(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    state = {"count": 0}

    def fake_is_cache_current(*_args, **_kwargs) -> bool:
        state["count"] += 1
        return state["count"] > 1

    monkeypatch.setattr("vexor.services.index_service.build_index", fake_build_index)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", fake_is_cache_current)

    request = SearchRequest(
        query="alpha",
        directory=subdir,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=False,
        top_k=5,
        model_name="model",
        batch_size=0,
        provider="openai",
        base_url=None,
        api_key="k",
        local_cuda=False,
        exclude_patterns=(),
        extensions=(),
        auto_index=True,
    )

    response = perform_search(request)

    assert calls["indexed"] == 1
    assert calls["directory"] == root
    assert calls["recursive"] is True
    assert [result.path.name for result in response.results] == ["a.py"]


def test_perform_search_reranks_with_bm25(monkeypatch, tmp_path: Path) -> None:
    def fake_load_index_vectors(*_args, **_kwargs):
        paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        vectors = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
        metadata = {
            "files": [
                {"path": "a.txt", "absolute": str(paths[0]), "mtime": 0.0, "size": 1},
                {"path": "b.txt", "absolute": str(paths[1]), "mtime": 0.0, "size": 1},
            ],
            "chunks": [
                {"path": "a.txt", "chunk_index": 0, "preview": "beta"},
                {"path": "b.txt", "chunk_index": 0, "preview": "alpha match"},
            ],
        }
        return paths, vectors, metadata

    monkeypatch.setattr("vexor.cache.load_index_vectors", fake_load_index_vectors)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", lambda *_a, **_k: True)
    import importlib

    search_module = importlib.import_module("vexor.search")
    monkeypatch.setattr(search_module, "VexorSearcher", DummySearcher)

    request = SearchRequest(
        query="alpha",
        directory=tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=1,
        model_name="model",
        batch_size=0,
        provider="gemini",
        base_url=None,
        api_key="k",
        local_cuda=False,
        exclude_patterns=(),
        extensions=(),
        auto_index=True,
        rerank="bm25",
    )
    response = perform_search(request)

    assert response.reranker == "bm25"
    assert len(response.results) == 1
    assert response.results[0].path.name == "b.txt"


def test_perform_search_reranks_with_remote(monkeypatch, tmp_path: Path) -> None:
    def fake_load_index_vectors(*_args, **_kwargs):
        paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        vectors = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
        metadata = {
            "files": [
                {"path": "a.txt", "absolute": str(paths[0]), "mtime": 0.0, "size": 1},
                {"path": "b.txt", "absolute": str(paths[1]), "mtime": 0.0, "size": 1},
            ],
            "chunks": [
                {"path": "a.txt", "chunk_index": 0, "preview": "beta"},
                {"path": "b.txt", "chunk_index": 0, "preview": "alpha match"},
            ],
        }
        return paths, vectors, metadata

    def fake_remote_rerank_request(*_args, **_kwargs):
        return {"data": [{"index": 1, "relevance_score": 0.9}]}

    monkeypatch.setattr("vexor.cache.load_index_vectors", fake_load_index_vectors)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "vexor.services.search_service._remote_rerank_request",
        fake_remote_rerank_request,
    )
    import importlib

    search_module = importlib.import_module("vexor.search")
    monkeypatch.setattr(search_module, "VexorSearcher", DummySearcher)

    request = SearchRequest(
        query="alpha",
        directory=tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=1,
        model_name="model",
        batch_size=0,
        provider="gemini",
        base_url=None,
        api_key="k",
        local_cuda=False,
        exclude_patterns=(),
        extensions=(),
        auto_index=True,
        rerank="remote",
        remote_rerank=RemoteRerankConfig(
            base_url="https://api.example.test/v1/rerank",
            api_key="remote-key",
            model="rerank-model",
        ),
    )
    response = perform_search(request)

    assert response.reranker == "remote"
    assert len(response.results) == 1
    assert response.results[0].path.name == "b.txt"


def test_remote_rerank_uses_env_api_key(monkeypatch, tmp_path: Path) -> None:
    def fake_load_index_vectors(*_args, **_kwargs):
        paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        vectors = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
        metadata = {
            "files": [
                {"path": "a.txt", "absolute": str(paths[0]), "mtime": 0.0, "size": 1},
                {"path": "b.txt", "absolute": str(paths[1]), "mtime": 0.0, "size": 1},
            ],
            "chunks": [
                {"path": "a.txt", "chunk_index": 0, "preview": "beta"},
                {"path": "b.txt", "chunk_index": 0, "preview": "alpha match"},
            ],
        }
        return paths, vectors, metadata

    def fake_remote_rerank_request(*_args, **_kwargs):
        return {"results": [{"index": 1, "relevance_score": 0.9}]}

    monkeypatch.setenv("VEXOR_REMOTE_RERANK_API_KEY", "env-remote-key")
    monkeypatch.setattr("vexor.cache.load_index_vectors", fake_load_index_vectors)
    monkeypatch.setattr("vexor.services.search_service.is_cache_current", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "vexor.services.search_service._remote_rerank_request",
        fake_remote_rerank_request,
    )
    import importlib

    search_module = importlib.import_module("vexor.search")
    monkeypatch.setattr(search_module, "VexorSearcher", DummySearcher)

    request = SearchRequest(
        query="alpha",
        directory=tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=1,
        model_name="model",
        batch_size=0,
        provider="gemini",
        base_url=None,
        api_key="k",
        local_cuda=False,
        exclude_patterns=(),
        extensions=(),
        auto_index=True,
        rerank="remote",
        remote_rerank=RemoteRerankConfig(
            base_url="https://api.example.test/v1/rerank",
            api_key=None,
            model="rerank-model",
        ),
    )
    response = perform_search(request)

    assert response.reranker == "remote"
    assert response.results[0].path.name == "b.txt"
