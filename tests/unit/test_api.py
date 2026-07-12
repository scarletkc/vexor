import json

import numpy as np
import pytest

from vexor import api as api_module
from vexor.config import Config, DEFAULT_GEMINI_MODEL, DEFAULT_MODEL, RemoteRerankConfig
from vexor.search import SearchResult
from vexor.services.index_service import IndexResult, IndexStatus
from vexor.services.search_service import SearchResponse


def test_search_uses_config_defaults(tmp_path, monkeypatch) -> None:
    cfg = Config(
        api_key="key",
        model=DEFAULT_MODEL,
        batch_size=7,
        embed_concurrency=3,
        extract_concurrency=4,
        extract_backend="process",
        provider="gemini",
        base_url="https://example.test",
        auto_index=False,
        local_cuda=True,
        rerank="bm25",
        flashrank_model="ms-marco-MultiBERT-L-12",
        remote_rerank=RemoteRerankConfig(
            base_url="https://api.example.test/v1/rerank",
            api_key="remote-key",
            model="rerank-model",
        ),
    )
    monkeypatch.setattr(api_module, "load_config", lambda _directory=None: cfg)
    captured: dict[str, object] = {}

    def fake_perform_search(request):
        captured["request"] = request
        return SearchResponse(
            base_path=tmp_path,
            backend=None,
            results=[SearchResult(path=tmp_path / "file.py", score=0.9)],
            is_stale=False,
            index_empty=False,
        )

    monkeypatch.setattr(api_module, "perform_search", fake_perform_search)

    response = api_module.search("hello", path=tmp_path, mode="name")

    req = captured["request"]
    assert response.results[0].path.name == "file.py"
    assert req.directory == tmp_path.resolve()
    assert req.provider == "gemini"
    assert req.model_name == DEFAULT_GEMINI_MODEL
    assert req.batch_size == 7
    assert req.embed_concurrency == 3
    assert req.extract_concurrency == 4
    assert req.extract_backend == "process"
    assert req.base_url == "https://example.test"
    assert req.api_key == "key"
    assert req.auto_index is False
    assert req.local_cuda is True
    assert req.rerank == "bm25"
    assert req.flashrank_model == "ms-marco-MultiBERT-L-12"
    assert req.remote_rerank is not None
    assert req.remote_rerank.base_url == "https://api.example.test/v1/rerank"


def test_search_overrides_config(tmp_path, monkeypatch) -> None:
    cfg = Config(
        api_key="config-key",
        model=DEFAULT_MODEL,
        batch_size=1,
        embed_concurrency=2,
        extract_concurrency=3,
        extract_backend="thread",
        provider="gemini",
        base_url="https://config.test",
        auto_index=True,
        local_cuda=False,
    )
    monkeypatch.setattr(api_module, "load_config", lambda _directory=None: cfg)
    captured: dict[str, object] = {}

    def fake_perform_search(request):
        captured["request"] = request
        return SearchResponse(
            base_path=tmp_path,
            backend=None,
            results=[],
            is_stale=False,
            index_empty=True,
        )

    monkeypatch.setattr(api_module, "perform_search", fake_perform_search)

    api_module.search(
        "hello",
        path=tmp_path,
        mode="name",
        provider="openai",
        model="text-embedding-3-large",
        batch_size=8,
        embed_concurrency=5,
        extract_concurrency=6,
        extract_backend="process",
        base_url="https://override.test",
        api_key="override-key",
        local_cuda=True,
        embedding_dimensions=512,
        auto_index=False,
    )

    req = captured["request"]
    assert req.provider == "openai"
    assert req.model_name == "text-embedding-3-large"
    assert req.batch_size == 8
    assert req.embed_concurrency == 5
    assert req.extract_concurrency == 6
    assert req.extract_backend == "process"
    assert req.base_url == "https://override.test"
    assert req.api_key == "override-key"
    assert req.auto_index is False
    assert req.local_cuda is True
    assert req.embedding_dimensions == 512


def test_project_config_applies_to_search_index_and_in_memory(
    tmp_path, monkeypatch
) -> None:
    config_dir = tmp_path / "global-config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "batch_size": 4,
            }
        ),
        encoding="utf-8",
    )
    project = tmp_path / "project"
    project.mkdir()
    project_config = project / ".vexor" / "config.json"
    project_config.parent.mkdir()
    project_config.write_text(
        json.dumps(
            {
                "model": "text-embedding-3-large",
                "batch_size": 9,
                "embed_concurrency": 3,
                "extract_concurrency": 2,
                "embedding_dimensions": 512,
                "auto_index": False,
                "rerank": "hybrid",
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_perform_search(request):
        captured["search"] = request
        return SearchResponse(
            base_path=project,
            backend=None,
            results=[],
            is_stale=False,
            index_empty=True,
        )

    def fake_build_index(_directory, **kwargs):
        captured["index"] = kwargs
        return IndexResult(status=IndexStatus.EMPTY)

    def fake_build_index_in_memory(directory, **kwargs):
        captured["memory"] = kwargs
        return [], np.empty((0, 0), dtype=np.float32), {
            "include_hidden": False,
            "respect_gitignore": True,
            "recursive": True,
            "mode": "name",
            "exclude_patterns": (),
            "extensions": (),
            "chunks": [],
        }

    monkeypatch.setattr(api_module, "perform_search", fake_perform_search)
    monkeypatch.setattr(api_module, "build_index", fake_build_index)
    monkeypatch.setattr(
        api_module, "build_index_in_memory", fake_build_index_in_memory
    )

    api_module.search("hello", path=project, mode="name", config_dir=config_dir)
    api_module.index(project, mode="name", config_dir=config_dir)
    api_module.index_in_memory(project, mode="name", config_dir=config_dir)

    search_request = captured["search"]
    assert search_request.model_name == "text-embedding-3-large"
    assert search_request.batch_size == 9
    assert search_request.embed_concurrency == 3
    assert search_request.extract_concurrency == 2
    assert search_request.embedding_dimensions == 512
    assert search_request.auto_index is False
    assert search_request.rerank == "hybrid"
    for key in ("index", "memory"):
        kwargs = captured[key]
        assert kwargs["model_name"] == "text-embedding-3-large"
        assert kwargs["batch_size"] == 9
        assert kwargs["embed_concurrency"] == 3
        assert kwargs["extract_concurrency"] == 2
        assert kwargs["embedding_dimensions"] == 512


def test_api_explicit_and_per_call_config_override_project_config(
    tmp_path, monkeypatch
) -> None:
    config_dir = tmp_path / "global-config"
    config_dir.mkdir()
    project = tmp_path / "project"
    project_config = project / ".vexor" / "config.json"
    project_config.parent.mkdir(parents=True)
    project_config.write_text(
        json.dumps(
            {
                "model": "text-embedding-3-large",
                "batch_size": 9,
                "auto_index": False,
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_perform_search(request):
        captured["request"] = request
        return SearchResponse(
            base_path=project,
            backend=None,
            results=[],
            is_stale=False,
            index_empty=True,
        )

    monkeypatch.setattr(api_module, "perform_search", fake_perform_search)

    api_module.search(
        "hello",
        path=project,
        mode="name",
        config_dir=config_dir,
        config={"batch_size": 18, "auto_index": False},
        model="text-embedding-3-small",
        batch_size=20,
        auto_index=True,
    )

    request = captured["request"]
    assert request.model_name == "text-embedding-3-small"
    assert request.batch_size == 20
    assert request.auto_index is True


def test_runtime_config_overrides_only_its_fields_after_project_config(
    tmp_path, monkeypatch
) -> None:
    config_dir = tmp_path / "global-config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"model": "text-embedding-3-small", "batch_size": 4}),
        encoding="utf-8",
    )
    project = tmp_path / "project"
    project_config = project / ".vexor" / "config.json"
    project_config.parent.mkdir(parents=True)
    project_config.write_text(
        json.dumps({"model": "text-embedding-3-large", "batch_size": 9}),
        encoding="utf-8",
    )
    captured: list[object] = []

    def fake_perform_search(request):
        captured.append(request)
        return SearchResponse(
            base_path=project,
            backend=None,
            results=[],
            is_stale=False,
            index_empty=True,
        )

    monkeypatch.setattr(api_module, "perform_search", fake_perform_search)
    client = api_module.VexorClient(config_dir=config_dir)
    client.set_config_json({"batch_size": 17})

    client.search("hello", path=project, mode="name")

    assert captured[-1].model_name == "text-embedding-3-large"
    assert captured[-1].batch_size == 17

    client.set_config_json({"batch_size": 5}, replace=True)
    client.search("hello", path=project, mode="name")

    assert captured[-1].model_name == DEFAULT_MODEL
    assert captured[-1].batch_size == 5


def test_api_wraps_project_config_errors(tmp_path) -> None:
    config_dir = tmp_path / "global-config"
    config_dir.mkdir()
    project_config = tmp_path / "project" / ".vexor" / "config.json"
    project_config.parent.mkdir(parents=True)
    project_config.write_text(
        json.dumps({"api_key": "must-not-load"}), encoding="utf-8"
    )

    with pytest.raises(api_module.VexorError, match="must not contain: api_key"):
        api_module.search(
            "hello",
            path=project_config.parent.parent,
            mode="name",
            config_dir=config_dir,
        )


def test_set_data_dir_updates_config_and_cache(tmp_path) -> None:
    from vexor import cache as cache_module
    from vexor import config as config_module

    original_config_dir = config_module.CONFIG_DIR
    original_cache_dir = cache_module.CACHE_DIR

    api_module.set_data_dir(tmp_path)
    try:
        assert config_module.CONFIG_DIR == tmp_path
        assert config_module.CONFIG_FILE == tmp_path / "config.json"
        assert cache_module.CACHE_DIR == tmp_path
    finally:
        config_module.set_config_dir(original_config_dir)
        cache_module.set_cache_dir(original_cache_dir)


def test_set_config_json_updates_runtime_config(tmp_path, monkeypatch) -> None:
    from vexor import cache as cache_module
    from vexor import config as config_module

    original_config_dir = config_module.CONFIG_DIR
    original_cache_dir = cache_module.CACHE_DIR

    api_module.set_data_dir(tmp_path)
    try:
        captured: dict[str, object] = {}

        def fake_perform_search(request):
            captured["request"] = request
            return SearchResponse(
                base_path=tmp_path,
                backend=None,
                results=[SearchResult(path=tmp_path / "file.py", score=0.9)],
                is_stale=False,
                index_empty=False,
            )

        monkeypatch.setattr(api_module, "perform_search", fake_perform_search)

        api_module.set_config_json(
            {"provider": "gemini", "api_key": "key", "rerank": "bm25"}
        )
        assert config_module.CONFIG_FILE.exists() is False

        api_module.search("hello", path=tmp_path, mode="name")
        req = captured["request"]
        assert req.provider == "gemini"
        assert req.api_key == "key"
        assert req.rerank == "bm25"
    finally:
        api_module.set_config_json(None)
        config_module.set_config_dir(original_config_dir)
        cache_module.set_cache_dir(original_cache_dir)


def test_set_config_json_rejects_invalid_payload(tmp_path) -> None:
    from vexor import cache as cache_module
    from vexor import config as config_module

    original_config_dir = config_module.CONFIG_DIR
    original_cache_dir = cache_module.CACHE_DIR

    api_module.set_data_dir(tmp_path)
    try:
        with pytest.raises(api_module.VexorError):
            api_module.set_config_json('["nope"]')
    finally:
        config_module.set_config_dir(original_config_dir)
        cache_module.set_cache_dir(original_cache_dir)


def test_search_accepts_config_override(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_perform_search(request):
        captured["request"] = request
        return SearchResponse(
            base_path=tmp_path,
            backend=None,
            results=[SearchResult(path=tmp_path / "file.py", score=0.9)],
            is_stale=False,
            index_empty=False,
        )

    monkeypatch.setattr(api_module, "perform_search", fake_perform_search)

    api_module.search(
        "hello",
        path=tmp_path,
        mode="name",
        config={
            "provider": "gemini",
            "api_key": "key",
            "rerank": "remote",
            "remote_rerank": {
                "base_url": "https://api.example.test/v1",
                "api_key": "remote-key",
                "model": "rerank-model",
            },
        },
    )

    req = captured["request"]
    assert req.provider == "gemini"
    assert req.api_key == "key"
    assert req.rerank == "remote"
    assert req.remote_rerank is not None
    assert req.remote_rerank.base_url == "https://api.example.test/v1/rerank"


def test_search_accepts_config_override_with_embedding_dimensions(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_perform_search(request):
        captured["request"] = request
        return SearchResponse(
            base_path=tmp_path,
            backend=None,
            results=[],
            is_stale=False,
            index_empty=True,
        )

    monkeypatch.setattr(api_module, "perform_search", fake_perform_search)

    api_module.search(
        "hello",
        path=tmp_path,
        mode="name",
        config={
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "key",
            "embedding_dimensions": 1024,
        },
    )

    req = captured["request"]
    assert req.provider == "openai"
    assert req.model_name == "text-embedding-3-small"
    assert req.embedding_dimensions == 1024


def test_search_rejects_invalid_config_override(tmp_path) -> None:
    with pytest.raises(api_module.VexorError):
        api_module.search("hello", path=tmp_path, config='["nope"]')


def test_search_validates_mode_and_query(tmp_path) -> None:
    with pytest.raises(api_module.VexorError):
        api_module.search("   ", path=tmp_path, use_config=False)

    with pytest.raises(api_module.VexorError) as excinfo:
        api_module.search("hello", path=tmp_path, mode="nope", use_config=False)

    assert "Unsupported mode" in str(excinfo.value)


def test_index_normalizes_extensions_and_excludes(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_index(directory, **kwargs):
        captured["directory"] = directory
        captured["kwargs"] = kwargs
        return IndexResult(status=IndexStatus.EMPTY)

    monkeypatch.setattr(api_module, "build_index", fake_build_index)

    api_module.index(
        tmp_path,
        mode="name",
        extensions=".md,.py",
        exclude_patterns="tests/**",
        use_config=False,
    )

    kwargs = captured["kwargs"]
    assert captured["directory"] == tmp_path.resolve()
    assert kwargs["extensions"] == (".md", ".py")
    assert kwargs["exclude_patterns"] == ("tests/**",)


def test_index_passes_embedding_dimensions_to_builder(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_index(directory, **kwargs):
        captured["directory"] = directory
        captured["kwargs"] = kwargs
        return IndexResult(status=IndexStatus.EMPTY)

    monkeypatch.setattr(api_module, "build_index", fake_build_index)

    api_module.index(
        tmp_path,
        mode="name",
        use_config=False,
        embedding_dimensions=1024,
    )

    kwargs = captured["kwargs"]
    assert captured["directory"] == tmp_path.resolve()
    assert kwargs["embedding_dimensions"] == 1024


def test_search_uses_data_dir_override(tmp_path, monkeypatch) -> None:
    from vexor import config as config_module

    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"provider": "gemini", "api_key": "key", "model": "text-embedding-3-small"}'
    )
    captured: dict[str, object] = {}

    def fake_perform_search(request):
        captured["request"] = request
        return SearchResponse(
            base_path=tmp_path,
            backend=None,
            results=[SearchResult(path=tmp_path / "file.py", score=0.9)],
            is_stale=False,
            index_empty=False,
        )

    original_config_dir = config_module.CONFIG_DIR
    monkeypatch.setattr(api_module, "perform_search", fake_perform_search)

    api_module.search("hello", path=tmp_path, mode="name", data_dir=tmp_path)

    req = captured["request"]
    assert req.provider == "gemini"
    assert config_module.CONFIG_DIR == original_config_dir


def test_client_config_context_scopes_runtime_config(tmp_path, monkeypatch) -> None:
    base_config = Config(provider="openai")
    monkeypatch.setattr(
        api_module, "load_config", lambda _directory=None: base_config
    )
    captured: list[object] = []

    def fake_perform_search(request):
        captured.append(request)
        return SearchResponse(
            base_path=tmp_path,
            backend=None,
            results=[SearchResult(path=tmp_path / "file.py", score=0.9)],
            is_stale=False,
            index_empty=False,
        )

    monkeypatch.setattr(api_module, "perform_search", fake_perform_search)

    client = api_module.VexorClient()
    client.search("hello", path=tmp_path, mode="name")
    assert captured[-1].provider == "openai"

    with client.config_context({"provider": "gemini", "api_key": "key"}):
        client.search("hello", path=tmp_path, mode="name")
        assert captured[-1].provider == "gemini"

    client.search("hello", path=tmp_path, mode="name")
    assert captured[-1].provider == "openai"


def test_index_in_memory_builds_index(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_index_in_memory(directory, **kwargs):
        captured["directory"] = directory
        captured["kwargs"] = kwargs
        paths = [directory / "file.py"]
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        metadata = {
            "include_hidden": False,
            "respect_gitignore": True,
            "recursive": True,
            "mode": "name",
            "exclude_patterns": (),
            "extensions": (),
            "chunks": [{"preview": "file.py", "chunk_index": 0}],
        }
        return paths, vectors, metadata

    monkeypatch.setattr(api_module, "build_index_in_memory", fake_build_index_in_memory)

    result = api_module.index_in_memory(
        tmp_path,
        mode="name",
        use_config=False,
        embedding_dimensions=512,
    )

    assert isinstance(result, api_module.InMemoryIndex)
    assert result.base_path == tmp_path.resolve()
    assert result.paths[0].name == "file.py"
    assert captured["directory"] == tmp_path.resolve()
    assert captured["kwargs"]["no_cache"] is True
    assert captured["kwargs"]["embedding_dimensions"] == 512
    assert result.embedding_dimensions == 512


def test_in_memory_index_search_uses_search_from_vectors(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_search_from_vectors(request, *, paths, file_vectors, metadata, is_stale=False):
        captured["request"] = request
        return SearchResponse(
            base_path=tmp_path,
            backend=None,
            results=[SearchResult(path=tmp_path / "file.py", score=0.9)],
            is_stale=False,
            index_empty=False,
        )

    monkeypatch.setattr(api_module, "search_from_vectors", fake_search_from_vectors)

    index = api_module.InMemoryIndex(
        base_path=tmp_path,
        paths=[tmp_path / "file.py"],
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        metadata={"mode": "name", "chunks": []},
        model_name="text-embedding-3-small",
        batch_size=1,
        embed_concurrency=1,
        provider="openai",
        base_url=None,
        api_key=None,
        local_cuda=False,
        embedding_dimensions=1536,
    )

    index.search("hello", top=3)

    req = captured["request"]
    assert req.top_k == 3
    assert req.mode == "name"
    assert req.no_cache is True
    assert req.embedding_dimensions == 1536


def test_search_rejects_invalid_embedding_dimensions(tmp_path) -> None:
    with pytest.raises(api_module.VexorError):
        api_module.search(
            "hello",
            path=tmp_path,
            mode="name",
            use_config=False,
            embedding_dimensions=-1,
        )


def test_search_rejects_unsupported_model_for_custom_dimensions(tmp_path) -> None:
    with pytest.raises(api_module.VexorError, match="does not support"):
        api_module.search(
            "hello",
            path=tmp_path,
            mode="name",
            use_config=False,
            model="text-embedding-ada-002",
            embedding_dimensions=512,
        )


def test_search_rejects_unsupported_dimension_for_model(tmp_path) -> None:
    with pytest.raises(api_module.VexorError, match="not supported"):
        api_module.search(
            "hello",
            path=tmp_path,
            mode="name",
            use_config=False,
            model="text-embedding-3-small",
            embedding_dimensions=3072,
        )


def test_config_context_yields_configured_client(tmp_path, monkeypatch) -> None:
    base_config = Config(provider="openai")
    monkeypatch.setattr(
        api_module, "load_config", lambda _directory=None: base_config
    )
    captured: dict[str, object] = {}

    def fake_perform_search(request):
        captured["request"] = request
        return SearchResponse(
            base_path=tmp_path,
            backend=None,
            results=[SearchResult(path=tmp_path / "file.py", score=0.9)],
            is_stale=False,
            index_empty=False,
        )

    monkeypatch.setattr(api_module, "perform_search", fake_perform_search)

    with api_module.config_context({"provider": "gemini", "api_key": "key"}) as client:
        client.search("hello", path=tmp_path, mode="name")

    assert captured["request"].provider == "gemini"


def test_index_explicit_cache_dir_wins_over_project_marker(tmp_path, monkeypatch) -> None:
    from vexor import cache as cache_module

    project = tmp_path / "project"
    (project / ".vexor").mkdir(parents=True)
    (project / "sample.txt").write_text("sample", encoding="utf-8")
    explicit = tmp_path / "explicit"

    def fake_build_index(*_args, **_kwargs):
        db_path = cache_module.cache_db_path()
        db_path.write_text("index", encoding="utf-8")
        return IndexResult(
            status=IndexStatus.STORED,
            files_indexed=1,
            cache_path=db_path,
        )

    monkeypatch.setattr(api_module, "build_index", fake_build_index)

    api_module.index(project, mode="name", use_config=False, cache_dir=explicit)

    assert (explicit / cache_module.DB_FILENAME).is_file()
    assert not (project / ".vexor" / cache_module.DB_FILENAME).exists()


def test_index_local_creates_project_cache_and_uses_it(tmp_path, monkeypatch) -> None:
    from vexor import cache as cache_module

    def fake_build_index(*_args, **_kwargs):
        return IndexResult(
            status=IndexStatus.STORED,
            cache_path=cache_module.cache_db_path(),
        )

    monkeypatch.setattr(api_module, "build_index", fake_build_index)

    result = api_module.index(tmp_path, mode="name", use_config=False, local=True)

    assert (tmp_path / ".vexor" / ".gitignore").is_file()
    expected_cache_path = tmp_path / ".vexor" / cache_module.DB_FILENAME
    assert result.cache_path == expected_cache_path.resolve()
