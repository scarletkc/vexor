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
    monkeypatch.setattr(api_module, "load_config", lambda: cfg)
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
        provider="gemini",
        base_url="https://config.test",
        auto_index=True,
        local_cuda=False,
    )
    monkeypatch.setattr(api_module, "load_config", lambda: cfg)
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
        base_url="https://override.test",
        api_key="override-key",
        local_cuda=True,
        auto_index=False,
    )

    req = captured["request"]
    assert req.provider == "openai"
    assert req.model_name == "text-embedding-3-large"
    assert req.batch_size == 8
    assert req.embed_concurrency == 5
    assert req.base_url == "https://override.test"
    assert req.api_key == "override-key"
    assert req.auto_index is False
    assert req.local_cuda is True


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
