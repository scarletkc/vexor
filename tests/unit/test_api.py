import pytest

from vexor import api as api_module
from vexor.config import Config, DEFAULT_GEMINI_MODEL, DEFAULT_MODEL
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
