import json

from vexor import config as config_module


def _prepare_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_file = config_dir / "config.json"
    monkeypatch.setattr(config_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    return config_file


def test_load_config_defaults(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    cfg = config_module.load_config()

    assert cfg.provider == config_module.DEFAULT_PROVIDER
    assert cfg.base_url is None
    assert cfg.auto_index is True
    assert cfg.local_cuda is False
    assert cfg.embed_concurrency == config_module.DEFAULT_EMBED_CONCURRENCY
    assert cfg.rerank == config_module.DEFAULT_RERANK


def test_resolve_default_model_gemini_defaults() -> None:
    assert (
        config_module.resolve_default_model("gemini", None)
        == config_module.DEFAULT_GEMINI_MODEL
    )
    assert (
        config_module.resolve_default_model("gemini", config_module.DEFAULT_MODEL)
        == config_module.DEFAULT_GEMINI_MODEL
    )


def test_set_provider_and_base_url(tmp_path, monkeypatch):
    config_file = _prepare_config(tmp_path, monkeypatch)

    config_module.set_provider("gemini")
    config_module.set_base_url("https://proxy.example.com")

    stored = json.loads(config_file.read_text())
    assert stored["provider"] == "gemini"
    assert stored["base_url"] == "https://proxy.example.com"
    assert stored["auto_index"] is True

    config_module.set_base_url(None)
    cfg = config_module.load_config()
    assert cfg.base_url is None


def test_save_and_load_auto_index(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    config_module.save_config(config_module.Config(auto_index=False))
    cfg = config_module.load_config()
    assert cfg.auto_index is False


def test_save_and_load_local_cuda(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    config_module.save_config(config_module.Config(local_cuda=True))
    cfg = config_module.load_config()
    assert cfg.local_cuda is True


def test_save_and_load_embed_concurrency(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    config_module.save_config(config_module.Config(embed_concurrency=4))
    cfg = config_module.load_config()
    assert cfg.embed_concurrency == 4


def test_save_and_load_rerank(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    config_module.save_config(config_module.Config(rerank="bm25"))
    cfg = config_module.load_config()
    assert cfg.rerank == "bm25"


def test_resolve_api_key_prefers_config(monkeypatch):
    assert config_module.resolve_api_key("cfg-key", "gemini") == "cfg-key"


def test_resolve_api_key_env_fallback(monkeypatch):
    monkeypatch.delenv(config_module.ENV_API_KEY, raising=False)
    monkeypatch.setenv(config_module.OPENAI_ENV, "env-openai")
    assert config_module.resolve_api_key(None, "openai") == "env-openai"


def test_resolve_api_key_custom_uses_openai_env(monkeypatch):
    monkeypatch.delenv(config_module.ENV_API_KEY, raising=False)
    monkeypatch.setenv(config_module.OPENAI_ENV, "env-openai")
    assert config_module.resolve_api_key(None, "custom") == "env-openai"


def test_resolve_api_key_general_env(monkeypatch):
    monkeypatch.setenv(config_module.ENV_API_KEY, "shared-key")
    assert config_module.resolve_api_key(None, "gemini") == "shared-key"


def test_resolve_api_key_legacy_gemini_env(monkeypatch):
    monkeypatch.delenv(config_module.ENV_API_KEY, raising=False)
    monkeypatch.delenv(config_module.OPENAI_ENV, raising=False)
    monkeypatch.setenv(config_module.LEGACY_GEMINI_ENV, "legacy-gemini")
    assert config_module.resolve_api_key(None, "gemini") == "legacy-gemini"


def test_resolve_api_key_local_ignores_keys(monkeypatch):
    monkeypatch.setenv(config_module.ENV_API_KEY, "shared-key")
    assert config_module.resolve_api_key("cfg-key", "local") is None
