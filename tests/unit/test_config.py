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
    assert cfg.extract_concurrency == config_module.DEFAULT_EXTRACT_CONCURRENCY
    assert cfg.rerank == config_module.DEFAULT_RERANK
    assert cfg.flashrank_model is None
    assert cfg.remote_rerank is None


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


def test_save_and_load_extract_concurrency(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    config_module.save_config(config_module.Config(extract_concurrency=5))
    cfg = config_module.load_config()
    assert cfg.extract_concurrency == 5


def test_save_and_load_rerank(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    config_module.save_config(config_module.Config(rerank="bm25"))
    cfg = config_module.load_config()
    assert cfg.rerank == "bm25"


def test_save_and_load_flashrank_model(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    config_module.save_config(
        config_module.Config(flashrank_model="ms-marco-MultiBERT-L-12")
    )
    cfg = config_module.load_config()
    assert cfg.flashrank_model == "ms-marco-MultiBERT-L-12"


def test_save_and_load_remote_rerank(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    config_module.save_config(
        config_module.Config(
            remote_rerank=config_module.RemoteRerankConfig(
                base_url="https://api.example.test/v1/rerank",
                api_key="remote-key",
                model="rerank-model",
            )
        )
    )
    cfg = config_module.load_config()
    assert cfg.remote_rerank is not None
    assert cfg.remote_rerank.base_url == "https://api.example.test/v1/rerank"
    assert cfg.remote_rerank.api_key == "remote-key"
    assert cfg.remote_rerank.model == "rerank-model"


def test_normalize_remote_rerank_url_appends_rerank():
    assert (
        config_module.normalize_remote_rerank_url("https://api.example.test/v1")
        == "https://api.example.test/v1/rerank"
    )


def test_normalize_remote_rerank_url_keeps_rerank():
    assert (
        config_module.normalize_remote_rerank_url("https://api.example.test/v1/rerank")
        == "https://api.example.test/v1/rerank"
    )


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


def test_update_config_from_json_merges(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    config_module.save_config(
        config_module.Config(
            provider="openai",
            model=config_module.DEFAULT_MODEL,
            batch_size=3,
            embed_concurrency=4,
            extract_concurrency=5,
            auto_index=True,
            local_cuda=False,
        )
    )

    payload = json.dumps(
        {
            "provider": "gemini",
            "api_key": "key",
            "batch_size": 9,
            "rerank": "remote",
            "remote_rerank": {
                "base_url": "https://api.example.test/v1",
                "api_key": "remote-key",
                "model": "rerank-model",
            },
        }
    )

    config_module.update_config_from_json(payload)

    cfg = config_module.load_config()
    assert cfg.provider == "gemini"
    assert cfg.api_key == "key"
    assert cfg.batch_size == 9
    assert cfg.embed_concurrency == 4
    assert cfg.extract_concurrency == 5
    assert cfg.rerank == "remote"
    assert cfg.remote_rerank is not None
    assert cfg.remote_rerank.base_url == "https://api.example.test/v1/rerank"


def test_resolve_remote_rerank_api_key_prefers_config(monkeypatch):
    monkeypatch.setenv(config_module.REMOTE_RERANK_ENV, "env-key")
    assert config_module.resolve_remote_rerank_api_key("cfg-key") == "cfg-key"


def test_resolve_remote_rerank_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv(config_module.REMOTE_RERANK_ENV, "env-key")
    assert config_module.resolve_remote_rerank_api_key(None) == "env-key"
