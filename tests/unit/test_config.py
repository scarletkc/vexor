import json
import pytest

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
    assert cfg.extract_backend == config_module.DEFAULT_EXTRACT_BACKEND
    assert cfg.rerank == config_module.DEFAULT_RERANK
    assert cfg.flashrank_model is None
    assert cfg.remote_rerank is None


def test_config_dir_context_overrides_config_file(tmp_path):
    original_config_dir = config_module.CONFIG_DIR
    with config_module.config_dir_context(tmp_path):
        config_module.save_config(config_module.Config(provider="gemini"))
        loaded = config_module.load_config()
        assert loaded.provider == "gemini"
        assert (tmp_path / "config.json").exists()
    assert config_module.CONFIG_DIR == original_config_dir


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


def test_save_and_load_extract_backend(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)

    config_module.save_config(config_module.Config(extract_backend="process"))
    cfg = config_module.load_config()
    assert cfg.extract_backend == "process"


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
            "extract_backend": "process",
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
    assert cfg.extract_backend == "process"
    assert cfg.rerank == "remote"
    assert cfg.remote_rerank is not None
    assert cfg.remote_rerank.base_url == "https://api.example.test/v1/rerank"


def test_resolve_remote_rerank_api_key_prefers_config(monkeypatch):
    monkeypatch.setenv(config_module.REMOTE_RERANK_ENV, "env-key")
    assert config_module.resolve_remote_rerank_api_key("cfg-key") == "cfg-key"


def test_resolve_remote_rerank_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv(config_module.REMOTE_RERANK_ENV, "env-key")
    assert config_module.resolve_remote_rerank_api_key(None) == "env-key"


# --- Embedding Dimension Tests ---


def test_supports_dimensions_voyage():
    """Test that voyage models are detected as supporting dimensions."""
    assert config_module.supports_dimensions("voyage-3")
    assert config_module.supports_dimensions("voyage-3-large")
    assert config_module.supports_dimensions("voyage-code-3")


def test_supports_dimensions_openai():
    """Test that OpenAI text-embedding-3 models support dimensions."""
    assert config_module.supports_dimensions("text-embedding-3-small")
    assert config_module.supports_dimensions("text-embedding-3-large")


def test_supports_dimensions_unsupported():
    """Test that unsupported models are correctly identified."""
    assert not config_module.supports_dimensions("text-embedding-ada-002")
    assert not config_module.supports_dimensions("some-other-model")


def test_get_supported_dimensions_voyage():
    """Test that voyage models return correct dimension options."""
    dims = config_module.get_supported_dimensions("voyage-3")
    assert dims == (256, 512, 1024, 2048)


def test_get_supported_dimensions_openai():
    """Test that OpenAI models return correct dimension options."""
    small_dims = config_module.get_supported_dimensions("text-embedding-3-small")
    large_dims = config_module.get_supported_dimensions("text-embedding-3-large")
    assert small_dims == (256, 512, 1024, 1536)
    assert large_dims == (256, 512, 1024, 1536, 3072)


def test_get_supported_dimensions_unsupported():
    """Test that unsupported models return None."""
    assert config_module.get_supported_dimensions("text-embedding-ada-002") is None


def test_set_embedding_dimensions_valid(tmp_path, monkeypatch):
    """Test setting a valid dimension for a supported model."""
    config_file = _prepare_config(tmp_path, monkeypatch)
    # First set a model that supports dimensions
    config_module.save_config(config_module.Config(model="voyage-3"))

    config_module.set_embedding_dimensions(512)

    cfg = config_module.load_config()
    assert cfg.embedding_dimensions == 512


def test_set_embedding_dimensions_clears_with_zero(tmp_path, monkeypatch):
    """Test that setting dimension to 0 clears it."""
    _prepare_config(tmp_path, monkeypatch)
    config_module.save_config(config_module.Config(model="voyage-3", embedding_dimensions=512))

    config_module.set_embedding_dimensions(0)

    cfg = config_module.load_config()
    assert cfg.embedding_dimensions is None


def test_set_embedding_dimensions_clears_with_none(tmp_path, monkeypatch):
    """Test that setting dimension to None clears it."""
    _prepare_config(tmp_path, monkeypatch)
    config_module.save_config(config_module.Config(model="voyage-3", embedding_dimensions=512))

    config_module.set_embedding_dimensions(None)

    cfg = config_module.load_config()
    assert cfg.embedding_dimensions is None


def test_set_embedding_dimensions_negative_raises(tmp_path, monkeypatch):
    """Test that negative dimensions raise ValueError."""
    _prepare_config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="non-negative"):
        config_module.set_embedding_dimensions(-1)


def test_set_embedding_dimensions_unsupported_model_raises(tmp_path, monkeypatch):
    """Test that setting dimensions for unsupported model raises ValueError."""
    _prepare_config(tmp_path, monkeypatch)
    config_module.save_config(config_module.Config(model="text-embedding-ada-002"))

    with pytest.raises(ValueError, match="does not support"):
        config_module.set_embedding_dimensions(512)


def test_set_embedding_dimensions_invalid_dimension_raises(tmp_path, monkeypatch):
    """Test that invalid dimension for model raises ValueError."""
    _prepare_config(tmp_path, monkeypatch)
    config_module.save_config(config_module.Config(model="voyage-3"))

    with pytest.raises(ValueError, match="not supported"):
        config_module.set_embedding_dimensions(999)  # Not a valid dimension


def test_set_embedding_dimensions_with_explicit_model(tmp_path, monkeypatch):
    """Test setting dimensions with explicit model override."""
    _prepare_config(tmp_path, monkeypatch)
    # Config has unsupported model, but we pass a supported model explicitly
    config_module.save_config(config_module.Config(model="text-embedding-ada-002"))

    config_module.set_embedding_dimensions(512, model="voyage-3")

    cfg = config_module.load_config()
    assert cfg.embedding_dimensions == 512


def test_set_embedding_dimensions_openai_small_rejects_3072(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)
    config_module.save_config(config_module.Config(model="text-embedding-3-small"))

    with pytest.raises(ValueError, match="not supported"):
        config_module.set_embedding_dimensions(3072)


def test_set_model_rejects_incompatible_existing_dimensions(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)
    config_module.save_config(
        config_module.Config(
            provider="openai",
            model="text-embedding-3-large",
            embedding_dimensions=3072,
        )
    )

    with pytest.raises(ValueError, match="incompatible"):
        config_module.set_model("text-embedding-3-small")


def test_set_provider_rejects_incompatible_existing_dimensions(tmp_path, monkeypatch):
    _prepare_config(tmp_path, monkeypatch)
    config_module.save_config(
        config_module.Config(
            provider="openai",
            model="text-embedding-3-small",
            embedding_dimensions=1536,
        )
    )

    with pytest.raises(ValueError, match="incompatible"):
        config_module.set_provider("gemini")


def test_apply_config_updates_allows_model_change_when_dimensions_cleared(tmp_path, monkeypatch):
    from vexor.services.config_service import apply_config_updates

    _prepare_config(tmp_path, monkeypatch)
    config_module.save_config(
        config_module.Config(
            provider="openai",
            model="text-embedding-3-large",
            embedding_dimensions=3072,
        )
    )

    updates = apply_config_updates(
        model="text-embedding-3-small",
        embedding_dimensions=0,
    )

    cfg = config_module.load_config()
    assert cfg.model == "text-embedding-3-small"
    assert cfg.embedding_dimensions is None
    assert updates.embedding_dimensions_set is False
    assert updates.embedding_dimensions_cleared is True


def test_config_private_coercion_helpers_cover_invalid_inputs(monkeypatch):
    with pytest.raises(ValueError):
        config_module._coerce_config_payload("{")
    with pytest.raises(ValueError):
        config_module._coerce_config_payload(["not", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        config_module._coerce_config_payload("[]")

    assert config_module._coerce_optional_str(None, "api_key") is None
    assert config_module._coerce_optional_str("  ", "api_key") is None
    assert config_module._coerce_optional_str(" value ", "api_key") == "value"
    with pytest.raises(ValueError):
        config_module._coerce_optional_str(1, "api_key")

    assert config_module._coerce_required_str(None, "model", "default") == "default"
    assert config_module._coerce_required_str("  ", "model", "default") == "default"
    assert config_module._coerce_required_str(" model ", "model", "default") == "model"
    with pytest.raises(ValueError):
        config_module._coerce_required_str(False, "model", "default")

    assert config_module._coerce_int(None, "batch_size", 7) == 7
    assert config_module._coerce_int(2, "batch_size", 7) == 2
    assert config_module._coerce_int(2.0, "batch_size", 7) == 2
    assert config_module._coerce_int("", "batch_size", 7) == 7
    assert config_module._coerce_int(" 3 ", "batch_size", 7) == 3
    for value in (True, 2.5, "bad", object()):
        with pytest.raises(ValueError):
            config_module._coerce_int(value, "batch_size", 7)

    assert config_module._coerce_bool(True, "auto_index") is True
    assert config_module._coerce_bool(0, "auto_index") is False
    assert config_module._coerce_bool("yes", "auto_index") is True
    assert config_module._coerce_bool("off", "auto_index") is False
    with pytest.raises(ValueError):
        config_module._coerce_bool("maybe", "auto_index")

    assert config_module._coerce_optional_int(None) is None
    assert config_module._coerce_optional_int(True) is None
    assert config_module._coerce_optional_int(-1) is None
    assert config_module._coerce_optional_int(4) == 4
    assert config_module._coerce_optional_int(4.0) == 4
    assert config_module._coerce_optional_int(4.5) is None
    assert config_module._coerce_optional_int("") is None
    assert config_module._coerce_optional_int("8") == 8
    assert config_module._coerce_optional_int("bad") is None
    assert config_module._coerce_optional_int(object()) is None


def test_config_normalizers_and_remote_rerank_parsing():
    assert config_module._normalize_extract_backend(None) == config_module.DEFAULT_EXTRACT_BACKEND
    assert config_module._normalize_extract_backend("THREAD") == "thread"
    with pytest.raises(ValueError):
        config_module._normalize_extract_backend("bad")

    assert config_module._coerce_extract_backend(None) == config_module.DEFAULT_EXTRACT_BACKEND
    assert config_module._coerce_extract_backend("process") == "process"
    assert config_module._coerce_extract_backend("bad") == config_module.DEFAULT_EXTRACT_BACKEND

    assert config_module._normalize_rerank(None) == config_module.DEFAULT_RERANK
    assert config_module._normalize_rerank(" BM25 ") == "bm25"
    assert config_module._normalize_rerank("bad") == config_module.DEFAULT_RERANK
    with pytest.raises(ValueError):
        config_module._normalize_rerank(3)

    assert config_module._coerce_remote_rerank(None) is None
    remote = config_module._coerce_remote_rerank(
        {
            "base_url": "https://api.example.com",
            "api_key": " secret ",
            "model": " model ",
        }
    )
    assert remote is not None
    assert remote.base_url == "https://api.example.com/rerank"
    assert remote.api_key == "secret"
    assert remote.model == "model"
    with pytest.raises(ValueError):
        config_module._coerce_remote_rerank("bad")
    incomplete = config_module._coerce_remote_rerank({"base_url": "", "model": "m"})
    assert incomplete is not None
    assert incomplete.base_url is None
    assert incomplete.model == "m"


def test_apply_config_payload_and_clone_cover_all_fields():
    config = config_module.Config()
    payload = {
        "api_key": " key ",
        "model": " model ",
        "batch_size": "4",
        "embed_concurrency": 3.0,
        "extract_concurrency": None,
        "extract_backend": "process",
        "provider": " voyageai ",
        "base_url": " https://proxy.example.com ",
        "auto_index": "false",
        "local_cuda": 1,
        "rerank": "remote",
        "flashrank_model": " ranker ",
        "remote_rerank": {
            "base_url": "https://rerank.example.com",
            "api_key": "rk",
            "model": "rm",
        },
        "embedding_dimensions": "512",
    }

    config_module._apply_config_payload(config, payload)
    cloned = config_module._clone_config(config)

    assert cloned.api_key == "key"
    assert cloned.model == "model"
    assert cloned.batch_size == 4
    assert cloned.embed_concurrency == 3
    assert cloned.extract_concurrency == config_module.DEFAULT_EXTRACT_CONCURRENCY
    assert cloned.extract_backend == "process"
    assert cloned.provider == "voyageai"
    assert cloned.base_url == "https://proxy.example.com"
    assert cloned.auto_index is False
    assert cloned.local_cuda is True
    assert cloned.rerank == "remote"
    assert cloned.flashrank_model == "ranker"
    assert cloned.remote_rerank is not None
    assert cloned.remote_rerank.base_url == "https://rerank.example.com/rerank"
    assert cloned.embedding_dimensions == 512
    assert cloned.remote_rerank is not config.remote_rerank


def test_api_key_resolution_voyage_and_empty_remote_key(monkeypatch):
    monkeypatch.delenv(config_module.ENV_API_KEY, raising=False)
    monkeypatch.delenv(config_module.OPENAI_ENV, raising=False)
    monkeypatch.setenv(config_module.VOYAGE_ENV, "voyage-env")
    assert config_module.resolve_api_key(None, "voyageai") == "voyage-env"

    monkeypatch.delenv(config_module.REMOTE_RERANK_ENV, raising=False)
    assert config_module.resolve_remote_rerank_api_key(None) is None
