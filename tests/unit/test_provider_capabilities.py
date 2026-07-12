import pytest

import vexor.config as config_module
import vexor.providers.capabilities as caps
from vexor.providers.capabilities import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL,
    VOYAGE_BASE_URL,
    get_supported_dimensions,
    resolve_api_key,
    resolve_base_url,
    resolve_default_model,
    supports_dimensions,
    validate_embedding_dimensions_for_model,
)


def test_resolve_default_model_direct_import():
    assert resolve_default_model("gemini", None) == DEFAULT_GEMINI_MODEL
    assert (
        resolve_default_model("gemini", "text-embedding-3-small")
        == DEFAULT_GEMINI_MODEL
    )
    assert resolve_default_model("openai", "custom-model") == "custom-model"


def test_resolve_base_url_direct_import():
    assert resolve_base_url("voyageai", None) == VOYAGE_BASE_URL
    assert resolve_base_url("openai", "http://x") == "http://x"


def test_resolve_api_key_direct_import(monkeypatch):
    for env_name in (
        caps.ENV_API_KEY,
        caps.LEGACY_GEMINI_ENV,
        caps.OPENAI_ENV,
        caps.VOYAGE_ENV,
    ):
        monkeypatch.delenv(env_name, raising=False)

    assert resolve_api_key(None, "local") is None
    assert resolve_api_key(None, "openai") is None

    monkeypatch.setenv(caps.OPENAI_ENV, "openai-env")
    assert resolve_api_key(None, "openai") == "openai-env"

    monkeypatch.setenv(caps.ENV_API_KEY, "vexor-env")
    assert resolve_api_key(None, "openai") == "vexor-env"
    assert resolve_api_key("configured", "openai") == "configured"


def test_dimension_capabilities_direct_import():
    assert supports_dimensions("text-embedding-3-small") is True
    assert supports_dimensions("custom-model") is False
    assert get_supported_dimensions("voyage-3-lite") == (256, 512, 1024, 2048)
    assert get_supported_dimensions("custom-model") is None


def test_validate_embedding_dimensions_direct_import():
    validate_embedding_dimensions_for_model(None, "custom-model")
    validate_embedding_dimensions_for_model(512, "text-embedding-3-small")

    with pytest.raises(ValueError, match="does not support custom dimensions"):
        validate_embedding_dimensions_for_model(512, "custom-model")
    with pytest.raises(ValueError, match="Dimension 3072 is not supported"):
        validate_embedding_dimensions_for_model(3072, "text-embedding-3-small")


def test_config_re_exports_provider_capabilities_by_identity():
    assert config_module.DEFAULT_MODEL is caps.DEFAULT_MODEL
    assert config_module.SUPPORTED_PROVIDERS is caps.SUPPORTED_PROVIDERS
    assert config_module.resolve_api_key is caps.resolve_api_key
    assert config_module.resolve_default_model is caps.resolve_default_model
