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


def test_set_provider_and_base_url(tmp_path, monkeypatch):
    config_file = _prepare_config(tmp_path, monkeypatch)

    config_module.set_provider("gemini")
    config_module.set_base_url("https://proxy.example.com")

    stored = json.loads(config_file.read_text())
    assert stored["provider"] == "gemini"
    assert stored["base_url"] == "https://proxy.example.com"

    config_module.set_base_url(None)
    cfg = config_module.load_config()
    assert cfg.base_url is None


def test_resolve_api_key_prefers_config(monkeypatch):
    assert config_module.resolve_api_key("cfg-key", "gemini") == "cfg-key"


def test_resolve_api_key_env_fallback(monkeypatch):
    monkeypatch.delenv(config_module.ENV_API_KEY, raising=False)
    monkeypatch.setenv(config_module.OPENAI_ENV, "env-openai")
    assert config_module.resolve_api_key(None, "openai") == "env-openai"


def test_resolve_api_key_general_env(monkeypatch):
    monkeypatch.setenv(config_module.ENV_API_KEY, "shared-key")
    assert config_module.resolve_api_key(None, "gemini") == "shared-key"


def test_resolve_api_key_legacy_gemini_env(monkeypatch):
    monkeypatch.delenv(config_module.ENV_API_KEY, raising=False)
    monkeypatch.delenv(config_module.OPENAI_ENV, raising=False)
    monkeypatch.setenv(config_module.LEGACY_GEMINI_ENV, "legacy-gemini")
    assert config_module.resolve_api_key(None, "gemini") == "legacy-gemini"
