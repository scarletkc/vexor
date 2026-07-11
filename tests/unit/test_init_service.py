from __future__ import annotations

from vexor.services import init_service
from vexor.services.init_service import should_auto_run_init
from vexor.text import Messages


def test_should_auto_run_init_skips_when_config_exists(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    assert should_auto_run_init([], config_path=config_path, is_tty=True) is False


def test_should_auto_run_init_requires_tty(tmp_path):
    config_path = tmp_path / "config.json"
    assert should_auto_run_init(["search"], config_path=config_path, is_tty=False) is False


def test_should_auto_run_init_skips_help(tmp_path):
    config_path = tmp_path / "config.json"
    assert should_auto_run_init(["--help"], config_path=config_path, is_tty=True) is False


def test_should_auto_run_init_runs_when_missing_config(tmp_path):
    config_path = tmp_path / "config.json"
    assert should_auto_run_init(["search"], config_path=config_path, is_tty=True) is True


def test_should_auto_run_init_skips_mcp_command(tmp_path):
    config_path = tmp_path / "config.json"
    assert should_auto_run_init(["mcp"], config_path=config_path, is_tty=True) is False


def test_collect_remote_settings_voyageai_prompts_voyage_api_key(monkeypatch):
    monkeypatch.setattr(init_service, "_print_step_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(init_service, "_print_option", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(init_service.console, "print", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        init_service,
        "_prompt_choice",
        lambda *_args, **_kwargs: "voyageai",
    )

    captured: dict[str, str] = {}

    def fake_prompt_api_key(prompt: str, provider: str) -> str:
        captured["prompt"] = prompt
        captured["provider"] = provider
        return "voyage-key"

    monkeypatch.setattr(init_service, "_prompt_api_key", fake_prompt_api_key)

    updates = init_service._collect_remote_settings()

    assert updates == {"provider": "voyageai", "api_key": "voyage-key"}
    assert captured == {
        "prompt": Messages.INIT_PROMPT_API_KEY_VOYAGE,
        "provider": "voyageai",
    }


def test_collect_remote_settings_custom_uses_custom_prompts(monkeypatch):
    monkeypatch.setattr(init_service, "_print_step_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(init_service, "_print_option", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(init_service.console, "print", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        init_service,
        "_prompt_choice",
        lambda *_args, **_kwargs: "custom",
    )
    monkeypatch.setattr(
        init_service,
        "_prompt_required",
        lambda prompt: (
            "https://api.example.com/v1"
            if prompt == Messages.INIT_PROMPT_CUSTOM_BASE_URL
            else "embed-model"
        ),
    )
    monkeypatch.setattr(init_service, "_prompt_api_key", lambda *_args, **_kwargs: "custom-key")

    updates = init_service._collect_remote_settings()

    assert updates == {
        "provider": "custom",
        "base_url": "https://api.example.com/v1",
        "model": "embed-model",
        "api_key": "custom-key",
    }
