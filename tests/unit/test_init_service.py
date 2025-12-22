from __future__ import annotations

from vexor.services.init_service import should_auto_run_init


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
