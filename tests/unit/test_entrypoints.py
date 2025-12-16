from __future__ import annotations

from typer.testing import CliRunner

import vexor
from vexor.cli import app


def test_get_version_matches_dunder():
    assert vexor.get_version() == vexor.__version__


def test_module_main_calls_run(monkeypatch):
    import vexor.__main__ as main_mod

    called = {"ok": False}

    def fake_run():
        called["ok"] = True

    monkeypatch.setattr(main_mod, "run", fake_run)
    main_mod.main()
    assert called["ok"] is True


def test_cli_version_flag_prints_version():
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Vexor v" in result.stdout
