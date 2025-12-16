from __future__ import annotations

import pytest
from typer.testing import CliRunner

from vexor.cli import app


def test_doctor_reports_found(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.find_command_on_path", lambda _: "/bin/vexor")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "available" in result.stdout.lower()


def test_doctor_reports_missing(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.find_command_on_path", lambda _: None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1


def test_update_reports_available(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_remote_version", lambda *_: "99.0.0")
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "new version" in result.stdout.lower()


def test_update_reports_up_to_date(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_remote_version", lambda *_: "0.0.0")
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "latest version" in result.stdout.lower()


def test_update_reports_fetch_error(monkeypatch):
    runner = CliRunner()

    def raise_fetch(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("vexor.cli.fetch_remote_version", raise_fetch)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 1


def test_search_rejects_invalid_mode(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "search",
            "hello",
            "--path",
            str(tmp_path),
            "--mode",
            "nope",
            "--format",
            "porcelain",
        ],
    )
    assert result.exit_code == 2
    assert "invalid value" in result.output.lower()


def test_search_rejects_empty_query(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "search",
            "   ",
            "--path",
            str(tmp_path),
            "--format",
            "porcelain",
        ],
    )
    assert result.exit_code == 1
