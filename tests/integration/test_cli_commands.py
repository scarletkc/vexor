from __future__ import annotations

import pytest
from typer.testing import CliRunner

from vexor.cli import app


def test_doctor_reports_found(monkeypatch):
    runner = CliRunner()
    # Mock all doctor checks to pass
    from vexor.services.system_service import DoctorCheckResult
    mock_results = [
        DoctorCheckResult(name="Command", passed=True, message="`vexor` found at /bin/vexor"),
        DoctorCheckResult(name="Config", passed=True, message="Config exists"),
        DoctorCheckResult(name="Cache Dir", passed=True, message="Cache writable"),
        DoctorCheckResult(name="API Key", passed=True, message="API key configured"),
    ]
    monkeypatch.setattr("vexor.cli.run_all_doctor_checks", lambda **kw: mock_results)
    result = runner.invoke(app, ["doctor", "--skip-api-test"])
    assert result.exit_code == 0
    assert "passed" in result.stdout.lower()


def test_doctor_reports_missing(monkeypatch):
    runner = CliRunner()
    # Mock doctor checks with command missing
    from vexor.services.system_service import DoctorCheckResult
    mock_results = [
        DoctorCheckResult(name="Command", passed=False, message="`vexor` not found on PATH"),
        DoctorCheckResult(name="Config", passed=True, message="Config exists"),
        DoctorCheckResult(name="Cache Dir", passed=True, message="Cache writable"),
        DoctorCheckResult(name="API Key", passed=False, message="API key not configured"),
    ]
    monkeypatch.setattr("vexor.cli.run_all_doctor_checks", lambda **kw: mock_results)
    result = runner.invoke(app, ["doctor", "--skip-api-test"])
    assert result.exit_code == 1
    assert "failed" in result.stdout.lower()


def test_update_reports_available(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_latest_pypi_version", lambda *_args, **_kwargs: "99.0.0")
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "new version" in result.stdout.lower()


def test_update_reports_up_to_date(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_latest_pypi_version", lambda *_args, **_kwargs: "0.0.0")
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "latest version" in result.stdout.lower()


def test_update_reports_fetch_error(monkeypatch):
    runner = CliRunner()

    def raise_fetch(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("vexor.cli.fetch_latest_pypi_version", raise_fetch)
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


def test_search_rejects_respect_gitignore_flag(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "search",
            "hello",
            "--path",
            str(tmp_path),
            "--respect-gitignore",
            "--format",
            "porcelain",
        ],
    )
    assert result.exit_code == 2
    assert "no such option" in result.output.lower()
