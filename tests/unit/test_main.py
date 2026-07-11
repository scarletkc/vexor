from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest


def test_main_invokes_run(monkeypatch):
    import vexor.__main__ as vexor_main

    called = {}

    def fake_run():
        called["ok"] = True

    monkeypatch.setattr("vexor.__main__.run", fake_run)

    vexor_main.main()

    assert called["ok"] is True


def test_module_runs_as_script(monkeypatch):
    import vexor.cli

    called = {"ok": False}

    def fake_run() -> None:
        called["ok"] = True

    monkeypatch.setattr(vexor.cli, "run", fake_run)

    sys.modules.pop("vexor.__main__", None)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("vexor.__main__", run_name="__main__")

    assert called["ok"] is True
    assert exc.value.code is None


def test_main_routes_mcp_without_loading_full_cli(monkeypatch):
    import vexor.__main__ as vexor_main

    called = {}
    monkeypatch.setattr(sys, "argv", ["vexor", "mcp", "--path", "."])
    monkeypatch.setattr(
        vexor_main,
        "_run_mcp",
        lambda args: called.setdefault("args", list(args)),
    )
    monkeypatch.setattr(
        vexor_main,
        "run",
        lambda: (_ for _ in ()).throw(AssertionError("full CLI should stay unloaded")),
    )

    vexor_main.main()

    assert called["args"] == ["--path", "."]


def test_mcp_help_uses_centralized_messages(monkeypatch, capsys):
    import vexor.__main__ as vexor_main
    from vexor.text import Messages

    monkeypatch.setattr(Messages, "HELP_MCP", "Centralized MCP help.")
    monkeypatch.setattr(Messages, "HELP_MCP_PATH", "Centralized path help.")

    with pytest.raises(SystemExit) as exc:
        vexor_main._run_mcp(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "Centralized MCP help." in output
    assert "Centralized path help." in output


def test_mcp_accepts_short_path_alias(monkeypatch, tmp_path):
    import vexor.__main__ as vexor_main
    import vexor.services.mcp_service as mcp_service

    captured = {}
    for name in vexor_main._NUMERIC_THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        mcp_service,
        "serve_stdio",
        lambda path: captured.setdefault("path", path),
    )

    vexor_main._run_mcp(["-p", str(tmp_path)])

    assert captured["path"] == tmp_path


def test_mcp_runs_when_main_file_is_executed_as_script(monkeypatch, tmp_path):
    import vexor.__main__ as vexor_main
    import vexor.services.mcp_service as mcp_service

    captured = {}
    script_path = Path(vexor_main.__file__)
    for name in vexor_main._NUMERIC_THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        mcp_service,
        "serve_stdio",
        lambda path: captured.setdefault("path", path),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script_path), "mcp", "--path", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script_path), run_name="__main__")

    assert exc.value.code is None
    assert captured["path"] == tmp_path


def test_mcp_numeric_threads_respect_explicit_backend_values(monkeypatch, tmp_path):
    import vexor.__main__ as vexor_main
    import vexor.services.mcp_service as mcp_service

    captured = {}
    for name in vexor_main._NUMERIC_THREAD_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VEXOR_MCP_NUM_THREADS", "3")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.setattr(
        mcp_service,
        "serve_stdio",
        lambda path: captured.update(path=path, mkl=os.environ["MKL_NUM_THREADS"]),
    )

    vexor_main._run_mcp(["--path", str(tmp_path)])

    assert captured == {"path": tmp_path, "mkl": "3"}
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"
