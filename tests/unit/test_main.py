from __future__ import annotations

import runpy
import sys

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
