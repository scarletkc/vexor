from __future__ import annotations

import os

import pytest

from vexor.services import system_service


class DummyResponse:
    def __init__(self, *, status: int, body: str) -> None:
        self.status = status
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_version_tuple_parses_release_and_suffix():
    assert system_service.version_tuple("1.2.3") == (1, 2, 3, 0)
    assert system_service.version_tuple("1.2") == (1, 2, 0, 0)
    assert system_service.version_tuple("1") == (1, 0, 0, 0)
    assert system_service.version_tuple("1.2.3rc4") == (1, 2, 3, 4)
    assert system_service.version_tuple("  2.0.0b12 ") == (2, 0, 0, 12)
    assert system_service.version_tuple("nonsense") == (0, 0, 0, 0)
    assert system_service.version_tuple("1.2.3.4.5") == (1, 2, 3, 4)


def test_parse_version_orders_prereleases_before_final():
    assert system_service.parse_version("1.0.0a1") < system_service.parse_version("1.0.0b1")
    assert system_service.parse_version("1.0.0b1") < system_service.parse_version("1.0.0rc1")
    assert system_service.parse_version("1.0.0rc1") < system_service.parse_version("1.0.0")


def test_select_latest_version_respects_prerelease_flag():
    versions = ["0.9.2", "0.10.0a1"]
    assert system_service.select_latest_version(versions, include_prerelease=False) == "0.9.2"
    assert system_service.select_latest_version(versions, include_prerelease=True) == "0.10.0a1"


def test_fetch_remote_version_success(monkeypatch):
    def fake_urlopen(url, timeout=10.0):
        assert "example.com" in url
        assert timeout == 0.5
        return DummyResponse(status=200, body="__version__ = '9.9.9'\\n")

    monkeypatch.setattr(system_service.request, "urlopen", fake_urlopen)

    assert system_service.fetch_remote_version("https://example.com/x.py", timeout=0.5) == "9.9.9"


def test_fetch_remote_version_raises_on_non_200(monkeypatch):
    monkeypatch.setattr(
        system_service.request,
        "urlopen",
        lambda *_args, **_kwargs: DummyResponse(status=404, body="not found"),
    )
    with pytest.raises(RuntimeError, match=r"HTTP 404"):
        system_service.fetch_remote_version("https://example.com/x.py")


def test_fetch_remote_version_raises_when_missing_version(monkeypatch):
    monkeypatch.setattr(
        system_service.request,
        "urlopen",
        lambda *_args, **_kwargs: DummyResponse(status=200, body="print('no version')"),
    )
    with pytest.raises(RuntimeError, match="Version string not found"):
        system_service.fetch_remote_version("https://example.com/x.py")


def test_resolve_editor_command_from_env(monkeypatch):
    monkeypatch.setenv("VISUAL", "code --wait")
    assert system_service.resolve_editor_command() == ("code", "--wait")


def test_resolve_editor_command_uses_fallback(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)

    def fake_which(cmd: str):
        if cmd == "nano":
            return "/usr/bin/nano"
        return None

    monkeypatch.setattr(system_service.shutil, "which", fake_which)
    assert system_service.resolve_editor_command() == ("/usr/bin/nano",)


def test_resolve_editor_command_none_when_missing(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setattr(system_service.shutil, "which", lambda *_: None)
    assert system_service.resolve_editor_command() is None


def test_find_command_on_path(monkeypatch):
    monkeypatch.setattr(system_service.shutil, "which", lambda cmd: f"/bin/{cmd}")
    assert system_service.find_command_on_path("vexor") == "/bin/vexor"
