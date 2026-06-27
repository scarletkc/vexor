from __future__ import annotations

from types import SimpleNamespace

from vexor import output


def test_encoding_supports_handles_missing_success_and_failure():
    assert output._encoding_supports("x", None) is False
    assert output._encoding_supports("✓", "utf-8") is True
    assert output._encoding_supports("✓", "ascii") is False


def test_supports_unicode_output_prefers_console_then_stdout(monkeypatch):
    assert output.supports_unicode_output(SimpleNamespace(encoding="utf-8")) is True

    monkeypatch.setattr(output.sys, "stdout", SimpleNamespace(encoding="ascii"))
    assert output.supports_unicode_output(SimpleNamespace(encoding="ascii")) is False


def test_format_status_icon_unicode_and_ascii(monkeypatch):
    monkeypatch.setattr(output, "supports_unicode_output", lambda console=None: True)
    assert "✓" in output.format_status_icon(True)
    assert "✗" in output.format_status_icon(False)

    monkeypatch.setattr(output, "supports_unicode_output", lambda console=None: False)
    assert output.format_status_icon(True) == "[green]OK[/green]"
    assert output.format_status_icon(False) == "[red]X[/red]"
