from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vexor import modes


def test_get_strategy_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported mode"):
        modes.get_strategy("nope")


def test_available_modes_contains_auto_and_is_sorted():
    available = modes.available_modes()
    assert "auto" in available
    assert available == sorted(available)


def test_trim_preview_truncates():
    text = "x" * 500
    trimmed = modes._trim_preview(text, limit=10)  # type: ignore[attr-defined]
    assert trimmed.endswith("…")
    assert len(trimmed) == 10


def test_protocol_methods_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        modes.IndexModeStrategy.payloads_for_files(None, [])  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        modes.IndexModeStrategy.payload_for_file(None, Path("x.txt"))  # type: ignore[arg-type]


def test_normalize_preview_chunk_returns_none_for_whitespace():
    assert modes._normalize_preview_chunk(" \n\t ") is None  # type: ignore[attr-defined]


def test_chunk_text_empty_and_multi_window():
    assert modes._chunk_text(" \n", chunk_size=10, overlap=0) == []  # type: ignore[attr-defined]
    assert modes._chunk_text("abcdefghijklmno", chunk_size=10, overlap=0) == [  # type: ignore[attr-defined]
        "abcdefghij",
        "klmno",
    ]


def test_outline_strategy_handles_empty_text(monkeypatch, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n", encoding="utf-8")

    monkeypatch.setattr(
        modes,
        "extract_outline_chunks",
        lambda *_a, **_k: [SimpleNamespace(breadcrumb="Title", text="", start_line=1, end_line=1)],
    )

    strategy = modes.OutlineStrategy()
    payloads = strategy.payloads_for_files([path])

    assert payloads
    assert payloads[0].preview == "Title"


def test_auto_strategy_stat_error_falls_back_to_head(monkeypatch, tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")

    original_stat = Path.stat

    def fake_stat(self: Path):
        if self == target:
            raise OSError("boom")
        return original_stat(self)

    monkeypatch.setattr(Path, "stat", fake_stat)

    strategy = modes.AutoStrategy()
    payload = strategy.payload_for_file(target)
    assert payload.label.startswith("note.txt ::")
    assert "[#1]" not in payload.label


def test_auto_strategy_payload_for_file_fallback(monkeypatch):
    monkeypatch.setattr(modes.AutoStrategy, "_payloads_for_file", lambda *_args, **_kwargs: [])
    strategy = modes.AutoStrategy()
    payload = strategy.payload_for_file(Path("x.txt"))
    assert payload.label == "x.txt"
