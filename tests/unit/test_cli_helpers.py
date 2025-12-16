from __future__ import annotations

import pytest

from vexor import cli
import typer


def test_format_lines_variants():
    assert cli._format_lines(None, None) == "-"  # type: ignore[attr-defined]
    assert cli._format_lines(5, None) == "L5"  # type: ignore[attr-defined]
    assert cli._format_lines(5, 4) == "L5"  # type: ignore[attr-defined]
    assert cli._format_lines(5, 5) == "L5"  # type: ignore[attr-defined]
    assert cli._format_lines(5, 8) == "L5-8"  # type: ignore[attr-defined]


def test_format_extensions_display():
    assert cli._format_extensions_display(None) == "all"  # type: ignore[attr-defined]
    assert cli._format_extensions_display((".py", ".md")) == ".py, .md"  # type: ignore[attr-defined]


def test_validate_mode_rejects_invalid():
    assert cli._validate_mode("auto") == "auto"  # type: ignore[attr-defined]
    with pytest.raises(typer.BadParameter):
        cli._validate_mode("nope")  # type: ignore[attr-defined]
