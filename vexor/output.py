"""Helpers for formatting CLI output safely across terminals."""

from __future__ import annotations

import sys

from rich.console import Console


def _encoding_supports(text: str, encoding: str | None) -> bool:
    if not encoding:
        return False
    try:
        text.encode(encoding)
    except Exception:
        return False
    return True


def supports_unicode_output(console: Console | None = None) -> bool:
    sample = "\u2713\u2717"
    if console is not None and _encoding_supports(sample, console.encoding):
        return True
    return _encoding_supports(sample, sys.stdout.encoding)


def format_status_icon(passed: bool, console: Console | None = None) -> str:
    if supports_unicode_output(console):
        return "[green]\u2713[/green]" if passed else "[red]\u2717[/red]"
    return "[green]OK[/green]" if passed else "[red]X[/red]"
