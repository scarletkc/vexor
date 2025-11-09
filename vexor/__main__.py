"""Entry point for `python -m vexor` and frozen builds."""

from __future__ import annotations

try:
    # Normal package execution path
    from .cli import run
except ImportError:  # pragma: no cover - happens in frozen single-file builds
    from vexor.cli import run  # type: ignore[import]


def main() -> None:
    """Execute the Typer application."""
    run()


if __name__ == "__main__":
    raise SystemExit(main())
