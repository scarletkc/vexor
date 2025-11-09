"""Entry point for `python -m vexor`."""

from __future__ import annotations

from .cli import run


def main() -> None:
    """Execute the Typer application."""
    run()


if __name__ == "__main__":
    main()
