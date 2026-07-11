"""Entry point for `python -m vexor` and frozen builds."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

_NUMERIC_THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def run() -> None:
    """Load and execute the full Typer CLI."""

    try:
        from .cli import run as run_cli
    except ImportError:  # pragma: no cover - frozen single-file builds
        from vexor.cli import run as run_cli  # type: ignore[import]
    run_cli()


def _run_mcp(args: Sequence[str]) -> None:
    """Parse the small MCP command without importing the full CLI stack."""

    try:
        from .text import Messages
    except ImportError:  # pragma: no cover - frozen single-file builds
        from vexor.text import Messages

    parser = argparse.ArgumentParser(
        prog="vexor mcp",
        description=Messages.HELP_MCP,
    )
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=Path.cwd(),
        help=Messages.HELP_MCP_PATH,
    )
    options = parser.parse_args(list(args))

    # Matrix-vector search rarely benefits from a machine-wide BLAS thread
    # pool, while every reserved worker stack is charged to each MCP process.
    # Respect explicit backend settings and allow one shared Vexor override.
    numeric_threads = os.environ.get("VEXOR_MCP_NUM_THREADS", "2").strip()
    if numeric_threads.isdigit() and int(numeric_threads) > 0:
        for name in _NUMERIC_THREAD_ENV:
            os.environ.setdefault(name, numeric_threads)

    try:
        from .services.mcp_service import serve_stdio
    except ImportError:  # pragma: no cover - frozen single-file builds
        from vexor.services.mcp_service import serve_stdio

    serve_stdio(options.path)


def main() -> None:
    """Execute the Typer application."""

    args = sys.argv[1:]
    if args and args[0] == "mcp":
        _run_mcp(args[1:])
        return
    run()


if __name__ == "__main__":
    raise SystemExit(main())
