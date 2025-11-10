"""Command line interface for Vexor."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    load_config,
)
from .modes import available_modes, get_strategy
from .services.config_service import apply_config_updates, get_config_snapshot
from .services.index_service import IndexStatus, build_index, clear_index_entries
from .services.search_service import SearchRequest, perform_search
from .services.system_service import (
    fetch_remote_version,
    find_command_on_path,
    version_tuple,
)
from .text import Messages, Styles
from .utils import resolve_directory, format_path, ensure_positive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .search import SearchResult

REMOTE_VERSION_URL = "https://raw.githubusercontent.com/scarletkc/vexor/refs/heads/main/vexor/__init__.py"
PROJECT_URL = "https://github.com/scarletkc/vexor"
PYPI_URL = "https://pypi.org/project/vexor/"

console = Console()
app = typer.Typer(
    help=Messages.APP_HELP,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)



def _version_callback(value: bool) -> None:
    if value:
        console.print(f"Vexor v{__version__}")
        raise typer.Exit()


def _validate_mode(mode: str) -> str:
    try:
        get_strategy(mode)
    except ValueError as exc:
        allowed = ", ".join(available_modes())
        raise typer.BadParameter(
            Messages.ERROR_MODE_INVALID.format(value=mode, allowed=allowed)
        ) from exc
    return mode


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    )
) -> None:
    """Global Typer callback for shared options."""
    return None


@app.command()
def search(
    query: str = typer.Argument(..., help=Messages.HELP_QUERY),
    path: Path = typer.Option(
        Path.cwd(),
        "--path",
        "-p",
        help=Messages.HELP_SEARCH_PATH,
    ),
    top: int = typer.Option(5, "--top", "-k", help=Messages.HELP_SEARCH_TOP),
    include_hidden: bool = typer.Option(
        False,
        "--include-hidden",
        help=Messages.HELP_INCLUDE_HIDDEN,
    ),
    mode: str = typer.Option(
        ...,
        "--mode",
        "-m",
        help=Messages.HELP_MODE,
    ),
    no_recursive: bool = typer.Option(
        False,
        "--no-recursive",
        "-n",
        help=Messages.HELP_RECURSIVE,
    ),
) -> None:
    """Run the semantic search using a cached index."""
    config = load_config()
    model_name = config.model or DEFAULT_MODEL
    batch_size = config.batch_size if config.batch_size is not None else DEFAULT_BATCH_SIZE

    clean_query = query.strip()
    if not clean_query:
        console.print(_styled(Messages.ERROR_EMPTY_QUERY, Styles.ERROR))
        raise typer.Exit(code=1)
    try:
        ensure_positive(top, "top")
    except ValueError as exc:  # pragma: no cover - validated by Typer
        raise typer.BadParameter(str(exc), param_name="top") from exc

    directory = resolve_directory(path)
    mode_value = _validate_mode(mode)
    recursive = not no_recursive
    console.print(_styled(Messages.INFO_SEARCH_RUNNING.format(path=directory), Styles.INFO))
    request = SearchRequest(
        query=clean_query,
        directory=directory,
        include_hidden=include_hidden,
        mode=mode_value,
        recursive=recursive,
        top_k=top,
        model_name=model_name,
        batch_size=batch_size,
    )
    try:
        response = perform_search(request)
    except FileNotFoundError:
        console.print(
            _styled(Messages.ERROR_INDEX_MISSING.format(path=directory), Styles.ERROR)
        )
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        console.print(_styled(str(exc), Styles.ERROR))
        raise typer.Exit(code=1)

    if response.index_empty:
        console.print(_styled(Messages.INFO_INDEX_EMPTY, Styles.WARNING))
        raise typer.Exit(code=0)
    if response.is_stale:
        console.print(
            _styled(Messages.WARNING_INDEX_STALE.format(path=directory), Styles.WARNING)
        )
    if not response.results:
        console.print(_styled(Messages.INFO_NO_RESULTS, Styles.WARNING))
        raise typer.Exit(code=0)

    _render_results(response.results, response.base_path, response.backend)


@app.command()
def index(
    path: Path = typer.Option(
        Path.cwd(),
        "--path",
        "-p",
        help=Messages.HELP_INDEX_PATH,
    ),
    include_hidden: bool = typer.Option(
        False,
        "--include-hidden",
        help=Messages.HELP_INDEX_INCLUDE,
    ),
    mode: str = typer.Option(
        ...,
        "--mode",
        "-m",
        help=Messages.HELP_MODE,
    ),
    no_recursive: bool = typer.Option(
        False,
        "--no-recursive",
        "-n",
        help=Messages.HELP_RECURSIVE,
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help=Messages.HELP_INDEX_CLEAR,
    ),
) -> None:
    """Create or refresh the cached index for the given directory."""
    config = load_config()
    model_name = config.model or DEFAULT_MODEL
    batch_size = config.batch_size if config.batch_size is not None else DEFAULT_BATCH_SIZE

    directory = resolve_directory(path)
    mode_value = _validate_mode(mode)
    recursive = not no_recursive
    if clear:
        removed = clear_index_entries(
            directory,
            include_hidden=include_hidden,
            mode=mode_value,
            recursive=recursive,
        )
        if removed:
            plural = "ies" if removed > 1 else "y"
            console.print(
                _styled(
                    Messages.INFO_INDEX_CLEARED.format(
                        path=directory,
                        count=removed,
                        plural=plural,
                    ),
                    Styles.SUCCESS,
                )
            )
        else:
            console.print(
                _styled(
                    Messages.INFO_INDEX_CLEAR_NONE.format(path=directory),
                    Styles.INFO,
                )
            )
        return

    console.print(_styled(Messages.INFO_INDEX_RUNNING.format(path=directory), Styles.INFO))
    result = build_index(
        directory,
        include_hidden=include_hidden,
        mode=mode_value,
        recursive=recursive,
        model_name=model_name,
        batch_size=batch_size,
    )
    if result.status == IndexStatus.EMPTY:
        console.print(_styled(Messages.INFO_NO_FILES, Styles.WARNING))
        raise typer.Exit(code=0)
    if result.status == IndexStatus.UP_TO_DATE:
        console.print(
            _styled(Messages.INFO_INDEX_UP_TO_DATE.format(path=directory), Styles.INFO)
        )
        return
    if result.cache_path is not None:
        console.print(_styled(Messages.INFO_INDEX_SAVED.format(path=result.cache_path), Styles.SUCCESS))


@app.command()
def config(
    set_api_key_option: str | None = typer.Option(
        None,
        "--set-api-key",
        help=Messages.HELP_SET_API_KEY,
    ),
    clear_api_key: bool = typer.Option(
        False,
        "--clear-api-key",
        help=Messages.HELP_CLEAR_API_KEY,
    ),
    set_model_option: str | None = typer.Option(
        None,
        "--set-model",
        help=Messages.HELP_SET_MODEL,
    ),
    set_batch_option: int | None = typer.Option(
        None,
        "--set-batch-size",
        help=Messages.HELP_SET_BATCH,
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help=Messages.HELP_SHOW_CONFIG,
    ),
) -> None:
    """Manage Vexor configuration stored in ~/.vexor/config.json."""
    if set_batch_option is not None and set_batch_option < 0:
        raise typer.BadParameter(Messages.ERROR_BATCH_NEGATIVE)

    updates = apply_config_updates(
        api_key=set_api_key_option,
        clear_api_key=clear_api_key,
        model=set_model_option,
        batch_size=set_batch_option,
    )

    if updates.api_key_set:
        console.print(_styled(Messages.INFO_API_SAVED, Styles.SUCCESS))
    if updates.api_key_cleared:
        console.print(_styled(Messages.INFO_API_CLEARED, Styles.SUCCESS))
    if updates.model_set and set_model_option is not None:
        console.print(
            _styled(Messages.INFO_MODEL_SET.format(value=set_model_option), Styles.SUCCESS)
        )
    if updates.batch_size_set and set_batch_option is not None:
        console.print(
            _styled(Messages.INFO_BATCH_SET.format(value=set_batch_option), Styles.SUCCESS)
        )

    if show or not updates.changed:
        cfg = get_config_snapshot()
        console.print(
            _styled(
                Messages.INFO_CONFIG_SUMMARY.format(
                    api="yes" if cfg.api_key else "no",
                    model=cfg.model or DEFAULT_MODEL,
                    batch=cfg.batch_size if cfg.batch_size is not None else DEFAULT_BATCH_SIZE,
                ),
                Styles.INFO,
            )
        )


@app.command()
def doctor() -> None:
    """Check whether the `vexor` command is available on PATH."""
    console.print(_styled(Messages.INFO_DOCTOR_CHECKING, Styles.INFO))
    command_path = find_command_on_path("vexor")
    if command_path:
        console.print(
            _styled(Messages.INFO_DOCTOR_FOUND.format(path=command_path), Styles.SUCCESS)
        )
        return
    console.print(_styled(Messages.ERROR_DOCTOR_MISSING, Styles.ERROR))
    raise typer.Exit(code=1)


@app.command()
def update() -> None:
    """Check whether a newer release is available online."""
    console.print(_styled(Messages.INFO_UPDATE_CHECKING, Styles.INFO))
    console.print(_styled(Messages.INFO_UPDATE_CURRENT.format(current=__version__), Styles.INFO))
    try:
        latest = fetch_remote_version(REMOTE_VERSION_URL)
    except RuntimeError as exc:
        console.print(
            _styled(Messages.ERROR_UPDATE_FETCH.format(reason=str(exc)), Styles.ERROR)
        )
        raise typer.Exit(code=1)

    if version_tuple(latest) > version_tuple(__version__):
        console.print(
            _styled(
                Messages.INFO_UPDATE_AVAILABLE.format(
                    latest=latest,
                    github=PROJECT_URL,
                    pypi=PYPI_URL,
                ),
                Styles.WARNING,
            )
        )
        return

    console.print(
        _styled(Messages.INFO_UPDATE_UP_TO_DATE.format(latest=latest), Styles.SUCCESS)
    )


def _render_results(results: Sequence["SearchResult"], base: Path, backend: str | None) -> None:
    console.print(_styled(Messages.TABLE_TITLE, Styles.TITLE))
    if backend:
        console.print(_styled(f"{Messages.TABLE_BACKEND_PREFIX}{backend}", Styles.INFO))
    table = Table(show_header=True, header_style=Styles.TABLE_HEADER)
    table.add_column(Messages.TABLE_HEADER_INDEX, justify="right")
    table.add_column(Messages.TABLE_HEADER_SIMILARITY, justify="right")
    table.add_column(Messages.TABLE_HEADER_PATH, overflow="fold")
    for idx, result in enumerate(results, start=1):
        table.add_row(
            str(idx),
            f"{result.score:.3f}",
            format_path(result.path, base),
        )
    console.print(table)


def _styled(text: str, style: str) -> str:
    return f"[{style}]{text}[/{style}]"


def run(argv: list[str] | None = None) -> None:
    """Entry point wrapper allowing optional argument override."""
    if argv is None:
        app()
    else:
        app(args=list(argv))
