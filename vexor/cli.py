"""Command line interface for Vexor."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, config as config_module
from .cache import clear_all_cache, list_cache_entries
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    load_config,
)
from .modes import available_modes, get_strategy
from .services.cache_service import load_index_metadata_safe
from .services.config_service import apply_config_updates, get_config_snapshot
from .services.index_service import IndexStatus, build_index, clear_index_entries
from .services.search_service import SearchRequest, perform_search
from .services.system_service import (
    fetch_remote_version,
    find_command_on_path,
    resolve_editor_command,
    version_tuple,
)
from .text import Messages, Styles
from .utils import resolve_directory, format_path, ensure_positive, normalize_extensions

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


def _format_extensions_display(values: Sequence[str] | None) -> str:
    if not values:
        return "all"
    return ", ".join(values)


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
        "-i",
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
    extensions: list[str] | None = typer.Option(
        None,
        "--ext",
        "-e",
        help=Messages.HELP_EXTENSIONS,
    ),
) -> None:
    """Run the semantic search using a cached index."""
    config = load_config()
    model_name = config.model or DEFAULT_MODEL
    batch_size = config.batch_size if config.batch_size is not None else DEFAULT_BATCH_SIZE
    provider = config.provider or DEFAULT_PROVIDER
    base_url = config.base_url
    api_key = config.api_key

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
    normalized_exts = normalize_extensions(extensions)
    if extensions and not normalized_exts:
        raise typer.BadParameter(Messages.ERROR_EXTENSIONS_EMPTY, param_name="ext")
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
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        extensions=normalized_exts,
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
        "-i",
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
    show_cache: bool = typer.Option(
        False,
        "--show",
        help=Messages.HELP_INDEX_SHOW,
    ),
    extensions: list[str] | None = typer.Option(
        None,
        "--ext",
        "-e",
        help=Messages.HELP_EXTENSIONS,
    ),
) -> None:
    """Create or refresh the cached index for the given directory."""
    config = load_config()
    model_name = config.model or DEFAULT_MODEL
    batch_size = config.batch_size if config.batch_size is not None else DEFAULT_BATCH_SIZE
    provider = config.provider or DEFAULT_PROVIDER
    base_url = config.base_url
    api_key = config.api_key

    directory = resolve_directory(path)
    mode_value = _validate_mode(mode)
    recursive = not no_recursive
    normalized_exts = normalize_extensions(extensions)
    if extensions and not normalized_exts:
        raise typer.BadParameter(Messages.ERROR_EXTENSIONS_EMPTY, param_name="ext")
    if clear and show_cache:
        raise typer.BadParameter(Messages.ERROR_INDEX_SHOW_CONFLICT)

    if show_cache:
        metadata = load_index_metadata_safe(
            directory,
            model_name,
            include_hidden,
            mode_value,
            recursive,
            extensions=normalized_exts,
        )
        if not metadata:
            console.print(
                _styled(
                    Messages.INFO_INDEX_CLEAR_NONE.format(path=directory),
                    Styles.INFO,
                )
            )
            return

        files = metadata.get("files", [])
        console.print(
            _styled(Messages.INFO_INDEX_SHOW_HEADER.format(path=directory), Styles.TITLE)
        )
        summary = Messages.INFO_INDEX_SHOW_SUMMARY.format(
            mode=metadata.get("mode"),
            model=metadata.get("model"),
            hidden="yes" if metadata.get("include_hidden") else "no",
            recursive="yes" if metadata.get("recursive") else "no",
            extensions=_format_extensions_display(metadata.get("extensions")),
            files=len(files),
            dimension=metadata.get("dimension"),
            version=metadata.get("version"),
            generated=metadata.get("generated_at"),
        )
        console.print(_styled(summary, Styles.INFO))
        return

    if clear:
        removed = clear_index_entries(
            directory,
            include_hidden=include_hidden,
            mode=mode_value,
            recursive=recursive,
            extensions=normalized_exts,
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
    try:
        result = build_index(
            directory,
            include_hidden=include_hidden,
            mode=mode_value,
            recursive=recursive,
            model_name=model_name,
            batch_size=batch_size,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            extensions=normalized_exts,
        )
    except RuntimeError as exc:
        console.print(_styled(str(exc), Styles.ERROR))
        raise typer.Exit(code=1)
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
    set_provider_option: str | None = typer.Option(
        None,
        "--set-provider",
        help=Messages.HELP_SET_PROVIDER,
    ),
    set_base_url_option: str | None = typer.Option(
        None,
        "--set-base-url",
        help=Messages.HELP_SET_BASE_URL,
    ),
    clear_base_url: bool = typer.Option(
        False,
        "--clear-base-url",
        help=Messages.HELP_CLEAR_BASE_URL,
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help=Messages.HELP_SHOW_CONFIG,
    ),
    show_index_all: bool = typer.Option(
        False,
        "--show-index-all",
        help=Messages.HELP_SHOW_INDEX_ALL,
    ),
    clear_index_all: bool = typer.Option(
        False,
        "--clear-index-all",
        help=Messages.HELP_CLEAR_INDEX_ALL,
    ),
) -> None:
    """Manage Vexor configuration stored in ~/.vexor/config.json."""
    if set_batch_option is not None and set_batch_option < 0:
        raise typer.BadParameter(Messages.ERROR_BATCH_NEGATIVE)
    if set_base_url_option and clear_base_url:
        raise typer.BadParameter(Messages.ERROR_BASE_URL_CONFLICT)
    if set_provider_option is not None:
        normalized_provider = set_provider_option.strip().lower()
        if normalized_provider not in SUPPORTED_PROVIDERS:
            allowed = ", ".join(SUPPORTED_PROVIDERS)
            raise typer.BadParameter(
                Messages.ERROR_PROVIDER_INVALID.format(
                    value=set_provider_option, allowed=allowed
                )
            )
        set_provider_option = normalized_provider

    updates = apply_config_updates(
        api_key=set_api_key_option,
        clear_api_key=clear_api_key,
        model=set_model_option,
        batch_size=set_batch_option,
        provider=set_provider_option,
        base_url=set_base_url_option,
        clear_base_url=clear_base_url,
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
    if updates.provider_set and set_provider_option is not None:
        console.print(
            _styled(Messages.INFO_PROVIDER_SET.format(value=set_provider_option), Styles.SUCCESS)
        )
    if updates.base_url_set and set_base_url_option is not None:
        console.print(
            _styled(Messages.INFO_BASE_URL_SET.format(value=set_base_url_option), Styles.SUCCESS)
        )
    if updates.base_url_cleared and clear_base_url:
        console.print(_styled(Messages.INFO_BASE_URL_CLEARED, Styles.SUCCESS))

    if clear_index_all:
        removed = clear_all_cache()
        if removed:
            plural = "ies" if removed > 1 else "y"
            console.print(
                _styled(
                    Messages.INFO_INDEX_ALL_CLEARED.format(count=removed, plural=plural),
                    Styles.SUCCESS,
                )
            )
        else:
            console.print(_styled(Messages.INFO_INDEX_ALL_CLEAR_NONE, Styles.INFO))

    should_edit = not any(
        (
            updates.changed,
            show,
            show_index_all,
            clear_index_all,
        )
    )
    if should_edit:
        _edit_config_file()
        return

    if show:
        cfg = get_config_snapshot()
        console.print(
            _styled(
                Messages.INFO_CONFIG_SUMMARY.format(
                    api="yes" if cfg.api_key else "no",
                    provider=cfg.provider or DEFAULT_PROVIDER,
                    model=cfg.model or DEFAULT_MODEL,
                    batch=cfg.batch_size if cfg.batch_size is not None else DEFAULT_BATCH_SIZE,
                    base_url=cfg.base_url or "none",
                ),
                Styles.INFO,
            )
        )

    if show_index_all:
        entries = list_cache_entries()
        if not entries:
            console.print(_styled(Messages.INFO_INDEX_ALL_EMPTY, Styles.INFO))
        else:
            console.print(_styled(Messages.INFO_INDEX_ALL_HEADER, Styles.TITLE))
            table = Table(show_header=True, header_style=Styles.TABLE_HEADER)
            table.add_column(Messages.TABLE_INDEX_HEADER_ROOT)
            table.add_column(Messages.TABLE_INDEX_HEADER_MODE)
            table.add_column(Messages.TABLE_INDEX_HEADER_MODEL)
            table.add_column(Messages.TABLE_INDEX_HEADER_HIDDEN, justify="center")
            table.add_column(Messages.TABLE_INDEX_HEADER_RECURSIVE, justify="center")
            table.add_column(Messages.TABLE_INDEX_HEADER_EXTENSIONS)
            table.add_column(Messages.TABLE_INDEX_HEADER_FILES, justify="right")
            table.add_column(Messages.TABLE_INDEX_HEADER_GENERATED)
            for entry in entries:
                table.add_row(
                    str(entry["root_path"]),
                    str(entry["mode"]),
                    str(entry["model"]),
                    "yes" if entry["include_hidden"] else "no",
                    "yes" if entry["recursive"] else "no",
                    _format_extensions_display(entry.get("extensions")),
                    str(entry["file_count"]),
                    str(entry["generated_at"]),
                )
            console.print(table)


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
    table.add_column(Messages.TABLE_HEADER_PREVIEW, overflow="fold")
    for idx, result in enumerate(results, start=1):
        table.add_row(
            str(idx),
            f"{result.score:.3f}",
            format_path(result.path, base),
            _format_preview(result.preview),
        )
    console.print(table)


def _styled(text: str, style: str) -> str:
    return f"[{style}]{text}[/{style}]"


def _format_preview(text: str | None, limit: int = 80) -> str:
    if not text:
        return "-"
    snippet = text.strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 1].rstrip() + "…"


def run(argv: list[str] | None = None) -> None:
    """Entry point wrapper allowing optional argument override."""
    if argv is None:
        app()
    else:
        app(args=list(argv))


def _format_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _ensure_config_file() -> Path:
    config_path = config_module.CONFIG_FILE
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text("{}\n", encoding="utf-8")
    return config_path


def _edit_config_file() -> None:
    command = resolve_editor_command()
    if not command:
        console.print(_styled(Messages.ERROR_CONFIG_EDITOR_NOT_FOUND, Styles.ERROR))
        raise typer.Exit(code=1)

    cmd_list = list(command)
    config_path = _ensure_config_file()
    console.print(
        _styled(
            Messages.INFO_CONFIG_EDITING.format(
                path=config_path,
                editor=_format_command(cmd_list),
            ),
            Styles.INFO,
        )
    )
    try:
        subprocess.run(cmd_list + [str(config_path)], check=True)
    except FileNotFoundError as exc:
        console.print(
            _styled(
                Messages.ERROR_CONFIG_EDITOR_LAUNCH.format(reason=str(exc)),
                Styles.ERROR,
            )
        )
        raise typer.Exit(code=1)
    except subprocess.CalledProcessError as exc:
        code = exc.returncode if exc.returncode is not None else 1
        console.print(
            _styled(
                Messages.ERROR_CONFIG_EDITOR_FAILED.format(code=code),
                Styles.ERROR,
            )
        )
        raise typer.Exit(code=code)
