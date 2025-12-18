"""Command line interface for Vexor."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import webbrowser
from enum import Enum
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
from .services.cache_service import is_cache_current, load_index_metadata_safe
from .services.config_service import apply_config_updates, get_config_snapshot
from .services.index_service import IndexStatus, build_index, clear_index_entries
from .services.search_service import SearchRequest, perform_search
from .services.system_service import (
    DoctorCheckResult,
    fetch_remote_version,
    find_command_on_path,
    resolve_editor_command,
    run_all_doctor_checks,
    version_tuple,
)
from .services.skill_service import (
    DEFAULT_SKILL_NAME,
    SkillInstallStatus,
    install_bundled_skill,
    resolve_skill_roots,
)
from .text import Messages, Styles
from .utils import resolve_directory, format_path, ensure_positive, normalize_extensions

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .search import SearchResult

REMOTE_VERSION_URL = "https://raw.githubusercontent.com/scarletkc/vexor/refs/heads/main/vexor/__init__.py"
PROJECT_URL = "https://github.com/scarletkc/vexor"
REPO_OWNER_AND_NAME = "scarletkc/vexor"
PYPI_URL = "https://pypi.org/project/vexor/"

console = Console()
app = typer.Typer(
    help=Messages.APP_HELP,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


class SearchOutputFormat(str, Enum):
    rich = "rich"
    porcelain = "porcelain"
    porcelain_z = "porcelain-z"



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


def _parse_boolean(value: str) -> bool:
    token = value.strip().lower()
    if token in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(Messages.ERROR_BOOLEAN_INVALID.format(value=value))


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
    no_respect_gitignore: bool = typer.Option(
        False,
        "--no-respect-gitignore",
        help=Messages.HELP_RESPECT_GITIGNORE,
    ),
    mode: str = typer.Option(
        "auto",
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
    output_format: SearchOutputFormat = typer.Option(
        SearchOutputFormat.rich,
        "--format",
        help=Messages.HELP_SEARCH_FORMAT,
    ),
) -> None:
    """Run the semantic search."""
    config = load_config()
    model_name = config.model or DEFAULT_MODEL
    batch_size = config.batch_size if config.batch_size is not None else DEFAULT_BATCH_SIZE
    provider = config.provider or DEFAULT_PROVIDER
    base_url = config.base_url
    api_key = config.api_key
    auto_index = bool(config.auto_index)
    respect_gitignore = not no_respect_gitignore

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
    if output_format == SearchOutputFormat.rich:
        should_index_first = False
        if auto_index:
            metadata = load_index_metadata_safe(
                directory,
                model_name,
                include_hidden,
                respect_gitignore,
                mode_value,
                recursive,
                extensions=normalized_exts,
            )
            file_snapshot = metadata.get("files", []) if metadata else []
            if metadata is None:
                should_index_first = True
            elif file_snapshot and not is_cache_current(
                directory,
                include_hidden,
                respect_gitignore,
                file_snapshot,
                recursive=recursive,
                extensions=normalized_exts,
            ):
                should_index_first = True
        if should_index_first:
            console.print(
                _styled(Messages.INFO_INDEX_RUNNING.format(path=directory), Styles.INFO)
            )
        else:
            console.print(
                _styled(Messages.INFO_SEARCH_RUNNING.format(path=directory), Styles.INFO)
            )
    request = SearchRequest(
        query=clean_query,
        directory=directory,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode_value,
        recursive=recursive,
        top_k=top,
        model_name=model_name,
        batch_size=batch_size,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        extensions=normalized_exts,
        auto_index=auto_index,
    )
    try:
        response = perform_search(request)
    except FileNotFoundError:
        message = Messages.ERROR_INDEX_MISSING.format(path=directory)
        if output_format == SearchOutputFormat.rich:
            console.print(_styled(message, Styles.ERROR))
        else:
            typer.echo(message, err=True)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        if output_format == SearchOutputFormat.rich:
            console.print(_styled(str(exc), Styles.ERROR))
        else:
            typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    if response.index_empty:
        if output_format == SearchOutputFormat.rich:
            console.print(_styled(Messages.INFO_INDEX_EMPTY, Styles.WARNING))
        else:
            typer.echo(Messages.INFO_INDEX_EMPTY, err=True)
        raise typer.Exit(code=0)
    if response.is_stale:
        warning = Messages.WARNING_INDEX_STALE.format(path=directory)
        if output_format == SearchOutputFormat.rich:
            console.print(_styled(warning, Styles.WARNING))
        else:
            typer.echo(warning, err=True)
    if not response.results:
        if output_format == SearchOutputFormat.rich:
            console.print(_styled(Messages.INFO_NO_RESULTS, Styles.WARNING))
        else:
            typer.echo(Messages.INFO_NO_RESULTS, err=True)
        raise typer.Exit(code=0)

    if output_format == SearchOutputFormat.porcelain:
        _render_results_porcelain(response.results, response.base_path)
        return
    if output_format == SearchOutputFormat.porcelain_z:
        _render_results_porcelain_z(response.results, response.base_path)
        return
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
    no_respect_gitignore: bool = typer.Option(
        False,
        "--no-respect-gitignore",
        help=Messages.HELP_RESPECT_GITIGNORE,
    ),
    mode: str = typer.Option(
        "auto",
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
    respect_gitignore = not no_respect_gitignore
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
            respect_gitignore,
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
            gitignore="yes" if metadata.get("respect_gitignore", True) else "no",
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
            respect_gitignore=respect_gitignore,
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
            respect_gitignore=respect_gitignore,
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
    set_auto_index_option: str | None = typer.Option(
        None,
        "--set-auto-index",
        help=Messages.HELP_SET_AUTO_INDEX,
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

    auto_index: bool | None = None
    if set_auto_index_option is not None:
        try:
            auto_index = _parse_boolean(set_auto_index_option)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_name="set-auto-index") from exc

    updates = apply_config_updates(
        api_key=set_api_key_option,
        clear_api_key=clear_api_key,
        model=set_model_option,
        batch_size=set_batch_option,
        provider=set_provider_option,
        base_url=set_base_url_option,
        clear_base_url=clear_base_url,
        auto_index=auto_index,
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
    if updates.auto_index_set and auto_index is not None:
        state = "enabled" if auto_index else "disabled"
        console.print(_styled(Messages.INFO_AUTO_INDEX_SET.format(value=state), Styles.SUCCESS))

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
                    auto_index="yes" if cfg.auto_index else "no",
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
            table.add_column(Messages.TABLE_INDEX_HEADER_ROOT, overflow="fold")
            table.add_column(Messages.TABLE_INDEX_HEADER_MODE, no_wrap=True)
            table.add_column(Messages.TABLE_INDEX_HEADER_MODEL, no_wrap=True)
            table.add_column(Messages.TABLE_INDEX_HEADER_HIDDEN, justify="center")
            table.add_column(Messages.TABLE_INDEX_HEADER_RECURSIVE, justify="center")
            table.add_column(Messages.TABLE_INDEX_HEADER_GITIGNORE, justify="center")
            table.add_column(Messages.TABLE_INDEX_HEADER_EXTENSIONS)
            table.add_column(Messages.TABLE_INDEX_HEADER_FILES, justify="right")
            table.add_column(Messages.TABLE_INDEX_HEADER_GENERATED, overflow="fold")
            for entry in entries:
                table.add_row(
                    str(entry["root_path"]),
                    str(entry["mode"]),
                    str(entry["model"]),
                    "yes" if entry["include_hidden"] else "no",
                    "yes" if entry["recursive"] else "no",
                    "yes" if entry.get("respect_gitignore", True) else "no",
                    _format_extensions_display(entry.get("extensions")),
                    str(entry["file_count"]),
                    str(entry["generated_at"]),
                )
            console.print(table)


@app.command()
def install(
    skills: str = typer.Option(
        ...,
        "--skills",
        help=Messages.HELP_INSTALL_SKILLS,
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help=Messages.HELP_INSTALL_FORCE,
    ),
) -> None:
    """Install Vexor Agent Skills for AI assistants."""

    try:
        roots = resolve_skill_roots(skills)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    failures = 0
    for root in roots:
        destination_root = root.expanduser()
        try:
            result = install_bundled_skill(
                skill_name=DEFAULT_SKILL_NAME,
                skills_dir=destination_root,
                force=force,
            )
        except FileExistsError as exc:
            console.print(
                _styled(
                    Messages.ERROR_INSTALL_SKILL_EXISTS.format(path=str(exc.args[0])),
                    Styles.ERROR,
                )
            )
            failures += 1
            continue
        except FileNotFoundError as exc:
            console.print(
                _styled(Messages.ERROR_INSTALL_SKILL_SOURCE.format(reason=str(exc)), Styles.ERROR)
            )
            raise typer.Exit(code=1)

        if result.status == SkillInstallStatus.up_to_date:
            console.print(
                _styled(
                    Messages.INFO_INSTALL_SKILL_UP_TO_DATE.format(path=result.destination),
                    Styles.INFO,
                )
            )
        else:
            console.print(
                _styled(
                    Messages.INFO_INSTALL_SKILL_DONE.format(path=result.destination),
                    Styles.SUCCESS,
                )
            )

    if failures:
        raise typer.Exit(code=1)


@app.command()
def doctor(
    skip_api_test: bool = typer.Option(
        False,
        "--skip-api-test",
        help=Messages.HELP_DOCTOR_SKIP_API,
    ),
) -> None:
    """Run diagnostic checks for Vexor installation and configuration."""
    from . import __version__

    console.print(_styled(Messages.DOCTOR_TITLE.format(version=__version__), Styles.TITLE))
    console.print()

    config_load_error: DoctorCheckResult | None = None
    try:
        config = load_config()
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        config = config_module.Config()
        config_load_error = DoctorCheckResult(
            name="Config JSON",
            passed=False,
            message=Messages.DOCTOR_CONFIG_INVALID.format(path=config_module.CONFIG_FILE),
            detail=str(exc),
        )

    provider = config.provider or DEFAULT_PROVIDER
    model = config.model or DEFAULT_MODEL

    results: list[DoctorCheckResult] = []
    if config_load_error is not None:
        results.append(config_load_error)

    results.extend(
        run_all_doctor_checks(
        provider=provider,
        model=model,
        api_key=config.api_key,
        base_url=config.base_url,
        skip_api_test=skip_api_test,
    )
    )

    has_failure = False
    for result in results:
        if result.passed:
            icon = "[green]✓[/green]"
        else:
            icon = "[red]✗[/red]"
            has_failure = True

        console.print(f"  {icon} [bold]{result.name}:[/bold] {result.message}")
        if result.detail:
            console.print(f"      [dim]{result.detail}[/dim]")

    console.print()
    if has_failure:
        console.print(_styled(Messages.DOCTOR_SOME_FAILED, Styles.WARNING))
        raise typer.Exit(code=1)
    console.print(_styled(Messages.DOCTOR_ALL_PASSED, Styles.SUCCESS))


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


@app.command()
def star() -> None:
    """Star the Vexor repository on GitHub."""
    gh_path = find_command_on_path("gh")
    if gh_path:
        try:
            result = subprocess.run(
                [gh_path, "repo", "star", REPO_OWNER_AND_NAME],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.returncode == 0:
                console.print(_styled(Messages.INFO_STAR_SUCCESS, Styles.SUCCESS))
                return
            # gh CLI failed, fall back to browser
        except subprocess.CalledProcessError:
            # gh CLI failed with a non-zero exit code; fall back to browser
            pass

    # Fall back to opening the browser
    console.print(_styled(Messages.INFO_STAR_BROWSER.format(url=PROJECT_URL), Styles.INFO))
    webbrowser.open(PROJECT_URL)


def _render_results(results: Sequence["SearchResult"], base: Path, backend: str | None) -> None:
    console.print(_styled(Messages.TABLE_TITLE, Styles.TITLE))
    if backend:
        console.print(_styled(f"{Messages.TABLE_BACKEND_PREFIX}{backend}", Styles.INFO))
    table = Table(show_header=True, header_style=Styles.TABLE_HEADER)
    table.add_column(Messages.TABLE_HEADER_INDEX, justify="right")
    table.add_column(Messages.TABLE_HEADER_SIMILARITY, justify="right")
    table.add_column(Messages.TABLE_HEADER_PATH, overflow="fold")
    table.add_column(Messages.TABLE_HEADER_LINES, justify="right")
    table.add_column(Messages.TABLE_HEADER_PREVIEW, overflow="fold")
    for idx, result in enumerate(results, start=1):
        table.add_row(
            str(idx),
            f"{result.score:.3f}",
            format_path(result.path, base),
            _format_lines(result.start_line, result.end_line),
            _format_preview(result.preview),
        )
    console.print(table)


def _escape_porcelain_field(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _render_results_porcelain(
    results: Sequence["SearchResult"],
    base: Path,
) -> None:
    for idx, result in enumerate(results, start=1):
        preview = result.preview if result.preview is not None else "-"
        start_line = str(result.start_line) if result.start_line is not None else "-"
        end_line = str(result.end_line) if result.end_line is not None else "-"
        fields = (
            str(idx),
            f"{result.score:.3f}",
            format_path(result.path, base),
            str(result.chunk_index),
            start_line,
            end_line,
            _escape_porcelain_field(preview),
        )
        typer.echo("\t".join(fields))


def _render_results_porcelain_z(results: Sequence["SearchResult"], base: Path) -> None:
    for idx, result in enumerate(results, start=1):
        preview = result.preview if result.preview is not None else "-"
        start_line = str(result.start_line) if result.start_line is not None else "-"
        end_line = str(result.end_line) if result.end_line is not None else "-"
        fields = (
            str(idx),
            f"{result.score:.3f}",
            format_path(result.path, base),
            str(result.chunk_index),
            start_line,
            end_line,
            preview.replace("\0", ""),
        )
        sys.stdout.write("\0".join(fields) + "\0")


def _styled(text: str, style: str) -> str:
    return f"[{style}]{text}[/{style}]"


def _format_preview(text: str | None, limit: int = 80) -> str:
    if not text:
        return "-"
    snippet = text.strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 1].rstrip() + "…"


def _format_lines(start_line: int | None, end_line: int | None) -> str:
    if start_line is None:
        return "-"
    if end_line is None or end_line <= start_line:
        return f"L{start_line}"
    return f"L{start_line}-{end_line}"


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
