"""Command line interface for Vexor."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from difflib import get_close_matches
from enum import Enum
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

import click
import typer
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from . import __version__, config as config_module
from .cache import clear_all_cache, list_cache_entries
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLASHRANK_MODEL,
    DEFAULT_FLASHRANK_MAX_LENGTH,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_RERANK,
    SUPPORTED_EXTRACT_BACKENDS,
    SUPPORTED_PROVIDERS,
    SUPPORTED_RERANKERS,
    flashrank_cache_dir,
    load_config,
    normalize_remote_rerank_url,
    resolve_remote_rerank_api_key,
    resolve_default_model,
)
from .modes import available_modes, get_strategy
from .services.cache_service import is_cache_current, load_index_metadata_safe
from .services.config_service import apply_config_updates, get_config_snapshot
from .services.init_service import run_init_wizard, should_auto_run_init
from .services.index_service import IndexStatus, build_index, clear_index_entries
from .services.search_service import SearchRequest, perform_search, _select_cache_superset
from .services.system_service import (
    DoctorCheckResult,
    build_standalone_download_url,
    build_upgrade_commands,
    detect_install_method,
    fetch_latest_pypi_version,
    find_command_on_path,
    git_worktree_is_dirty,
    InstallMethod,
    parse_version,
    resolve_editor_command,
    run_upgrade_commands,
    run_all_doctor_checks,
    version_tuple,
)
from .services.skill_service import (
    DEFAULT_SKILL_NAME,
    SkillInstallStatus,
    install_bundled_skill,
    resolve_skill_roots,
)
from .providers.local import LocalEmbeddingBackend, resolve_fastembed_cache_dir
from .output import format_status_icon
from .text import Messages, Styles
from .utils import (
    resolve_directory,
    format_path,
    ensure_positive,
    build_exclude_spec,
    is_excluded_path,
    normalize_extensions,
    normalize_exclude_patterns,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .search import SearchResult

REMOTE_VERSION_URL = "https://raw.githubusercontent.com/scarletkc/vexor/refs/heads/main/vexor/__init__.py"
PROJECT_URL = "https://github.com/scarletkc/vexor"
REPO_OWNER_AND_NAME = "scarletkc/vexor"
PYPI_URL = "https://pypi.org/project/vexor/"

console = Console()


def _normalize_flashrank_model_args(args: list[str]) -> list[str]:
    normalized: list[str] = []
    idx = 0
    while idx < len(args):
        arg = args[idx]
        normalized.append(arg)
        if arg == "--":
            normalized.extend(args[idx + 1 :])
            break
        if arg == "--set-flashrank-model":
            next_arg = args[idx + 1] if idx + 1 < len(args) else None
            if next_arg is None or next_arg.startswith("-"):
                normalized.append("")
        idx += 1
    return normalized


class DefaultSearchGroup(TyperGroup):
    """Treat unknown subcommands as search queries."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        return super().parse_args(ctx, _normalize_flashrank_model_args(args))

    def resolve_command(
        self,
        ctx: click.Context,
        args: list[str],
    ) -> tuple[str | None, click.Command | None, list[str]]:
        original_args = list(args)
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if not original_args:
                raise
            token = original_args[0]
            if token.startswith("-"):
                raise
            if self.suggest_commands and self.commands:
                matches = get_close_matches(
                    token,
                    list(self.commands.keys()),
                    cutoff=0.8,
                )
                if matches:
                    raise
            command = self.get_command(ctx, "search")
            if command is None:
                raise
            return "search", command, original_args


app = typer.Typer(
    help=Messages.APP_HELP,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    cls=DefaultSearchGroup,
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


def _prepare_flashrank_model(model_name: str | None) -> None:
    try:
        from flashrank import Ranker
    except ImportError as exc:
        raise RuntimeError(Messages.ERROR_FLASHRANK_MISSING) from exc
    cache_dir = flashrank_cache_dir()
    try:
        effective_model = model_name or DEFAULT_FLASHRANK_MODEL
        kwargs = {
            "max_length": DEFAULT_FLASHRANK_MAX_LENGTH,
            "cache_dir": str(cache_dir),
            "model_name": effective_model,
        }
        Ranker(**kwargs)
    except Exception as exc:
        raise RuntimeError(Messages.ERROR_FLASHRANK_SETUP.format(reason=str(exc))) from exc


def _format_extensions_display(values: Sequence[str] | None) -> str:
    if not values:
        return "all"
    return ", ".join(values)


def _format_patterns_display(values: Sequence[str] | None) -> str:
    if not values:
        return "none"
    return ", ".join(values)


def _filter_snapshot_by_directory(
    entries: Sequence[dict],
    relative_dir: Path,
    *,
    recursive: bool,
) -> list[dict]:
    filtered: list[dict] = []
    for entry in entries:
        rel_path = entry.get("path", "")
        try:
            rel_subpath = Path(rel_path).relative_to(relative_dir)
        except ValueError:
            continue
        if not recursive and len(rel_subpath.parts) > 1:
            continue
        updated = dict(entry)
        updated["path"] = rel_subpath.as_posix()
        filtered.append(updated)
    return filtered


def _filter_snapshot_by_extensions(
    entries: Sequence[dict],
    extensions: Sequence[str],
) -> list[dict]:
    ext_set = {ext.lower() for ext in extensions if ext}
    if not ext_set:
        return list(entries)
    filtered: list[dict] = []
    for entry in entries:
        rel_path = entry.get("path", "")
        if Path(rel_path).suffix.lower() in ext_set:
            filtered.append(entry)
    return filtered


def _filter_snapshot_by_exclude_patterns(
    entries: Sequence[dict],
    exclude_spec,
) -> list[dict]:
    if exclude_spec is None:
        return list(entries)
    filtered: list[dict] = []
    for entry in entries:
        rel_path = entry.get("path", "")
        rel_posix = Path(rel_path).as_posix() if rel_path else ""
        if is_excluded_path(exclude_spec, rel_posix, is_dir=False):
            continue
        filtered.append(entry)
    return filtered


def _should_index_before_search(request: SearchRequest) -> bool:
    metadata = load_index_metadata_safe(
        request.directory,
        request.model_name,
        request.include_hidden,
        request.respect_gitignore,
        request.mode,
        request.recursive,
        exclude_patterns=request.exclude_patterns,
        extensions=request.extensions,
    )
    file_snapshot = metadata.get("files", []) if metadata else []
    if metadata is None:
        superset_entry = _select_cache_superset(request, list_cache_entries)
        if superset_entry is None:
            return True
        superset_root = Path(superset_entry.get("root_path", "")).expanduser().resolve()
        superset_recursive = bool(superset_entry.get("recursive"))
        superset_extensions = tuple(superset_entry.get("extensions") or ())
        superset_excludes = tuple(superset_entry.get("exclude_patterns") or ())
        superset_metadata = load_index_metadata_safe(
            superset_root,
            request.model_name,
            request.include_hidden,
            request.respect_gitignore,
            request.mode,
            superset_recursive,
            exclude_patterns=superset_excludes,
            extensions=superset_extensions,
        )
        if not superset_metadata:
            return True
        file_snapshot = superset_metadata.get("files", [])
        if superset_root != request.directory:
            try:
                relative_dir = request.directory.resolve().relative_to(superset_root)
            except ValueError:
                return True
            file_snapshot = _filter_snapshot_by_directory(
                file_snapshot,
                relative_dir,
                recursive=request.recursive,
            )
    if request.extensions:
        file_snapshot = _filter_snapshot_by_extensions(file_snapshot, request.extensions)
    exclude_spec = build_exclude_spec(request.exclude_patterns)
    if exclude_spec is not None:
        file_snapshot = _filter_snapshot_by_exclude_patterns(file_snapshot, exclude_spec)
    if file_snapshot and not is_cache_current(
        request.directory,
        request.include_hidden,
        request.respect_gitignore,
        file_snapshot,
        recursive=request.recursive,
        exclude_patterns=request.exclude_patterns,
        extensions=request.extensions,
    ):
        return True
    return False


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
    exclude_patterns: list[str] | None = typer.Option(
        None,
        "--exclude-pattern",
        help=Messages.HELP_EXCLUDE_PATTERNS,
    ),
    output_format: SearchOutputFormat = typer.Option(
        SearchOutputFormat.rich,
        "--format",
        help=Messages.HELP_SEARCH_FORMAT,
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help=Messages.HELP_NO_CACHE,
    ),
) -> None:
    """Run the semantic search."""
    config = load_config()
    provider = (config.provider or DEFAULT_PROVIDER).lower()
    model_name = resolve_default_model(provider, config.model)
    batch_size = config.batch_size if config.batch_size is not None else DEFAULT_BATCH_SIZE
    embed_concurrency = config.embed_concurrency
    extract_concurrency = config.extract_concurrency
    extract_backend = config.extract_backend
    base_url = config.base_url
    api_key = config.api_key
    auto_index = bool(config.auto_index)
    flashrank_model = config.flashrank_model
    remote_rerank = config.remote_rerank
    rerank = (config.rerank or DEFAULT_RERANK).strip().lower()
    if rerank not in SUPPORTED_RERANKERS:
        rerank = DEFAULT_RERANK
    respect_gitignore = not no_respect_gitignore

    clean_query = query.strip()
    if not clean_query:
        console.print(_styled(Messages.ERROR_EMPTY_QUERY, Styles.ERROR))
        raise typer.Exit(code=1)
    try:
        ensure_positive(top, "top")
    except ValueError as exc:  # pragma: no cover - validated by Typer
        raise typer.BadParameter(str(exc)) from exc

    directory = resolve_directory(path)
    mode_value = _validate_mode(mode)
    recursive = not no_recursive
    normalized_exts = normalize_extensions(extensions)
    normalized_excludes = normalize_exclude_patterns(exclude_patterns)
    if extensions and not normalized_exts:
        raise typer.BadParameter(Messages.ERROR_EXTENSIONS_EMPTY)
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
        embed_concurrency=embed_concurrency,
        extract_concurrency=extract_concurrency,
        extract_backend=extract_backend,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        local_cuda=bool(config.local_cuda),
        exclude_patterns=normalized_excludes,
        extensions=normalized_exts,
        auto_index=auto_index,
        no_cache=no_cache,
        rerank=rerank,
        flashrank_model=flashrank_model,
        remote_rerank=remote_rerank,
    )
    if output_format == SearchOutputFormat.rich:
        if no_cache:
            console.print(
                _styled(
                    Messages.INFO_SEARCH_RUNNING_NO_CACHE.format(path=directory),
                    Styles.INFO,
                )
            )
        else:
            should_index_first = (
                _should_index_before_search(request) if auto_index else False
            )
            if should_index_first:
                console.print(
                    _styled(
                        Messages.INFO_INDEX_RUNNING.format(path=directory), Styles.INFO
                    )
                )
            else:
                console.print(
                    _styled(
                        Messages.INFO_SEARCH_RUNNING.format(path=directory), Styles.INFO
                    )
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
    _render_results(response.results, response.base_path, response.backend, response.reranker)


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
    exclude_patterns: list[str] | None = typer.Option(
        None,
        "--exclude-pattern",
        help=Messages.HELP_EXCLUDE_PATTERNS,
    ),
) -> None:
    """Create or refresh the cached index for the given directory."""
    config = load_config()
    provider = (config.provider or DEFAULT_PROVIDER).lower()
    model_name = resolve_default_model(provider, config.model)
    batch_size = config.batch_size if config.batch_size is not None else DEFAULT_BATCH_SIZE
    embed_concurrency = config.embed_concurrency
    extract_concurrency = config.extract_concurrency
    extract_backend = config.extract_backend
    base_url = config.base_url
    api_key = config.api_key

    directory = resolve_directory(path)
    mode_value = _validate_mode(mode)
    recursive = not no_recursive
    respect_gitignore = not no_respect_gitignore
    normalized_exts = normalize_extensions(extensions)
    normalized_excludes = normalize_exclude_patterns(exclude_patterns)
    if extensions and not normalized_exts:
        raise typer.BadParameter(Messages.ERROR_EXTENSIONS_EMPTY)
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
            exclude_patterns=normalized_excludes,
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
            exclude_patterns=_format_patterns_display(metadata.get("exclude_patterns")),
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
            exclude_patterns=normalized_excludes,
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
            embed_concurrency=embed_concurrency,
            extract_concurrency=extract_concurrency,
            extract_backend=extract_backend,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            local_cuda=bool(config.local_cuda),
            exclude_patterns=normalized_excludes,
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


@app.command(help=Messages.HELP_INIT)
def init(
    dry: bool = typer.Option(
        False,
        "--dry",
        help=Messages.HELP_INIT_DRY,
    ),
) -> None:
    """Run the interactive setup wizard."""
    run_init_wizard(dry_run=dry)


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
    set_embed_concurrency_option: int | None = typer.Option(
        None,
        "--set-embed-concurrency",
        help=Messages.HELP_SET_EMBED_CONCURRENCY,
    ),
    set_extract_concurrency_option: int | None = typer.Option(
        None,
        "--set-extract-concurrency",
        help=Messages.HELP_SET_EXTRACT_CONCURRENCY,
    ),
    set_extract_backend_option: str | None = typer.Option(
        None,
        "--set-extract-backend",
        help=Messages.HELP_SET_EXTRACT_BACKEND,
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
    clear_flashrank: bool = typer.Option(
        False,
        "--clear-flashrank",
        help=Messages.HELP_CLEAR_FLASHRANK,
    ),
    set_rerank_option: str | None = typer.Option(
        None,
        "--rerank",
        help=Messages.HELP_SET_RERANK,
    ),
    set_flashrank_model_option: str | None = typer.Option(
        None,
        "--set-flashrank-model",
        help=Messages.HELP_SET_FLASHRANK_MODEL,
    ),
    set_remote_rerank_url_option: str | None = typer.Option(
        None,
        "--set-remote-rerank-url",
        help=Messages.HELP_SET_REMOTE_RERANK_URL,
    ),
    set_remote_rerank_model_option: str | None = typer.Option(
        None,
        "--set-remote-rerank-model",
        help=Messages.HELP_SET_REMOTE_RERANK_MODEL,
    ),
    set_remote_rerank_api_key_option: str | None = typer.Option(
        None,
        "--set-remote-rerank-api-key",
        help=Messages.HELP_SET_REMOTE_RERANK_API_KEY,
    ),
    clear_remote_rerank: bool = typer.Option(
        False,
        "--clear-remote-rerank",
        help=Messages.HELP_CLEAR_REMOTE_RERANK,
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
    if set_embed_concurrency_option is not None and set_embed_concurrency_option < 1:
        raise typer.BadParameter(Messages.ERROR_CONCURRENCY_INVALID)
    if set_extract_concurrency_option is not None and set_extract_concurrency_option < 1:
        raise typer.BadParameter(Messages.ERROR_EXTRACT_CONCURRENCY_INVALID)
    if set_base_url_option and clear_base_url:
        raise typer.BadParameter(Messages.ERROR_BASE_URL_CONFLICT)
    flashrank_model_reset = False
    if set_flashrank_model_option is not None:
        normalized_flashrank_model = set_flashrank_model_option.strip()
        if not normalized_flashrank_model:
            set_flashrank_model_option = ""
            flashrank_model_reset = True
        else:
            set_flashrank_model_option = normalized_flashrank_model
    if set_remote_rerank_url_option is not None:
        normalized_remote_url = normalize_remote_rerank_url(set_remote_rerank_url_option)
        if not normalized_remote_url:
            raise typer.BadParameter(Messages.ERROR_REMOTE_RERANK_URL_EMPTY)
        set_remote_rerank_url_option = normalized_remote_url
    if set_remote_rerank_model_option is not None:
        normalized_remote_model = set_remote_rerank_model_option.strip()
        if not normalized_remote_model:
            raise typer.BadParameter(Messages.ERROR_REMOTE_RERANK_MODEL_EMPTY)
        set_remote_rerank_model_option = normalized_remote_model
    if set_remote_rerank_api_key_option is not None:
        normalized_remote_key = set_remote_rerank_api_key_option.strip()
        if not normalized_remote_key:
            raise typer.BadParameter(Messages.ERROR_REMOTE_RERANK_API_KEY_EMPTY)
        set_remote_rerank_api_key_option = normalized_remote_key
    if set_extract_backend_option is not None:
        normalized_backend = set_extract_backend_option.strip().lower()
        if normalized_backend not in SUPPORTED_EXTRACT_BACKENDS:
            allowed = ", ".join(SUPPORTED_EXTRACT_BACKENDS)
            raise typer.BadParameter(
                Messages.ERROR_EXTRACT_BACKEND_INVALID.format(
                    value=set_extract_backend_option, allowed=allowed
                )
            )
        set_extract_backend_option = normalized_backend
    if clear_remote_rerank and any(
        (
            set_remote_rerank_url_option is not None,
            set_remote_rerank_model_option is not None,
            set_remote_rerank_api_key_option is not None,
        )
    ):
        raise typer.BadParameter(Messages.ERROR_REMOTE_RERANK_CLEAR_CONFLICT)
    if clear_flashrank and any(
        (
            set_api_key_option is not None,
            clear_api_key,
            set_model_option is not None,
            set_batch_option is not None,
            set_embed_concurrency_option is not None,
            set_extract_concurrency_option is not None,
            set_extract_backend_option is not None,
            set_provider_option is not None,
            set_base_url_option is not None,
            clear_base_url,
            set_auto_index_option is not None,
            set_rerank_option is not None,
            set_flashrank_model_option is not None,
            set_remote_rerank_url_option is not None,
            set_remote_rerank_model_option is not None,
            set_remote_rerank_api_key_option is not None,
            clear_remote_rerank,
            show,
            show_index_all,
            clear_index_all,
        )
    ):
        raise typer.BadParameter(
            Messages.ERROR_FLASHRANK_CLEAR_CONFLICT,
            param_hint="--clear-flashrank",
        )
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
    if set_rerank_option is not None:
        normalized_rerank = set_rerank_option.strip().lower()
        if normalized_rerank not in SUPPORTED_RERANKERS:
            allowed = ", ".join(SUPPORTED_RERANKERS)
            raise typer.BadParameter(
                Messages.ERROR_RERANK_INVALID.format(
                    value=set_rerank_option, allowed=allowed
                )
            )
        if normalized_rerank == "flashrank":
            if importlib.util.find_spec("flashrank") is None:
                raise typer.BadParameter(Messages.ERROR_FLASHRANK_MISSING)
        set_rerank_option = normalized_rerank

    config_snapshot = load_config()
    current_provider = (config_snapshot.provider or DEFAULT_PROVIDER).lower()
    pending_provider = set_provider_option or current_provider
    pending_model = set_model_option if set_model_option is not None else config_snapshot.model
    if set_provider_option == "gemini" and set_model_option is None:
        if (pending_model or "").strip() == DEFAULT_MODEL:
            set_model_option = DEFAULT_GEMINI_MODEL
            pending_model = set_model_option
    pending_base_url = None
    if not clear_base_url:
        pending_base_url = (
            set_base_url_option
            if set_base_url_option is not None
            else config_snapshot.base_url
        )
    provider_mutation = any(
        (
            set_provider_option is not None,
            set_model_option is not None,
            set_base_url_option is not None,
            clear_base_url,
        )
    )
    if pending_provider == "custom" and provider_mutation:
        if set_provider_option == "custom" and set_model_option is None:
            raise typer.BadParameter(Messages.ERROR_CUSTOM_MODEL_REQUIRED)
        if not (pending_model and pending_model.strip()):
            raise typer.BadParameter(Messages.ERROR_CUSTOM_MODEL_REQUIRED)
        if not (pending_base_url and pending_base_url.strip()):
            raise typer.BadParameter(Messages.ERROR_CUSTOM_BASE_URL_REQUIRED)

    if clear_remote_rerank:
        active_rerank = set_rerank_option or config_snapshot.rerank
        if (active_rerank or "").strip().lower() == "remote":
            raise typer.BadParameter(Messages.ERROR_REMOTE_RERANK_INCOMPLETE)
    if set_rerank_option == "remote":
        existing_remote = config_snapshot.remote_rerank
        pending_remote_url = (
            set_remote_rerank_url_option
            if set_remote_rerank_url_option is not None
            else (existing_remote.base_url if existing_remote else None)
        )
        pending_remote_model = (
            set_remote_rerank_model_option
            if set_remote_rerank_model_option is not None
            else (existing_remote.model if existing_remote else None)
        )
        pending_remote_key = (
            set_remote_rerank_api_key_option
            if set_remote_rerank_api_key_option is not None
            else (existing_remote.api_key if existing_remote else None)
        )
        pending_remote_key = resolve_remote_rerank_api_key(pending_remote_key)
        if not (pending_remote_url and pending_remote_model and pending_remote_key):
            raise typer.BadParameter(Messages.ERROR_REMOTE_RERANK_INCOMPLETE)

    auto_index: bool | None = None
    if set_auto_index_option is not None:
        try:
            auto_index = _parse_boolean(set_auto_index_option)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    updates = apply_config_updates(
        api_key=set_api_key_option,
        clear_api_key=clear_api_key,
        model=set_model_option,
        batch_size=set_batch_option,
        embed_concurrency=set_embed_concurrency_option,
        extract_concurrency=set_extract_concurrency_option,
        extract_backend=set_extract_backend_option,
        provider=set_provider_option,
        base_url=set_base_url_option,
        clear_base_url=clear_base_url,
        auto_index=auto_index,
        rerank=set_rerank_option,
        flashrank_model=set_flashrank_model_option,
        remote_rerank_url=set_remote_rerank_url_option,
        remote_rerank_model=set_remote_rerank_model_option,
        remote_rerank_api_key=set_remote_rerank_api_key_option,
        clear_remote_rerank=clear_remote_rerank,
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
    if updates.embed_concurrency_set and set_embed_concurrency_option is not None:
        console.print(
            _styled(
                Messages.INFO_EMBED_CONCURRENCY_SET.format(value=set_embed_concurrency_option),
                Styles.SUCCESS,
            )
        )
    if updates.extract_concurrency_set and set_extract_concurrency_option is not None:
        console.print(
            _styled(
                Messages.INFO_EXTRACT_CONCURRENCY_SET.format(
                    value=set_extract_concurrency_option
                ),
                Styles.SUCCESS,
            )
        )
    if updates.extract_backend_set and set_extract_backend_option is not None:
        console.print(
            _styled(
                Messages.INFO_EXTRACT_BACKEND_SET.format(value=set_extract_backend_option),
                Styles.SUCCESS,
            )
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
    if updates.rerank_set and set_rerank_option is not None:
        console.print(
            _styled(Messages.INFO_RERANK_SET.format(value=set_rerank_option), Styles.SUCCESS)
        )
        if set_rerank_option == "flashrank":
            flashrank_model = (
                set_flashrank_model_option
                if set_flashrank_model_option is not None
                else get_config_snapshot().flashrank_model
            )
            console.print(_styled(Messages.INFO_FLASHRANK_SETUP_START, Styles.INFO))
            try:
                _prepare_flashrank_model(flashrank_model)
            except RuntimeError as exc:
                console.print(_styled(str(exc), Styles.ERROR))
                raise typer.Exit(code=1)
            console.print(_styled(Messages.INFO_FLASHRANK_SETUP_DONE, Styles.SUCCESS))
    if updates.flashrank_model_set and set_flashrank_model_option is not None:
        if flashrank_model_reset:
            console.print(
                _styled(
                    Messages.INFO_FLASHRANK_MODEL_RESET.format(value=DEFAULT_FLASHRANK_MODEL),
                    Styles.SUCCESS,
                )
            )
        else:
            console.print(
                _styled(
                    Messages.INFO_FLASHRANK_MODEL_SET.format(value=set_flashrank_model_option),
                    Styles.SUCCESS,
                )
            )
    if updates.remote_rerank_url_set and set_remote_rerank_url_option is not None:
        console.print(
            _styled(
                Messages.INFO_REMOTE_RERANK_URL_SET.format(value=set_remote_rerank_url_option),
                Styles.SUCCESS,
            )
        )
    if updates.remote_rerank_model_set and set_remote_rerank_model_option is not None:
        console.print(
            _styled(
                Messages.INFO_REMOTE_RERANK_MODEL_SET.format(
                    value=set_remote_rerank_model_option
                ),
                Styles.SUCCESS,
            )
        )
    if updates.remote_rerank_api_key_set and set_remote_rerank_api_key_option is not None:
        console.print(_styled(Messages.INFO_REMOTE_RERANK_API_KEY_SET, Styles.SUCCESS))
    if updates.remote_rerank_cleared and clear_remote_rerank:
        console.print(_styled(Messages.INFO_REMOTE_RERANK_CLEARED, Styles.SUCCESS))

    if clear_flashrank:
        cache_dir = flashrank_cache_dir(create=False)
        if not cache_dir.exists():
            console.print(
                _styled(Messages.INFO_FLASHRANK_CACHE_EMPTY.format(path=cache_dir), Styles.INFO)
            )
        else:
            try:
                shutil.rmtree(cache_dir)
            except OSError as exc:
                console.print(
                    _styled(
                        Messages.ERROR_FLASHRANK_CACHE_CLEANUP.format(
                            path=cache_dir, reason=str(exc)
                        ),
                        Styles.ERROR,
                    )
                )
                raise typer.Exit(code=1)
            console.print(
                _styled(Messages.INFO_FLASHRANK_CACHE_CLEARED.format(path=cache_dir), Styles.SUCCESS)
            )

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
            clear_flashrank,
        )
    )
    if should_edit:
        _edit_config_file()
        return

    if show:
        cfg = get_config_snapshot()
        provider = (cfg.provider or DEFAULT_PROVIDER).lower()
        rerank = (cfg.rerank or DEFAULT_RERANK).lower()
        flashrank_line = ""
        remote_rerank_line = ""
        if rerank == "flashrank":
            model_label = cfg.flashrank_model or f"default ({DEFAULT_FLASHRANK_MODEL})"
            flashrank_line = (
                f"{Messages.INFO_FLASHRANK_MODEL_SUMMARY.format(value=model_label)}\n"
            )
        if rerank == "remote":
            remote_cfg = cfg.remote_rerank
            if remote_cfg is None:
                remote_label = "not configured"
            else:
                url_label = remote_cfg.base_url or "unset"
                model_label = remote_cfg.model or "unset"
                key_label = "yes" if remote_cfg.api_key else "no"
                remote_label = f"{url_label} (model {model_label}, key {key_label})"
            remote_rerank_line = (
                f"{Messages.INFO_REMOTE_RERANK_SUMMARY.format(value=remote_label)}\n"
            )
        console.print(
            _styled(
                Messages.INFO_CONFIG_SUMMARY.format(
                    api="yes" if cfg.api_key else "no",
                    provider=provider,
                    model=resolve_default_model(provider, cfg.model),
                    batch=cfg.batch_size if cfg.batch_size is not None else DEFAULT_BATCH_SIZE,
                    concurrency=cfg.embed_concurrency,
                    extract_concurrency=cfg.extract_concurrency,
                    extract_backend=cfg.extract_backend,
                    auto_index="yes" if cfg.auto_index else "no",
                    rerank=rerank,
                    flashrank_line=flashrank_line,
                    remote_rerank_line=remote_rerank_line,
                    local_cuda="yes" if cfg.local_cuda else "no",
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
            table.add_column(Messages.TABLE_INDEX_HEADER_EXCLUDES)
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
                    _format_patterns_display(entry.get("exclude_patterns")),
                    _format_extensions_display(entry.get("extensions")),
                    str(entry["file_count"]),
                    str(entry["generated_at"]),
                )
            console.print(table)


@app.command("local", help=Messages.HELP_LOCAL)
def local(
    setup: bool = typer.Option(
        False,
        "--setup",
        help=Messages.HELP_SETUP_LOCAL,
    ),
    clean_up: bool = typer.Option(
        False,
        "--clean-up",
        help=Messages.HELP_LOCAL_CLEANUP,
    ),
    cuda: bool = typer.Option(
        False,
        "--cuda",
        help=Messages.HELP_LOCAL_CUDA,
    ),
    cpu: bool = typer.Option(
        False,
        "--cpu",
        help=Messages.HELP_LOCAL_CPU,
    ),
    model: str = typer.Option(
        DEFAULT_LOCAL_MODEL,
        "--model",
        "-m",
        help=Messages.HELP_SETUP_LOCAL_MODEL,
    ),
) -> None:
    """Manage local embedding models."""
    if cuda and cpu:
        raise typer.BadParameter(Messages.ERROR_LOCAL_CUDA_CONFLICT)
    if clean_up and (setup or cuda or cpu):
        raise typer.BadParameter(Messages.ERROR_LOCAL_OPTIONS_CONFLICT)
    local_cuda: bool | None = None
    if cuda:
        local_cuda = True
    if cpu:
        local_cuda = False
    if clean_up:
        cache_dir = resolve_fastembed_cache_dir(create=False)
        if not cache_dir.exists():
            console.print(
                _styled(Messages.INFO_LOCAL_CACHE_EMPTY.format(path=cache_dir), Styles.INFO)
            )
            raise typer.Exit(code=0)
        try:
            shutil.rmtree(cache_dir)
        except OSError as exc:
            console.print(
                _styled(
                    Messages.ERROR_LOCAL_CACHE_CLEANUP.format(
                        path=cache_dir,
                        reason=str(exc),
                    ),
                    Styles.ERROR,
                )
            )
            raise typer.Exit(code=1)
        console.print(
            _styled(Messages.INFO_LOCAL_CACHE_CLEARED.format(path=cache_dir), Styles.SUCCESS)
        )
        raise typer.Exit(code=0)
    if not setup and local_cuda is None:
        console.print(_styled(Messages.INFO_LOCAL_SETUP_HINT, Styles.INFO))
        raise typer.Exit(code=0)
    if not setup and local_cuda is not None:
        apply_config_updates(local_cuda=local_cuda)
        message = (
            Messages.INFO_LOCAL_CUDA_ENABLED
            if local_cuda
            else Messages.INFO_LOCAL_CUDA_DISABLED
        )
        console.print(_styled(message, Styles.SUCCESS))
        raise typer.Exit(code=0)

    clean_model = model.strip()
    if not clean_model:
        raise typer.BadParameter(Messages.ERROR_LOCAL_MODEL_EMPTY)

    console.print(_styled(Messages.INFO_LOCAL_SETUP_START.format(model=clean_model), Styles.INFO))
    if local_cuda is None:
        config_snapshot = load_config()
        effective_cuda = bool(config_snapshot.local_cuda)
    else:
        effective_cuda = local_cuda
    if effective_cuda:
        try:
            import onnxruntime as ort
        except Exception as exc:
            console.print(
                _styled(Messages.DOCTOR_LOCAL_CUDA_IMPORT_FAILED, Styles.ERROR)
            )
            console.print(
                _styled(
                    Messages.DOCTOR_LOCAL_CUDA_IMPORT_DETAIL.format(reason=str(exc)),
                    Styles.ERROR,
                )
            )
            raise typer.Exit(code=1)
        try:
            providers = ort.get_available_providers()
        except Exception as exc:
            console.print(
                _styled(Messages.DOCTOR_LOCAL_CUDA_MISSING, Styles.ERROR)
            )
            console.print(
                _styled(
                    Messages.DOCTOR_LOCAL_CUDA_IMPORT_DETAIL.format(reason=str(exc)),
                    Styles.ERROR,
                )
            )
            raise typer.Exit(code=1)
        if "CUDAExecutionProvider" not in providers:
            console.print(_styled(Messages.DOCTOR_LOCAL_CUDA_MISSING, Styles.ERROR))
            console.print(
                _styled(
                    Messages.DOCTOR_LOCAL_CUDA_MISSING_DETAIL.format(
                        providers=", ".join(providers) if providers else "none"
                    ),
                    Styles.ERROR,
                )
            )
            raise typer.Exit(code=1)
    try:
        backend = LocalEmbeddingBackend(model_name=clean_model, cuda=effective_cuda)
        vectors = backend.embed(["test"])
    except RuntimeError as exc:
        console.print(_styled(str(exc), Styles.ERROR))
        raise typer.Exit(code=1)

    if vectors.size == 0:
        console.print(_styled(Messages.ERROR_NO_EMBEDDINGS, Styles.ERROR))
        raise typer.Exit(code=1)

    apply_config_updates(
        provider="local",
        model=clean_model,
        local_cuda=local_cuda,
    )
    if local_cuda is not None:
        message = (
            Messages.INFO_LOCAL_CUDA_ENABLED
            if local_cuda
            else Messages.INFO_LOCAL_CUDA_DISABLED
        )
        console.print(_styled(message, Styles.SUCCESS))
    cache_dir = resolve_fastembed_cache_dir()
    console.print(_styled(Messages.INFO_LOCAL_CACHE_DIR.format(path=cache_dir), Styles.INFO))
    console.print(_styled(Messages.INFO_LOCAL_SETUP_DONE.format(model=clean_model), Styles.SUCCESS))


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

    provider = (config.provider or DEFAULT_PROVIDER).lower()
    model = resolve_default_model(provider, config.model)

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
            local_cuda=bool(config.local_cuda),
            rerank=config.rerank,
            flashrank_model=config.flashrank_model,
            remote_rerank=config.remote_rerank,
        )
    )

    has_failure = False
    for result in results:
        icon = format_status_icon(result.passed, console=console)
        if not result.passed:
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
def update(
    upgrade: bool = typer.Option(
        False,
        "--upgrade",
        help=Messages.HELP_UPDATE_UPGRADE,
    ),
    pre: bool = typer.Option(
        False,
        "--pre",
        help=Messages.HELP_UPDATE_PRE,
    ),
) -> None:
    """Check whether a newer release is available online."""
    console.print(_styled(Messages.INFO_UPDATE_CHECKING, Styles.INFO))
    console.print(_styled(Messages.INFO_UPDATE_CURRENT.format(current=__version__), Styles.INFO))
    try:
        latest = fetch_latest_pypi_version("vexor", include_prerelease=pre)
    except RuntimeError as exc:
        console.print(
            _styled(Messages.ERROR_UPDATE_FETCH.format(reason=str(exc)), Styles.ERROR)
        )
        raise typer.Exit(code=1)

    latest_parsed = parse_version(latest)
    current_parsed = parse_version(__version__)
    is_newer = False
    if latest_parsed and current_parsed:
        is_newer = latest_parsed > current_parsed
    elif version_tuple(latest) > version_tuple(__version__):
        is_newer = True

    if is_newer:
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
        if not upgrade:
            return

        install_info = detect_install_method()
        console.print(
            _styled(
                Messages.INFO_UPDATE_INSTALL_METHOD.format(method=install_info.method.value),
                Styles.INFO,
            )
        )
        if install_info.method == InstallMethod.GIT_EDITABLE and install_info.editable_root:
            if git_worktree_is_dirty(install_info.editable_root):
                console.print(
                    _styled(
                        Messages.WARNING_UPDATE_GIT_DIRTY.format(path=install_info.editable_root),
                        Styles.WARNING,
                    )
                )
        if install_info.requires_admin:
            console.print(_styled(Messages.WARNING_UPDATE_ADMIN, Styles.WARNING))

        if install_info.method == InstallMethod.STANDALONE:
            asset, url = build_standalone_download_url(latest)
            console.print(
                _styled(
                    Messages.WARNING_UPDATE_STANDALONE.format(path=sys.executable),
                    Styles.WARNING,
                )
            )
            if asset:
                console.print(_styled(Messages.INFO_UPDATE_STANDALONE_ASSET.format(asset=asset), Styles.INFO))
            console.print(_styled(Messages.INFO_UPDATE_STANDALONE_URL.format(url=url), Styles.INFO))
            return

        commands = build_upgrade_commands(install_info, include_prerelease=pre)
        for command in commands:
            console.print(
                _styled(
                    Messages.INFO_UPDATE_UPGRADE_CMD.format(cmd=shlex.join(command)),
                    Styles.INFO,
                )
            )

        if not typer.confirm(Messages.PROMPT_UPDATE_UPGRADE_CONFIRM):
            console.print(_styled(Messages.INFO_UPDATE_UPGRADE_CANCELLED, Styles.INFO))
            return

        console.print(_styled(Messages.INFO_UPDATE_UPGRADING, Styles.INFO))
        exit_code = run_upgrade_commands(commands)
        if exit_code != 0:
            console.print(
                _styled(Messages.ERROR_UPDATE_UPGRADE_FAILED.format(code=exit_code), Styles.ERROR)
            )
            raise typer.Exit(code=1)
        console.print(_styled(Messages.INFO_UPDATE_UPGRADE_DONE, Styles.SUCCESS))
        return

    console.print(
        _styled(Messages.INFO_UPDATE_UP_TO_DATE.format(latest=latest), Styles.SUCCESS)
    )


@app.command(help=Messages.HELP_ALIAS)
def alias() -> None:
    """Print a shell alias that maps `vx` to `vexor`."""
    shell_name = _detect_shell_name()
    alias_command = _resolve_alias_command(shell_name)
    typer.echo(alias_command)
    if not sys.stdin.isatty():
        return
    if not typer.confirm(Messages.PROMPT_ALIAS_APPLY):
        return

    profile_path = _resolve_alias_profile(shell_name)
    if profile_path is None:
        console.print(_styled(Messages.WARNING_ALIAS_PROFILE_MISSING, Styles.WARNING))
        return
    try:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        existing = profile_path.read_text(encoding="utf-8") if profile_path.exists() else ""
        if alias_command in existing:
            console.print(
                _styled(
                    Messages.INFO_ALIAS_ALREADY_SET.format(path=profile_path),
                    Styles.INFO,
                )
            )
            return
        with profile_path.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(alias_command + "\n")
        console.print(
            _styled(
                Messages.INFO_ALIAS_APPLIED.format(path=profile_path),
                Styles.SUCCESS,
            )
        )
    except OSError as exc:
        console.print(
            _styled(
                Messages.ERROR_ALIAS_WRITE.format(path=profile_path, reason=str(exc)),
                Styles.ERROR,
            )
        )
        raise typer.Exit(code=1)


@app.command()
def star() -> None:
    """Star the Vexor repository on GitHub (or use `gh` if available)."""
    gh_path = find_command_on_path("gh")
    if gh_path:
        try:
            subprocess.run(
                [gh_path, "repo", "star", REPO_OWNER_AND_NAME],
                capture_output=True,
                text=True,
                check=True,
            )
            console.print(_styled(Messages.INFO_STAR_SUCCESS, Styles.SUCCESS))
            return
        except subprocess.CalledProcessError:
            # gh CLI failed with a non-zero exit code; fall back to browser
            pass

    # Fall back to opening the browser
    console.print(_styled(Messages.INFO_STAR_BROWSER.format(url=PROJECT_URL), Styles.INFO))
    try:
        typer.launch(PROJECT_URL)
    except Exception as exc:  # pragma: no cover - depends on system setup
        console.print(
            _styled(
                f"Failed to open your browser for {PROJECT_URL}: {exc}",
                Styles.ERROR,
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def feedback() -> None:
    """Open the GitHub issue form for feedback."""

    url = f"{PROJECT_URL}/issues/new"
    gh = find_command_on_path("gh")
    if gh:
        console.print(_styled(Messages.INFO_FEEDBACK_GH, Styles.INFO))
        try:
            completed = subprocess.run(
                [gh, "issue", "create", "--repo", "scarletkc/vexor", "--web"],
                check=False,
            )
        except Exception as exc:  # pragma: no cover - OS specific
            console.print(
                _styled(
                    Messages.WARNING_FEEDBACK_GH_FAILED.format(reason=str(exc)),
                    Styles.WARNING,
                )
            )
        else:
            if completed.returncode == 0:
                return
            console.print(
                _styled(
                    Messages.WARNING_FEEDBACK_GH_FAILED.format(
                        reason=f"exit code {completed.returncode}"
                    ),
                    Styles.WARNING,
                )
            )

    console.print(_styled(Messages.INFO_FEEDBACK_OPENING.format(url=url), Styles.INFO))
    try:
        typer.launch(url)
    except Exception as exc:  # pragma: no cover - depends on system setup
        console.print(
            _styled(Messages.ERROR_FEEDBACK_LAUNCH.format(url=url, reason=str(exc)), Styles.ERROR)
        )
        raise typer.Exit(code=1) from exc


def _render_results(
    results: Sequence["SearchResult"],
    base: Path,
    backend: str | None,
    reranker: str | None,
) -> None:
    console.print(_styled(Messages.TABLE_TITLE, Styles.TITLE))
    if backend:
        line = f"{Messages.TABLE_BACKEND_PREFIX}{backend}"
        if reranker:
            line = f"{line} | {Messages.TABLE_RERANKER_PREFIX}{reranker}"
        console.print(_styled(line, Styles.INFO))
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


def _detect_shell_name() -> str | None:
    shell_env = os.environ.get("SHELL", "")
    if shell_env:
        name = Path(shell_env).name.lower()
        if name in {"bash", "zsh", "fish"}:
            return name
    if os.name == "nt":
        return "powershell"
    return None


def _resolve_powershell_profile() -> Path:
    home = Path.home()
    ps7_dir = home / "Documents" / "PowerShell"
    ps5_dir = home / "Documents" / "WindowsPowerShell"
    if ps7_dir.exists():
        return ps7_dir / "Microsoft.PowerShell_profile.ps1"
    if ps5_dir.exists():
        return ps5_dir / "Microsoft.PowerShell_profile.ps1"
    return ps7_dir / "Microsoft.PowerShell_profile.ps1"


def _resolve_alias_profile(shell_name: str | None) -> Path | None:
    if shell_name == "bash":
        return Path("~/.bashrc").expanduser()
    if shell_name == "zsh":
        return Path("~/.zshrc").expanduser()
    if shell_name == "fish":
        return Path("~/.config/fish/config.fish").expanduser()
    if shell_name == "powershell":
        return _resolve_powershell_profile()
    return None


def _resolve_alias_command(shell_name: str | None) -> str:
    if shell_name == "fish":
        return Messages.INFO_ALIAS_FISH
    if shell_name == "powershell":
        return Messages.INFO_ALIAS_POWERSHELL
    return Messages.INFO_ALIAS_VX


def run(argv: list[str] | None = None) -> None:
    """Entry point wrapper allowing optional argument override."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if should_auto_run_init(args, config_path=config_module.CONFIG_FILE):
        run_init_wizard()
        resume_cmd = "vexor"
        if args:
            resume_cmd = f"vexor {_format_command(args)}"
        console.print(_styled(Messages.INIT_RESUME_TITLE, Styles.INFO))
        console.print(resume_cmd)
        return
    if argv is None:
        app()
    else:
        app(args=args)


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
