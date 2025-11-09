"""Command line interface for Vexor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    load_config,
    set_api_key,
    set_batch_size,
    set_model,
)
from .text import Messages, Styles
from .utils import collect_files, resolve_directory, format_path, ensure_positive

console = Console()
app = typer.Typer(
    help=Messages.APP_HELP,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@dataclass(slots=True)
class DisplayResult:
    path: Path
    score: float


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"Vexor v{__version__}")
        raise typer.Exit()


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
    try:
        cached_paths, file_vectors, meta = _load_index(directory, model_name, include_hidden)
    except FileNotFoundError:
        console.print(
            _styled(Messages.ERROR_INDEX_MISSING.format(path=directory), Styles.ERROR)
        )
        raise typer.Exit(code=1)

    _warn_if_stale(directory, include_hidden, meta.get("files", []))

    if not cached_paths:
        console.print(_styled(Messages.INFO_INDEX_EMPTY, Styles.WARNING))
        raise typer.Exit(code=0)

    searcher = _create_searcher(model_name=model_name, batch_size=batch_size)
    try:
        query_vector = searcher.embed_texts([clean_query])[0]
    except RuntimeError as exc:
        console.print(_styled(str(exc), Styles.ERROR))
        raise typer.Exit(code=1)

    from sklearn.metrics.pairwise import cosine_similarity  # local import

    similarities = cosine_similarity(
        query_vector.reshape(1, -1), file_vectors
    )[0]
    scored = [
        DisplayResult(path=path, score=float(score))
        for path, score in zip(cached_paths, similarities)
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    results = scored[:top]

    if not results:
        console.print(_styled(Messages.INFO_NO_RESULTS, Styles.WARNING))
        raise typer.Exit(code=0)

    _render_results(results, directory, searcher.device)


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
) -> None:
    """Create or refresh the cached index for the given directory."""
    config = load_config()
    model_name = config.model or DEFAULT_MODEL
    batch_size = config.batch_size if config.batch_size is not None else DEFAULT_BATCH_SIZE

    directory = resolve_directory(path)
    files = collect_files(directory, include_hidden=include_hidden)
    if not files:
        console.print(_styled(Messages.INFO_NO_FILES, Styles.WARNING))
        raise typer.Exit(code=0)

    existing_meta = _load_index_metadata_safe(directory, model_name, include_hidden)
    if existing_meta:
        cached_files = existing_meta.get("files", [])
        if cached_files and _is_cache_current(
            directory, include_hidden, cached_files, current_files=files
        ):
            console.print(
                _styled(Messages.INFO_INDEX_UP_TO_DATE.format(path=directory), Styles.INFO)
            )
            return

    searcher = _create_searcher(model_name=model_name, batch_size=batch_size)
    file_labels = [_label_for_path(file) for file in files]
    embeddings = searcher.embed_texts(file_labels)

    cache_path = _store_index(
        root=directory,
        model=model_name,
        include_hidden=include_hidden,
        files=files,
        embeddings=embeddings,
    )
    console.print(_styled(Messages.INFO_INDEX_SAVED.format(path=cache_path), Styles.SUCCESS))


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
    changed = False

    if set_api_key_option is not None:
        set_api_key(set_api_key_option)
        console.print(_styled(Messages.INFO_API_SAVED, Styles.SUCCESS))
        changed = True
    if clear_api_key:
        set_api_key(None)
        console.print(_styled(Messages.INFO_API_CLEARED, Styles.SUCCESS))
        changed = True
    if set_model_option is not None:
        set_model(set_model_option)
        console.print(
            _styled(Messages.INFO_MODEL_SET.format(value=set_model_option), Styles.SUCCESS)
        )
        changed = True
    if set_batch_option is not None:
        if set_batch_option < 0:
            raise typer.BadParameter(Messages.ERROR_BATCH_NEGATIVE)
        set_batch_size(set_batch_option)
        console.print(
            _styled(Messages.INFO_BATCH_SET.format(value=set_batch_option), Styles.SUCCESS)
        )
        changed = True

    if show or not changed:
        cfg = load_config()
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


def _render_results(results: Sequence[DisplayResult], base: Path, backend: str | None) -> None:
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


def _create_searcher(model_name: str, batch_size: int):
    from .search import VexorSearcher  # Local import keeps CLI startup fast

    return VexorSearcher(model_name=model_name, batch_size=batch_size)


def _label_for_path(path: Path) -> str:
    return path.name.replace("_", " ")


def _load_index(root: Path, model: str, include_hidden: bool):
    from .cache import load_index_vectors  # local import

    return load_index_vectors(root, model, include_hidden)


def _load_index_metadata_safe(root: Path, model: str, include_hidden: bool):
    from .cache import load_index  # local import

    try:
        return load_index(root, model, include_hidden)
    except FileNotFoundError:
        return None


def _store_index(**kwargs):
    from .cache import store_index  # local import

    return store_index(**kwargs)


def _is_cache_current(
    root: Path,
    include_hidden: bool,
    cached_files: Sequence[dict],
    *,
    current_files: Sequence[Path] | None = None,
) -> bool:
    if not cached_files:
        return False
    from .cache import compare_snapshot  # local import

    return compare_snapshot(
        root,
        include_hidden,
        cached_files,
        current_files=current_files,
    )


def _warn_if_stale(root: Path, include_hidden: bool, cached_files: Sequence[dict]) -> None:
    if not cached_files:
        return
    if not _is_cache_current(root, include_hidden, cached_files):
        console.print(
            _styled(Messages.WARNING_INDEX_STALE.format(path=root), Styles.WARNING)
        )


def _styled(text: str, style: str) -> str:
    return f"[{style}]{text}[/{style}]"


def run(argv: list[str] | None = None) -> None:
    """Entry point wrapper allowing optional argument override."""
    if argv is None:
        app()
    else:
        app(args=list(argv))
