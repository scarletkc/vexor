"""Public Python API for Vexor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Sequence

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PROVIDER,
    DEFAULT_RERANK,
    Config,
    RemoteRerankConfig,
    SUPPORTED_RERANKERS,
    config_from_json,
    load_config,
    resolve_default_model,
    set_config_dir,
)
from .cache import set_cache_dir
from .modes import available_modes, get_strategy
from .services.index_service import IndexResult, build_index, clear_index_entries
from .services.search_service import SearchRequest, SearchResponse, perform_search
from .text import Messages
from .utils import (
    ensure_positive,
    normalize_exclude_patterns,
    normalize_extensions,
    resolve_directory,
)


class VexorError(ValueError):
    """Raised when the Vexor public API input is invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    provider: str
    model_name: str
    batch_size: int
    embed_concurrency: int
    extract_concurrency: int
    extract_backend: str
    base_url: str | None
    api_key: str | None
    local_cuda: bool
    auto_index: bool
    rerank: str
    flashrank_model: str | None
    remote_rerank: RemoteRerankConfig | None


_RUNTIME_CONFIG: Config | None = None


def set_data_dir(path: Path | str | None) -> None:
    """Set the base directory for config and cache data."""
    set_config_dir(path)
    set_cache_dir(path)


def set_config_json(
    payload: Mapping[str, object] | str | None, *, replace: bool = False
) -> None:
    """Set in-memory config for API calls from a JSON string or mapping."""
    global _RUNTIME_CONFIG
    if payload is None:
        _RUNTIME_CONFIG = None
        return
    base = None if replace else (_RUNTIME_CONFIG or load_config())
    try:
        _RUNTIME_CONFIG = config_from_json(payload, base=base)
    except ValueError as exc:
        raise VexorError(str(exc)) from exc


def search(
    query: str,
    *,
    path: Path | str = Path.cwd(),
    top: int = 5,
    include_hidden: bool = False,
    respect_gitignore: bool = True,
    mode: str = "auto",
    recursive: bool = True,
    extensions: Sequence[str] | str | None = None,
    exclude_patterns: Sequence[str] | str | None = None,
    provider: str | None = None,
    model: str | None = None,
    batch_size: int | None = None,
    embed_concurrency: int | None = None,
    extract_concurrency: int | None = None,
    extract_backend: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    local_cuda: bool | None = None,
    auto_index: bool | None = None,
    use_config: bool = True,
    config: Config | Mapping[str, object] | str | None = None,
    temporary_index: bool = False,
    no_cache: bool = False,
) -> SearchResponse:
    """Run a semantic search and return ranked results."""

    clean_query = query.strip()
    if not clean_query:
        raise VexorError(Messages.ERROR_EMPTY_QUERY)
    try:
        ensure_positive(top, "top")
    except ValueError as exc:
        raise VexorError(str(exc)) from exc

    directory = resolve_directory(path)
    mode_value = _validate_mode(mode)
    normalized_exts = _normalize_extensions(extensions)
    normalized_excludes = _normalize_excludes(exclude_patterns)
    if extensions and not normalized_exts:
        raise VexorError(Messages.ERROR_EXTENSIONS_EMPTY)

    settings = _resolve_settings(
        provider=provider,
        model=model,
        batch_size=batch_size,
        embed_concurrency=embed_concurrency,
        extract_concurrency=extract_concurrency,
        extract_backend=extract_backend,
        base_url=base_url,
        api_key=api_key,
        local_cuda=local_cuda,
        auto_index=auto_index,
        use_config=use_config,
        runtime_config=_RUNTIME_CONFIG,
        config_override=config,
    )

    request = SearchRequest(
        query=clean_query,
        directory=directory,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode_value,
        recursive=recursive,
        top_k=top,
        model_name=settings.model_name,
        batch_size=settings.batch_size,
        embed_concurrency=settings.embed_concurrency,
        extract_concurrency=settings.extract_concurrency,
        extract_backend=settings.extract_backend,
        provider=settings.provider,
        base_url=settings.base_url,
        api_key=settings.api_key,
        local_cuda=settings.local_cuda,
        exclude_patterns=normalized_excludes,
        extensions=normalized_exts,
        auto_index=settings.auto_index,
        temporary_index=temporary_index,
        no_cache=no_cache,
        rerank=settings.rerank,
        flashrank_model=settings.flashrank_model,
        remote_rerank=settings.remote_rerank,
    )
    return perform_search(request)


def index(
    path: Path | str = Path.cwd(),
    *,
    include_hidden: bool = False,
    respect_gitignore: bool = True,
    mode: str = "auto",
    recursive: bool = True,
    extensions: Sequence[str] | str | None = None,
    exclude_patterns: Sequence[str] | str | None = None,
    provider: str | None = None,
    model: str | None = None,
    batch_size: int | None = None,
    embed_concurrency: int | None = None,
    extract_concurrency: int | None = None,
    extract_backend: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    local_cuda: bool | None = None,
    use_config: bool = True,
    config: Config | Mapping[str, object] | str | None = None,
) -> IndexResult:
    """Build or refresh the index for the given directory."""

    directory = resolve_directory(path)
    mode_value = _validate_mode(mode)
    normalized_exts = _normalize_extensions(extensions)
    normalized_excludes = _normalize_excludes(exclude_patterns)
    if extensions and not normalized_exts:
        raise VexorError(Messages.ERROR_EXTENSIONS_EMPTY)

    settings = _resolve_settings(
        provider=provider,
        model=model,
        batch_size=batch_size,
        embed_concurrency=embed_concurrency,
        extract_concurrency=extract_concurrency,
        extract_backend=extract_backend,
        base_url=base_url,
        api_key=api_key,
        local_cuda=local_cuda,
        auto_index=None,
        use_config=use_config,
        runtime_config=_RUNTIME_CONFIG,
        config_override=config,
    )

    return build_index(
        directory,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode_value,
        recursive=recursive,
        model_name=settings.model_name,
        batch_size=settings.batch_size,
        embed_concurrency=settings.embed_concurrency,
        extract_concurrency=settings.extract_concurrency,
        extract_backend=settings.extract_backend,
        provider=settings.provider,
        base_url=settings.base_url,
        api_key=settings.api_key,
        local_cuda=settings.local_cuda,
        exclude_patterns=normalized_excludes,
        extensions=normalized_exts,
    )


def clear_index(
    path: Path | str = Path.cwd(),
    *,
    include_hidden: bool = False,
    respect_gitignore: bool = True,
    mode: str = "auto",
    recursive: bool = True,
    extensions: Sequence[str] | str | None = None,
    exclude_patterns: Sequence[str] | str | None = None,
) -> int:
    """Clear cached index entries for the given directory."""

    directory = resolve_directory(path)
    mode_value = _validate_mode(mode)
    normalized_exts = _normalize_extensions(extensions)
    normalized_excludes = _normalize_excludes(exclude_patterns)
    if extensions and not normalized_exts:
        raise VexorError(Messages.ERROR_EXTENSIONS_EMPTY)

    return clear_index_entries(
        directory,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode_value,
        recursive=recursive,
        exclude_patterns=normalized_excludes,
        extensions=normalized_exts,
    )


def _validate_mode(mode: str) -> str:
    try:
        get_strategy(mode)
    except ValueError as exc:
        allowed = ", ".join(available_modes())
        raise VexorError(
            Messages.ERROR_MODE_INVALID.format(value=mode, allowed=allowed)
        ) from exc
    return mode




def _normalize_extensions(values: Sequence[str] | str | None) -> tuple[str, ...]:
    return normalize_extensions(_coerce_iterable(values))


def _normalize_excludes(values: Sequence[str] | str | None) -> tuple[str, ...]:
    return normalize_exclude_patterns(_coerce_iterable(values))


def _coerce_iterable(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def _resolve_settings(
    *,
    provider: str | None,
    model: str | None,
    batch_size: int | None,
    embed_concurrency: int | None,
    extract_concurrency: int | None,
    extract_backend: str | None,
    base_url: str | None,
    api_key: str | None,
    local_cuda: bool | None,
    auto_index: bool | None,
    use_config: bool,
    runtime_config: Config | None = None,
    config_override: Config | Mapping[str, object] | str | None = None,
) -> RuntimeSettings:
    config = (
        runtime_config if (use_config and runtime_config is not None) else None
    )
    if config is None:
        config = load_config() if use_config else Config()
    if config_override is not None:
        config = _apply_config_override(config, config_override)
    provider_value = (provider or config.provider or DEFAULT_PROVIDER).lower()
    rerank_value = (config.rerank or DEFAULT_RERANK).strip().lower()
    if rerank_value not in SUPPORTED_RERANKERS:
        rerank_value = DEFAULT_RERANK
    model_name = resolve_default_model(
        provider_value,
        model if model is not None else config.model,
    )
    batch_value = (
        batch_size
        if batch_size is not None
        else (config.batch_size if config.batch_size is not None else DEFAULT_BATCH_SIZE)
    )
    embed_value = (
        embed_concurrency if embed_concurrency is not None else config.embed_concurrency
    )
    extract_value = (
        extract_concurrency
        if extract_concurrency is not None
        else config.extract_concurrency
    )
    extract_backend_value = (
        extract_backend if extract_backend is not None else config.extract_backend
    )
    return RuntimeSettings(
        provider=provider_value,
        model_name=model_name,
        batch_size=batch_value,
        embed_concurrency=embed_value,
        extract_concurrency=extract_value,
        extract_backend=extract_backend_value,
        base_url=base_url if base_url is not None else config.base_url,
        api_key=api_key if api_key is not None else config.api_key,
        local_cuda=bool(local_cuda if local_cuda is not None else config.local_cuda),
        auto_index=bool(auto_index if auto_index is not None else config.auto_index),
        rerank=rerank_value,
        flashrank_model=config.flashrank_model,
        remote_rerank=config.remote_rerank,
    )


def _apply_config_override(
    base: Config,
    override: Config | Mapping[str, object] | str,
) -> Config:
    if isinstance(override, Config):
        return override
    try:
        return config_from_json(override, base=base)
    except ValueError as exc:
        raise VexorError(str(exc)) from exc
