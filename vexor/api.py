"""Public Python API for Vexor."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack, contextmanager
from pathlib import Path
from collections.abc import Mapping
from typing import Sequence

import numpy as np

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EXTRACT_BACKEND,
    DEFAULT_EXTRACT_CONCURRENCY,
    DEFAULT_PROVIDER,
    DEFAULT_RERANK,
    Config,
    RemoteRerankConfig,
    SUPPORTED_RERANKERS,
    config_from_json,
    config_dir_context,
    load_config,
    resolve_default_model,
    set_config_dir,
)
from .cache import cache_dir_context, set_cache_dir
from .modes import available_modes, get_strategy
from .services.index_service import (
    IndexResult,
    build_index,
    build_index_in_memory,
    clear_index_entries,
)
from .services.search_service import (
    SearchRequest,
    SearchResponse,
    perform_search,
    search_from_vectors,
)
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


@dataclass(slots=True)
class InMemoryIndex:
    base_path: Path
    paths: Sequence[Path]
    vectors: np.ndarray
    metadata: dict[str, object]
    model_name: str
    batch_size: int
    embed_concurrency: int
    provider: str
    base_url: str | None
    api_key: str | None
    local_cuda: bool
    rerank: str = DEFAULT_RERANK
    flashrank_model: str | None = None
    remote_rerank: RemoteRerankConfig | None = None

    def search(
        self,
        query: str,
        *,
        top: int = 5,
        rerank: str | None = None,
        flashrank_model: str | None = None,
        remote_rerank: RemoteRerankConfig | None = None,
        no_cache: bool = True,
    ) -> SearchResponse:
        """Search against the in-memory index without touching disk."""

        clean_query = query.strip()
        if not clean_query:
            raise VexorError(Messages.ERROR_EMPTY_QUERY)
        try:
            ensure_positive(top, "top")
        except ValueError as exc:
            raise VexorError(str(exc)) from exc

        effective_rerank = (rerank or self.rerank or DEFAULT_RERANK).strip().lower()
        if effective_rerank not in SUPPORTED_RERANKERS:
            effective_rerank = DEFAULT_RERANK

        include_hidden = bool(self.metadata.get("include_hidden", False))
        respect_gitignore = bool(self.metadata.get("respect_gitignore", True))
        mode = str(self.metadata.get("mode", "auto"))
        recursive = bool(self.metadata.get("recursive", True))
        exclude_patterns = tuple(self.metadata.get("exclude_patterns") or ())
        extensions = tuple(self.metadata.get("extensions") or ())

        request = SearchRequest(
            query=clean_query,
            directory=self.base_path,
            include_hidden=include_hidden,
            respect_gitignore=respect_gitignore,
            mode=mode,
            recursive=recursive,
            top_k=top,
            model_name=self.model_name,
            batch_size=self.batch_size,
            embed_concurrency=self.embed_concurrency,
            extract_concurrency=DEFAULT_EXTRACT_CONCURRENCY,
            extract_backend=DEFAULT_EXTRACT_BACKEND,
            provider=self.provider,
            base_url=self.base_url,
            api_key=self.api_key,
            local_cuda=self.local_cuda,
            exclude_patterns=exclude_patterns,
            extensions=extensions,
            auto_index=False,
            temporary_index=True,
            no_cache=no_cache,
            rerank=effective_rerank,
            flashrank_model=(
                flashrank_model
                if flashrank_model is not None
                else self.flashrank_model
            ),
            remote_rerank=(
                remote_rerank if remote_rerank is not None else self.remote_rerank
            ),
        )
        return search_from_vectors(
            request,
            paths=self.paths,
            file_vectors=self.vectors,
            metadata=self.metadata,
            is_stale=False,
        )


_RUNTIME_CONFIG: Config | None = None


@contextmanager
def _data_dir_context(
    data_dir: Path | str | None,
    *,
    config_dir: Path | str | None,
    cache_dir: Path | str | None,
):
    if data_dir is None and config_dir is None and cache_dir is None:
        yield
        return
    effective_config_dir = config_dir if config_dir is not None else data_dir
    effective_cache_dir = cache_dir if cache_dir is not None else data_dir
    with ExitStack() as stack:
        if effective_config_dir is not None:
            stack.enter_context(config_dir_context(effective_config_dir))
        if effective_cache_dir is not None:
            stack.enter_context(cache_dir_context(effective_cache_dir))
        yield


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


class VexorClient:
    """Session-style API wrapper for library use."""

    def __init__(
        self,
        *,
        data_dir: Path | str | None = None,
        config_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
        use_config: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.cache_dir = cache_dir
        self.use_config = use_config
        self._runtime_config: Config | None = None

    def set_config_json(
        self,
        payload: Mapping[str, object] | str | None,
        *,
        replace: bool = False,
    ) -> None:
        """Set in-memory config for this client from a JSON string or mapping."""
        if payload is None:
            self._runtime_config = None
            return
        base = None if replace else (self._runtime_config or load_config())
        try:
            self._runtime_config = config_from_json(payload, base=base)
        except ValueError as exc:
            raise VexorError(str(exc)) from exc

    @contextmanager
    def config_context(
        self,
        payload: Mapping[str, object] | str | None,
        *,
        replace: bool = False,
    ):
        """Temporarily override this client's in-memory config."""
        previous = self._runtime_config
        self.set_config_json(payload, replace=replace)
        try:
            yield self
        finally:
            self._runtime_config = previous

    def _resolve_dir_overrides(
        self,
        data_dir: Path | str | None,
        config_dir: Path | str | None,
        cache_dir: Path | str | None,
    ) -> tuple[Path | str | None, Path | str | None, Path | str | None]:
        resolved_data_dir = data_dir if data_dir is not None else self.data_dir
        resolved_config_dir = config_dir if config_dir is not None else self.config_dir
        resolved_cache_dir = cache_dir if cache_dir is not None else self.cache_dir
        return resolved_data_dir, resolved_config_dir, resolved_cache_dir

    def search(
        self,
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
        use_config: bool | None = None,
        config: Config | Mapping[str, object] | str | None = None,
        temporary_index: bool = False,
        no_cache: bool = False,
        data_dir: Path | str | None = None,
        config_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> SearchResponse:
        """Run a semantic search and return ranked results."""

        resolved_use_config = self.use_config if use_config is None else use_config
        resolved_data_dir, resolved_config_dir, resolved_cache_dir = (
            self._resolve_dir_overrides(data_dir, config_dir, cache_dir)
        )
        return _search_with_settings(
            query,
            path=path,
            top=top,
            include_hidden=include_hidden,
            respect_gitignore=respect_gitignore,
            mode=mode,
            recursive=recursive,
            extensions=extensions,
            exclude_patterns=exclude_patterns,
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
            use_config=resolved_use_config,
            config=config,
            temporary_index=temporary_index,
            no_cache=no_cache,
            runtime_config=self._runtime_config,
            data_dir=resolved_data_dir,
            config_dir=resolved_config_dir,
            cache_dir=resolved_cache_dir,
        )

    def index(
        self,
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
        use_config: bool | None = None,
        config: Config | Mapping[str, object] | str | None = None,
        data_dir: Path | str | None = None,
        config_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> IndexResult:
        """Build or refresh the index for the given directory."""

        resolved_use_config = self.use_config if use_config is None else use_config
        resolved_data_dir, resolved_config_dir, resolved_cache_dir = (
            self._resolve_dir_overrides(data_dir, config_dir, cache_dir)
        )
        return _index_with_settings(
            path=path,
            include_hidden=include_hidden,
            respect_gitignore=respect_gitignore,
            mode=mode,
            recursive=recursive,
            extensions=extensions,
            exclude_patterns=exclude_patterns,
            provider=provider,
            model=model,
            batch_size=batch_size,
            embed_concurrency=embed_concurrency,
            extract_concurrency=extract_concurrency,
            extract_backend=extract_backend,
            base_url=base_url,
            api_key=api_key,
            local_cuda=local_cuda,
            use_config=resolved_use_config,
            config=config,
            runtime_config=self._runtime_config,
            data_dir=resolved_data_dir,
            config_dir=resolved_config_dir,
            cache_dir=resolved_cache_dir,
        )

    def index_in_memory(
        self,
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
        use_config: bool | None = None,
        config: Config | Mapping[str, object] | str | None = None,
        no_cache: bool = True,
        data_dir: Path | str | None = None,
        config_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> InMemoryIndex:
        """Build an index in memory without writing to disk."""

        resolved_use_config = self.use_config if use_config is None else use_config
        resolved_data_dir, resolved_config_dir, resolved_cache_dir = (
            self._resolve_dir_overrides(data_dir, config_dir, cache_dir)
        )
        return _index_in_memory_with_settings(
            path=path,
            include_hidden=include_hidden,
            respect_gitignore=respect_gitignore,
            mode=mode,
            recursive=recursive,
            extensions=extensions,
            exclude_patterns=exclude_patterns,
            provider=provider,
            model=model,
            batch_size=batch_size,
            embed_concurrency=embed_concurrency,
            extract_concurrency=extract_concurrency,
            extract_backend=extract_backend,
            base_url=base_url,
            api_key=api_key,
            local_cuda=local_cuda,
            use_config=resolved_use_config,
            config=config,
            no_cache=no_cache,
            runtime_config=self._runtime_config,
            data_dir=resolved_data_dir,
            config_dir=resolved_config_dir,
            cache_dir=resolved_cache_dir,
        )

    def clear_index(
        self,
        path: Path | str = Path.cwd(),
        *,
        include_hidden: bool = False,
        respect_gitignore: bool = True,
        mode: str = "auto",
        recursive: bool = True,
        extensions: Sequence[str] | str | None = None,
        exclude_patterns: Sequence[str] | str | None = None,
        data_dir: Path | str | None = None,
        config_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> int:
        """Clear cached index entries for the given directory."""

        resolved_data_dir, resolved_config_dir, resolved_cache_dir = (
            self._resolve_dir_overrides(data_dir, config_dir, cache_dir)
        )
        return _clear_index_with_settings(
            path=path,
            include_hidden=include_hidden,
            respect_gitignore=respect_gitignore,
            mode=mode,
            recursive=recursive,
            extensions=extensions,
            exclude_patterns=exclude_patterns,
            data_dir=resolved_data_dir,
            config_dir=resolved_config_dir,
            cache_dir=resolved_cache_dir,
        )


@contextmanager
def config_context(
    payload: Mapping[str, object] | str | None,
    *,
    replace: bool = False,
    data_dir: Path | str | None = None,
    config_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    use_config: bool = True,
):
    """Yield a configured client for scoped API usage."""
    client = VexorClient(
        data_dir=data_dir,
        config_dir=config_dir,
        cache_dir=cache_dir,
        use_config=use_config,
    )
    client.set_config_json(payload, replace=replace)
    try:
        yield client
    finally:
        client.set_config_json(None)


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
    data_dir: Path | str | None = None,
    config_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> SearchResponse:
    """Run a semantic search and return ranked results."""
    return _search_with_settings(
        query,
        path=path,
        top=top,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode,
        recursive=recursive,
        extensions=extensions,
        exclude_patterns=exclude_patterns,
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
        config=config,
        temporary_index=temporary_index,
        no_cache=no_cache,
        runtime_config=_RUNTIME_CONFIG,
        data_dir=data_dir,
        config_dir=config_dir,
        cache_dir=cache_dir,
    )


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
    data_dir: Path | str | None = None,
    config_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> IndexResult:
    """Build or refresh the index for the given directory."""
    return _index_with_settings(
        path=path,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode,
        recursive=recursive,
        extensions=extensions,
        exclude_patterns=exclude_patterns,
        provider=provider,
        model=model,
        batch_size=batch_size,
        embed_concurrency=embed_concurrency,
        extract_concurrency=extract_concurrency,
        extract_backend=extract_backend,
        base_url=base_url,
        api_key=api_key,
        local_cuda=local_cuda,
        use_config=use_config,
        config=config,
        runtime_config=_RUNTIME_CONFIG,
        data_dir=data_dir,
        config_dir=config_dir,
        cache_dir=cache_dir,
    )


def index_in_memory(
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
    no_cache: bool = True,
    data_dir: Path | str | None = None,
    config_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> InMemoryIndex:
    """Build an index in memory without writing to disk."""
    return _index_in_memory_with_settings(
        path=path,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode,
        recursive=recursive,
        extensions=extensions,
        exclude_patterns=exclude_patterns,
        provider=provider,
        model=model,
        batch_size=batch_size,
        embed_concurrency=embed_concurrency,
        extract_concurrency=extract_concurrency,
        extract_backend=extract_backend,
        base_url=base_url,
        api_key=api_key,
        local_cuda=local_cuda,
        use_config=use_config,
        config=config,
        no_cache=no_cache,
        runtime_config=_RUNTIME_CONFIG,
        data_dir=data_dir,
        config_dir=config_dir,
        cache_dir=cache_dir,
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
    data_dir: Path | str | None = None,
    config_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> int:
    """Clear cached index entries for the given directory."""
    return _clear_index_with_settings(
        path=path,
        include_hidden=include_hidden,
        respect_gitignore=respect_gitignore,
        mode=mode,
        recursive=recursive,
        extensions=extensions,
        exclude_patterns=exclude_patterns,
        data_dir=data_dir,
        config_dir=config_dir,
        cache_dir=cache_dir,
    )


def _search_with_settings(
    query: str,
    *,
    path: Path | str,
    top: int,
    include_hidden: bool,
    respect_gitignore: bool,
    mode: str,
    recursive: bool,
    extensions: Sequence[str] | str | None,
    exclude_patterns: Sequence[str] | str | None,
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
    config: Config | Mapping[str, object] | str | None,
    temporary_index: bool,
    no_cache: bool,
    runtime_config: Config | None,
    data_dir: Path | str | None,
    config_dir: Path | str | None,
    cache_dir: Path | str | None,
) -> SearchResponse:
    with _data_dir_context(data_dir, config_dir=config_dir, cache_dir=cache_dir):
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
            runtime_config=runtime_config,
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


def _index_with_settings(
    *,
    path: Path | str,
    include_hidden: bool,
    respect_gitignore: bool,
    mode: str,
    recursive: bool,
    extensions: Sequence[str] | str | None,
    exclude_patterns: Sequence[str] | str | None,
    provider: str | None,
    model: str | None,
    batch_size: int | None,
    embed_concurrency: int | None,
    extract_concurrency: int | None,
    extract_backend: str | None,
    base_url: str | None,
    api_key: str | None,
    local_cuda: bool | None,
    use_config: bool,
    config: Config | Mapping[str, object] | str | None,
    runtime_config: Config | None,
    data_dir: Path | str | None,
    config_dir: Path | str | None,
    cache_dir: Path | str | None,
) -> IndexResult:
    with _data_dir_context(data_dir, config_dir=config_dir, cache_dir=cache_dir):
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
            runtime_config=runtime_config,
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


def _index_in_memory_with_settings(
    *,
    path: Path | str,
    include_hidden: bool,
    respect_gitignore: bool,
    mode: str,
    recursive: bool,
    extensions: Sequence[str] | str | None,
    exclude_patterns: Sequence[str] | str | None,
    provider: str | None,
    model: str | None,
    batch_size: int | None,
    embed_concurrency: int | None,
    extract_concurrency: int | None,
    extract_backend: str | None,
    base_url: str | None,
    api_key: str | None,
    local_cuda: bool | None,
    use_config: bool,
    config: Config | Mapping[str, object] | str | None,
    no_cache: bool,
    runtime_config: Config | None,
    data_dir: Path | str | None,
    config_dir: Path | str | None,
    cache_dir: Path | str | None,
) -> InMemoryIndex:
    with _data_dir_context(data_dir, config_dir=config_dir, cache_dir=cache_dir):
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
            runtime_config=runtime_config,
            config_override=config,
        )

        paths, vectors, metadata = build_index_in_memory(
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
            no_cache=no_cache,
        )

        return InMemoryIndex(
            base_path=directory,
            paths=paths,
            vectors=vectors,
            metadata=metadata,
            model_name=settings.model_name,
            batch_size=settings.batch_size,
            embed_concurrency=settings.embed_concurrency,
            provider=settings.provider,
            base_url=settings.base_url,
            api_key=settings.api_key,
            local_cuda=settings.local_cuda,
            rerank=settings.rerank,
            flashrank_model=settings.flashrank_model,
            remote_rerank=settings.remote_rerank,
        )


def _clear_index_with_settings(
    *,
    path: Path | str,
    include_hidden: bool,
    respect_gitignore: bool,
    mode: str,
    recursive: bool,
    extensions: Sequence[str] | str | None,
    exclude_patterns: Sequence[str] | str | None,
    data_dir: Path | str | None,
    config_dir: Path | str | None,
    cache_dir: Path | str | None,
) -> int:
    with _data_dir_context(data_dir, config_dir=config_dir, cache_dir=cache_dir):
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
