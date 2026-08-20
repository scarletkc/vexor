"""Public Python API for Vexor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import (
    IndexVectorCache,
    cache_dir_context,
    create_project_cache_dir,
    project_cache_context,
    set_cache_dir,
)
from .collection_store import CollectionError, CollectionInfo, StoredRecord
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBED_CONCURRENCY,
    DEFAULT_EXTRACT_BACKEND,
    DEFAULT_EXTRACT_CONCURRENCY,
    DEFAULT_RERANK,
    SUPPORTED_RERANKERS,
    Config,
    RemoteRerankConfig,
    _coerce_config_payload,
    config_dir_context,
    config_from_json,
    load_config,
    set_config_dir,
)
from .modes import available_modes, get_strategy
from .providers.capabilities import (
    DEFAULT_PROVIDER,
    resolve_default_model,
    validate_embedding_dimensions_for_model,
)
from .search import VexorSearcher
from .services import collection_service
from .services.collection_service import (
    DEFAULT_COLLECTION_RERANK,
    RecordResult,
    UpsertReport,
)
from .services.freshness_service import FreshnessTracker
from .services.index_service import (
    IndexResult,
    build_index,
    build_index_in_memory,
    clear_index_entries,
)
from .services.search_service import (
    DEFAULT_CONTENT_CHARS_PER_RESULT,
    DEFAULT_CONTENT_CHARS_TOTAL,
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
    embedding_dimensions: int | None


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
    embedding_dimensions: int | None = None
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
        include_content: bool = False,
        content_chars_per_result: int = DEFAULT_CONTENT_CHARS_PER_RESULT,
        content_chars_total: int = DEFAULT_CONTENT_CHARS_TOTAL,
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
            embedding_dimensions=self.embedding_dimensions,
            flashrank_model=(
                flashrank_model
                if flashrank_model is not None
                else self.flashrank_model
            ),
            remote_rerank=(
                remote_rerank if remote_rerank is not None else self.remote_rerank
            ),
            include_content=include_content,
            content_chars_per_result=content_chars_per_result,
            content_chars_total=content_chars_total,
        )
        return search_from_vectors(
            request,
            paths=self.paths,
            file_vectors=self.vectors,
            metadata=self.metadata,
            is_stale=False,
        )


@dataclass(frozen=True, slots=True)
class _RuntimeConfigOverride:
    payload: Mapping[str, object]
    replace: bool = False


_RUNTIME_CONFIG: _RuntimeConfigOverride | None = None


def _update_runtime_config_override(
    current: _RuntimeConfigOverride | None,
    payload: Mapping[str, object] | str,
    *,
    replace: bool,
) -> _RuntimeConfigOverride:
    """Validate and accumulate a deferred in-memory config override."""

    try:
        data = dict(_coerce_config_payload(payload))
        if current is None or replace:
            combined = data
            effective_replace = replace
        else:
            combined = {**current.payload, **data}
            effective_replace = current.replace
        base = Config() if effective_replace else load_config()
        config_from_json(combined, base=base)
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        raise VexorError(str(exc)) from exc
    return _RuntimeConfigOverride(
        payload=combined,
        replace=effective_replace,
    )


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
    _RUNTIME_CONFIG = _update_runtime_config_override(
        _RUNTIME_CONFIG,
        payload,
        replace=replace,
    )


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
        self._runtime_config: _RuntimeConfigOverride | None = None
        self._index_vector_cache = IndexVectorCache()
        self._freshness_tracker = FreshnessTracker()

    def close(self) -> None:
        """Release background resources owned by this client."""

        self._freshness_tracker.close()
        self._index_vector_cache.clear()

    def __enter__(self) -> VexorClient:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

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
        self._runtime_config = _update_runtime_config_override(
            self._runtime_config,
            payload,
            replace=replace,
        )

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
        path: Path | str = ".",
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
        embedding_dimensions: int | None = None,
        auto_index: bool | None = None,
        use_config: bool | None = None,
        config: Config | Mapping[str, object] | str | None = None,
        temporary_index: bool = False,
        no_cache: bool = False,
        include_content: bool = False,
        content_chars_per_result: int = DEFAULT_CONTENT_CHARS_PER_RESULT,
        content_chars_total: int = DEFAULT_CONTENT_CHARS_TOTAL,
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
            embedding_dimensions=embedding_dimensions,
            auto_index=auto_index,
            use_config=resolved_use_config,
            config=config,
            temporary_index=temporary_index,
            no_cache=no_cache,
            include_content=include_content,
            content_chars_per_result=content_chars_per_result,
            content_chars_total=content_chars_total,
            runtime_config=self._runtime_config,
            data_dir=resolved_data_dir,
            config_dir=resolved_config_dir,
            cache_dir=resolved_cache_dir,
            index_vector_cache=self._index_vector_cache,
            freshness_tracker=self._freshness_tracker,
        )

    def index(
        self,
        path: Path | str = ".",
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
        embedding_dimensions: int | None = None,
        local: bool = False,
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
        self._index_vector_cache.clear()
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
            embedding_dimensions=embedding_dimensions,
            local=local,
            use_config=resolved_use_config,
            config=config,
            runtime_config=self._runtime_config,
            data_dir=resolved_data_dir,
            config_dir=resolved_config_dir,
            cache_dir=resolved_cache_dir,
        )

    def index_in_memory(
        self,
        path: Path | str = ".",
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
        embedding_dimensions: int | None = None,
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
            embedding_dimensions=embedding_dimensions,
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
        path: Path | str = ".",
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
        self._index_vector_cache.clear()
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

    def collection(
        self,
        name: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        embed_concurrency: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        local_cuda: bool | None = None,
        embedding_dimensions: int | None = None,
        use_config: bool | None = None,
        config: Config | Mapping[str, object] | str | None = None,
        no_cache: bool = False,
        data_dir: Path | str | None = None,
        config_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> CollectionHandle:
        """Return a handle to the named record collection.

        The collection is created on first write, not here, so asking for a
        handle never touches the database.
        """

        resolved_use_config = self.use_config if use_config is None else use_config
        resolved_data_dir, resolved_config_dir, resolved_cache_dir = (
            self._resolve_dir_overrides(data_dir, config_dir, cache_dir)
        )
        return CollectionHandle(
            name,
            provider=provider,
            model=model,
            batch_size=batch_size,
            embed_concurrency=embed_concurrency,
            base_url=base_url,
            api_key=api_key,
            local_cuda=local_cuda,
            embedding_dimensions=embedding_dimensions,
            use_config=resolved_use_config,
            config=config,
            runtime_config=self._runtime_config,
            no_cache=no_cache,
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
        client.close()


def search(
    query: str,
    *,
    path: Path | str = ".",
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
    embedding_dimensions: int | None = None,
    auto_index: bool | None = None,
    use_config: bool = True,
    config: Config | Mapping[str, object] | str | None = None,
    temporary_index: bool = False,
    no_cache: bool = False,
    include_content: bool = False,
    content_chars_per_result: int = DEFAULT_CONTENT_CHARS_PER_RESULT,
    content_chars_total: int = DEFAULT_CONTENT_CHARS_TOTAL,
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
        embedding_dimensions=embedding_dimensions,
        auto_index=auto_index,
        use_config=use_config,
        config=config,
        temporary_index=temporary_index,
        no_cache=no_cache,
        include_content=include_content,
        content_chars_per_result=content_chars_per_result,
        content_chars_total=content_chars_total,
        runtime_config=_RUNTIME_CONFIG,
        data_dir=data_dir,
        config_dir=config_dir,
        cache_dir=cache_dir,
    )


def index(
    path: Path | str = ".",
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
    embedding_dimensions: int | None = None,
    local: bool = False,
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
        embedding_dimensions=embedding_dimensions,
        local=local,
        use_config=use_config,
        config=config,
        runtime_config=_RUNTIME_CONFIG,
        data_dir=data_dir,
        config_dir=config_dir,
        cache_dir=cache_dir,
    )


def index_in_memory(
    path: Path | str = ".",
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
    embedding_dimensions: int | None = None,
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
        embedding_dimensions=embedding_dimensions,
        use_config=use_config,
        config=config,
        no_cache=no_cache,
        runtime_config=_RUNTIME_CONFIG,
        data_dir=data_dir,
        config_dir=config_dir,
        cache_dir=cache_dir,
    )


def clear_index(
    path: Path | str = ".",
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
    embedding_dimensions: int | None,
    auto_index: bool | None,
    use_config: bool,
    config: Config | Mapping[str, object] | str | None,
    temporary_index: bool,
    no_cache: bool,
    include_content: bool = False,
    content_chars_per_result: int = DEFAULT_CONTENT_CHARS_PER_RESULT,
    content_chars_total: int = DEFAULT_CONTENT_CHARS_TOTAL,
    runtime_config: _RuntimeConfigOverride | None,
    data_dir: Path | str | None,
    config_dir: Path | str | None,
    cache_dir: Path | str | None,
    index_vector_cache: IndexVectorCache | None = None,
    freshness_tracker: FreshnessTracker | None = None,
) -> SearchResponse:
    with (
        _data_dir_context(data_dir, config_dir=config_dir, cache_dir=cache_dir),
        project_cache_context(directory := resolve_directory(path)),
    ):
        clean_query = query.strip()
        if not clean_query:
            raise VexorError(Messages.ERROR_EMPTY_QUERY)
        try:
            ensure_positive(top, "top")
        except ValueError as exc:
            raise VexorError(str(exc)) from exc

        mode_value = _validate_mode(mode)
        normalized_exts = _normalize_extensions(extensions)
        normalized_excludes = _normalize_excludes(exclude_patterns)
        if extensions and not normalized_exts:
            raise VexorError(Messages.ERROR_EXTENSIONS_EMPTY)

        settings = _resolve_settings(
            directory=directory,
            provider=provider,
            model=model,
            batch_size=batch_size,
            embed_concurrency=embed_concurrency,
            extract_concurrency=extract_concurrency,
            extract_backend=extract_backend,
            base_url=base_url,
            api_key=api_key,
            local_cuda=local_cuda,
            embedding_dimensions=embedding_dimensions,
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
            embedding_dimensions=settings.embedding_dimensions,
            flashrank_model=settings.flashrank_model,
            remote_rerank=settings.remote_rerank,
            include_content=include_content,
            content_chars_per_result=content_chars_per_result,
            content_chars_total=content_chars_total,
            index_vector_cache=index_vector_cache,
            freshness_tracker=freshness_tracker,
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
    embedding_dimensions: int | None,
    local: bool,
    use_config: bool,
    config: Config | Mapping[str, object] | str | None,
    runtime_config: _RuntimeConfigOverride | None,
    data_dir: Path | str | None,
    config_dir: Path | str | None,
    cache_dir: Path | str | None,
) -> IndexResult:
    directory = resolve_directory(path)
    if local:
        create_project_cache_dir(directory)
    with (
        _data_dir_context(data_dir, config_dir=config_dir, cache_dir=cache_dir),
        project_cache_context(directory),
    ):
        mode_value = _validate_mode(mode)
        normalized_exts = _normalize_extensions(extensions)
        normalized_excludes = _normalize_excludes(exclude_patterns)
        if extensions and not normalized_exts:
            raise VexorError(Messages.ERROR_EXTENSIONS_EMPTY)

        settings = _resolve_settings(
            directory=directory,
            provider=provider,
            model=model,
            batch_size=batch_size,
            embed_concurrency=embed_concurrency,
            extract_concurrency=extract_concurrency,
            extract_backend=extract_backend,
            base_url=base_url,
            api_key=api_key,
            local_cuda=local_cuda,
            embedding_dimensions=embedding_dimensions,
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
            embedding_dimensions=settings.embedding_dimensions,
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
    embedding_dimensions: int | None,
    use_config: bool,
    config: Config | Mapping[str, object] | str | None,
    no_cache: bool,
    runtime_config: _RuntimeConfigOverride | None,
    data_dir: Path | str | None,
    config_dir: Path | str | None,
    cache_dir: Path | str | None,
) -> InMemoryIndex:
    with (
        _data_dir_context(data_dir, config_dir=config_dir, cache_dir=cache_dir),
        project_cache_context(directory := resolve_directory(path)),
    ):
        mode_value = _validate_mode(mode)
        normalized_exts = _normalize_extensions(extensions)
        normalized_excludes = _normalize_excludes(exclude_patterns)
        if extensions and not normalized_exts:
            raise VexorError(Messages.ERROR_EXTENSIONS_EMPTY)

        settings = _resolve_settings(
            directory=directory,
            provider=provider,
            model=model,
            batch_size=batch_size,
            embed_concurrency=embed_concurrency,
            extract_concurrency=extract_concurrency,
            extract_backend=extract_backend,
            base_url=base_url,
            api_key=api_key,
            local_cuda=local_cuda,
            embedding_dimensions=embedding_dimensions,
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
            embedding_dimensions=settings.embedding_dimensions,
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
            embedding_dimensions=settings.embedding_dimensions,
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
    with (
        _data_dir_context(data_dir, config_dir=config_dir, cache_dir=cache_dir),
        project_cache_context(directory := resolve_directory(path)),
    ):
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
    directory: Path | str | None,
    provider: str | None,
    model: str | None,
    batch_size: int | None,
    embed_concurrency: int | None,
    extract_concurrency: int | None,
    extract_backend: str | None,
    base_url: str | None,
    api_key: str | None,
    local_cuda: bool | None,
    embedding_dimensions: int | None,
    auto_index: bool | None,
    use_config: bool,
    runtime_config: _RuntimeConfigOverride | None = None,
    config_override: Config | Mapping[str, object] | str | None = None,
) -> RuntimeSettings:
    try:
        if not use_config:
            config = Config()
        elif runtime_config is not None and runtime_config.replace:
            config = config_from_json(runtime_config.payload, base=Config())
        else:
            config = load_config(directory)
            if runtime_config is not None:
                config = config_from_json(runtime_config.payload, base=config)
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        raise VexorError(str(exc)) from exc
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
    resolved_embedding_dimensions = _coerce_embedding_dimensions(
        embedding_dimensions
        if embedding_dimensions is not None
        else config.embedding_dimensions
    )
    try:
        validate_embedding_dimensions_for_model(
            resolved_embedding_dimensions,
            model_name,
        )
    except ValueError as exc:
        raise VexorError(str(exc)) from exc

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
        embedding_dimensions=resolved_embedding_dimensions,
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


def _coerce_embedding_dimensions(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise VexorError(Messages.ERROR_EMBEDDING_DIMENSIONS_INVALID)
    if not isinstance(value, int):
        raise VexorError(Messages.ERROR_EMBEDDING_DIMENSIONS_INVALID)
    if value == 0:
        return None
    if value < 0:
        raise VexorError(Messages.ERROR_EMBEDDING_DIMENSIONS_INVALID)
    return value


@contextmanager
def _collection_errors():
    """Translate store/service contract errors into the public error type."""

    try:
        yield
    except CollectionError as exc:
        raise VexorError(str(exc)) from exc


class CollectionHandle:
    """Session handle for one named collection.

    A collection holds caller-supplied records instead of files, so nothing here
    takes a path. The embedding provider, model, and vector width are pinned on
    first write and verified on every later one; changing any of them raises
    rather than mixing incompatible vectors into one corpus.
    """

    def __init__(
        self,
        name: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        embed_concurrency: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        local_cuda: bool | None = None,
        embedding_dimensions: int | None = None,
        use_config: bool = True,
        config: Config | Mapping[str, object] | str | None = None,
        runtime_config: _RuntimeConfigOverride | None = None,
        no_cache: bool = False,
        data_dir: Path | str | None = None,
        config_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.name = name
        self._provider = provider
        self._model = model
        self._batch_size = batch_size
        self._embed_concurrency = embed_concurrency
        self._base_url = base_url
        self._api_key = api_key
        self._local_cuda = local_cuda
        self._embedding_dimensions = embedding_dimensions
        self._use_config = use_config
        self._config = config
        self._runtime_config = runtime_config
        self._no_cache = no_cache
        self._data_dir = data_dir
        self._config_dir = config_dir
        self._cache_dir = cache_dir

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CollectionHandle(name={self.name!r})"

    def _settings(self) -> RuntimeSettings:
        # ``directory=None`` on purpose: collections live in the shared cache
        # directory, so picking up a project-local config from the working
        # directory would make the pinned provider depend on where Python
        # happened to be started.
        return _resolve_settings(
            directory=None,
            provider=self._provider,
            model=self._model,
            batch_size=self._batch_size,
            embed_concurrency=self._embed_concurrency,
            extract_concurrency=None,
            extract_backend=None,
            base_url=self._base_url,
            api_key=self._api_key,
            local_cuda=self._local_cuda,
            embedding_dimensions=self._embedding_dimensions,
            auto_index=None,
            use_config=self._use_config,
            runtime_config=self._runtime_config,
            config_override=self._config,
        )

    def _dir_context(self):
        return _data_dir_context(
            self._data_dir,
            config_dir=self._config_dir,
            cache_dir=self._cache_dir,
        )

    def _searcher(self, settings: RuntimeSettings) -> VexorSearcher:
        return VexorSearcher(
            model_name=settings.model_name,
            batch_size=settings.batch_size,
            embed_concurrency=settings.embed_concurrency or DEFAULT_EMBED_CONCURRENCY,
            provider=settings.provider,
            base_url=settings.base_url,
            api_key=settings.api_key,
            local_cuda=settings.local_cuda,
            embedding_dimensions=settings.embedding_dimensions,
        )

    def upsert_many(
        self,
        records: Sequence[Mapping[str, object]],
    ) -> UpsertReport:
        """Insert or replace records, embedding only the texts that changed.

        Each record is a mapping with ``id`` and ``text``, plus an optional flat
        ``metadata`` mapping of scalars. A record whose text is unchanged since
        the last upsert keeps its stored vector and postings; its metadata is
        still replaced, so a pure metadata edit costs nothing to embed.
        """

        with self._dir_context(), _collection_errors():
            settings = self._settings()
            return collection_service.upsert_records(
                name=self.name,
                records=records,
                searcher=self._searcher(settings),
                model_name=settings.model_name,
                provider=settings.provider,
                embedding_dimension=settings.embedding_dimensions,
                no_cache=self._no_cache,
            )

    def upsert(
        self,
        record_id: str,
        text: str,
        metadata: Mapping[str, object] | None = None,
    ) -> UpsertReport:
        """Insert or replace a single record."""

        return self.upsert_many(
            [{"id": record_id, "text": text, "metadata": metadata}]
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, object] | None = None,
        rerank: str = DEFAULT_COLLECTION_RERANK,
    ) -> list[RecordResult]:
        """Search this collection, applying *filters* before anything is scored.

        Filtering is strict and resolves to a candidate id set first, so a
        record that never reaches the global head is still findable inside its
        own filtered subset.
        """

        with self._dir_context(), _collection_errors():
            settings = self._settings()
            return collection_service.search_records(
                name=self.name,
                query=query,
                searcher=self._searcher(settings),
                model_name=settings.model_name,
                provider=settings.provider,
                top_k=top_k,
                filters=filters,
                rerank=rerank,
                embedding_dimension=settings.embedding_dimensions,
                no_cache=self._no_cache,
            )

    def delete_many(self, record_ids: Sequence[str]) -> int:
        """Delete records by id, returning how many existed."""

        with self._dir_context(), _collection_errors():
            return collection_service.delete_records(
                name=self.name, record_keys=record_ids
            )

    def delete(self, record_id: str) -> int:
        """Delete a single record by id."""

        return self.delete_many([record_id])

    def get_many(self, record_ids: Sequence[str]) -> list[StoredRecord]:
        """Return stored records by id, skipping ids that are absent."""

        with self._dir_context(), _collection_errors():
            return collection_service.get_records(
                name=self.name, record_keys=record_ids
            )

    def get(self, record_id: str) -> StoredRecord | None:
        """Return one stored record, or ``None`` when it is absent."""

        found = self.get_many([record_id])
        return found[0] if found else None

    def count(self) -> int:
        """Return how many records this collection holds."""

        with self._dir_context(), _collection_errors():
            return collection_service.count_records(name=self.name)

    def info(self) -> CollectionInfo | None:
        """Return the pinned provider/model/dimension contract, or ``None``."""

        with self._dir_context(), _collection_errors():
            return collection_service.collection_info(name=self.name)

    def drop(self) -> bool:
        """Delete this collection and every record in it."""

        with self._dir_context(), _collection_errors():
            return collection_service.drop_collection(name=self.name)
