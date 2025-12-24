"""Logic helpers for the `vexor config` command."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    Config,
    load_config,
    set_api_key,
    set_base_url,
    set_batch_size,
    set_embed_concurrency,
    set_extract_concurrency,
    set_extract_backend,
    set_auto_index,
    set_flashrank_model,
    set_local_cuda,
    set_model,
    set_provider,
    set_rerank,
    update_remote_rerank,
)


@dataclass(slots=True)
class ConfigUpdateResult:
    api_key_set: bool = False
    api_key_cleared: bool = False
    model_set: bool = False
    batch_size_set: bool = False
    embed_concurrency_set: bool = False
    extract_concurrency_set: bool = False
    extract_backend_set: bool = False
    provider_set: bool = False
    base_url_set: bool = False
    base_url_cleared: bool = False
    auto_index_set: bool = False
    local_cuda_set: bool = False
    rerank_set: bool = False
    flashrank_model_set: bool = False
    remote_rerank_url_set: bool = False
    remote_rerank_model_set: bool = False
    remote_rerank_api_key_set: bool = False
    remote_rerank_cleared: bool = False

    @property
    def changed(self) -> bool:
        return any(
            (
                self.api_key_set,
                self.api_key_cleared,
                self.model_set,
                self.batch_size_set,
                self.embed_concurrency_set,
                self.extract_concurrency_set,
                self.extract_backend_set,
                self.provider_set,
                self.base_url_set,
                self.base_url_cleared,
                self.auto_index_set,
                self.local_cuda_set,
                self.rerank_set,
                self.flashrank_model_set,
                self.remote_rerank_url_set,
                self.remote_rerank_model_set,
                self.remote_rerank_api_key_set,
                self.remote_rerank_cleared,
            )
        )


def apply_config_updates(
    *,
    api_key: str | None = None,
    clear_api_key: bool = False,
    model: str | None = None,
    batch_size: int | None = None,
    embed_concurrency: int | None = None,
    extract_concurrency: int | None = None,
    extract_backend: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    clear_base_url: bool = False,
    auto_index: bool | None = None,
    local_cuda: bool | None = None,
    rerank: str | None = None,
    flashrank_model: str | None = None,
    remote_rerank_url: str | None = None,
    remote_rerank_model: str | None = None,
    remote_rerank_api_key: str | None = None,
    clear_remote_rerank: bool = False,
) -> ConfigUpdateResult:
    """Apply config mutations and report which fields were updated."""

    result = ConfigUpdateResult()
    if api_key is not None:
        set_api_key(api_key)
        result.api_key_set = True
    if clear_api_key:
        set_api_key(None)
        result.api_key_cleared = True
    if model is not None:
        set_model(model)
        result.model_set = True
    if batch_size is not None:
        set_batch_size(batch_size)
        result.batch_size_set = True
    if embed_concurrency is not None:
        set_embed_concurrency(embed_concurrency)
        result.embed_concurrency_set = True
    if extract_concurrency is not None:
        set_extract_concurrency(extract_concurrency)
        result.extract_concurrency_set = True
    if extract_backend is not None:
        set_extract_backend(extract_backend)
        result.extract_backend_set = True
    if provider is not None:
        set_provider(provider)
        result.provider_set = True
    if base_url is not None:
        set_base_url(base_url)
        result.base_url_set = True
    if clear_base_url:
        set_base_url(None)
        result.base_url_cleared = True
    if auto_index is not None:
        set_auto_index(auto_index)
        result.auto_index_set = True
    if local_cuda is not None:
        set_local_cuda(local_cuda)
        result.local_cuda_set = True
    if rerank is not None:
        set_rerank(rerank)
        result.rerank_set = True
    if flashrank_model is not None:
        set_flashrank_model(flashrank_model)
        result.flashrank_model_set = True
    if (
        clear_remote_rerank
        or remote_rerank_url is not None
        or remote_rerank_model is not None
        or remote_rerank_api_key is not None
    ):
        update_remote_rerank(
            base_url=remote_rerank_url,
            api_key=remote_rerank_api_key,
            model=remote_rerank_model,
            clear=clear_remote_rerank,
        )
        result.remote_rerank_url_set = remote_rerank_url is not None
        result.remote_rerank_model_set = remote_rerank_model is not None
        result.remote_rerank_api_key_set = remote_rerank_api_key is not None
        result.remote_rerank_cleared = clear_remote_rerank
    return result


def get_config_snapshot() -> Config:
    """Return the current configuration dataclass."""

    return load_config()
