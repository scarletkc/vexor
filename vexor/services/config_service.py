"""Logic helpers for the `vexor config` command."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    Config,
    load_config,
    set_api_key,
    set_base_url,
    set_batch_size,
    set_model,
    set_provider,
)


@dataclass(slots=True)
class ConfigUpdateResult:
    api_key_set: bool = False
    api_key_cleared: bool = False
    model_set: bool = False
    batch_size_set: bool = False
    provider_set: bool = False
    base_url_set: bool = False
    base_url_cleared: bool = False

    @property
    def changed(self) -> bool:
        return any(
            (
                self.api_key_set,
                self.api_key_cleared,
                self.model_set,
                self.batch_size_set,
                self.provider_set,
                self.base_url_set,
                self.base_url_cleared,
            )
        )


def apply_config_updates(
    *,
    api_key: str | None = None,
    clear_api_key: bool = False,
    model: str | None = None,
    batch_size: int | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    clear_base_url: bool = False,
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
    if provider is not None:
        set_provider(provider)
        result.provider_set = True
    if base_url is not None:
        set_base_url(base_url)
        result.base_url_set = True
    if clear_base_url:
        set_base_url(None)
        result.base_url_cleared = True
    return result


def get_config_snapshot() -> Config:
    """Return the current configuration dataclass."""

    return load_config()
