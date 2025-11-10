"""Logic helpers for the `vexor config` command."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config, load_config, set_api_key, set_batch_size, set_model


@dataclass(slots=True)
class ConfigUpdateResult:
    api_key_set: bool = False
    api_key_cleared: bool = False
    model_set: bool = False
    batch_size_set: bool = False

    @property
    def changed(self) -> bool:
        return any((self.api_key_set, self.api_key_cleared, self.model_set, self.batch_size_set))


def apply_config_updates(
    *,
    api_key: str | None = None,
    clear_api_key: bool = False,
    model: str | None = None,
    batch_size: int | None = None,
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
    return result


def get_config_snapshot() -> Config:
    """Return the current configuration dataclass."""

    return load_config()

