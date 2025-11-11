"""Global configuration management for Vexor."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(os.path.expanduser("~")) / ".vexor"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_BATCH_SIZE = 0
DEFAULT_PROVIDER = "gemini"
SUPPORTED_PROVIDERS: tuple[str, ...] = (DEFAULT_PROVIDER, "openai")
ENV_API_KEY = "VEXOR_API_KEY"
LEGACY_GEMINI_ENV = "GOOGLE_GENAI_API_KEY"
OPENAI_ENV = "OPENAI_API_KEY"


@dataclass
class Config:
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    provider: str = DEFAULT_PROVIDER
    base_url: str | None = None


def load_config() -> Config:
    if not CONFIG_FILE.exists():
        return Config()
    raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return Config(
        api_key=raw.get("api_key") or None,
        model=raw.get("model") or DEFAULT_MODEL,
        batch_size=int(raw.get("batch_size", DEFAULT_BATCH_SIZE)),
        provider=raw.get("provider") or DEFAULT_PROVIDER,
        base_url=raw.get("base_url") or None,
    )


def save_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if config.api_key:
        data["api_key"] = config.api_key
    if config.model:
        data["model"] = config.model
    data["batch_size"] = config.batch_size
    if config.provider:
        data["provider"] = config.provider
    if config.base_url:
        data["base_url"] = config.base_url
    CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_api_key(value: str | None) -> None:
    config = load_config()
    config.api_key = value
    save_config(config)


def set_model(value: str) -> None:
    config = load_config()
    config.model = value
    save_config(config)


def set_batch_size(value: int) -> None:
    config = load_config()
    config.batch_size = value
    save_config(config)


def set_provider(value: str) -> None:
    config = load_config()
    config.provider = value
    save_config(config)


def set_base_url(value: str | None) -> None:
    config = load_config()
    config.base_url = value
    save_config(config)


def resolve_api_key(configured: str | None, provider: str) -> str | None:
    """Return the first available API key from config or environment."""

    if configured:
        return configured
    general = os.getenv(ENV_API_KEY)
    if general:
        return general
    normalized = (provider or DEFAULT_PROVIDER).lower()
    if normalized == "gemini":
        legacy = os.getenv(LEGACY_GEMINI_ENV)
        if legacy:
            return legacy
    if normalized == "openai":
        openai_key = os.getenv(OPENAI_ENV)
        if openai_key:
            return openai_key
    return None
