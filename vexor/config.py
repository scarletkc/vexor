"""Global configuration management for Vexor."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse, urlunparse

CONFIG_DIR = Path(os.path.expanduser("~")) / ".vexor"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_BATCH_SIZE = 64
DEFAULT_EMBED_CONCURRENCY = 2
DEFAULT_PROVIDER = "openai"
DEFAULT_RERANK = "off"
DEFAULT_FLASHRANK_MODEL = "ms-marco-TinyBERT-L-2-v2"
DEFAULT_FLASHRANK_MAX_LENGTH = 256
SUPPORTED_PROVIDERS: tuple[str, ...] = (DEFAULT_PROVIDER, "gemini", "custom", "local")
SUPPORTED_RERANKERS: tuple[str, ...] = ("off", "bm25", "flashrank", "remote")
ENV_API_KEY = "VEXOR_API_KEY"
REMOTE_RERANK_ENV = "VEXOR_REMOTE_RERANK_API_KEY"
LEGACY_GEMINI_ENV = "GOOGLE_GENAI_API_KEY"
OPENAI_ENV = "OPENAI_API_KEY"


@dataclass
class RemoteRerankConfig:
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


@dataclass
class Config:
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    embed_concurrency: int = DEFAULT_EMBED_CONCURRENCY
    provider: str = DEFAULT_PROVIDER
    base_url: str | None = None
    auto_index: bool = True
    local_cuda: bool = False
    rerank: str = DEFAULT_RERANK
    flashrank_model: str | None = None
    remote_rerank: RemoteRerankConfig | None = None


def _parse_remote_rerank(raw: object) -> RemoteRerankConfig | None:
    if not isinstance(raw, dict):
        return None
    base_url = normalize_remote_rerank_url(raw.get("base_url"))
    api_key = (raw.get("api_key") or "").strip() or None
    model = (raw.get("model") or "").strip() or None
    if not any((base_url, api_key, model)):
        return None
    return RemoteRerankConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


def load_config() -> Config:
    if not CONFIG_FILE.exists():
        return Config()
    raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    rerank = (raw.get("rerank") or DEFAULT_RERANK).strip().lower()
    if rerank not in SUPPORTED_RERANKERS:
        rerank = DEFAULT_RERANK
    return Config(
        api_key=raw.get("api_key") or None,
        model=raw.get("model") or DEFAULT_MODEL,
        batch_size=int(raw.get("batch_size", DEFAULT_BATCH_SIZE)),
        embed_concurrency=int(raw.get("embed_concurrency", DEFAULT_EMBED_CONCURRENCY)),
        provider=raw.get("provider") or DEFAULT_PROVIDER,
        base_url=raw.get("base_url") or None,
        auto_index=bool(raw.get("auto_index", True)),
        local_cuda=bool(raw.get("local_cuda", False)),
        rerank=rerank,
        flashrank_model=raw.get("flashrank_model") or None,
        remote_rerank=_parse_remote_rerank(raw.get("remote_rerank")),
    )


def save_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if config.api_key:
        data["api_key"] = config.api_key
    if config.model:
        data["model"] = config.model
    data["batch_size"] = config.batch_size
    data["embed_concurrency"] = config.embed_concurrency
    if config.provider:
        data["provider"] = config.provider
    if config.base_url:
        data["base_url"] = config.base_url
    data["auto_index"] = bool(config.auto_index)
    data["local_cuda"] = bool(config.local_cuda)
    data["rerank"] = config.rerank
    if config.flashrank_model:
        data["flashrank_model"] = config.flashrank_model
    if config.remote_rerank is not None:
        remote_data: Dict[str, Any] = {}
        if config.remote_rerank.base_url:
            remote_data["base_url"] = config.remote_rerank.base_url
        if config.remote_rerank.api_key:
            remote_data["api_key"] = config.remote_rerank.api_key
        if config.remote_rerank.model:
            remote_data["model"] = config.remote_rerank.model
        if remote_data:
            data["remote_rerank"] = remote_data
    CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def local_model_dir() -> Path:
    return CONFIG_DIR / "models"


def flashrank_cache_dir(*, create: bool = True) -> Path:
    cache_dir = CONFIG_DIR / "flashrank"
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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


def set_embed_concurrency(value: int) -> None:
    config = load_config()
    config.embed_concurrency = value
    save_config(config)


def set_provider(value: str) -> None:
    config = load_config()
    config.provider = value
    save_config(config)


def set_base_url(value: str | None) -> None:
    config = load_config()
    config.base_url = value
    save_config(config)


def set_auto_index(value: bool) -> None:
    config = load_config()
    config.auto_index = bool(value)
    save_config(config)


def set_local_cuda(value: bool) -> None:
    config = load_config()
    config.local_cuda = bool(value)
    save_config(config)


def set_rerank(value: str) -> None:
    config = load_config()
    normalized = (value or DEFAULT_RERANK).strip().lower()
    if normalized not in SUPPORTED_RERANKERS:
        normalized = DEFAULT_RERANK
    config.rerank = normalized
    save_config(config)


def set_flashrank_model(value: str | None) -> None:
    config = load_config()
    clean_value = (value or "").strip()
    config.flashrank_model = clean_value or None
    save_config(config)


def update_remote_rerank(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    clear: bool = False,
) -> None:
    config = load_config()
    if clear:
        config.remote_rerank = None
        save_config(config)
        return
    if any(value is not None for value in (base_url, api_key, model)):
        if config.remote_rerank is None:
            config.remote_rerank = RemoteRerankConfig()
        if base_url is not None:
            config.remote_rerank.base_url = normalize_remote_rerank_url(base_url)
        if api_key is not None:
            config.remote_rerank.api_key = api_key.strip() or None
        if model is not None:
            config.remote_rerank.model = model.strip() or None
    save_config(config)


def normalize_remote_rerank_url(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        base = cleaned.rstrip("/")
        if base.endswith("/rerank") or base.endswith("/reranker"):
            return base
        return f"{base}/rerank"
    path = parsed.path or ""
    trimmed = path.rstrip("/")
    if trimmed.endswith("/rerank") or trimmed.endswith("/reranker"):
        new_path = trimmed
    else:
        new_path = f"{trimmed}/rerank" if trimmed else "/rerank"
    normalized = parsed._replace(path=new_path)
    return urlunparse(normalized)


def resolve_default_model(provider: str | None, model: str | None) -> str:
    """Return the effective model name for the selected provider."""
    clean_model = (model or "").strip()
    normalized = (provider or DEFAULT_PROVIDER).lower()
    if normalized == "gemini" and (not clean_model or clean_model == DEFAULT_MODEL):
        return DEFAULT_GEMINI_MODEL
    if clean_model:
        return clean_model
    return DEFAULT_MODEL


def resolve_api_key(configured: str | None, provider: str) -> str | None:
    """Return the first available API key from config or environment."""

    normalized = (provider or DEFAULT_PROVIDER).lower()
    if normalized == "local":
        return None
    if configured:
        return configured
    general = os.getenv(ENV_API_KEY)
    if general:
        return general
    if normalized == "gemini":
        legacy = os.getenv(LEGACY_GEMINI_ENV)
        if legacy:
            return legacy
    if normalized in {"openai", "custom"}:
        openai_key = os.getenv(OPENAI_ENV)
        if openai_key:
            return openai_key
    return None


def resolve_remote_rerank_api_key(configured: str | None) -> str | None:
    """Return the remote rerank API key from config or environment."""

    if configured:
        return configured
    env_key = os.getenv(REMOTE_RERANK_ENV)
    if env_key:
        return env_key
    return None
