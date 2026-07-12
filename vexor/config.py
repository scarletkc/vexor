"""Global configuration management for Vexor."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse, urlunparse

# Permanent public re-exports preserve existing imports from vexor.config.
from .providers.capabilities import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_VOYAGE_MODEL,
    DIMENSION_SUPPORTED_MODELS,
    ENV_API_KEY,
    LEGACY_GEMINI_ENV,
    OPENAI_ENV,
    SUPPORTED_PROVIDERS,
    VOYAGE_BASE_URL,
    VOYAGE_ENV,
    get_supported_dimensions,
    resolve_api_key,
    resolve_base_url,
    resolve_default_model,
    supports_dimensions,
    validate_embedding_dimensions_for_model,
)
from .text import Messages

DEFAULT_CONFIG_DIR = Path(os.path.expanduser("~")) / ".vexor"
CONFIG_DIR = DEFAULT_CONFIG_DIR
CONFIG_FILE = CONFIG_DIR / "config.json"
_CONFIG_DIR_OVERRIDE: ContextVar[Path | None] = ContextVar(
    "vexor_config_dir_override",
    default=None,
)
DEFAULT_BATCH_SIZE = 64
DEFAULT_EMBED_CONCURRENCY = 4
DEFAULT_EXTRACT_CONCURRENCY = max(1, min(4, os.cpu_count() or 1))
DEFAULT_EXTRACT_BACKEND = "auto"
DEFAULT_RERANK = "off"
DEFAULT_FLASHRANK_MODEL = "ms-marco-TinyBERT-L-2-v2"
DEFAULT_FLASHRANK_MAX_LENGTH = 256
SUPPORTED_RERANKERS: tuple[str, ...] = (
    "off",
    "bm25",
    "flashrank",
    "remote",
    "hybrid",
)
SUPPORTED_EXTRACT_BACKENDS: tuple[str, ...] = ("auto", "thread", "process")
ENV_CONFIG_JSON = "VEXOR_CONFIG_JSON"
ENV_NO_UPDATE_CHECK = "VEXOR_NO_UPDATE_CHECK"
REMOTE_RERANK_ENV = "VEXOR_REMOTE_RERANK_API_KEY"
PROJECT_CONFIG_FILENAME = "config.json"
PROJECT_CONFIG_FIELDS = frozenset(
    {
        "auto_index",
        "batch_size",
        "embed_concurrency",
        "embedding_dimensions",
        "extract_concurrency",
        "model",
        "rerank",
    }
)
PROJECT_CONFIG_SENSITIVE_FIELDS = frozenset(
    {"api_key", "base_url", "remote_rerank"}
)


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
    extract_concurrency: int = DEFAULT_EXTRACT_CONCURRENCY
    extract_backend: str = DEFAULT_EXTRACT_BACKEND
    provider: str = DEFAULT_PROVIDER
    base_url: str | None = None
    auto_index: bool = True
    local_cuda: bool = False
    update_check: bool = True
    rerank: str = DEFAULT_RERANK
    flashrank_model: str | None = None
    remote_rerank: RemoteRerankConfig | None = None
    embedding_dimensions: int | None = None


class ConfigOrigin(str, Enum):
    """Source of an effective configuration field."""

    DEFAULT = "default"
    GLOBAL = "global"
    PROJECT = "project"
    ENVIRONMENT = "environment"


CONFIG_FIELD_NAMES = tuple(Config.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class ConfigResolution:
    """Effective configuration plus source metadata for each field."""

    config: Config
    origins: Mapping[str, ConfigOrigin]
    global_file: Path
    project_file: Path | None = None

    def origin_for(self, field: str) -> ConfigOrigin:
        return self.origins.get(field, ConfigOrigin.DEFAULT)


class ProjectConfigError(ValueError):
    """Raised when a project-controlled config cannot be applied safely."""

    def __init__(self, message: str, *, path: Path) -> None:
        super().__init__(message)
        self.path = path


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


def _resolve_config_dir() -> Path:
    override = _CONFIG_DIR_OVERRIDE.get()
    return override if override is not None else CONFIG_DIR


def _resolve_config_file() -> Path:
    override = _CONFIG_DIR_OVERRIDE.get()
    if override is not None:
        return override / "config.json"
    return CONFIG_FILE


def _resolve_project_config_file(
    directory: Path | str | None,
) -> Path | None:
    """Return the config candidate under the nearest project marker."""

    if directory is None:
        return None

    # Import lazily so importing the standalone config module does not pull in
    # the cache's NumPy/SQLite dependencies.
    from .cache import find_project_cache_dir

    project_dir = find_project_cache_dir(Path(directory))
    if project_dir is None:
        return None
    project_file = project_dir / PROJECT_CONFIG_FILENAME
    if project_file.resolve() == _resolve_config_file().resolve():
        return None
    return project_file


@contextmanager
def config_dir_context(path: Path | str | None):
    """Temporarily override the config directory for the current context."""

    if path is None:
        yield
        return
    dir_path = Path(path).expanduser().resolve()
    if dir_path.exists() and not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    token = _CONFIG_DIR_OVERRIDE.set(dir_path)
    try:
        yield
    finally:
        _CONFIG_DIR_OVERRIDE.reset(token)


def _apply_env_overrides(
    config: Config,
    *,
    origins: dict[str, ConfigOrigin] | None = None,
) -> Config:
    """Merge the VEXOR_CONFIG_JSON environment override over *config*.

    Lets MCP client configs (and CI) inject any non-secret config field via
    a single environment variable without touching ~/.vexor/config.json.
    Credentials are rejected so they stay on their dedicated variables
    (VEXOR_API_KEY, VEXOR_REMOTE_RERANK_API_KEY), which clients treat as
    secrets.
    """
    payload = (os.getenv(ENV_CONFIG_JSON) or "").strip()
    if payload:
        try:
            data = _coerce_config_payload(payload)
            if "api_key" in data:
                raise ValueError(
                    Messages.ERROR_ENV_CONFIG_JSON_SECRET.format(
                        field="api_key", env=ENV_API_KEY
                    )
                )
            remote_rerank = data.get("remote_rerank")
            if isinstance(remote_rerank, Mapping) and "api_key" in remote_rerank:
                raise ValueError(
                    Messages.ERROR_ENV_CONFIG_JSON_SECRET.format(
                        field="remote_rerank.api_key", env=REMOTE_RERANK_ENV
                    )
                )
            config = config_from_json(data, base=config)
            if origins is not None:
                for field in data:
                    if field in origins:
                        origins[field] = ConfigOrigin.ENVIRONMENT
        except ValueError as exc:
            raise ValueError(
                Messages.ERROR_ENV_CONFIG_JSON_INVALID.format(reason=exc)
            ) from exc

    # The dedicated secret variables override credentials from the persisted
    # config. Explicit API arguments are applied after load_config(), so they
    # can still take precedence over these process-level defaults.
    api_key = os.getenv(ENV_API_KEY)
    if api_key:
        config.api_key = api_key
        if origins is not None:
            origins["api_key"] = ConfigOrigin.ENVIRONMENT
    remote_rerank_api_key = os.getenv(REMOTE_RERANK_ENV)
    if remote_rerank_api_key and config.remote_rerank is not None:
        config.remote_rerank.api_key = remote_rerank_api_key
        if origins is not None:
            origins["remote_rerank"] = ConfigOrigin.ENVIRONMENT
    return config


def _load_stored_config_payload() -> Mapping[str, object]:
    """Load the persisted global JSON object without applying it."""

    config_file = _resolve_config_file()
    if not config_file.exists():
        return {}
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(Messages.ERROR_CONFIG_JSON_INVALID)
    return raw


def _config_from_stored_payload(raw: Mapping[str, object]) -> Config:
    """Build a Config while preserving the global file's legacy coercions."""

    rerank = (raw.get("rerank") or DEFAULT_RERANK).strip().lower()
    if rerank not in SUPPORTED_RERANKERS:
        rerank = DEFAULT_RERANK
    return Config(
        api_key=raw.get("api_key") or None,
        model=raw.get("model") or DEFAULT_MODEL,
        batch_size=int(raw.get("batch_size", DEFAULT_BATCH_SIZE)),
        embed_concurrency=int(raw.get("embed_concurrency", DEFAULT_EMBED_CONCURRENCY)),
        extract_concurrency=int(
            raw.get("extract_concurrency", DEFAULT_EXTRACT_CONCURRENCY)
        ),
        extract_backend=_coerce_extract_backend(raw.get("extract_backend")),
        provider=raw.get("provider") or DEFAULT_PROVIDER,
        base_url=raw.get("base_url") or None,
        auto_index=bool(raw.get("auto_index", True)),
        local_cuda=bool(raw.get("local_cuda", False)),
        update_check=bool(raw.get("update_check", True)),
        rerank=rerank,
        flashrank_model=raw.get("flashrank_model") or None,
        remote_rerank=_parse_remote_rerank(raw.get("remote_rerank")),
        embedding_dimensions=_coerce_optional_int(raw.get("embedding_dimensions")),
    )


def _load_stored_config() -> Config:
    """Load the persisted config without applying environment overrides."""

    return _config_from_stored_payload(_load_stored_config_payload())


def _load_project_config(
    base: Config,
    directory: Path | str | None,
) -> tuple[Config, Path | None, tuple[str, ...]]:
    project_file = _resolve_project_config_file(directory)
    if project_file is None or not project_file.exists():
        return base, project_file, ()

    try:
        raw = json.loads(project_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ProjectConfigError(
            Messages.ERROR_PROJECT_CONFIG_INVALID.format(
                path=project_file,
                reason=exc,
            ),
            path=project_file,
        ) from exc
    if not isinstance(raw, Mapping):
        raise ProjectConfigError(
            Messages.ERROR_PROJECT_CONFIG_INVALID.format(
                path=project_file,
                reason=Messages.ERROR_CONFIG_JSON_INVALID,
            ),
            path=project_file,
        )

    fields = set(raw)
    sensitive = sorted(fields & PROJECT_CONFIG_SENSITIVE_FIELDS)
    if sensitive:
        raise ProjectConfigError(
            Messages.ERROR_PROJECT_CONFIG_SENSITIVE.format(
                path=project_file,
                fields=", ".join(sensitive),
            ),
            path=project_file,
        )
    unsupported = sorted(fields - PROJECT_CONFIG_FIELDS)
    if unsupported:
        raise ProjectConfigError(
            Messages.ERROR_PROJECT_CONFIG_UNSUPPORTED.format(
                path=project_file,
                fields=", ".join(unsupported),
                allowed=", ".join(sorted(PROJECT_CONFIG_FIELDS)),
            ),
            path=project_file,
        )

    try:
        config = config_from_json(raw, base=base)
    except ValueError as exc:
        raise ProjectConfigError(
            Messages.ERROR_PROJECT_CONFIG_INVALID.format(
                path=project_file,
                reason=exc,
            ),
            path=project_file,
        ) from exc
    return config, project_file, tuple(sorted(fields))


def resolve_config(
    directory: Path | str | None = None,
) -> ConfigResolution:
    """Resolve global, project, and environment config with field origins."""

    global_file = _resolve_config_file()
    stored = _load_stored_config_payload()
    config = _config_from_stored_payload(stored)
    origins = {field: ConfigOrigin.DEFAULT for field in CONFIG_FIELD_NAMES}
    for field in stored:
        if field in origins:
            origins[field] = ConfigOrigin.GLOBAL

    config, project_file, project_fields = _load_project_config(config, directory)
    for field in project_fields:
        origins[field] = ConfigOrigin.PROJECT

    config = _apply_env_overrides(config, origins=origins)
    return ConfigResolution(
        config=config,
        origins=origins,
        global_file=global_file,
        project_file=project_file,
    )


def load_config(directory: Path | str | None = None) -> Config:
    """Load effective config for *directory* and apply environment overrides."""

    return resolve_config(directory).config


def save_config(config: Config) -> None:
    config_dir = _resolve_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if config.api_key:
        data["api_key"] = config.api_key
    if config.model:
        data["model"] = config.model
    data["batch_size"] = config.batch_size
    data["embed_concurrency"] = config.embed_concurrency
    data["extract_concurrency"] = config.extract_concurrency
    data["extract_backend"] = config.extract_backend
    if config.provider:
        data["provider"] = config.provider
    if config.base_url:
        data["base_url"] = config.base_url
    data["auto_index"] = bool(config.auto_index)
    data["local_cuda"] = bool(config.local_cuda)
    data["update_check"] = bool(config.update_check)
    data["rerank"] = config.rerank
    if config.flashrank_model:
        data["flashrank_model"] = config.flashrank_model
    if config.embedding_dimensions is not None:
        data["embedding_dimensions"] = config.embedding_dimensions
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
    config_file = _resolve_config_file()
    config_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def local_model_dir() -> Path:
    return _resolve_config_dir() / "models"


def flashrank_cache_dir(*, create: bool = True) -> Path:
    cache_dir = _resolve_config_dir() / "flashrank"
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def update_check_file() -> Path:
    """Path of the cached update-check state."""
    return _resolve_config_dir() / "update_check.json"


def set_config_dir(path: Path | str | None) -> None:
    global CONFIG_DIR, CONFIG_FILE
    if path is None:
        CONFIG_DIR = DEFAULT_CONFIG_DIR
    else:
        dir_path = Path(path).expanduser().resolve()
        if dir_path.exists() and not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {dir_path}")
        CONFIG_DIR = dir_path
    CONFIG_FILE = CONFIG_DIR / "config.json"


def config_from_json(
    payload: str | Mapping[str, object], *, base: Config | None = None
) -> Config:
    """Return a Config from a JSON string or mapping without saving it."""
    data = _coerce_config_payload(payload)
    config = Config() if base is None else _clone_config(base)
    _apply_config_payload(config, data)
    return config


def update_config_from_json(
    payload: str | Mapping[str, object], *, replace: bool = False
) -> Config:
    """Update config from a JSON string or mapping and persist it."""
    base = None if replace else _load_stored_config()
    config = config_from_json(payload, base=base)
    save_config(config)
    return config


def set_api_key(value: str | None) -> None:
    config = _load_stored_config()
    config.api_key = value
    save_config(config)


def set_model(value: str, *, validate_embedding_dimensions: bool = True) -> None:
    config = _load_stored_config()
    config.model = value
    if validate_embedding_dimensions:
        _validate_config_embedding_dimensions(config)
    save_config(config)


def set_batch_size(value: int) -> None:
    config = _load_stored_config()
    config.batch_size = value
    save_config(config)


def set_embed_concurrency(value: int) -> None:
    config = _load_stored_config()
    config.embed_concurrency = value
    save_config(config)


def set_extract_concurrency(value: int) -> None:
    config = _load_stored_config()
    config.extract_concurrency = value
    save_config(config)


def set_extract_backend(value: str) -> None:
    config = _load_stored_config()
    config.extract_backend = _normalize_extract_backend(value)
    save_config(config)


def set_provider(value: str, *, validate_embedding_dimensions: bool = True) -> None:
    config = _load_stored_config()
    config.provider = value
    if validate_embedding_dimensions:
        _validate_config_embedding_dimensions(config)
    save_config(config)


def set_base_url(value: str | None) -> None:
    config = _load_stored_config()
    config.base_url = value
    save_config(config)


def set_auto_index(value: bool) -> None:
    config = _load_stored_config()
    config.auto_index = bool(value)
    save_config(config)


def set_update_check(value: bool) -> None:
    config = _load_stored_config()
    config.update_check = bool(value)
    save_config(config)


def set_local_cuda(value: bool) -> None:
    config = _load_stored_config()
    config.local_cuda = bool(value)
    save_config(config)


def set_rerank(value: str) -> None:
    config = _load_stored_config()
    normalized = (value or DEFAULT_RERANK).strip().lower()
    if normalized not in SUPPORTED_RERANKERS:
        normalized = DEFAULT_RERANK
    config.rerank = normalized
    save_config(config)


def set_flashrank_model(value: str | None) -> None:
    config = _load_stored_config()
    clean_value = (value or "").strip()
    config.flashrank_model = clean_value or None
    save_config(config)


def set_embedding_dimensions(
    value: int | None,
    model: str | None = None,
    provider: str | None = None,
) -> None:
    """Set the embedding dimensions for providers that support it (e.g., Voyage AI).

    Args:
        value: The dimension to set, or None/0 to clear
        model: Optional model to validate against. If not provided, uses config model.
        provider: Optional provider to resolve effective model. If not provided, uses config provider.

    Raises:
        ValueError: If value is negative, model doesn't support dimensions,
                   or dimension is not valid for the model.
    """
    config = _load_stored_config()

    # Reject negative values explicitly
    if value is not None and value < 0:
        raise ValueError(f"embedding_dimensions must be non-negative, got {value}")

    # Treat 0 and None as "clear"
    if not value or value <= 0:
        config.embedding_dimensions = None
        save_config(config)
        return

    # Validate against effective model (resolved from provider + model)
    effective_provider = provider if provider else config.provider
    effective_model = resolve_default_model(effective_provider, model if model else config.model)
    validate_embedding_dimensions_for_model(value, effective_model)

    config.embedding_dimensions = value
    save_config(config)


def update_remote_rerank(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    clear: bool = False,
) -> None:
    config = _load_stored_config()
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


def _validate_config_embedding_dimensions(config: Config) -> None:
    """Ensure stored embedding dimensions remain compatible with provider/model."""
    if config.embedding_dimensions is None:
        return
    effective_model = resolve_default_model(config.provider, config.model)
    try:
        validate_embedding_dimensions_for_model(
            config.embedding_dimensions,
            effective_model,
        )
    except ValueError as exc:
        raise ValueError(
            f"Current embedding_dimensions ({config.embedding_dimensions}) is incompatible with "
            f"model '{effective_model}'. Clear it with "
            "`vexor config --clear-embedding-dimensions` or set a supported value."
        ) from exc


def resolve_remote_rerank_api_key(configured: str | None) -> str | None:
    """Return the remote rerank API key from config or environment."""

    if configured:
        return configured
    env_key = os.getenv(REMOTE_RERANK_ENV)
    if env_key:
        return env_key
    return None


def _coerce_config_payload(payload: str | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(Messages.ERROR_CONFIG_JSON_INVALID) from exc
    elif isinstance(payload, Mapping):
        data = dict(payload)
    else:
        raise ValueError(Messages.ERROR_CONFIG_JSON_INVALID)
    if not isinstance(data, Mapping):
        raise ValueError(Messages.ERROR_CONFIG_JSON_INVALID)
    return data


def _clone_config(config: Config) -> Config:
    remote = config.remote_rerank
    return Config(
        api_key=config.api_key,
        model=config.model,
        batch_size=config.batch_size,
        embed_concurrency=config.embed_concurrency,
        extract_concurrency=config.extract_concurrency,
        extract_backend=config.extract_backend,
        provider=config.provider,
        base_url=config.base_url,
        auto_index=config.auto_index,
        local_cuda=config.local_cuda,
        update_check=config.update_check,
        rerank=config.rerank,
        flashrank_model=config.flashrank_model,
        remote_rerank=(
            None
            if remote is None
            else RemoteRerankConfig(
                base_url=remote.base_url,
                api_key=remote.api_key,
                model=remote.model,
            )
        ),
        embedding_dimensions=config.embedding_dimensions,
    )


def _apply_config_payload(config: Config, payload: Mapping[str, object]) -> None:
    if "api_key" in payload:
        config.api_key = _coerce_optional_str(payload["api_key"], "api_key")
    if "model" in payload:
        config.model = _coerce_required_str(payload["model"], "model", DEFAULT_MODEL)
    if "batch_size" in payload:
        config.batch_size = _coerce_int(
            payload["batch_size"], "batch_size", DEFAULT_BATCH_SIZE
        )
    if "embed_concurrency" in payload:
        config.embed_concurrency = _coerce_int(
            payload["embed_concurrency"],
            "embed_concurrency",
            DEFAULT_EMBED_CONCURRENCY,
        )
    if "extract_concurrency" in payload:
        config.extract_concurrency = _coerce_int(
            payload["extract_concurrency"],
            "extract_concurrency",
            DEFAULT_EXTRACT_CONCURRENCY,
        )
    if "extract_backend" in payload:
        config.extract_backend = _normalize_extract_backend(payload["extract_backend"])
    if "provider" in payload:
        config.provider = _coerce_required_str(
            payload["provider"], "provider", DEFAULT_PROVIDER
        )
    if "base_url" in payload:
        config.base_url = _coerce_optional_str(payload["base_url"], "base_url")
    if "auto_index" in payload:
        config.auto_index = _coerce_bool(payload["auto_index"], "auto_index")
    if "update_check" in payload:
        config.update_check = _coerce_bool(payload["update_check"], "update_check")
    if "local_cuda" in payload:
        config.local_cuda = _coerce_bool(payload["local_cuda"], "local_cuda")
    if "rerank" in payload:
        config.rerank = _normalize_rerank(payload["rerank"])
    if "flashrank_model" in payload:
        config.flashrank_model = _coerce_optional_str(
            payload["flashrank_model"], "flashrank_model"
        )
    if "remote_rerank" in payload:
        config.remote_rerank = _coerce_remote_rerank(payload["remote_rerank"])
    if "embedding_dimensions" in payload:
        config.embedding_dimensions = _coerce_optional_int(payload["embedding_dimensions"])


def _coerce_optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    raise ValueError(Messages.ERROR_CONFIG_VALUE_INVALID.format(field=field))


def _coerce_required_str(value: object, field: str, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or default
    raise ValueError(Messages.ERROR_CONFIG_VALUE_INVALID.format(field=field))


def _coerce_int(value: object, field: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(Messages.ERROR_CONFIG_VALUE_INVALID.format(field=field))
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(Messages.ERROR_CONFIG_VALUE_INVALID.format(field=field))
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return default
        try:
            return int(cleaned)
        except ValueError as exc:
            raise ValueError(Messages.ERROR_CONFIG_VALUE_INVALID.format(field=field)) from exc
    raise ValueError(Messages.ERROR_CONFIG_VALUE_INVALID.format(field=field))


def _coerce_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "on"}:
            return True
        if cleaned in {"false", "0", "no", "off"}:
            return False
    raise ValueError(Messages.ERROR_CONFIG_VALUE_INVALID.format(field=field))


def _coerce_optional_int(value: object) -> int | None:
    """Coerce a value to an optional integer, returning None for empty/null values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if value.is_integer() and value > 0:
            return int(value)
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            parsed = int(cleaned)
            return parsed if parsed > 0 else None
        except ValueError:
            return None
    return None


def _normalize_extract_backend(value: object) -> str:
    if value is None:
        return DEFAULT_EXTRACT_BACKEND
    if isinstance(value, str):
        normalized = value.strip().lower() or DEFAULT_EXTRACT_BACKEND
        if normalized in SUPPORTED_EXTRACT_BACKENDS:
            return normalized
    raise ValueError(Messages.ERROR_CONFIG_VALUE_INVALID.format(field="extract_backend"))


def _coerce_extract_backend(value: object) -> str:
    if value is None:
        return DEFAULT_EXTRACT_BACKEND
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SUPPORTED_EXTRACT_BACKENDS:
            return normalized
    return DEFAULT_EXTRACT_BACKEND


def _normalize_rerank(value: object) -> str:
    if value is None:
        normalized = DEFAULT_RERANK
    elif isinstance(value, str):
        normalized = value.strip().lower() or DEFAULT_RERANK
    else:
        raise ValueError(Messages.ERROR_CONFIG_VALUE_INVALID.format(field="rerank"))
    if normalized not in SUPPORTED_RERANKERS:
        normalized = DEFAULT_RERANK
    return normalized


def _coerce_remote_rerank(value: object) -> RemoteRerankConfig | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return _parse_remote_rerank(dict(value))
    raise ValueError(Messages.ERROR_CONFIG_VALUE_INVALID.format(field="remote_rerank"))
