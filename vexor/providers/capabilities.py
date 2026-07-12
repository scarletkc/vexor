"""Provider capability metadata.

This module is the home for default models, provider environment variables,
base URLs, and dimension rules. New provider-specific rules belong here, not
in config.py. Convert this module to a per-provider registry when the next
provider (for example, Azure) lands.
"""

import os

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
DEFAULT_VOYAGE_MODEL = "voyage-3-large"
DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_PROVIDER = "openai"
VOYAGE_BASE_URL = "https://api.voyageai.com/v1"
SUPPORTED_PROVIDERS: tuple[str, ...] = (DEFAULT_PROVIDER, "gemini", "voyageai", "custom", "local")

# Models that support the dimensions parameter (model prefix/name -> supported dimensions)
DIMENSION_SUPPORTED_MODELS: dict[str, tuple[int, ...]] = {
    "text-embedding-3-small": (256, 512, 1024, 1536),
    "text-embedding-3-large": (256, 512, 1024, 1536, 3072),
    "voyage-3": (256, 512, 1024, 2048),
    "voyage-code-3": (256, 512, 1024, 2048),
}

ENV_API_KEY = "VEXOR_API_KEY"
LEGACY_GEMINI_ENV = "GOOGLE_GENAI_API_KEY"
OPENAI_ENV = "OPENAI_API_KEY"
VOYAGE_ENV = "VOYAGE_API_KEY"


def resolve_default_model(provider: str | None, model: str | None) -> str:
    """Return the effective model name for the selected provider."""
    clean_model = (model or "").strip()
    normalized = (provider or DEFAULT_PROVIDER).lower()
    if normalized == "gemini" and (not clean_model or clean_model == DEFAULT_MODEL):
        return DEFAULT_GEMINI_MODEL
    if normalized == "voyageai" and (not clean_model or clean_model == DEFAULT_MODEL):
        return DEFAULT_VOYAGE_MODEL
    if clean_model:
        return clean_model
    return DEFAULT_MODEL


def resolve_base_url(provider: str | None, configured_url: str | None) -> str | None:
    """Return the effective base URL for the selected provider."""
    if configured_url:
        return configured_url
    normalized = (provider or DEFAULT_PROVIDER).lower()
    if normalized == "voyageai":
        return VOYAGE_BASE_URL
    return None


def supports_dimensions(model: str) -> bool:
    """Check if a model supports the dimensions parameter."""
    return get_supported_dimensions(model) is not None


def get_supported_dimensions(model: str) -> tuple[int, ...] | None:
    """Return the supported dimensions for a model, or None if not supported."""
    model_lower = model.lower()
    for prefix, dims in DIMENSION_SUPPORTED_MODELS.items():
        if model_lower.startswith(prefix):
            return dims
    return None


def validate_embedding_dimensions_for_model(value: int | None, model: str) -> None:
    """Validate that `value` is supported by `model` when value is set."""
    if value is None:
        return
    supported = get_supported_dimensions(model)
    if not supported:
        raise ValueError(
            f"Model '{model}' does not support custom dimensions. "
            f"Supported model names/prefixes: {', '.join(DIMENSION_SUPPORTED_MODELS.keys())}"
        )
    if value not in supported:
        raise ValueError(
            f"Dimension {value} is not supported for model '{model}'. "
            f"Supported dimensions: {supported}"
        )


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
    if normalized == "voyageai":
        voyage_key = os.getenv(VOYAGE_ENV)
        if voyage_key:
            return voyage_key
    if normalized in {"openai", "custom"}:
        openai_key = os.getenv(OPENAI_ENV)
        if openai_key:
            return openai_key
    return None
