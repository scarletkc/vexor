"""Prepare optional reranker models for setup commands."""

from ..config import DEFAULT_FLASHRANK_MAX_LENGTH, DEFAULT_FLASHRANK_MODEL, flashrank_cache_dir
from ..text import Messages


def prepare_flashrank_model(model_name: str | None) -> None:
    try:
        from flashrank import Ranker
    except ImportError as exc:
        raise RuntimeError(Messages.ERROR_FLASHRANK_MISSING) from exc
    cache_dir = flashrank_cache_dir()
    try:
        effective_model = model_name or DEFAULT_FLASHRANK_MODEL
        kwargs = {
            "max_length": DEFAULT_FLASHRANK_MAX_LENGTH,
            "cache_dir": str(cache_dir),
            "model_name": effective_model,
        }
        Ranker(**kwargs)
    except Exception as exc:
        raise RuntimeError(Messages.ERROR_FLASHRANK_SETUP.format(reason=str(exc))) from exc
