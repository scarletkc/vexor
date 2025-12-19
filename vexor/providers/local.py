"""Local embedding backend for Vexor."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..text import Messages


def _load_fastembed():
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError(Messages.ERROR_LOCAL_DEP_MISSING) from exc
    return TextEmbedding


def resolve_fastembed_cache_dir() -> Path:
    """Match fastembed's default cache resolution."""
    override = os.getenv("FASTEMBED_CACHE_PATH")
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir()) / "fastembed_cache"


class LocalEmbeddingBackend:
    """Embedding backend that runs a lightweight local model via fastembed."""

    def __init__(
        self,
        *,
        model_name: str,
        chunk_size: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else None
        TextEmbedding = _load_fastembed()
        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception as exc:
            raise RuntimeError(
                Messages.ERROR_LOCAL_MODEL_LOAD.format(model=model_name, reason=str(exc))
            ) from exc

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for chunk in _chunk(texts, self.chunk_size):
            try:
                for embedding in self._model.embed(list(chunk)):
                    vectors.append(np.asarray(embedding, dtype=np.float32))
            except Exception as exc:
                raise RuntimeError(
                    Messages.ERROR_LOCAL_MODEL_EMBED.format(reason=str(exc))
                ) from exc
        if not vectors:
            raise RuntimeError(Messages.ERROR_NO_EMBEDDINGS)
        return np.vstack(vectors)


def _chunk(items: Sequence[str], size: int | None) -> Iterator[Sequence[str]]:
    if size is None or size <= 0:
        yield items
        return
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]
