"""Local embedding backend for Vexor."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..config import local_model_dir
from ..text import Messages


def _load_fastembed():
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError(Messages.ERROR_LOCAL_DEP_MISSING) from exc
    return TextEmbedding


def resolve_fastembed_cache_dir(*, create: bool = True) -> Path:
    """Return the fixed cache directory used for local models."""
    cache_dir = local_model_dir()
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


_CUSTOM_TEXT_MODELS: dict[str, dict[str, object]] = {
    "intfloat/multilingual-e5-small": {
        "model": "intfloat/multilingual-e5-small",
        "pooling": "MEAN",
        "normalization": True,
        "hf": "intfloat/multilingual-e5-small",
        "dim": 384,
        "model_file": "onnx/model.onnx",
        "description": "Multilingual E5 model for cross-lingual retrieval",
        "license": "MIT",
        "size_in_gb": 0.12,
    },
}


def _is_unsupported_model_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "not supported in TextEmbedding" in str(exc)


def _register_custom_model(text_embedding_cls, model_name: str) -> bool:
    spec = _CUSTOM_TEXT_MODELS.get(model_name.strip().lower())
    if not spec:
        return False
    try:
        from fastembed.common.model_description import ModelSource, PoolingType
    except Exception as exc:
        raise RuntimeError(Messages.ERROR_LOCAL_MODEL_LOAD.format(model=model_name, reason=str(exc))) from exc
    try:
        text_embedding_cls.add_custom_model(
            model=spec["model"],
            pooling=getattr(PoolingType, str(spec["pooling"])),
            normalization=bool(spec["normalization"]),
            sources=ModelSource(hf=str(spec["hf"])),
            dim=int(spec["dim"]),
            model_file=str(spec["model_file"]),
            description=str(spec["description"]),
            license=str(spec["license"]),
            size_in_gb=float(spec["size_in_gb"]),
        )
    except ValueError as exc:
        if "already registered" not in str(exc).lower():
            raise
    return True


class LocalEmbeddingBackend:
    """Embedding backend that runs a lightweight local model via fastembed."""

    def __init__(
        self,
        *,
        model_name: str,
        chunk_size: int | None = None,
        concurrency: int = 1,
        cuda: bool = False,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else None
        self.concurrency = max(int(concurrency or 1), 1)
        self.cuda = bool(cuda)
        TextEmbedding = _load_fastembed()
        cache_dir = resolve_fastembed_cache_dir()
        try:
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=str(cache_dir),
                cuda=self.cuda,
            )
        except Exception as exc:
            if _is_unsupported_model_error(exc) and _register_custom_model(
                TextEmbedding, model_name
            ):
                try:
                    self._model = TextEmbedding(
                        model_name=model_name,
                        cache_dir=str(cache_dir),
                        cuda=self.cuda,
                    )
                except Exception as retry_exc:
                    raise RuntimeError(
                        Messages.ERROR_LOCAL_MODEL_LOAD.format(
                            model=model_name, reason=str(retry_exc)
                        )
                    ) from retry_exc
            else:
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
