"""OpenAI-backed embedding backend for Vexor."""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from ..text import Messages


class OpenAIEmbeddingBackend:
    """Embedding backend that calls OpenAI's embeddings API."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None,
        chunk_size: int | None = None,
        base_url: str | None = None,
    ) -> None:
        load_dotenv()
        self.model_name = model_name
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else None
        self.api_key = api_key
        if not self.api_key:
            raise RuntimeError(Messages.ERROR_API_KEY_MISSING)
        client_kwargs: dict[str, object] = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")
        self._client = OpenAI(**client_kwargs)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for chunk in _chunk(texts, self.chunk_size):
            try:
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=list(chunk),
                )
            except Exception as exc:  # pragma: no cover - API client variations
                raise RuntimeError(_format_openai_error(exc)) from exc
            data = getattr(response, "data", None) or []
            if not data:
                raise RuntimeError(Messages.ERROR_NO_EMBEDDINGS)
            for item in data:
                embedding = getattr(item, "embedding", None)
                if embedding is None:
                    continue
                vectors.append(np.asarray(embedding, dtype=np.float32))
        return np.vstack(vectors)


def _chunk(items: Sequence[str], size: int | None) -> Iterator[Sequence[str]]:
    if size is None or size <= 0:
        yield items
        return
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _format_openai_error(exc: Exception) -> str:
    message = getattr(exc, "message", None) or str(exc)
    return f"{Messages.ERROR_OPENAI_PREFIX}{message}"
