"""Gemini-backed embedding backend for Vexor."""

from __future__ import annotations

import os
from typing import Iterator, Sequence

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from ..config import DEFAULT_MODEL, ENV_API_KEY, load_config
from ..text import Messages


class GeminiEmbeddingBackend:
    """Embedding backend that calls the Gemini API via google-genai."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        api_key: str | None = None,
        chunk_size: int | None = None,
    ) -> None:
        load_dotenv()
        config = load_config()
        self.model_name = model_name
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else None
        env_key = os.getenv(ENV_API_KEY)
        configured_key = getattr(config, "api_key", None)
        self.api_key = api_key or configured_key or env_key
        if not self.api_key or self.api_key.strip().lower() == "your_api_key_here":
            raise RuntimeError(Messages.ERROR_API_KEY_MISSING)
        self._client = genai.Client(api_key=self.api_key)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for chunk in _chunk(texts, self.chunk_size):
            try:
                response = self._client.models.embed_content(
                    model=self.model_name,
                    contents=list(chunk),
                )
            except genai_errors.ClientError as exc:
                raise RuntimeError(_format_genai_error(exc)) from exc
            embeddings = getattr(response, "embeddings", None)
            if not embeddings:
                raise RuntimeError(Messages.ERROR_NO_EMBEDDINGS)
            for embedding in embeddings:
                values = getattr(embedding, "values", None) or getattr(
                    embedding, "value", None
                )
                vectors.append(np.asarray(values, dtype=np.float32))
        return np.vstack(vectors)


def _chunk(items: Sequence[str], size: int | None) -> Iterator[Sequence[str]]:
    if size is None or size <= 0:
        yield items
        return
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _format_genai_error(exc: genai_errors.ClientError) -> str:
    message = getattr(exc, "message", None) or str(exc)
    if "API key" in message:
        return Messages.ERROR_API_KEY_INVALID
    return f"{Messages.ERROR_GENAI_PREFIX}{message}"

