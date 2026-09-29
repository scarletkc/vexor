"""Gemini-backed embedding backend for Vexor."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from time import sleep as _sleep

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from ..text import Messages
from .batching import embed_batches
from .capabilities import DEFAULT_GEMINI_MODEL
from .retry import MAX_RETRIES, backoff_delay, should_retry_error


class GeminiEmbeddingBackend:
    """Embedding backend that calls the Gemini API via google-genai."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        chunk_size: int | None = None,
        concurrency: int = 1,
        base_url: str | None = None,
    ) -> None:
        load_dotenv()
        self.model_name = model_name
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else None
        self.concurrency = max(int(concurrency or 1), 1)
        self.api_key = api_key
        if not self.api_key or self.api_key.strip().lower() == "your_api_key_here":
            raise RuntimeError(Messages.ERROR_API_KEY_MISSING)
        client_kwargs: dict[str, object] = {"api_key": self.api_key}
        if base_url:
            client_kwargs["http_options"] = genai_types.HttpOptions(base_url=base_url)
        self._client = genai.Client(**client_kwargs)
        self._executor: ThreadPoolExecutor | None = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return embed_batches(self, texts)

    def _embed_batch(self, batch: Sequence[str]) -> list[np.ndarray]:
        attempt = 0
        while True:
            try:
                response = self._client.models.embed_content(
                    model=self.model_name,
                    contents=list(batch),
                )
                break
            except genai_errors.ClientError as exc:
                if should_retry_error(exc) and attempt < MAX_RETRIES:
                    _sleep(backoff_delay(attempt))
                    attempt += 1
                    continue
                raise RuntimeError(_format_genai_error(exc)) from exc
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise RuntimeError(Messages.ERROR_NO_EMBEDDINGS)
        vectors: list[np.ndarray] = []
        for embedding in embeddings:
            values = getattr(embedding, "values", None) or getattr(
                embedding, "value", None
            )
            vectors.append(np.asarray(values, dtype=np.float32))
        return vectors


def _format_genai_error(exc: genai_errors.ClientError) -> str:
    message = getattr(exc, "message", None) or str(exc)
    if "API key" in message:
        return Messages.ERROR_API_KEY_INVALID
    return f"{Messages.ERROR_GENAI_PREFIX}{message}"
