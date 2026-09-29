"""OpenAI-backed embedding backend for Vexor."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from time import sleep as _sleep

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from ..text import Messages
from .batching import embed_batches
from .retry import MAX_RETRIES, backoff_delay, should_retry_error


class OpenAIEmbeddingBackend:
    """Embedding backend that calls OpenAI's embeddings API."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None,
        chunk_size: int | None = None,
        concurrency: int = 1,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        load_dotenv()
        self.model_name = model_name
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else None
        self.concurrency = max(int(concurrency or 1), 1)
        self.api_key = api_key
        self.dimensions = dimensions if dimensions and dimensions > 0 else None
        if not self.api_key:
            raise RuntimeError(Messages.ERROR_API_KEY_MISSING)
        client_kwargs: dict[str, object] = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")
        self._client = OpenAI(**client_kwargs)
        self._executor: ThreadPoolExecutor | None = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return embed_batches(self, texts)

    def _embed_batch(self, batch: Sequence[str]) -> list[np.ndarray]:
        attempt = 0
        while True:
            try:
                create_kwargs: dict[str, object] = {
                    "model": self.model_name,
                    "input": list(batch),
                }
                if self.dimensions is not None:
                    # Voyage AI uses output_dimension, OpenAI uses dimensions
                    if self.model_name.startswith("voyage"):
                        # Pass Voyage-specific params via extra_body
                        create_kwargs["extra_body"] = {"output_dimension": self.dimensions}
                    else:
                        create_kwargs["dimensions"] = self.dimensions
                response = self._client.embeddings.create(**create_kwargs)
                break
            except Exception as exc:  # pragma: no cover - API client variations
                if should_retry_error(exc) and attempt < MAX_RETRIES:
                    _sleep(backoff_delay(attempt))
                    attempt += 1
                    continue
                raise RuntimeError(_format_openai_error(exc)) from exc
        data = getattr(response, "data", None) or []
        if not data:
            raise RuntimeError(Messages.ERROR_NO_EMBEDDINGS)
        # Batch responses identify inputs by index; response order need not be
        # input order. Never skip a missing row and shift every later query.
        if len(data) != len(batch):
            raise RuntimeError(Messages.ERROR_EMBEDDING_RESPONSE_INDEX)
        vectors: dict[int, np.ndarray] = {}
        for item in data:
            index = getattr(item, "index", None)
            embedding = getattr(item, "embedding", None)
            if (
                type(index) is not int
                or not 0 <= index < len(batch)
                or index in vectors
                or embedding is None
            ):
                raise RuntimeError(Messages.ERROR_EMBEDDING_RESPONSE_INDEX)
            vectors[index] = np.asarray(embedding, dtype=np.float32)
        return [vectors[index] for index in range(len(batch))]


def _format_openai_error(exc: Exception) -> str:
    message = getattr(exc, "message", None) or str(exc)
    return f"{Messages.ERROR_OPENAI_PREFIX}{message}"
