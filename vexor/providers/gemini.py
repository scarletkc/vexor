"""Gemini-backed embedding backend for Vexor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Sequence

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from ..config import DEFAULT_GEMINI_MODEL
from ..text import Messages


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

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        batches = list(_chunk(texts, self.chunk_size))
        if self.concurrency > 1 and len(batches) > 1:
            vectors_by_batch: list[list[np.ndarray] | None] = [None] * len(batches)
            with ThreadPoolExecutor(max_workers=min(self.concurrency, len(batches))) as executor:
                future_map = {
                    executor.submit(self._embed_batch, batch): idx
                    for idx, batch in enumerate(batches)
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    vectors_by_batch[idx] = future.result()
            vectors = [vec for batch in vectors_by_batch if batch for vec in batch]
        else:
            vectors = []
            for batch in batches:
                vectors.extend(self._embed_batch(batch))
        if not vectors:
            raise RuntimeError(Messages.ERROR_NO_EMBEDDINGS)
        return np.vstack(vectors)

    def _embed_batch(self, batch: Sequence[str]) -> list[np.ndarray]:
        try:
            response = self._client.models.embed_content(
                model=self.model_name,
                contents=list(batch),
            )
        except genai_errors.ClientError as exc:
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
