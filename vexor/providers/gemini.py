"""Gemini-backed embedding backend for Vexor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
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
        self._executor: ThreadPoolExecutor | None = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.concurrency > 1:
            batches = list(_chunk(texts, self.chunk_size))
            if len(batches) > 1:
                vectors_by_batch: list[list[np.ndarray] | None] = [None] * len(batches)
                executor = self._executor
                if executor is None:
                    executor = ThreadPoolExecutor(max_workers=self.concurrency)
                    self._executor = executor
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
        else:
            vectors = []
            for batch in _chunk(texts, self.chunk_size):
                vectors.extend(self._embed_batch(batch))
        if not vectors:
            raise RuntimeError(Messages.ERROR_NO_EMBEDDINGS)
        return np.vstack(vectors)

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
                if _should_retry_genai_error(exc) and attempt < _MAX_RETRIES:
                    _sleep(_backoff_delay(attempt))
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


def _chunk(items: Sequence[str], size: int | None) -> Iterator[Sequence[str]]:
    if size is None or size <= 0:
        yield items
        return
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 0.5
_RETRY_MAX_DELAY = 4.0


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _backoff_delay(attempt: int) -> float:
    return min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * (2**attempt))


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def _should_retry_genai_error(exc: Exception) -> bool:
    status = _extract_status_code(exc)
    if status in _RETRYABLE_STATUS_CODES:
        return True
    name = exc.__class__.__name__.lower()
    if "ratelimit" in name or "timeout" in name or "temporarily" in name:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "rate limit",
            "timeout",
            "temporar",
            "overload",
            "try again",
            "too many requests",
            "service unavailable",
        )
    )


def _format_genai_error(exc: genai_errors.ClientError) -> str:
    message = getattr(exc, "message", None) or str(exc)
    if "API key" in message:
        return Messages.ERROR_API_KEY_INVALID
    return f"{Messages.ERROR_GENAI_PREFIX}{message}"
