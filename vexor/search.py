"""Semantic search helpers backed by the Google Gemini embedding API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Protocol, Sequence

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from sklearn.metrics.pairwise import cosine_similarity

from .config import DEFAULT_MODEL, ENV_API_KEY, load_config
from .text import Messages


@dataclass(slots=True)
class SearchResult:
    """Container describing a single semantic search hit."""

    path: Path
    score: float


class EmbeddingBackend(Protocol):
    """Minimal protocol for components that can embed text batches."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings for *texts* as a 2D numpy array."""
        raise NotImplementedError  # pragma: no cover


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


class VexorSearcher:
    """Encapsulates embedding generation and similarity computation."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        backend: EmbeddingBackend | None = None,
        batch_size: int = 0,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(batch_size, 0)
        self._backend = backend or GeminiEmbeddingBackend(
            model_name=model_name, chunk_size=self.batch_size
        )
        self._device = f"{self.model_name} via Gemini API"

    @property
    def device(self) -> str:
        """Return a description of the remote backend in use."""
        return self._device

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._backend.embed(texts)
        if embeddings.size == 0:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Public helper to encode arbitrary text batches."""
        return self._encode(texts)

    def search(self, query: str, files: Sequence[Path], top_k: int = 5) -> List[SearchResult]:
        """Return the *top_k* most similar files for *query*."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query text must not be empty")
        if not files:
            return []
        file_labels = [self._prepare_text(path) for path in files]
        file_vectors = self._encode(file_labels)
        query_vector = self._encode([clean_query])[0]
        similarities = cosine_similarity(
            query_vector.reshape(1, -1), file_vectors
        )[0]
        scored = [
            SearchResult(path=path, score=float(score))
            for path, score in zip(files, similarities)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _prepare_text(path: Path) -> str:
        """Return the text representation of a file path for embedding."""
        return path.name.replace("_", " ")


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
