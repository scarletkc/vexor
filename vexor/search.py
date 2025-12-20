"""Semantic search helpers backed by pluggable embedding backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    DEFAULT_EMBED_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    resolve_api_key,
)
from .providers.gemini import GeminiEmbeddingBackend
from .providers.local import LocalEmbeddingBackend
from .providers.openai import OpenAIEmbeddingBackend
from .text import Messages


@dataclass(slots=True)
class SearchResult:
    """Container describing a single semantic search hit."""

    path: Path
    score: float
    preview: str | None = None
    chunk_index: int = 0
    start_line: int | None = None
    end_line: int | None = None


class EmbeddingBackend(Protocol):
    """Minimal protocol for components that can embed text batches."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings for *texts* as a 2D numpy array."""
        raise NotImplementedError  # pragma: no cover


class VexorSearcher:
    """Encapsulates embedding generation and similarity computation."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        backend: EmbeddingBackend | None = None,
        batch_size: int = 0,
        embed_concurrency: int = DEFAULT_EMBED_CONCURRENCY,
        provider: str = DEFAULT_PROVIDER,
        base_url: str | None = None,
        api_key: str | None = None,
        local_cuda: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(batch_size, 0)
        self.embed_concurrency = max(int(embed_concurrency or 1), 1)
        self.provider = (provider or DEFAULT_PROVIDER).lower()
        self.base_url = base_url
        self.api_key = resolve_api_key(api_key, self.provider)
        self.local_cuda = bool(local_cuda)
        if backend is not None:
            self._backend = backend
            self._device = getattr(backend, "device", "Custom embedding backend")
        else:
            self._backend = self._create_backend()

    @property
    def device(self) -> str:
        """Return a description of the remote backend in use."""
        return self._device

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        unique_texts, inverse = self._dedupe_texts(texts)
        embeddings = self._backend.embed(unique_texts)
        if embeddings.size == 0:
            return embeddings
        if embeddings.shape[0] != len(unique_texts):
            embeddings = self._backend.embed(texts)
            if embeddings.size == 0:
                return embeddings
        else:
            if len(unique_texts) != len(texts):
                embeddings = embeddings[inverse]
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

    def _create_backend(self) -> EmbeddingBackend:
        if self.provider == "gemini":
            self._device = f"{self.model_name} via Gemini API"
            return GeminiEmbeddingBackend(
                model_name=self.model_name,
                chunk_size=self.batch_size,
                concurrency=self.embed_concurrency,
                base_url=self.base_url,
                api_key=self.api_key,
            )
        if self.provider == "local":
            self._device = f"{self.model_name} via local model"
            return LocalEmbeddingBackend(
                model_name=self.model_name,
                chunk_size=self.batch_size,
                concurrency=self.embed_concurrency,
                cuda=self.local_cuda,
            )
        if self.provider == "custom":
            base_url = (self.base_url or "").strip()
            if not base_url:
                raise RuntimeError(Messages.ERROR_CUSTOM_BASE_URL_REQUIRED)
            if not self.model_name or not self.model_name.strip():
                raise RuntimeError(Messages.ERROR_CUSTOM_MODEL_REQUIRED)
            self._device = f"{self.model_name} via OpenAI-compatible API"
            return OpenAIEmbeddingBackend(
                model_name=self.model_name,
                chunk_size=self.batch_size,
                concurrency=self.embed_concurrency,
                base_url=base_url,
                api_key=self.api_key,
            )
        if self.provider == "openai":
            self._device = f"{self.model_name} via OpenAI API"
            return OpenAIEmbeddingBackend(
                model_name=self.model_name,
                chunk_size=self.batch_size,
                concurrency=self.embed_concurrency,
                base_url=self.base_url,
                api_key=self.api_key,
            )
        allowed = ", ".join(SUPPORTED_PROVIDERS)
        raise RuntimeError(
            Messages.ERROR_PROVIDER_INVALID.format(value=self.provider, allowed=allowed)
        )

    @staticmethod
    def _dedupe_texts(texts: Sequence[str]) -> tuple[list[str], list[int]]:
        unique_texts: list[str] = []
        index_map: dict[str, int] = {}
        inverse: list[int] = []
        for text in texts:
            position = index_map.get(text)
            if position is None:
                position = len(unique_texts)
                unique_texts.append(text)
                index_map[text] = position
            inverse.append(position)
        return unique_texts, inverse
