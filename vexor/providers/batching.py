"""Ordered batch execution shared by remote embedding adapters."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

import numpy as np

from ..text import Messages


class _BatchBackend(Protocol):
    chunk_size: int | None
    concurrency: int
    _executor: ThreadPoolExecutor | None

    def _embed_batch(self, batch: Sequence[str]) -> list[np.ndarray]: ...


def chunk_texts(items: Sequence[str], size: int | None) -> Iterator[Sequence[str]]:
    if size is None or size <= 0:
        yield items
        return
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def embed_batches(backend: _BatchBackend, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    if backend.concurrency > 1:
        batches = list(chunk_texts(texts, backend.chunk_size))
        if len(batches) > 1:
            vectors_by_batch: list[list[np.ndarray] | None] = [None] * len(batches)
            executor = backend._executor
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=backend.concurrency)
                backend._executor = executor
            future_map = {
                executor.submit(backend._embed_batch, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                vectors_by_batch[idx] = future.result()
            vectors = [vec for batch in vectors_by_batch if batch for vec in batch]
        else:
            vectors = []
            for batch in batches:
                vectors.extend(backend._embed_batch(batch))
    else:
        vectors = []
        for batch in chunk_texts(texts, backend.chunk_size):
            vectors.extend(backend._embed_batch(batch))
    if not vectors:
        raise RuntimeError(Messages.ERROR_NO_EMBEDDINGS)
    return np.vstack(vectors)
