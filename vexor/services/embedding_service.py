"""Cached embedding helper shared by the file index and the collection store.

Both consumers key the shared ``embedding_cache`` table by ``(model, text_hash)``
so a text embedded once is never paid for twice, whichever entry point saw it
first. ``embedding_dimension`` is the *configured* dimension used to segment the
cache, not the width of the returned vectors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def embed_texts_with_cache(
    *,
    searcher,
    model_name: str,
    labels: Sequence[str],
    no_cache: bool = False,
    embedding_dimension: int | None = None,
) -> np.ndarray:
    """Embed labels with caching support.

    Args:
        searcher: The embedding searcher instance
        model_name: Name of the embedding model
        labels: Sequence of label strings to embed
        no_cache: If True, bypass cache entirely
        embedding_dimension: Embedding dimension for cache segmentation (prevents
            cross-dimension cache pollution when dimension settings change)
    """
    if not labels:
        return np.empty((0, 0), dtype=np.float32)
    if no_cache:
        vectors = searcher.embed_texts(labels)
        return np.asarray(vectors, dtype=np.float32)
    from ..cache import embedding_cache_key, load_embedding_cache, store_embedding_cache

    # Include dimension in cache key to prevent cross-dimension cache pollution
    hashes = [embedding_cache_key(label, dimension=embedding_dimension) for label in labels]
    cached = load_embedding_cache(model_name, hashes, dimension=embedding_dimension)
    missing: dict[str, str] = {}
    for label, text_hash in zip(labels, hashes, strict=True):
        vector = cached.get(text_hash)
        if (vector is None or vector.size == 0) and text_hash not in missing:
            missing[text_hash] = label

    if missing:
        missing_items = list(missing.items())
        missing_labels = [label for _, label in missing_items]
        new_vectors = searcher.embed_texts(missing_labels)
        stored: dict[str, np.ndarray] = {}
        for idx, (text_hash, _) in enumerate(missing_items):
            vector = np.asarray(new_vectors[idx], dtype=np.float32)
            cached[text_hash] = vector
            stored[text_hash] = vector
        store_embedding_cache(
            model=model_name, embeddings=stored, dimension=embedding_dimension
        )

    vectors = [cached[text_hash] for text_hash in hashes]
    return np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])
