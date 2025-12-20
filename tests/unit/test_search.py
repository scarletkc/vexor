from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vexor.config import DEFAULT_EMBED_CONCURRENCY
from vexor.search import SearchResult, VexorSearcher


class DummyBackend:
    device = "dummy-device"

    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = vectors

    def embed(self, texts):
        # Return vectors sized to input length.
        count = len(texts)
        if count == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return self._vectors[:count].copy()


def test_vexor_searcher_encode_normalizes():
    backend = DummyBackend(np.array([[3.0, 4.0]], dtype=np.float32))
    searcher = VexorSearcher(model_name="m", backend=backend)
    vec = searcher.embed_texts(["x"])[0]
    assert np.allclose(vec, np.array([0.6, 0.8], dtype=np.float32))
    assert searcher.device == "dummy-device"

    empty = searcher.embed_texts([])
    assert empty.shape == (0, 0)


def test_vexor_searcher_search_ranks(monkeypatch, tmp_path):
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    backend = DummyBackend(vectors)
    searcher = VexorSearcher(model_name="m", backend=backend)

    files = [tmp_path / "a_config.py", tmp_path / "b_other.py"]
    results = searcher.search("config", files, top_k=2)
    assert [r.path for r in results] == files
    assert all(isinstance(r, SearchResult) for r in results)


def test_vexor_searcher_search_validates_inputs(tmp_path):
    backend = DummyBackend(np.array([[1.0, 0.0]], dtype=np.float32))
    searcher = VexorSearcher(model_name="m", backend=backend)
    with pytest.raises(ValueError, match="must not be empty"):
        searcher.search("   ", [tmp_path / "a.txt"])
    assert searcher.search("ok", []) == []


def test_vexor_searcher_invalid_provider_raises():
    with pytest.raises(RuntimeError, match="provider"):
        VexorSearcher(model_name="m", provider="invalid-provider")


def test_vexor_searcher_creates_gemini_backend(monkeypatch):
    created = {}

    class DummyGeminiBackend:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def embed(self, texts):
            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr("vexor.search.GeminiEmbeddingBackend", DummyGeminiBackend)
    searcher = VexorSearcher(model_name="m", provider="gemini", api_key="k", batch_size=2)
    assert "Gemini" in searcher.device
    assert created["model_name"] == "m"
    assert created["chunk_size"] == 2
    assert created["concurrency"] == DEFAULT_EMBED_CONCURRENCY


def test_vexor_searcher_creates_openai_backend(monkeypatch):
    created = {}

    class DummyOpenAIBackend:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def embed(self, texts):
            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr("vexor.search.OpenAIEmbeddingBackend", DummyOpenAIBackend)
    searcher = VexorSearcher(model_name="m", provider="openai", api_key="k")
    assert "OpenAI" in searcher.device
    assert created["model_name"] == "m"
    assert created["concurrency"] == DEFAULT_EMBED_CONCURRENCY


def test_vexor_searcher_creates_custom_backend(monkeypatch):
    created = {}

    class DummyOpenAIBackend:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def embed(self, texts):
            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr("vexor.search.OpenAIEmbeddingBackend", DummyOpenAIBackend)
    searcher = VexorSearcher(
        model_name="m",
        provider="custom",
        api_key="k",
        base_url="https://example.com",
    )
    assert "OpenAI-compatible" in searcher.device
    assert created["base_url"] == "https://example.com"
    assert created["concurrency"] == DEFAULT_EMBED_CONCURRENCY


def test_vexor_searcher_creates_local_backend(monkeypatch):
    created = {}

    class DummyLocalBackend:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def embed(self, texts):
            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr("vexor.search.LocalEmbeddingBackend", DummyLocalBackend)
    searcher = VexorSearcher(model_name="m", provider="local")
    assert "local" in searcher.device.lower()
    assert created["model_name"] == "m"
    assert created["concurrency"] == DEFAULT_EMBED_CONCURRENCY


def test_vexor_searcher_custom_requires_base_url():
    with pytest.raises(RuntimeError, match="base URL"):
        VexorSearcher(model_name="m", provider="custom", api_key="k")
