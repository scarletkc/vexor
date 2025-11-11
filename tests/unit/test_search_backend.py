from types import SimpleNamespace

import numpy as np
import pytest

from vexor import search
from vexor.providers import gemini as gemini_backend
from vexor.providers import openai as openai_backend


class FakeModels:
    def __init__(self, batches):
        self.batches = batches
        self.calls = []
        self.index = 0

    def embed_content(self, model, contents):
        self.calls.append(list(contents))
        vectors = self.batches[self.index]
        self.index += 1
        embeddings = [SimpleNamespace(values=vec) for vec in vectors]
        return SimpleNamespace(embeddings=embeddings)


def _install_fake_client(monkeypatch, batches):
    models = FakeModels(batches)
    monkeypatch.setattr(
        gemini_backend.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(models=models),
    )
    return models


def test_gemini_backend_chunks_requests(monkeypatch):
    models = _install_fake_client(
        monkeypatch,
        batches=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5]],
        ],
    )
    backend = gemini_backend.GeminiEmbeddingBackend(model_name="demo", chunk_size=2, api_key="cfg-key")

    vectors = backend.embed(["a", "bb", "ccc"])

    assert vectors.shape == (3, 2)
    assert len(models.calls) == 2  # chunked as 2 + 1
    assert models.calls[0] == ["a", "bb"]


def test_gemini_backend_empty(monkeypatch):
    _install_fake_client(monkeypatch, batches=[])
    backend = gemini_backend.GeminiEmbeddingBackend(model_name="demo", chunk_size=2, api_key="cfg-key")

    result = backend.embed([])

    assert result.shape == (0, 0)


def test_gemini_backend_no_embeddings(monkeypatch):
    models = _install_fake_client(monkeypatch, batches=[[]])
    backend = gemini_backend.GeminiEmbeddingBackend(model_name="demo", chunk_size=None, api_key="cfg-key")

    with pytest.raises(RuntimeError) as exc:
        backend.embed(["file.txt"])

    assert "no embeddings" in str(exc.value)


def test_format_genai_error_messages():
    class FakeError:
        def __init__(self, message):
            self.message = message

    msg = gemini_backend._format_genai_error(FakeError("API key invalid"))
    assert "invalid" in msg

    general = gemini_backend._format_genai_error(FakeError("quota exceeded"))
    assert "quota" in general


def test_chunk_helper():
    items = ["a", "b", "c", "d"]
    assert list(gemini_backend._chunk(items, None)) == [items]
    assert list(gemini_backend._chunk(items, 2)) == [["a", "b"], ["c", "d"]]


class FakeOpenAIEmbeddings:
    def __init__(self, batches):
        self.batches = batches
        self.calls = []
        self.index = 0

    def create(self, model, input):
        self.calls.append(list(input))
        vectors = self.batches[self.index]
        self.index += 1
        data = [SimpleNamespace(embedding=vec) for vec in vectors]
        return SimpleNamespace(data=data)


def _install_fake_openai_client(monkeypatch, batches):
    embeddings = FakeOpenAIEmbeddings(batches)

    class FakeClient:
        def __init__(self, **kwargs):
            self.embeddings = embeddings

    monkeypatch.setattr(openai_backend, "OpenAI", FakeClient)
    return embeddings


def test_openai_backend_chunks_requests(monkeypatch):
    embeddings = _install_fake_openai_client(
        monkeypatch,
        batches=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5]],
        ],
    )
    backend = openai_backend.OpenAIEmbeddingBackend(
        model_name="text-embedding-3-small",
        api_key="sk-test",
        chunk_size=2,
    )

    vectors = backend.embed(["a", "bb", "ccc"])

    assert vectors.shape == (3, 2)
    assert embeddings.calls[0] == ["a", "bb"]
    assert len(embeddings.calls) == 2


def test_openai_backend_no_embeddings(monkeypatch):
    _install_fake_openai_client(monkeypatch, batches=[[]])
    backend = openai_backend.OpenAIEmbeddingBackend(
        model_name="text-embedding-3-small",
        api_key="sk-test",
    )

    with pytest.raises(RuntimeError) as exc:
        backend.embed(["input"])

    assert "no embeddings" in str(exc.value).lower()


def test_vexor_searcher_embed_texts(monkeypatch):
    class DummyBackend:
        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return np.asarray([[3.0, 4.0]], dtype=np.float32)

    searcher = search.VexorSearcher(backend=DummyBackend())
    vector = searcher.embed_texts(["name"])

    assert np.allclose(vector[0], [0.6, 0.8])  # normalized
