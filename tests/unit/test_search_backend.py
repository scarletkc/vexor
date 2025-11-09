from types import SimpleNamespace

import numpy as np
import pytest

from vexor import search


@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    class DummyConfig:
        api_key = "cfg-key"
        model = search.DEFAULT_MODEL
        batch_size = 0

    monkeypatch.setattr(search, "load_config", lambda: DummyConfig())
    monkeypatch.delenv(search.ENV_API_KEY, raising=False)


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
        search.genai,
        "Client",
        lambda api_key: SimpleNamespace(models=models),
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
    backend = search.GeminiEmbeddingBackend(model_name="demo", chunk_size=2)

    vectors = backend.embed(["a", "bb", "ccc"])

    assert vectors.shape == (3, 2)
    assert len(models.calls) == 2  # chunked as 2 + 1
    assert models.calls[0] == ["a", "bb"]


def test_gemini_backend_empty(monkeypatch):
    _install_fake_client(monkeypatch, batches=[])
    backend = search.GeminiEmbeddingBackend(model_name="demo", chunk_size=2)

    result = backend.embed([])

    assert result.shape == (0, 0)


def test_gemini_backend_no_embeddings(monkeypatch):
    models = _install_fake_client(monkeypatch, batches=[[]])
    backend = search.GeminiEmbeddingBackend(model_name="demo", chunk_size=None)

    with pytest.raises(RuntimeError) as exc:
        backend.embed(["file.txt"])

    assert "no embeddings" in str(exc.value)


def test_format_genai_error_messages():
    class FakeError:
        def __init__(self, message):
            self.message = message

    msg = search._format_genai_error(FakeError("API key invalid"))
    assert "invalid" in msg

    general = search._format_genai_error(FakeError("quota exceeded"))
    assert "quota" in general


def test_chunk_helper():
    items = ["a", "b", "c", "d"]
    assert list(search._chunk(items, None)) == [items]
    assert list(search._chunk(items, 2)) == [["a", "b"], ["c", "d"]]


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
