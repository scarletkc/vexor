from types import SimpleNamespace

import numpy as np
import pytest

from vexor import search
from vexor.providers import gemini as gemini_backend
from vexor.providers import local as local_backend
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


def test_gemini_backend_rejects_placeholder_api_key():
    with pytest.raises(RuntimeError) as exc:
        gemini_backend.GeminiEmbeddingBackend(model_name="demo", chunk_size=2, api_key="your_api_key_here")
    assert "api key" in str(exc.value).lower()


def test_gemini_backend_passes_base_url(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(models=FakeModels([[[1.0, 2.0]]]))

    monkeypatch.setattr(gemini_backend.genai, "Client", fake_client)

    backend = gemini_backend.GeminiEmbeddingBackend(
        model_name="demo",
        chunk_size=None,
        api_key="cfg-key",
        base_url="https://example.com/",
    )
    vectors = backend.embed(["x"])
    assert vectors.shape == (1, 2)
    assert "http_options" in captured


def test_gemini_backend_formats_client_errors(monkeypatch):
    class DummyClientError(Exception):
        def __init__(self, message: str):
            super().__init__(message)
            self.message = message

    monkeypatch.setattr(gemini_backend.genai_errors, "ClientError", DummyClientError)

    class BoomModels:
        def embed_content(self, *_args, **_kwargs):
            raise DummyClientError("API key invalid")

    monkeypatch.setattr(
        gemini_backend.genai,
        "Client",
        lambda **_kwargs: SimpleNamespace(models=BoomModels()),
    )
    backend = gemini_backend.GeminiEmbeddingBackend(model_name="demo", chunk_size=None, api_key="cfg-key")

    with pytest.raises(RuntimeError) as exc:
        backend.embed(["x"])

    assert "invalid" in str(exc.value).lower()


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


def test_gemini_backend_retries_transient_errors(monkeypatch):
    calls = {"count": 0}

    class DummyClientError(Exception):
        def __init__(self, message: str, status: int = 503) -> None:
            super().__init__(message)
            self.message = message
            self.status = status

    monkeypatch.setattr(gemini_backend.genai_errors, "ClientError", DummyClientError)

    class FlakyModels:
        def embed_content(self, *_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise DummyClientError("rate limit")
            embeddings = [SimpleNamespace(values=[1.0, 0.0])]
            return SimpleNamespace(embeddings=embeddings)

    monkeypatch.setattr(
        gemini_backend.genai,
        "Client",
        lambda **_kwargs: SimpleNamespace(models=FlakyModels()),
    )
    monkeypatch.setattr(gemini_backend, "_sleep", lambda _seconds: None)

    backend = gemini_backend.GeminiEmbeddingBackend(model_name="demo", chunk_size=None, api_key="cfg-key")
    vectors = backend.embed(["x"])

    assert calls["count"] == 2
    assert vectors.shape == (1, 2)


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


def test_openai_backend_rejects_missing_api_key():
    with pytest.raises(RuntimeError) as exc:
        openai_backend.OpenAIEmbeddingBackend(
            model_name="text-embedding-3-small",
            api_key=None,
        )
    assert "api key" in str(exc.value).lower()


def test_openai_backend_passes_base_url(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.embeddings = FakeOpenAIEmbeddings([[[1.0, 2.0]]])

    monkeypatch.setattr(openai_backend, "OpenAI", FakeClient)

    backend = openai_backend.OpenAIEmbeddingBackend(
        model_name="text-embedding-3-small",
        api_key="sk-test",
        base_url="https://example.com/",
    )
    vectors = backend.embed(["x"])
    assert vectors.shape == (1, 2)
    assert captured["base_url"] == "https://example.com"


def test_openai_backend_empty_texts(monkeypatch):
    _install_fake_openai_client(monkeypatch, batches=[])
    backend = openai_backend.OpenAIEmbeddingBackend(
        model_name="text-embedding-3-small",
        api_key="sk-test",
    )

    result = backend.embed([])

    assert result.shape == (0, 0)


def test_openai_backend_skips_none_embeddings(monkeypatch):
    class MixedEmbeddings:
        def create(self, *_args, **_kwargs):
            data = [
                SimpleNamespace(embedding=None),
                SimpleNamespace(embedding=[1.0, 0.0]),
            ]
            return SimpleNamespace(data=data)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.embeddings = MixedEmbeddings()

    monkeypatch.setattr(openai_backend, "OpenAI", FakeClient)

    backend = openai_backend.OpenAIEmbeddingBackend(
        model_name="text-embedding-3-small",
        api_key="sk-test",
    )
    vectors = backend.embed(["x"])
    assert vectors.shape == (1, 2)


def test_openai_backend_no_embeddings(monkeypatch):
    _install_fake_openai_client(monkeypatch, batches=[[]])
    backend = openai_backend.OpenAIEmbeddingBackend(
        model_name="text-embedding-3-small",
        api_key="sk-test",
    )

    with pytest.raises(RuntimeError) as exc:
        backend.embed(["input"])

    assert "no embeddings" in str(exc.value).lower()


def test_openai_backend_retries_transient_errors(monkeypatch):
    calls = {"count": 0}

    class DummyError(Exception):
        def __init__(self) -> None:
            super().__init__("rate limit")
            self.message = "rate limit"
            self.status_code = 429

    class FlakyEmbeddings:
        def create(self, *_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise DummyError()
            data = [SimpleNamespace(embedding=[1.0, 0.0])]
            return SimpleNamespace(data=data)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.embeddings = FlakyEmbeddings()

    monkeypatch.setattr(openai_backend, "OpenAI", FakeClient)
    monkeypatch.setattr(openai_backend, "_sleep", lambda _seconds: None)

    backend = openai_backend.OpenAIEmbeddingBackend(
        model_name="text-embedding-3-small",
        api_key="sk-test",
    )
    vectors = backend.embed(["x"])

    assert calls["count"] == 2
    assert vectors.shape == (1, 2)


def test_openai_chunk_helper():
    items = ["a", "b"]
    assert list(openai_backend._chunk(items, None)) == [items]


def test_format_openai_error_prefers_message_attr():
    class FakeError(Exception):
        def __init__(self, message: str) -> None:
            super().__init__("ignored")
            self.message = message

        def __str__(self) -> str:
            return "fallback"

    msg = openai_backend._format_openai_error(FakeError("boom"))
    assert "boom" in msg


def test_local_backend_chunks_requests(monkeypatch, tmp_path):
    class FakeEmbeddingModel:
        def __init__(
            self,
            model_name: str,
            cache_dir: str | None = None,
            cuda: bool | None = None,
        ) -> None:
            self.model_name = model_name
            self.cache_dir = cache_dir
            self.cuda = cuda
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            for _ in texts:
                yield np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(local_backend, "_load_fastembed", lambda: FakeEmbeddingModel)
    def _cache_dir(create: bool = True):
        path = tmp_path / "models"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(local_backend, "resolve_fastembed_cache_dir", _cache_dir)

    backend = local_backend.LocalEmbeddingBackend(model_name="demo", chunk_size=2)
    vectors = backend.embed(["a", "bb", "ccc"])

    assert vectors.shape == (3, 2)
    assert backend._model.calls == [["a", "bb"], ["ccc"]]


def test_local_backend_requires_dependency(monkeypatch):
    def _boom():
        raise RuntimeError("missing")

    monkeypatch.setattr(local_backend, "_load_fastembed", _boom)
    with pytest.raises(RuntimeError, match="missing"):
        local_backend.LocalEmbeddingBackend(model_name="demo")


def test_local_backend_registers_custom_model(monkeypatch, tmp_path):
    calls = {"registered": 0}

    class FakeTextEmbedding:
        def __init__(
            self,
            model_name: str,
            cache_dir: str | None = None,
            cuda: bool | None = None,
        ) -> None:
            if calls["registered"] == 0:
                raise ValueError(
                    "Model intfloat/multilingual-e5-small is not supported in TextEmbedding."
                )
            self.model_name = model_name
            self.cache_dir = cache_dir
            self.cuda = cuda

        def embed(self, texts):
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]

    monkeypatch.setattr(local_backend, "_load_fastembed", lambda: FakeTextEmbedding)
    def _cache_dir(create: bool = True):
        path = tmp_path / "models"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(local_backend, "resolve_fastembed_cache_dir", _cache_dir)

    def fake_register(_cls, model_name: str) -> bool:
        assert model_name == "intfloat/multilingual-e5-small"
        calls["registered"] += 1
        return True

    monkeypatch.setattr(local_backend, "_register_custom_model", fake_register)

    backend = local_backend.LocalEmbeddingBackend(model_name="intfloat/multilingual-e5-small")
    vectors = backend.embed(["x"])

    assert calls["registered"] == 1
    assert vectors.shape == (1, 2)


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
