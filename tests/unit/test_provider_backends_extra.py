from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vexor.providers import gemini, local, openai


class DummyOpenAIClient:
    created_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs):
        DummyOpenAIClient.created_kwargs = kwargs
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, *, model, input):
        data = [
            SimpleNamespace(embedding=[float(idx), float(idx + 1)])
            for idx, _text in enumerate(input)
        ]
        return SimpleNamespace(data=data)


def test_openai_backend_batches_and_passes_base_url(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", DummyOpenAIClient)
    backend = openai.OpenAIEmbeddingBackend(
        model_name="text-embedding-test",
        api_key="secret",
        base_url="https://proxy.example.com/",
        chunk_size=2,
        concurrency=1,
    )

    vectors = backend.embed(["a", "b", "c"])

    assert vectors.shape == (3, 2)
    assert DummyOpenAIClient.created_kwargs == {
        "api_key": "secret",
        "base_url": "https://proxy.example.com",
    }


def test_openai_backend_uses_concurrent_batches(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", DummyOpenAIClient)
    backend = openai.OpenAIEmbeddingBackend(
        model_name="text-embedding-test",
        api_key="secret",
        chunk_size=1,
        concurrency=2,
    )

    vectors = backend.embed(["a", "b", "c"])

    assert vectors.shape == (3, 2)
    assert backend._executor is not None


def test_openai_backend_empty_and_missing_key(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", DummyOpenAIClient)

    with pytest.raises(RuntimeError):
        openai.OpenAIEmbeddingBackend(model_name="m", api_key=None)

    backend = openai.OpenAIEmbeddingBackend(model_name="m", api_key="secret")
    assert backend.embed([]).shape == (0, 0)


def test_openai_retry_helpers_accept_status_name_and_message():
    class StatusError(Exception):
        status_code = 429

    class TimeoutProblem(Exception):
        pass

    class ResponseProblem(Exception):
        response = SimpleNamespace(status_code=503)

    assert openai._should_retry_openai_error(StatusError("nope")) is True
    assert openai._should_retry_openai_error(TimeoutProblem("ordinary")) is True
    assert openai._should_retry_openai_error(ResponseProblem("down")) is True
    assert openai._should_retry_openai_error(Exception("service unavailable")) is True
    assert openai._should_retry_openai_error(Exception("bad request")) is False
    assert openai._backoff_delay(20) == openai._RETRY_MAX_DELAY
    assert "boom" in openai._format_openai_error(Exception("boom"))


class DummyGeminiModels:
    def embed_content(self, *, model, contents):
        embeddings = [
            SimpleNamespace(values=[float(idx), float(idx + 1)])
            for idx, _text in enumerate(contents)
        ]
        return SimpleNamespace(embeddings=embeddings)


class DummyGeminiClient:
    created_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs):
        DummyGeminiClient.created_kwargs = kwargs
        self.models = DummyGeminiModels()


def test_gemini_backend_batches_and_base_url(monkeypatch):
    monkeypatch.setattr(gemini.genai, "Client", DummyGeminiClient)
    backend = gemini.GeminiEmbeddingBackend(
        model_name="gemini-test",
        api_key="secret",
        base_url="https://proxy.example.com",
        chunk_size=2,
        concurrency=1,
    )

    vectors = backend.embed(["a", "b", "c"])

    assert vectors.shape == (3, 2)
    assert DummyGeminiClient.created_kwargs["api_key"] == "secret"
    assert "http_options" in DummyGeminiClient.created_kwargs


def test_gemini_backend_uses_concurrent_batches(monkeypatch):
    monkeypatch.setattr(gemini.genai, "Client", DummyGeminiClient)
    backend = gemini.GeminiEmbeddingBackend(
        model_name="gemini-test",
        api_key="secret",
        chunk_size=1,
        concurrency=2,
    )

    vectors = backend.embed(["a", "b", "c"])

    assert vectors.shape == (3, 2)
    assert backend._executor is not None


def test_gemini_backend_empty_missing_key_and_retry_helpers(monkeypatch):
    monkeypatch.setattr(gemini.genai, "Client", DummyGeminiClient)

    with pytest.raises(RuntimeError):
        gemini.GeminiEmbeddingBackend(api_key="your_api_key_here")

    backend = gemini.GeminiEmbeddingBackend(model_name="m", api_key="secret")
    assert backend.embed([]).shape == (0, 0)
    assert gemini._extract_status_code(SimpleNamespace(status=500)) == 500
    assert gemini._should_retry_genai_error(Exception("too many requests")) is True
    assert gemini._should_retry_genai_error(Exception("bad request")) is False
    assert gemini._backoff_delay(20) == gemini._RETRY_MAX_DELAY


def test_gemini_backend_raises_when_response_has_no_embeddings(monkeypatch):
    class EmptyModels:
        def embed_content(self, **_kwargs):
            return SimpleNamespace(embeddings=[])

    class EmptyClient:
        def __init__(self, **_kwargs):
            self.models = EmptyModels()

    monkeypatch.setattr(gemini.genai, "Client", EmptyClient)
    backend = gemini.GeminiEmbeddingBackend(model_name="m", api_key="secret")

    with pytest.raises(RuntimeError):
        backend.embed(["a"])


def test_gemini_format_error_handles_api_key_message():
    exc = SimpleNamespace(message="API key is invalid")
    assert gemini._format_genai_error(exc) == gemini.Messages.ERROR_API_KEY_INVALID


class DummyTextEmbedding:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs):
        DummyTextEmbedding.calls.append(kwargs)

    def embed(self, texts):
        for idx, _text in enumerate(texts):
            yield [float(idx), float(idx + 1)]


def test_local_backend_success_empty_and_embed_error(monkeypatch, tmp_path):
    monkeypatch.setattr(local, "_load_fastembed", lambda: DummyTextEmbedding)
    monkeypatch.setattr(local, "local_model_dir", lambda: tmp_path / "models")
    backend = local.LocalEmbeddingBackend(model_name="local-test", chunk_size=1, cuda=True)

    vectors = backend.embed(["a", "b"])

    assert vectors.shape == (2, 2)
    assert backend.embed([]).shape == (0, 0)
    assert DummyTextEmbedding.calls[-1]["cuda"] is True

    class BrokenModel:
        def embed(self, _texts):
            raise ValueError("broken")

    backend._model = BrokenModel()
    with pytest.raises(RuntimeError, match="broken"):
        backend.embed(["x"])


def test_local_backend_wraps_model_load_error(monkeypatch, tmp_path):
    class BrokenTextEmbedding:
        def __init__(self, **_kwargs):
            raise ValueError("cannot load")

    monkeypatch.setattr(local, "_load_fastembed", lambda: BrokenTextEmbedding)
    monkeypatch.setattr(local, "local_model_dir", lambda: tmp_path / "models")

    with pytest.raises(RuntimeError, match="cannot load"):
        local.LocalEmbeddingBackend(model_name="missing")


def test_local_load_fastembed_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)

    with pytest.raises(RuntimeError):
        local._load_fastembed()


def test_local_register_custom_model_success_and_already_registered(monkeypatch):
    model_description = ModuleType("fastembed.common.model_description")

    class ModelSource:
        def __init__(self, *, hf):
            self.hf = hf

    class PoolingType:
        MEAN = "MEAN"

    model_description.ModelSource = ModelSource
    model_description.PoolingType = PoolingType
    monkeypatch.setitem(sys.modules, "fastembed.common.model_description", model_description)

    class TextEmbedding:
        calls = []

        @classmethod
        def add_custom_model(cls, **kwargs):
            cls.calls.append(kwargs)

    assert local._register_custom_model(TextEmbedding, "intfloat/multilingual-e5-small") is True
    assert TextEmbedding.calls[0]["model"] == "intfloat/multilingual-e5-small"
    assert local._register_custom_model(TextEmbedding, "unknown/model") is False

    class AlreadyRegistered:
        @classmethod
        def add_custom_model(cls, **_kwargs):
            raise ValueError("already registered")

    assert local._register_custom_model(AlreadyRegistered, "intfloat/multilingual-e5-small") is True


def test_local_register_custom_model_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed.common.model_description", None)

    with pytest.raises(RuntimeError):
        local._register_custom_model(object, "intfloat/multilingual-e5-small")

    model_description = ModuleType("fastembed.common.model_description")
    model_description.ModelSource = lambda **_kwargs: object()
    model_description.PoolingType = SimpleNamespace(MEAN="MEAN")
    monkeypatch.setitem(sys.modules, "fastembed.common.model_description", model_description)

    class FailingTextEmbedding:
        @classmethod
        def add_custom_model(cls, **_kwargs):
            raise ValueError("different error")

    with pytest.raises(ValueError, match="different error"):
        local._register_custom_model(FailingTextEmbedding, "intfloat/multilingual-e5-small")


def test_local_backend_registers_custom_model_after_unsupported_error(monkeypatch, tmp_path):
    class TextEmbedding:
        calls = 0

        def __init__(self, **_kwargs):
            TextEmbedding.calls += 1
            if TextEmbedding.calls == 1:
                raise ValueError("not supported in TextEmbedding")

        @classmethod
        def add_custom_model(cls, **_kwargs):
            return None

        def embed(self, texts):
            for _text in texts:
                yield [1.0, 2.0]

    monkeypatch.setattr(local, "_load_fastembed", lambda: TextEmbedding)
    monkeypatch.setattr(local, "_register_custom_model", lambda _cls, _model: True)
    monkeypatch.setattr(local, "local_model_dir", lambda: tmp_path / "models")

    backend = local.LocalEmbeddingBackend(model_name="intfloat/multilingual-e5-small")

    assert TextEmbedding.calls == 2
    assert backend.embed(["x"]).shape == (1, 2)


def test_local_chunk_and_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(local, "local_model_dir", lambda: tmp_path / "models")

    cache_dir = local.resolve_fastembed_cache_dir(create=True)

    assert cache_dir.exists()
    assert list(local._chunk(["a", "b", "c"], 2)) == [["a", "b"], ["c"]]
    assert list(local._chunk(["a"], None)) == [["a"]]
