from collections.abc import Iterator, Sequence
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vexor.providers import gemini, openai

RemoteBackend = openai.OpenAIEmbeddingBackend | gemini.GeminiEmbeddingBackend


@pytest.fixture(params=["openai", "gemini"])
def backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[RemoteBackend]:
    monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: None)
    monkeypatch.setattr(gemini.genai, "Client", lambda **_kwargs: None)
    cls = (openai.OpenAIEmbeddingBackend if request.param == "openai"
           else gemini.GeminiEmbeddingBackend)
    instance = cls(model_name="test", api_key="test", chunk_size=2, concurrency=2)
    yield instance
    if instance._executor is not None:
        instance._executor.shutdown(wait=True)


def test_batches_preserve_order_when_completion_is_reversed(
    backend: RemoteBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    second_finished = Event()

    def embed_batch(texts: Sequence[str]) -> list[np.ndarray]:
        if texts[0] == "0":
            assert second_finished.wait(timeout=5), "batches were not concurrent"
        else:
            second_finished.set()
        return [np.array([float(text), 1.0], dtype=np.float32) for text in texts]

    monkeypatch.setattr(backend, "_embed_batch", embed_batch)
    assert backend.embed([]).shape == (0, 0)
    assert backend._executor is None
    result = backend.embed(["0", "1", "2", "3", "4"])
    assert result[:, 0].tolist() == [0, 1, 2, 3, 4]
    executor = backend._executor
    assert backend.embed(["2", "3", "4"]).shape == (3, 2)
    assert backend._executor is executor


def test_failed_batch_propagates_without_partial_results(
    backend: RemoteBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("batch failed")

    def embed_batch(texts: Sequence[str]) -> list[np.ndarray]:
        if "bad" in texts:
            raise error
        return [np.ones(2, dtype=np.float32) for _ in texts]

    monkeypatch.setattr(backend, "_embed_batch", embed_batch)
    with pytest.raises(RuntimeError) as caught:
        backend.embed(["ok", "ok", "bad"])
    assert caught.value is error


@pytest.mark.parametrize("status,attempts", [(429, 3), (400, 1)])
def test_provider_retry_exhaustion_and_permanent_errors(
    backend: RemoteBackend, monkeypatch: pytest.MonkeyPatch, status: int, attempts: int,
) -> None:
    class ClientError(Exception):
        status_code = status

    error = ClientError("request failed")
    request = Mock(side_effect=error)
    sleep = Mock()
    monkeypatch.setattr(gemini.genai_errors, "ClientError", ClientError)
    monkeypatch.setattr(openai, "_sleep", sleep)
    monkeypatch.setattr(gemini, "_sleep", sleep)
    backend._client = SimpleNamespace(
        embeddings=SimpleNamespace(create=request),
        models=SimpleNamespace(embed_content=request),
    )
    with pytest.raises(RuntimeError) as caught:
        backend.embed(["one"])
    assert caught.value.__cause__ is error
    assert request.call_count == attempts
    assert [args.args[0] for args in sleep.call_args_list] == ([0.5, 1.0] if status == 429 else [])


def test_gemini_keeps_non_client_errors_outside_retry_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = TimeoutError("timeout")
    request = Mock(side_effect=error)
    monkeypatch.setattr(gemini.genai, "Client", lambda **_kwargs: SimpleNamespace(
        models=SimpleNamespace(embed_content=request)
    ))
    backend = gemini.GeminiEmbeddingBackend(model_name="test", api_key="test")
    with pytest.raises(TimeoutError) as caught:
        backend.embed(["one"])
    assert caught.value is error
    request.assert_called_once()
