from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vexor.services import search_service
from vexor.services.freshness_service import FreshnessTracker
from vexor.services.search_service import SearchRequest


class FakeObserver:
    def __init__(self) -> None:
        self.handlers: list[object] = []
        self.started = False
        self.stopped = False
        self.joined = False

    def schedule(self, handler, _path: str, *, recursive: bool) -> None:
        assert recursive is True
        self.handlers.append(handler)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self) -> None:
        self.joined = True

    def emit(
        self,
        event_type: str,
        path: Path,
        *,
        destination: Path | None = None,
    ) -> None:
        event = SimpleNamespace(
            event_type=event_type,
            src_path=str(path),
            dest_path=str(destination) if destination is not None else None,
        )
        for handler in self.handlers:
            handler.on_any_event(event)


def _request(tmp_path: Path, tracker: FreshnessTracker) -> SearchRequest:
    return SearchRequest(
        query="alpha",
        directory=tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        top_k=2,
        model_name="model",
        batch_size=0,
        provider="openai",
        base_url=None,
        api_key="key",
        local_cuda=False,
        exclude_patterns=(),
        extensions=(),
        auto_index=False,
        freshness_tracker=tracker,
    )


def test_tracker_invalidates_only_on_source_mutations(tmp_path: Path) -> None:
    observer = FakeObserver()
    tracker = FreshnessTracker(observer_factory=lambda: observer)
    key = ("index", 1)

    token = tracker.begin_validation(tmp_path)
    assert tracker.finish_validation(tmp_path, key, token) is True
    assert tracker.is_fresh(tmp_path, key) is True

    observer.emit("opened", tmp_path / "a.py")
    observer.emit("modified", tmp_path / ".vexor" / "index.db")
    assert tracker.is_fresh(tmp_path, key) is True

    observer.emit("modified", tmp_path / "a.py")
    assert tracker.is_fresh(tmp_path, key) is False

    tracker.close()
    tracker.close()
    assert observer.started is True
    assert observer.stopped is True
    assert observer.joined is True
    with pytest.raises(RuntimeError, match="closed"):
        tracker.begin_validation(tmp_path)


def test_repeated_index_load_skips_snapshot_scan_until_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observer = FakeObserver()
    tracker = FreshnessTracker(observer_factory=lambda: observer)
    request = _request(tmp_path, tracker)
    calls = {"snapshots": 0}

    def load_index_vectors(*_args, **_kwargs):
        paths = [tmp_path / "a.py"]
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        metadata = {
            "index_id": 1,
            "generated_at": "generation-1",
            "vector_file": "vectors/one.npy",
            "files": [
                {
                    "path": "a.py",
                    "absolute": str(paths[0]),
                    "mtime": 1.0,
                    "size": 1,
                }
            ],
            "chunks": [],
            "chunk_ids": [1],
        }
        return paths, vectors, metadata

    def is_cache_current(*_args, **_kwargs) -> bool:
        calls["snapshots"] += 1
        return True

    monkeypatch.setattr(search_service, "is_cache_current", is_cache_current)

    first = search_service._load_filtered_index(
        request,
        None,
        load_index_vectors=load_index_vectors,
        list_cache_entries=lambda: [],
    )
    second = search_service._load_filtered_index(
        request,
        None,
        load_index_vectors=load_index_vectors,
        list_cache_entries=lambda: [],
    )
    assert first.stale is False
    assert second.stale is False
    assert calls["snapshots"] == 1

    observer.emit("modified", tmp_path / "a.py")
    third = search_service._load_filtered_index(
        request,
        None,
        load_index_vectors=load_index_vectors,
        list_cache_entries=lambda: [],
    )
    assert third.stale is False
    assert calls["snapshots"] == 2
    tracker.close()


def test_event_during_snapshot_validation_marks_index_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observer = FakeObserver()
    tracker = FreshnessTracker(observer_factory=lambda: observer)
    request = _request(tmp_path, tracker)

    def load_index_vectors(*_args, **_kwargs):
        return (
            [tmp_path / "a.py"],
            np.array([[1.0, 0.0]], dtype=np.float32),
            {
                "index_id": 1,
                "generated_at": "generation-1",
                "vector_file": "vectors/one.npy",
                "files": [{"path": "a.py", "mtime": 1.0, "size": 1}],
                "chunks": [],
                "chunk_ids": [1],
            },
        )

    def mutate_during_validation(*_args, **_kwargs) -> bool:
        observer.emit("modified", tmp_path / "a.py")
        return True

    monkeypatch.setattr(
        search_service,
        "is_cache_current",
        mutate_during_validation,
    )
    state = search_service._load_filtered_index(
        request,
        None,
        load_index_vectors=load_index_vectors,
        list_cache_entries=lambda: [],
    )
    assert state.stale is True
    tracker.close()
