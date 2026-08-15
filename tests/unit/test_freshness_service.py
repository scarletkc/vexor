from __future__ import annotations

from pathlib import Path
from threading import Event, Lock, Thread
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


class LockingObserver(FakeObserver):
    """Model watchdog's dispatcher/schedule lock ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.internal_lock = Lock()
        self.schedule_attempted = Event()
        self.dispatch_ready = Event()
        self.dispatch_handler = Event()

    def schedule(self, handler, _path: str, *, recursive: bool) -> None:
        assert recursive is True
        self.schedule_attempted.set()
        with self.internal_lock:
            self.handlers.append(handler)

    def dispatch(self, path: Path) -> None:
        with self.internal_lock:
            self.dispatch_ready.set()
            assert self.dispatch_handler.wait(2)
            event = SimpleNamespace(
                event_type="modified",
                src_path=str(path),
                dest_path=None,
            )
            self.handlers[0].on_any_event(event)


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


def test_registering_second_root_does_not_deadlock_dispatcher(tmp_path: Path) -> None:
    observer = LockingObserver()
    tracker = FreshnessTracker(observer_factory=lambda: observer)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    assert tracker.begin_validation(first_root) == 0
    observer.schedule_attempted.clear()

    dispatcher = Thread(
        target=observer.dispatch,
        args=(first_root / "changed.py",),
        daemon=True,
    )
    dispatcher.start()
    assert observer.dispatch_ready.wait(2)

    registration_result: list[int | None] = []
    registration = Thread(
        target=lambda: registration_result.append(tracker.begin_validation(second_root)),
        daemon=True,
    )
    registration.start()
    assert observer.schedule_attempted.wait(2)
    observer.dispatch_handler.set()

    dispatcher.join(2)
    registration.join(2)
    assert dispatcher.is_alive() is False
    assert registration.is_alive() is False
    assert registration_result == [0]
    tracker.close()


def test_schedule_failure_falls_back_to_snapshot_scans(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FailingObserver(FakeObserver):
        def __init__(self) -> None:
            super().__init__()
            self.schedule_calls = 0

        def schedule(self, handler, _path: str, *, recursive: bool) -> None:
            self.schedule_calls += 1
            raise OSError(28, "inotify watch limit reached")

    observer = FailingObserver()
    tracker = FreshnessTracker(observer_factory=lambda: observer)
    request = _request(tmp_path, tracker)
    snapshot_calls = 0

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

    def is_cache_current(*_args, **_kwargs) -> bool:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return True

    monkeypatch.setattr(search_service, "is_cache_current", is_cache_current)
    for _ in range(2):
        state = search_service._load_filtered_index(
            request,
            None,
            load_index_vectors=load_index_vectors,
            list_cache_entries=lambda: [],
        )
        assert state.stale is False

    assert observer.schedule_calls == 1
    assert snapshot_calls == 2
    tracker.close()


def test_missing_watchdog_falls_back_without_creating_observer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observer_created = False

    def create_observer():
        nonlocal observer_created
        observer_created = True
        return FakeObserver()

    def missing_handler(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'watchdog'", name="watchdog")

    monkeypatch.setattr(
        "vexor.services.freshness_service._build_event_handler",
        missing_handler,
    )
    tracker = FreshnessTracker(observer_factory=create_observer)

    assert tracker.begin_validation(tmp_path) is None
    assert tracker.begin_validation(tmp_path) is None
    assert observer_created is False
    tracker.close()


def test_fresh_hint_expires_by_age_and_reuse_budget(tmp_path: Path) -> None:
    observer = FakeObserver()
    now = [10.0]
    tracker = FreshnessTracker(
        observer_factory=lambda: observer,
        max_reuses=1,
        max_age_seconds=5.0,
        clock=lambda: now[0],
    )
    token = tracker.begin_validation(tmp_path)
    assert token == 0
    assert tracker.finish_validation(tmp_path, "by-count", token) is True
    assert tracker.is_fresh(tmp_path, "by-count") is True
    assert tracker.is_fresh(tmp_path, "by-count") is False

    assert tracker.finish_validation(tmp_path, "by-age", token) is True
    now[0] += 5.0
    assert tracker.is_fresh(tmp_path, "by-age") is False
    tracker.close()


def test_validation_cache_is_bounded_and_dirty_event_clears_root(tmp_path: Path) -> None:
    observer = FakeObserver()
    tracker = FreshnessTracker(
        observer_factory=lambda: observer,
        max_validations=2,
    )
    token = tracker.begin_validation(tmp_path)
    assert token == 0
    for key in ("old", "middle", "new"):
        assert tracker.finish_validation(tmp_path, key, token) is True

    assert tracker.is_fresh(tmp_path, "old") is False
    assert tracker.is_fresh(tmp_path, "middle") is True
    assert tracker.is_fresh(tmp_path, "new") is True
    observer.emit("modified", tmp_path / "a.py")
    assert tracker.is_fresh(tmp_path, "middle") is False
    assert tracker.is_fresh(tmp_path, "new") is False
    tracker.close()
