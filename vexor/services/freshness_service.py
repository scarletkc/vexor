"""Process-local filesystem freshness tracking for repeated searches."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Callable, Hashable


_MUTATION_EVENTS = {"created", "deleted", "modified", "moved"}


def _is_cache_event(root: Path, raw_path: str | bytes | None) -> bool:
    if not raw_path:
        return False
    try:
        relative = Path(raw_path).resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return bool(relative.parts) and relative.parts[0] == ".vexor"


def _build_event_handler(root: Path, callback: Callable[[], None]):
    from watchdog.events import FileSystemEventHandler

    class MutationHandler(FileSystemEventHandler):
        def on_any_event(self, event) -> None:
            if event.event_type not in _MUTATION_EVENTS:
                return
            paths = (getattr(event, "src_path", None), getattr(event, "dest_path", None))
            if all(_is_cache_event(root, path) for path in paths if path):
                return
            callback()

    return MutationHandler()


class FreshnessTracker:
    """Track whether a validated filesystem snapshot has received mutations."""

    def __init__(self, observer_factory=None) -> None:
        if observer_factory is None:
            from watchdog.observers import Observer

            observer_factory = Observer
        self._observer = observer_factory()
        self._lock = Lock()
        self._versions: dict[Path, int] = {}
        self._validated: dict[tuple[Path, Hashable], int] = {}
        self._started = False
        self._closed = False

    def _mark_dirty(self, root: Path) -> None:
        with self._lock:
            if self._closed:
                return
            self._versions[root] = self._versions.get(root, 0) + 1

    def _ensure_root(self, root: Path) -> Path:
        resolved = root.resolve()
        with self._lock:
            if self._closed:
                raise RuntimeError("Freshness tracker is closed")
            if resolved in self._versions:
                return resolved
            handler = _build_event_handler(
                resolved,
                lambda: self._mark_dirty(resolved),
            )
            self._observer.schedule(handler, str(resolved), recursive=True)
            self._versions[resolved] = 0
            if not self._started:
                self._observer.start()
                self._started = True
        return resolved

    def begin_validation(self, root: Path) -> int:
        resolved = self._ensure_root(root)
        with self._lock:
            return self._versions[resolved]

    def is_fresh(self, root: Path, key: Hashable) -> bool:
        resolved = self._ensure_root(root)
        with self._lock:
            return self._validated.get((resolved, key)) == self._versions[resolved]

    def finish_validation(self, root: Path, key: Hashable, token: int) -> bool:
        resolved = self._ensure_root(root)
        with self._lock:
            if self._versions[resolved] != token:
                return False
            self._validated[(resolved, key)] = token
            return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
        if started:
            self._observer.stop()
            self._observer.join()
