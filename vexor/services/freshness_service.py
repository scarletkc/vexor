"""Process-local filesystem freshness tracking for repeated searches."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from time import monotonic
from typing import Callable, Hashable


_MUTATION_EVENTS = {"created", "deleted", "modified", "moved"}
DEFAULT_MAX_REUSES = 32
DEFAULT_MAX_AGE_SECONDS = 5.0
DEFAULT_MAX_VALIDATIONS = 128


@dataclass(slots=True)
class _ValidationState:
    version: int
    validated_at: float
    reuse_count: int = 0


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


def _is_missing_watchdog(exc: ModuleNotFoundError) -> bool:
    return bool(
        exc.name
        and (exc.name == "watchdog" or exc.name.startswith("watchdog."))
    )


class _WatchingUnavailable(Exception):
    """Signal that this tracker cannot start a filesystem observer."""


class FreshnessTracker:
    """Use filesystem events as a bounded hint between full snapshot scans."""

    def __init__(
        self,
        observer_factory=None,
        *,
        max_reuses: int = DEFAULT_MAX_REUSES,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        max_validations: int = DEFAULT_MAX_VALIDATIONS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_reuses < 0:
            raise ValueError("max_reuses must be non-negative")
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        if max_validations <= 0:
            raise ValueError("max_validations must be positive")
        self._observer_factory = observer_factory
        self._observer = None
        self._observer_lock = Lock()
        self._lock = Lock()
        self._versions: dict[Path, int] = {}
        self._validated: OrderedDict[
            tuple[Path, Hashable], _ValidationState
        ] = OrderedDict()
        self._pending_roots: set[Path] = set()
        self._disabled_roots: set[Path] = set()
        self._started = False
        self._closed = False
        self._closed_event = Event()
        self._watching_unavailable = Event()
        self._max_reuses = max_reuses
        self._max_age_seconds = max_age_seconds
        self._max_validations = max_validations
        self._clock = clock

    def _mark_dirty(self, root: Path) -> None:
        with self._lock:
            if self._closed:
                return
            self._versions[root] = self._versions.get(root, 0) + 1
            stale_keys = [key for key in self._validated if key[0] == root]
            for key in stale_keys:
                del self._validated[key]

    def _disable_root(self, root: Path, *, watching_unavailable: bool = False) -> None:
        with self._lock:
            self._pending_roots.discard(root)
            self._disabled_roots.add(root)
            if watching_unavailable:
                self._watching_unavailable.set()

    def _new_observer(self):
        if self._observer_factory is None:
            try:
                from watchdog.observers import Observer
            except ModuleNotFoundError as exc:
                if _is_missing_watchdog(exc):
                    self._watching_unavailable.set()
                raise

            return Observer()
        return self._observer_factory()

    def _ensure_root(self, root: Path) -> Path | None:
        resolved = root.resolve()
        with self._lock:
            if self._closed:
                raise RuntimeError("Freshness tracker is closed")
            if (
                self._watching_unavailable.is_set()
                or resolved in self._disabled_roots
            ):
                return None
            if resolved in self._versions:
                return resolved
            if resolved in self._pending_roots:
                return None
            self._pending_roots.add(resolved)

        try:
            handler = _build_event_handler(
                resolved,
                lambda: self._mark_dirty(resolved),
            )
        except ModuleNotFoundError as exc:
            if not _is_missing_watchdog(exc):
                with self._lock:
                    self._pending_roots.discard(resolved)
                raise
            self._disable_root(resolved, watching_unavailable=True)
            return None

        watch = None
        start_failed = False
        try:
            # Never acquire the tracker lock while calling into watchdog. Its
            # dispatcher invokes handlers while holding watchdog's own lock.
            with self._observer_lock:
                if self._closed_event.is_set():
                    raise RuntimeError("Freshness tracker is closed")
                if self._watching_unavailable.is_set():
                    raise _WatchingUnavailable
                if self._observer is None:
                    self._observer = self._new_observer()
                watch = self._observer.schedule(handler, str(resolved), recursive=True)
                if not self._started:
                    try:
                        self._observer.start()
                    except OSError:
                        start_failed = True
                        self._watching_unavailable.set()
                        unschedule = getattr(self._observer, "unschedule", None)
                        if watch is not None and callable(unschedule):
                            try:
                                unschedule(watch)
                            except OSError:
                                pass
                        raise
                    self._started = True
        except _WatchingUnavailable:
            self._disable_root(resolved, watching_unavailable=True)
            return None
        except ModuleNotFoundError as exc:
            if not _is_missing_watchdog(exc):
                with self._lock:
                    self._pending_roots.discard(resolved)
                raise
            self._disable_root(resolved, watching_unavailable=True)
            return None
        except OSError:
            self._disable_root(resolved, watching_unavailable=start_failed)
            return None
        except Exception:
            with self._lock:
                self._pending_roots.discard(resolved)
            raise

        with self._lock:
            self._pending_roots.discard(resolved)
            if self._closed:
                raise RuntimeError("Freshness tracker is closed")
            self._versions.setdefault(resolved, 0)
        return resolved

    def begin_validation(self, root: Path) -> int | None:
        resolved = self._ensure_root(root)
        if resolved is None:
            return None
        with self._lock:
            return self._versions[resolved]

    def is_fresh(self, root: Path, key: Hashable) -> bool:
        resolved = self._ensure_root(root)
        if resolved is None:
            return False
        validation_key = (resolved, key)
        with self._lock:
            state = self._validated.get(validation_key)
            if state is None or state.version != self._versions[resolved]:
                return False
            expired = self._clock() - state.validated_at >= self._max_age_seconds
            exhausted = state.reuse_count >= self._max_reuses
            if expired or exhausted:
                del self._validated[validation_key]
                return False
            state.reuse_count += 1
            self._validated.move_to_end(validation_key)
            return True

    def finish_validation(self, root: Path, key: Hashable, token: int) -> bool:
        resolved = self._ensure_root(root)
        if resolved is None:
            return False
        with self._lock:
            if self._versions[resolved] != token:
                return False
            validation_key = (resolved, key)
            self._validated[validation_key] = _ValidationState(
                version=token,
                validated_at=self._clock(),
            )
            self._validated.move_to_end(validation_key)
            while len(self._validated) > self._max_validations:
                self._validated.popitem(last=False)
            return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._closed_event.set()
        with self._observer_lock:
            if self._started:
                self._observer.stop()
                self._observer.join()
