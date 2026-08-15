"""Measure Vexor index-loading and freshness-check costs locally."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from vexor.cache import (
    IndexedChunk,
    IndexVectorCache,
    cache_dir_context,
    compare_snapshot,
    load_index_vectors,
    store_index,
)
from vexor.services.index_service import _snapshot_current_files
from vexor.services.freshness_service import FreshnessTracker
from vexor.utils import collect_files


def _milliseconds(samples: list[float]) -> str:
    return f"min={min(samples) * 1000:.2f} median={median(samples) * 1000:.2f}"


def _measure(repeats: int, operation) -> tuple[list[float], object]:
    samples: list[float] = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = operation()
        samples.append(perf_counter() - started)
    return samples, result


def _build_files(root: Path, count: int) -> None:
    directory_count = max(min(count // 100, 100), 1)
    per_directory = (count + directory_count - 1) // directory_count
    created = 0
    for directory_index in range(directory_count):
        directory = root / f"d{directory_index:03d}"
        directory.mkdir()
        for file_index in range(per_directory):
            if created >= count:
                return
            (directory / f"f{file_index:05d}.py").touch()
            created += 1


def run_benchmark(
    *,
    vector_count: int,
    dimension: int,
    file_count: int,
    repeats: int,
) -> None:
    rng = np.random.default_rng(7)
    with TemporaryDirectory(prefix="vexor-search-benchmark-") as temporary:
        base = Path(temporary)
        project = base / "project"
        project.mkdir()
        cache_dir = base / "cache"
        cache_dir.mkdir()
        vector = rng.standard_normal(dimension, dtype=np.float32)
        entries = [
            IndexedChunk(
                path=project / f"vector-{index}.py",
                rel_path=f"vector-{index}.py",
                chunk_index=0,
                preview="",
                embedding=vector,
                size_bytes=1,
                mtime=1.0,
            )
            for index in range(vector_count)
        ]
        memory_cache = IndexVectorCache()
        with cache_dir_context(cache_dir):
            started = perf_counter()
            database = store_index(
                root=project,
                model="benchmark",
                include_hidden=False,
                respect_gitignore=False,
                mode="full",
                recursive=True,
                entries=entries,
            )
            store_seconds = perf_counter() - started
            first_samples, loaded = _measure(
                1,
                lambda: load_index_vectors(
                    project,
                    "benchmark",
                    False,
                    "full",
                    True,
                    respect_gitignore=False,
                    memory_cache=memory_cache,
                ),
            )
            repeated_samples, loaded = _measure(
                repeats,
                lambda: load_index_vectors(
                    project,
                    "benchmark",
                    False,
                    "full",
                    True,
                    respect_gitignore=False,
                    memory_cache=memory_cache,
                ),
            )
            vectors = loaded[1]
            vector_file = cache_dir / loaded[2]["vector_file"]
            print(
                f"vectors rows={vector_count} dimension={dimension} "
                f"database_mb={database.stat().st_size / 1024 / 1024:.1f} "
                f"sidecar_mb={vector_file.stat().st_size / 1024 / 1024:.1f}"
            )
            print(f"store seconds={store_seconds:.3f}")
            print(f"first_load_ms {_milliseconds(first_samples)}")
            print(f"cached_load_ms {_milliseconds(repeated_samples)}")
            del vectors
            del loaded
            memory_cache.clear()
            gc.collect()

        scan_root = base / "scan"
        scan_root.mkdir()
        _build_files(scan_root, file_count)
        files = collect_files(scan_root, recursive=True, respect_gitignore=False)
        snapshot = _snapshot_current_files(files, scan_root)
        cached_files = [
            {"path": relative, "mtime": entry.mtime, "size": entry.size}
            for relative, entry in snapshot.items()
        ]
        snapshot_samples, current = _measure(
            repeats,
            lambda: compare_snapshot(
                scan_root,
                False,
                cached_files,
                recursive=True,
                respect_gitignore=False,
            ),
        )
        tracker = FreshnessTracker()
        token = tracker.begin_validation(scan_root)
        tracker.finish_validation(scan_root, "benchmark", token)
        event_samples, fresh = _measure(
            repeats,
            lambda: tracker.is_fresh(scan_root, "benchmark"),
        )
        tracker.close()
        print(f"snapshot files={file_count} current={current} {_milliseconds(snapshot_samples)}")
        print(f"event_check fresh={fresh} {_milliseconds(event_samples)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-count", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--file-count", type=int, default=2_000)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    for name in ("vector_count", "dimension", "file_count", "repeats"):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    run_benchmark(
        vector_count=arguments.vector_count,
        dimension=arguments.dimension,
        file_count=arguments.file_count,
        repeats=arguments.repeats,
    )


if __name__ == "__main__":
    main()
