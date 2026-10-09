"""Exercise sidecar ownership with separate processes sharing one SQLite cache."""

from __future__ import annotations

import multiprocessing
import sqlite3
from pathlib import Path

import pytest

from vexor import cache
from vexor.services.cache_service import load_index_metadata_safe


def _options(root: Path) -> dict:
    """Use identical index settings across writer and reader processes."""
    return {
        "root": root,
        "model": "model",
        "include_hidden": False,
        "mode": "name",
        "recursive": True,
    }


def _entry(root: Path, value: float) -> cache.IndexedChunk:
    """Create one vector row whose value distinguishes index generations."""
    return cache.IndexedChunk(root / "a.txt", "a.txt", 0, "text", [value, 1.0])


def _replace_generation(root: Path, update: str, value: float = 2.0) -> None:
    """Commit a real replacement and release Windows mmaps before pruning its old file."""
    if update == "full":
        cache.store_index(**_options(root), entries=[_entry(root, value)])
    elif update == "incremental":
        cache.apply_index_updates(
            **_options(root),
            ordered_entries=[("a.txt", 0), ("b.txt", 0)],
            changed_entries=[
                _entry(root, value),
                cache.IndexedChunk(root / "b.txt", "b.txt", 0, "text", [value + 1.0, 1.0]),
            ],
            removed_rel_paths=[],
        )
        cache._prune_unreferenced_vector_files(cache.cache_db_path())
    else:
        assert cache.clear_index(root, False, "name", True, model="model") == 1


def _write_paused(cache_dir, root, incremental, stage, paused, release):
    """Hold a writer at a sidecar publication stage until the parent releases it."""
    with cache.cache_dir_context(cache_dir):
        if stage == "temporary":
            original = cache.np.lib.format.open_memmap

            def pause(*args, **kwargs):
                """Pause after the temporary mmap exists but before its rows are written."""
                result = original(*args, **kwargs)
                if kwargs.get("mode") == "w+":
                    paused.set()
                    if not release.wait(10):
                        raise TimeoutError("Writer was not released")
                return result

            cache.np.lib.format.open_memmap = pause
        else:
            original = cache._write_vector_file

            def pause(*args, **kwargs):
                """Pause after the final sidecar exists but before metadata is committed."""
                result = original(*args, **kwargs)
                paused.set()
                if not release.wait(10):
                    raise TimeoutError("Writer was not released")
                return result

            cache._write_vector_file = pause
        if incremental:
            cache.apply_index_updates(
                **_options(root),
                ordered_entries=[("a.txt", 0)],
                changed_entries=[_entry(root, 2.0)],
                removed_rel_paths=[],
            )
        else:
            cache.store_index(**_options(root), entries=[_entry(root, 2.0)])


def _read_and_prune(cache_dir, root, started, finished):
    """Load a peer index while its memory-cache path attempts orphan cleanup."""
    with cache.cache_dir_context(cache_dir):
        started.set()
        cache.load_index_vectors(
            root, "model", False, "name", True, memory_cache=cache.IndexVectorCache()
        )
        finished.set()


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("stage", ["temporary", "final"])
def test_pruning_preserves_other_process_pending_generation(tmp_path, incremental, stage):
    """Force cleanup during pending publication and verify both committed generations."""
    cache_dir = tmp_path / "shared-cache"
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()
        (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(cache_dir):
        for root in roots:
            cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
    orphan = cache_dir / "vectors" / "orphan.npy"
    orphan.write_bytes(b"abandoned")
    crashed_temporary = cache_dir / "vectors" / ".crashed.tmp.npy"
    crashed_temporary.write_bytes(b"abandoned temporary file")
    ctx = multiprocessing.get_context("spawn")
    paused, release, started, finished = [ctx.Event() for _ in range(4)]
    writer = ctx.Process(
        target=_write_paused, args=(cache_dir, roots[0], incremental, stage, paused, release)
    )
    pruner = ctx.Process(target=_read_and_prune, args=(cache_dir, roots[1], started, finished))
    writer.start()
    try:
        assert paused.wait(10), "Writer did not reach pending sidecar"
        pruner.start()
        assert started.wait(10), "Pruner did not start"
        assert finished.wait(10), "Healthy reads must succeed while cleanup is deferred"
        pending = (
            list((cache_dir / "vectors").glob(".*.tmp.npy"))
            if stage == "temporary"
            else list((cache_dir / "vectors").glob("*.npy"))
        )
        assert len(pending) == (2 if stage == "temporary" else 5)
        assert orphan.exists(), "Cleanup must defer while a pending sidecar is uncommitted"
        assert crashed_temporary.exists()
    finally:
        release.set()
        writer.join(10)
        if pruner.pid is not None:
            pruner.join(10)
        for process in (writer, pruner):
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(5)
    assert writer.exitcode == pruner.exitcode == 0
    with cache.cache_dir_context(cache_dir):
        for root, expected in zip(roots, [2.0, 1.0], strict=True):
            _, vectors, _ = cache.load_index_vectors(root, "model", False, "name", True)
            assert vectors.tolist() == [[expected, 1.0]]
            del vectors
    assert not orphan.exists()
    assert not crashed_temporary.exists()
    assert not list((cache_dir / "vectors").glob(".*.tmp.npy"))


def test_healthy_memory_cache_read_defers_cleanup_under_write_lock(tmp_path):
    """Keep reads usable under contention and clean abandoned files after lock release."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        orphan = tmp_path / "cache" / "vectors" / "orphan.npy"
        orphan.write_bytes(b"abandoned")
        blocker = cache._connect(cache.cache_db_path())
        try:
            blocker.execute("BEGIN IMMEDIATE;")
            memory_cache = cache.IndexVectorCache()
            _, vectors, _ = cache.load_index_vectors(
                root, "model", False, "name", True, memory_cache=memory_cache
            )
            assert vectors.tolist() == [[1.0, 1.0]]
            assert orphan.exists()
        finally:
            blocker.rollback()
            blocker.close()
        memory_cache.prune()
        assert not orphan.exists()


def test_readonly_database_can_load_into_memory_cache(tmp_path, monkeypatch):
    """Allow valid reads when opportunistic cleanup cannot obtain write permission."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        connect = cache._connect
        monkeypatch.setattr(cache, "_connect", lambda path, **kwargs: connect(path, readonly=True))
        _, vectors, _ = cache.load_index_vectors(
            root, "model", False, "name", True, memory_cache=cache.IndexVectorCache()
        )
        assert vectors.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("incremental", [False, True])
def test_writer_lock_timeout_preserves_existing_index(
    tmp_path, monkeypatch, incremental,
):
    """Fail a competing write without replacing the previously committed vectors."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        connect = cache._connect

        def no_wait(path, **kwargs):
            """Make contention fail immediately while retaining the real SQLite connection."""
            conn = connect(path, **kwargs)
            conn.execute("PRAGMA busy_timeout = 0;")
            return conn

        monkeypatch.setattr(cache, "_connect", no_wait)
        blocker = connect(cache.cache_db_path())
        blocker.execute("BEGIN IMMEDIATE;")
        try:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                if incremental:
                    cache.apply_index_updates(
                        **_options(root), ordered_entries=[("a.txt", 0)],
                        changed_entries=[_entry(root, 2.0)], removed_rel_paths=[],
                    )
                else:
                    cache.store_index(**_options(root), entries=[_entry(root, 2.0)])
        finally:
            blocker.rollback()
            blocker.close()
        _, vectors, _ = cache.load_index_vectors(root, "model", False, "name", True)
        assert vectors.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("command", ["index", "search", "json", "clear"])
def test_cli_reports_cache_write_timeout_without_traceback(tmp_path, monkeypatch, command):
    """Report real lock contention with retry guidance and preserve JSON stdout."""
    from typer.testing import CliRunner

    from vexor import cli, config

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    monkeypatch.setenv("VEXOR_CONFIG_JSON", '{"provider":"local","model":"model"}')
    with (
        cache.cache_dir_context(tmp_path / "cache"),
        config.config_dir_context(tmp_path / "config"),
    ):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        connect = cache._connect

        def no_wait(path, **kwargs):
            """Make contention fail immediately while retaining the real SQLite connection."""
            conn = connect(path, **kwargs)
            conn.execute("PRAGMA busy_timeout = 0;")
            return conn

        def write_index(*args, **kwargs):
            """Exercise a real cache write from the stubbed CLI service entry point."""
            cache.store_index(**_options(root), entries=[_entry(root, 2.0)])

        monkeypatch.setattr(cache, "_connect", no_wait)
        monkeypatch.setattr(cli, "build_index", write_index)
        monkeypatch.setattr(cli, "perform_search", write_index)
        blocker = connect(cache.cache_db_path())
        blocker.execute("BEGIN IMMEDIATE;")
        try:
            args = ["index"] if command in {"index", "clear"} else ["search", "query"]
            args.extend(["--path", str(root), "--mode", "name"])
            if command == "json":
                args.append("--format=json")
            if command == "clear":
                args.append("--clear")
            result = CliRunner().invoke(cli.app, args)
        finally:
            blocker.rollback()
            blocker.close()
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "retry" in result.output
        assert "Traceback" not in result.output
        if command == "json":
            assert result.stdout == ""
            assert "retry" in result.stderr


def test_cli_timeout_from_embedding_cache_has_retry_hint(tmp_path, monkeypatch):
    """Handle contention before index publication in the real build orchestration."""
    import importlib

    import numpy as np
    from typer.testing import CliRunner

    from vexor import cli, config

    class OfflineSearcher:
        def __init__(self, **kwargs):
            """Accept production searcher options without loading a model."""
            pass

        def embed_texts(self, labels):
            """Supply deterministic vectors while keeping embedding-cache writes real."""
            return np.ones((len(labels), 2), dtype=np.float32)

    root, peer = tmp_path / "new-project", tmp_path / "peer"
    for project in (root, peer):
        project.mkdir()
        (project / "a.txt").write_text("text", encoding="utf-8")
    monkeypatch.setenv(
        "VEXOR_CONFIG_JSON", '{"provider":"local","model":"model","update_check":false}',
    )
    monkeypatch.setattr(importlib.import_module("vexor.search"), "VexorSearcher", OfflineSearcher)
    with (
        cache.cache_dir_context(tmp_path / "cache"),
        config.config_dir_context(tmp_path / "config"),
    ):
        cache.store_index(**_options(peer), entries=[_entry(peer, 1.0)])
        connect = cache._connect

        def no_wait(path, **kwargs):
            """Make contention fail immediately while retaining the real SQLite connection."""
            conn = connect(path, **kwargs)
            conn.execute("PRAGMA busy_timeout = 0;")
            return conn

        monkeypatch.setattr(cache, "_connect", no_wait)
        blocker = connect(cache.cache_db_path())
        blocker.execute("BEGIN IMMEDIATE;")
        try:
            result = CliRunner().invoke(cli.app, ["index", "--path", str(root), "--mode", "name"])
        finally:
            blocker.rollback()
            blocker.close()
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "retry" in result.stdout
        _, vectors, _ = cache.load_index_vectors(peer, "model", False, "name", True)
        assert vectors.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("command", ["index", "search", "json"])
def test_cli_does_not_relabel_other_sqlite_errors_as_busy(tmp_path, monkeypatch, command):
    """Propagate an invalid SQL query instead of suggesting a misleading busy retry."""
    from contextlib import closing

    from typer.testing import CliRunner

    from vexor import cli

    monkeypatch.setenv("VEXOR_CONFIG_JSON", '{"provider":"local","model":"model"}')

    def invalid_query(*args, **kwargs):
        """Raise a real SQLite error unrelated to writer contention."""
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("SELECT nonexistent_column")

    monkeypatch.setattr(cli, "build_index", invalid_query)
    monkeypatch.setattr(cli, "perform_search", invalid_query)
    args = ["index"] if command == "index" else ["search", "query"]
    args.extend(["--path", str(tmp_path), "--mode", "name"])
    if command == "json":
        args.append("--format=json")
    result = CliRunner().invoke(cli.app, args)
    assert isinstance(result.exception, sqlite3.OperationalError)
    assert "no such column" in str(result.exception)
    assert "retry" not in result.output


def test_metadata_read_handles_index_cleared_mid_read(tmp_path, monkeypatch):
    """Report an absent index when another connection clears it during metadata loading."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        connect = cache._connect
        cleared = False

        class ClearAfterFetch:
            """Release the metadata cursor before clearing through another connection."""

            def __init__(self, cursor):
                """Keep the real SQLite cursor for the scheduled metadata read."""
                self.cursor = cursor

            def fetchone(self):
                """Return a real row after another connection commits index clearing."""
                nonlocal cleared
                row = self.cursor.fetchone()
                assert cache.clear_index(root, False, "name", True, model="model") == 1
                cleared = True
                return row

        class InterleavedReader:
            """Schedule clearing at the first metadata read using a real connection."""

            def __init__(self, conn):
                """Retain the read-only SQLite connection used by production loading."""
                self.conn = conn

            def execute(self, query, *args):
                """Intercept metadata fetching while leaving all SQL execution real."""
                cursor = self.conn.execute(query, *args)
                if "FROM index_metadata" in query and "cache_key = ?" in query:
                    return ClearAfterFetch(cursor)
                return cursor

            def close(self):
                """Close the underlying reader when production loading exits."""
                self.conn.close()

        def interleaved_connect(path, **kwargs):
            """Wrap only readers so index clearing uses an ordinary writer connection."""
            conn = connect(path, **kwargs)
            return InterleavedReader(conn) if kwargs.get("readonly") else conn

        monkeypatch.setattr(cache, "_connect", interleaved_connect)
        with pytest.raises(FileNotFoundError) as error:
            cache.load_index(root, "model", False, "name", True)
        assert cleared
        assert error.value.args == (cache.cache_db_path(),)


@pytest.mark.parametrize("empty", [False, True])
def test_committed_missing_sidecar_is_damage_not_missing_index(tmp_path, empty):
    """Distinguish damaged committed generations from genuinely absent indexes."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[] if empty else [_entry(root, 1.0)])
        sidecar = next((tmp_path / "cache" / "vectors").glob("*.npy"))
        sidecar.unlink()
        for load in (
            lambda: cache.load_index_vectors(root, "model", False, "name", True),
            lambda: load_index_metadata_safe(root, "model", False, True, "name", True),
        ):
            with pytest.raises(RuntimeError, match="vexor index --clear") as error:
                load()
            assert str(sidecar) in str(error.value)


@pytest.mark.parametrize("reader", ["load_index", "load_index_vectors"])
@pytest.mark.parametrize("update", ["full", "incremental", "clear"])
def test_replaced_sidecar_is_not_reported_as_damage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str, update: str,
) -> None:
    """Replace a real generation before file checking and distinguish it from damage."""
    root = tmp_path / "project"
    root.mkdir()
    for name in ("a.txt", "b.txt"):
        (root / name).write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        require = cache._require_vector_file
        updated = False

        def update_before_check(db_path: Path, stored_path: str, **kwargs) -> Path:
            """Perform one committed rewrite before the reader checks its old sidecar."""
            nonlocal updated
            if not updated:
                updated = True
                _replace_generation(root, update)
            return require(db_path, stored_path, **kwargs)

        monkeypatch.setattr(cache, "_require_vector_file", update_before_check)
        load = getattr(cache, reader)
        if update == "incremental":
            loaded = load(root, "model", False, "name", True)
            if reader == "load_index":
                assert [chunk["path"] for chunk in loaded["chunks"]] == ["a.txt", "b.txt"]
            else:
                paths, vectors, metadata = loaded
                assert paths == [root / "a.txt", root / "b.txt"]
                assert vectors.tolist() == [[2.0, 1.0], [3.0, 1.0]]
                assert len(metadata["chunk_ids"]) == 2
        else:
            with pytest.raises(FileNotFoundError):
                load(root, "model", False, "name", True)
        assert updated
        if update != "clear":
            _, current, _ = cache.load_index_vectors(root, "model", False, "name", True)
            assert current.tolist() == (
                [[2.0, 1.0], [3.0, 1.0]] if update == "incremental" else [[2.0, 1.0]]
            )


@pytest.mark.parametrize("update", ["full", "incremental", "clear"])
def test_replacement_between_file_check_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, update: str,
) -> None:
    """Let pruning remove the checked file immediately before NumPy opens it."""
    root = tmp_path / "project"
    root.mkdir()
    for name in ("a.txt", "b.txt"):
        (root / name).write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        load = cache.np.load
        updated = False

        def update_before_open(*args, **kwargs):
            """Open actual arrays after one writer removes the reader's observed file."""
            nonlocal updated
            if not updated:
                updated = True
                _replace_generation(root, update)
            return load(*args, **kwargs)

        monkeypatch.setattr(cache.np, "load", update_before_open)
        if update == "incremental":
            _, vectors, metadata = cache.load_index_vectors(root, "model", False, "name", True)
            assert vectors.tolist() == [[2.0, 1.0], [3.0, 1.0]]
            assert len(metadata["chunk_ids"]) == 2
        else:
            with pytest.raises(FileNotFoundError):
                cache.load_index_vectors(root, "model", False, "name", True)
        assert updated


@pytest.mark.parametrize("reader", ["load_index", "load_index_vectors"])
def test_replacement_retry_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str,
) -> None:
    """Stop after one complete retry while retaining the writer's healthy generation."""
    root = tmp_path / "project"
    root.mkdir()
    for name in ("a.txt", "b.txt"):
        (root / name).write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        require = cache._require_vector_file
        attempts = 0

        def replace_each_read(db_path: Path, stored_path: str, **kwargs) -> Path:
            """Replace only reader generations; leave the updater's locked read alone."""
            nonlocal attempts
            if kwargs.get("index_id") is not None:
                attempts += 1
                assert attempts <= 2
                _replace_generation(root, "incremental", float(attempts + 1))
            return require(db_path, stored_path, **kwargs)

        monkeypatch.setattr(cache, "_require_vector_file", replace_each_read)
        with pytest.raises(FileNotFoundError) as error:
            getattr(cache, reader)(root, "model", False, "name", True)
        assert type(error.value) is FileNotFoundError
        assert attempts == 2
        monkeypatch.setattr(cache, "_require_vector_file", require)
        _, current, _ = cache.load_index_vectors(root, "model", False, "name", True)
        assert current.tolist() == [[3.0, 1.0], [4.0, 1.0]]


def test_invalid_sidecar_is_not_retried_as_replacement(tmp_path: Path) -> None:
    """Preserve the invalid-array error when the committed file exists but is corrupt."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        sidecar = next((tmp_path / "cache" / "vectors").glob("*.npy"))
        sidecar.write_bytes(b"not a numpy array")
        with pytest.raises(RuntimeError) as error:
            cache.load_index_vectors(root, "model", False, "name", True)
        assert str(sidecar) in str(error.value)
        assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("command", ["index", "show", "search", "json"])
def test_cli_reports_committed_sidecar_damage_and_can_clear(tmp_path, monkeypatch, command):
    """Expose damaged-index recovery for each output mode and permit explicit clearing."""
    from typer.testing import CliRunner

    from vexor import cli, config

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    monkeypatch.setenv("VEXOR_CONFIG_JSON", '{"provider":"local","model":"model"}')
    with (
        cache.cache_dir_context(tmp_path / "cache"),
        config.config_dir_context(tmp_path / "config"),
    ):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        next((tmp_path / "cache" / "vectors").glob("*.npy")).unlink()
        args = ["search", "query"] if command in {"search", "json"} else ["index"]
        args.extend(["--path", str(root), "--mode", "name"])
        if command == "show":
            args.append("--show")
        if command == "json":
            args.append("--format=json")
        result = CliRunner().invoke(cli.app, args)
        assert result.exit_code == 1, result.output
        assert "Damaged index cache" in result.output
        assert "vexor index --clear" in " ".join(result.output.split())
        if command == "json":
            assert result.stdout == ""
            assert "Damaged index cache" in result.stderr
        result = CliRunner().invoke(
            cli.app, ["index", "--path", str(root), "--mode", "name", "--clear"]
        )
        assert result.exit_code == 0, result.output
        assert load_index_metadata_safe(root, "model", False, True, "name", True) is None
