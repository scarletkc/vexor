"""Shared SQLite connection and batching helpers.

Extracted from ``cache.py`` so the file index and the collection store open
databases with identical WAL, ``busy_timeout``, and foreign-key behavior. Two
stores writing under the same pragmas is the whole point: a divergence here
shows up as a lock error under concurrent writers, not as a test failure.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

# SQLite's default host parameter limit is 999; stay under it with room for the
# fixed parameters callers prepend to a batch.
MAX_SQL_PARAMS = 900


def _is_lock_contention(exc: sqlite3.OperationalError) -> bool:
    """True when *exc* means "someone else holds it", not "something is broken"."""

    message = str(exc).lower()
    return any(token in message for token in ("readonly", "locked", "busy"))


def _ensure_wal(conn: sqlite3.Connection) -> bool:
    """Try to put *conn* in WAL mode. Returns whether the database ended up there.

    Changing ``journal_mode`` needs a brief exclusive lock, and SQLite does not
    run the busy handler for it: a concurrent switch returns SQLITE_BUSY
    immediately no matter what ``busy_timeout`` says. Reading the current mode
    takes no lock, so the overwhelmingly common case — a database already in WAL
    — never contends at all.

    Failing to switch is deliberately tolerated rather than raised. WAL is a
    concurrency optimization, not a correctness requirement: a database left on
    the rollback journal still reads and writes correctly, it just serializes
    readers against writers. Note that an unsuccessful switch does *not* raise —
    SQLite reports the mode still in force — so the result is inspected rather
    than assumed. Errors that are not lock contention do propagate; a disk or
    corruption failure here is real and must not be deferred to some later,
    less obviously related query.
    """

    try:
        row = conn.execute("PRAGMA journal_mode;").fetchone()
    except sqlite3.OperationalError as exc:
        if not _is_lock_contention(exc):
            raise
        return False
    if row is not None and str(row[0]).lower() == "wal":
        return True
    try:
        result = conn.execute("PRAGMA journal_mode = WAL;").fetchone()
    except sqlite3.OperationalError as exc:
        if not _is_lock_contention(exc):
            raise
        return False
    return result is not None and str(result[0]).lower() == "wal"


def connect(
    db_path: Path,
    *,
    readonly: bool = False,
    query_only: bool = False,
) -> sqlite3.Connection:
    """Open *db_path* with Vexor's standard pragmas."""

    if readonly:
        db_uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # busy_timeout must come first. Switching journal_mode needs a brief
    # exclusive lock even when the database is already in WAL, so setting the
    # timeout afterwards leaves that one statement unprotected: concurrent
    # openers get SQLITE_BUSY immediately instead of waiting their turn.
    conn.execute("PRAGMA busy_timeout = 5000;")
    _ensure_wal(conn)
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA foreign_keys = ON;")
    if readonly or query_only:
        conn.execute("PRAGMA query_only = ON;")
    return conn


def chunk_values(
    values: Sequence[object],
    size: int = MAX_SQL_PARAMS,
) -> Iterable[Sequence[object]]:
    """Yield *values* in slices that fit SQLite's host parameter limit."""

    for idx in range(0, len(values), size):
        yield values[idx : idx + size]
