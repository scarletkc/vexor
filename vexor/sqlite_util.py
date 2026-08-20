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


def _ensure_wal(conn: sqlite3.Connection) -> None:
    """Put *conn* in WAL mode without fighting other connections over the switch.

    Changing ``journal_mode`` needs a brief exclusive lock, and SQLite does not
    run the busy handler for it: a concurrent switch returns SQLITE_BUSY
    immediately no matter what ``busy_timeout`` says. Reading the current mode
    takes no lock, so the overwhelmingly common case — a database already in WAL
    — never contends at all.

    Losing the race on a freshly created database is deliberately tolerated.
    WAL is a concurrency optimization, not a correctness requirement; whichever
    connection won the switch leaves the database in WAL for everyone after it,
    and a connection that stayed on the rollback journal still reads and writes
    correctly. Any other operational error still propagates.
    """

    try:
        row = conn.execute("PRAGMA journal_mode;").fetchone()
    except sqlite3.OperationalError:
        return
    if row is not None and str(row[0]).lower() == "wal":
        return
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if not any(token in message for token in ("readonly", "locked", "busy")):
            raise


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
