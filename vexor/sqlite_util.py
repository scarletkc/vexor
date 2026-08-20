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
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError as exc:
        if "readonly" not in str(exc).lower():
            raise
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA busy_timeout = 5000;")
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
