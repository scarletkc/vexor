"""SQLite storage for filesystem-independent text collections.

Collections live in their own ``collections.db`` next to ``index.db`` rather than
in new tables inside it. A file index is rebuildable from disk at any time; the
records in a collection are only recoverable by making the caller re-upsert
everything and pay for embeddings a second time, so the two must not share a
lifecycle. For the same reason this module never bumps ``CACHE_VERSION``: every
table below is additive under ``CREATE TABLE IF NOT EXISTS`` and cannot
invalidate an existing user's file index.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .cache import ensure_cache_dir
from .sqlite_util import chunk_values, connect
from .text import Messages

COLLECTIONS_DB_FILENAME = "collections.db"
COLLECTION_SCHEMA_VERSION = 1

# Metadata values are restricted to flat scalars. Nested values are rejected at
# write time rather than stored, because a value that can never be filtered on
# surfaces later as a phantom recall bug instead of an error.
ScalarValue = str | int | float | bool | datetime | None

FILTER_OPERATORS: tuple[str, ...] = (
    "eq",
    "ne",
    "in",
    "nin",
    "gt",
    "gte",
    "lt",
    "lte",
    "exists",
)
_SET_OPERATORS = frozenset({"in", "nin"})
_RANGE_OPERATORS = frozenset({"gt", "gte", "lt", "lte"})
_RANGE_SQL = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
# ``ne``/``nin`` match records that lack the key entirely. Requiring the key to
# exist would silently drop unlabeled records from a negative filter, which is
# the phantom recall failure the metadata contract exists to prevent.
_NEGATIVE_OPERATORS = frozenset({"ne", "nin"})


class CollectionError(ValueError):
    """Raised for collection contract violations (schema, metadata, filters)."""


@dataclass(slots=True)
class CollectionInfo:
    """Describes one collection's pinned embedding contract."""

    id: int
    name: str
    provider: str
    model: str
    dimension: int
    schema_version: int
    created_at: str


@dataclass(slots=True)
class StoredRecord:
    """A record as persisted, without any relevance score."""

    id: str
    text: str
    metadata: dict[str, ScalarValue]


@dataclass(slots=True)
class PreparedRecord:
    """A record with everything the store needs, computed by the service layer."""

    record_key: str
    text: str
    text_hash: str
    metadata: dict[str, ScalarValue]
    vector: np.ndarray | None = None
    bm25_terms: dict[str, int] = field(default_factory=dict)
    token_count: int = 0
    # ``False`` when the text is unchanged and only metadata is being rewritten.
    # Metadata always replaces wholesale (upsert replaces the record), but the
    # embedding and postings behind an unchanged text must survive, or a pure
    # metadata edit would silently drop the record out of lexical search.
    refresh_embedding: bool = True


def collections_db_path() -> Path:
    """Return the absolute path to the collection database."""

    return ensure_cache_dir() / COLLECTIONS_DB_FILENAME


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS collection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            schema_version INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS collection_record (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_id INTEGER NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
            record_key TEXT NOT NULL,
            text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(collection_id, record_key)
        );

        CREATE TABLE IF NOT EXISTS collection_embedding (
            record_id INTEGER PRIMARY KEY
                REFERENCES collection_record(id) ON DELETE CASCADE,
            vector_blob BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS collection_bm25_doc (
            record_id INTEGER PRIMARY KEY
                REFERENCES collection_record(id) ON DELETE CASCADE,
            token_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS collection_bm25_posting (
            collection_id INTEGER NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
            record_id INTEGER NOT NULL
                REFERENCES collection_record(id) ON DELETE CASCADE,
            term TEXT NOT NULL,
            tf INTEGER NOT NULL,
            PRIMARY KEY (collection_id, term, record_id)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS collection_meta (
            record_id INTEGER NOT NULL
                REFERENCES collection_record(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            value_text TEXT,
            value_num REAL,
            PRIMARY KEY (record_id, key)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_collection_meta_text
            ON collection_meta(key, value_text);

        CREATE INDEX IF NOT EXISTS idx_collection_meta_num
            ON collection_meta(key, value_num);

        CREATE INDEX IF NOT EXISTS idx_collection_bm25_posting_record
            ON collection_bm25_posting(record_id);
        """
    )


def _open(readonly: bool = False) -> sqlite3.Connection:
    db_path = collections_db_path()
    conn = connect(db_path, readonly=readonly)
    if not readonly:
        # Hand transaction control to us so writes can open with BEGIN
        # IMMEDIATE; see _write_transaction for why that matters.
        conn.isolation_level = None
        _ensure_schema(conn)
    return conn


@contextmanager
def _write_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a write transaction that takes the write lock up front.

    SQLite's default DEFERRED transaction takes a read lock on the first SELECT
    and only tries to upgrade on the first write. Two writers that each hold a
    read lock can never both upgrade, so SQLite returns SQLITE_BUSY immediately
    and ``busy_timeout`` does not apply — waiting cannot resolve a deadlock.
    ``BEGIN IMMEDIATE`` takes the write lock at the start, which ``busy_timeout``
    does cover, so concurrent writers queue instead of failing.
    """

    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


def _open_readonly() -> sqlite3.Connection | None:
    """Open the database read-only, or return ``None`` when it does not exist."""

    db_path = collections_db_path()
    if not db_path.exists():
        return None
    try:
        return connect(db_path, readonly=True)
    except sqlite3.OperationalError:
        return None


# --------------------------------------------------------------------------
# Metadata encoding
# --------------------------------------------------------------------------


def normalize_metadata(metadata: Mapping[str, object] | None) -> dict[str, ScalarValue]:
    """Validate that *metadata* is a flat scalar mapping and return a copy."""

    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise CollectionError(Messages.ERROR_COLLECTION_METADATA_NOT_MAPPING)
    normalized: dict[str, ScalarValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise CollectionError(Messages.ERROR_COLLECTION_METADATA_KEY_INVALID)
        if not _is_scalar(value):
            raise CollectionError(
                Messages.ERROR_COLLECTION_METADATA_VALUE_INVALID.format(
                    key=key,
                    type=type(value).__name__,
                )
            )
        normalized[key] = value
    return normalized


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, datetime))


def _encode_value(value: ScalarValue) -> tuple[str | None, float | None]:
    """Return the ``(value_text, value_num)`` pair stored for *value*.

    Only the column a query can actually read is written. Strings compare by
    equality on ``value_text``; numbers, booleans, and timestamps compare and
    sort on ``value_num``. A ``datetime`` writes both so callers can filter on a
    range and still read back the original ISO string.
    """

    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, 1.0 if value else 0.0
    if isinstance(value, datetime):
        return value.isoformat(), value.timestamp()
    if isinstance(value, (int, float)):
        return None, float(value)
    return str(value), None


def _query_column(value: object) -> tuple[str, object]:
    """Return the column and bind parameter used to compare against *value*."""

    if isinstance(value, bool):
        return "value_num", 1.0 if value else 0.0
    if isinstance(value, datetime):
        return "value_num", value.timestamp()
    if isinstance(value, (int, float)):
        return "value_num", float(value)
    if isinstance(value, str):
        return "value_text", value
    raise CollectionError(
        Messages.ERROR_COLLECTION_FILTER_VALUE_INVALID.format(type=type(value).__name__)
    )


# --------------------------------------------------------------------------
# Collection lifecycle
# --------------------------------------------------------------------------


def _row_to_info(row: sqlite3.Row) -> CollectionInfo:
    return CollectionInfo(
        id=int(row["id"]),
        name=str(row["name"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        dimension=int(row["dimension"]),
        schema_version=int(row["schema_version"]),
        created_at=str(row["created_at"]),
    )


def ensure_collection(
    name: str,
    *,
    provider: str,
    model: str,
    dimension: int,
) -> CollectionInfo:
    """Create *name* if absent, else verify its pinned embedding contract.

    Provider, model, and dimension are pinned together. Changing any of them
    would mix vector widths or semantics inside one collection, so this raises
    and tells the caller to recreate instead of migrating silently.
    """

    clean_name = (name or "").strip()
    if not clean_name:
        raise CollectionError(Messages.ERROR_COLLECTION_NAME_REQUIRED)
    if dimension <= 0:
        raise CollectionError(Messages.ERROR_COLLECTION_DIMENSION_INVALID)
    conn = _open()
    try:
        with _write_transaction(conn):
            row = conn.execute(
                "SELECT * FROM collection WHERE name = ?", (clean_name,)
            ).fetchone()
            if row is not None:
                info = _row_to_info(row)
                if (
                    info.provider != provider
                    or info.model != model
                    or info.dimension != dimension
                ):
                    raise CollectionError(
                        Messages.ERROR_COLLECTION_CONTRACT_MISMATCH.format(
                            name=clean_name,
                            stored=f"{info.provider}/{info.model}/{info.dimension}",
                            requested=f"{provider}/{model}/{dimension}",
                        )
                    )
                return info
            created_at = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                INSERT INTO collection (
                    name, provider, model, dimension, schema_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_name,
                    provider,
                    model,
                    int(dimension),
                    COLLECTION_SCHEMA_VERSION,
                    created_at,
                ),
            )
            return CollectionInfo(
                id=int(cursor.lastrowid),
                name=clean_name,
                provider=provider,
                model=model,
                dimension=int(dimension),
                schema_version=COLLECTION_SCHEMA_VERSION,
                created_at=created_at,
            )
    finally:
        conn.close()


def get_collection(name: str) -> CollectionInfo | None:
    """Return the stored contract for *name*, or ``None`` when it does not exist."""

    conn = _open_readonly()
    if conn is None:
        return None
    try:
        try:
            row = conn.execute(
                "SELECT * FROM collection WHERE name = ?", ((name or "").strip(),)
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return _row_to_info(row) if row is not None else None
    finally:
        conn.close()


def list_collections() -> list[CollectionInfo]:
    """Return every stored collection, ordered by name."""

    conn = _open_readonly()
    if conn is None:
        return []
    try:
        try:
            rows = conn.execute("SELECT * FROM collection ORDER BY name").fetchall()
        except sqlite3.OperationalError:
            return []
        return [_row_to_info(row) for row in rows]
    finally:
        conn.close()


def drop_collection(name: str) -> bool:
    """Delete *name* and everything under it. Returns ``False`` when absent."""

    if not collections_db_path().exists():
        # Dropping something that was never created must not create a database.
        return False
    conn = _open()
    try:
        with _write_transaction(conn):
            cursor = conn.execute(
                "DELETE FROM collection WHERE name = ?", ((name or "").strip(),)
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def count_records(collection_id: int) -> int:
    """Return how many records *collection_id* holds."""

    conn = _open_readonly()
    if conn is None:
        return 0
    try:
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM collection_record WHERE collection_id = ?",
                (int(collection_id),),
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["total"]) if row is not None else 0
    finally:
        conn.close()


def clear_all_collections() -> None:
    """Remove the collection database entirely.

    Deliberately not wired into ``vexor config --clear-index-all``: a file index
    can be rebuilt from disk, but these records can only be restored by the
    caller re-upserting everything and paying for embeddings again.
    """

    db_path = collections_db_path()
    if not db_path.exists():
        return
    # Empty the tables first. Unlinking alone is not enough on Windows, where an
    # open reader anywhere in the process holds the file and unlink raises; the
    # caller asked for the records to be gone, and that must succeed either way.
    conn = _open()
    try:
        with _write_transaction(conn):
            conn.execute("DELETE FROM collection")
    finally:
        conn.close()
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{db_path}{suffix}")
        try:
            if candidate.exists():
                candidate.unlink()
        except OSError:
            # The rows are already gone; a leftover empty file is harmless and
            # will be reused by the next write.
            pass


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------


def load_record_hashes(
    collection_id: int,
    record_keys: Sequence[str],
) -> dict[str, str]:
    """Return ``{record_key: text_hash}`` for the keys already stored."""

    keys = [key for key in dict.fromkeys(record_keys) if key]
    if not keys:
        return {}
    conn = _open_readonly()
    if conn is None:
        return {}
    try:
        results: dict[str, str] = {}
        for batch in chunk_values(keys):
            placeholders = ", ".join("?" for _ in batch)
            try:
                rows = conn.execute(
                    f"""
                    SELECT record_key, text_hash
                    FROM collection_record
                    WHERE collection_id = ? AND record_key IN ({placeholders})
                    """,
                    (int(collection_id), *batch),
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
            for row in rows:
                results[str(row["record_key"])] = str(row["text_hash"])
        return results
    finally:
        conn.close()


def upsert_records(collection_id: int, records: Sequence[PreparedRecord]) -> int:
    """Insert or replace *records*, returning how many rows were written."""

    if not records:
        return 0
    conn = _open()
    try:
        updated_at = datetime.now(timezone.utc).isoformat()
        with _write_transaction(conn):
            for record in records:
                conn.execute(
                    """
                    INSERT INTO collection_record (
                        collection_id, record_key, text, text_hash,
                        metadata_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(collection_id, record_key) DO UPDATE SET
                        text = excluded.text,
                        text_hash = excluded.text_hash,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        int(collection_id),
                        record.record_key,
                        record.text,
                        record.text_hash,
                        json.dumps(_metadata_to_json(record.metadata)),
                        updated_at,
                    ),
                )
                row = conn.execute(
                    """
                    SELECT id FROM collection_record
                    WHERE collection_id = ? AND record_key = ?
                    """,
                    (int(collection_id), record.record_key),
                ).fetchone()
                record_id = int(row["id"])
                # Metadata replaces wholesale on every upsert.
                conn.execute(
                    "DELETE FROM collection_meta WHERE record_id = ?", (record_id,)
                )
                if record.refresh_embedding:
                    # New text means the old postings score a string that no
                    # longer exists, so they go before the new ones land.
                    conn.execute(
                        "DELETE FROM collection_bm25_posting WHERE record_id = ?",
                        (record_id,),
                    )
                meta_rows = []
                for key, value in record.metadata.items():
                    value_text, value_num = _encode_value(value)
                    meta_rows.append((record_id, key, value_text, value_num))
                if meta_rows:
                    conn.executemany(
                        """
                        INSERT INTO collection_meta (record_id, key, value_text, value_num)
                        VALUES (?, ?, ?, ?)
                        """,
                        meta_rows,
                    )
                if record.vector is not None:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO collection_embedding (record_id, vector_blob)
                        VALUES (?, ?)
                        """,
                        (
                            record_id,
                            np.asarray(record.vector, dtype=np.float32).tobytes(),
                        ),
                    )
                if record.bm25_terms:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO collection_bm25_doc (record_id, token_count)
                        VALUES (?, ?)
                        """,
                        (record_id, int(record.token_count)),
                    )
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO collection_bm25_posting (
                            collection_id, record_id, term, tf
                        ) VALUES (?, ?, ?, ?)
                        """,
                        [
                            (int(collection_id), record_id, term, int(tf))
                            for term, tf in record.bm25_terms.items()
                        ],
                    )
        return len(records)
    finally:
        conn.close()


def _metadata_to_json(metadata: Mapping[str, ScalarValue]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in metadata.items():
        payload[key] = value.isoformat() if isinstance(value, datetime) else value
    return payload


def delete_records(collection_id: int, record_keys: Sequence[str]) -> int:
    """Delete the named records, returning how many rows were removed."""

    keys = [key for key in dict.fromkeys(record_keys) if key]
    if not keys:
        return 0
    conn = _open()
    try:
        removed = 0
        with _write_transaction(conn):
            for batch in chunk_values(keys):
                placeholders = ", ".join("?" for _ in batch)
                cursor = conn.execute(
                    f"""
                    DELETE FROM collection_record
                    WHERE collection_id = ? AND record_key IN ({placeholders})
                    """,
                    (int(collection_id), *batch),
                )
                removed += cursor.rowcount
        return removed
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def _decode_metadata(payload: str) -> dict[str, ScalarValue]:
    try:
        loaded = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def fetch_records(collection_id: int, record_keys: Sequence[str]) -> list[StoredRecord]:
    """Return stored records for *record_keys*, skipping keys that are absent."""

    keys = [key for key in dict.fromkeys(record_keys) if key]
    if not keys:
        return []
    conn = _open_readonly()
    if conn is None:
        return []
    try:
        found: dict[str, StoredRecord] = {}
        for batch in chunk_values(keys):
            placeholders = ", ".join("?" for _ in batch)
            try:
                rows = conn.execute(
                    f"""
                    SELECT record_key, text, metadata_json
                    FROM collection_record
                    WHERE collection_id = ? AND record_key IN ({placeholders})
                    """,
                    (int(collection_id), *batch),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            for row in rows:
                found[str(row["record_key"])] = StoredRecord(
                    id=str(row["record_key"]),
                    text=str(row["text"]),
                    metadata=_decode_metadata(row["metadata_json"]),
                )
        return [found[key] for key in keys if key in found]
    finally:
        conn.close()


def _compile_filter(
    filters: Mapping[str, object] | None,
) -> tuple[list[str], list[object]]:
    """Compile *filters* into SQL fragments ANDed against ``collection_record``."""

    if not filters:
        return [], []
    clauses: list[str] = []
    params: list[object] = []
    for key, condition in filters.items():
        if not isinstance(key, str) or not key.strip():
            raise CollectionError(Messages.ERROR_COLLECTION_FILTER_KEY_INVALID)
        operations = _normalize_condition(condition)
        for operator, value in operations:
            clause, clause_params = _compile_operation(key, operator, value)
            clauses.append(clause)
            params.extend(clause_params)
    return clauses, params


def _normalize_condition(condition: object) -> list[tuple[str, object]]:
    """Return ``[(operator, value)]`` for a filter condition."""

    if isinstance(condition, Mapping):
        operations: list[tuple[str, object]] = []
        for operator, value in condition.items():
            normalized = str(operator).lower()
            if normalized not in FILTER_OPERATORS:
                raise CollectionError(
                    Messages.ERROR_COLLECTION_FILTER_OPERATOR_INVALID.format(
                        operator=operator,
                        allowed=", ".join(FILTER_OPERATORS),
                    )
                )
            operations.append((normalized, value))
        if not operations:
            raise CollectionError(Messages.ERROR_COLLECTION_FILTER_EMPTY)
        return operations
    # A bare value is shorthand for equality.
    return [("eq", condition)]


def _compile_operation(
    key: str,
    operator: str,
    value: object,
) -> tuple[str, list[object]]:
    if operator == "exists":
        if not isinstance(value, bool):
            raise CollectionError(Messages.ERROR_COLLECTION_FILTER_EXISTS_INVALID)
        predicate = "IN" if value else "NOT IN"
        return (
            f"r.id {predicate} (SELECT record_id FROM collection_meta WHERE key = ?)",
            [key],
        )
    if operator in _SET_OPERATORS:
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise CollectionError(
                Messages.ERROR_COLLECTION_FILTER_SET_INVALID.format(operator=operator)
            )
        members = list(value)
        if not members:
            raise CollectionError(
                Messages.ERROR_COLLECTION_FILTER_SET_EMPTY.format(operator=operator)
            )
        text_values: list[object] = []
        num_values: list[object] = []
        for member in members:
            column, bind = _query_column(member)
            (text_values if column == "value_text" else num_values).append(bind)
        sub_clauses: list[str] = []
        params: list[object] = []
        if text_values:
            placeholders = ", ".join("?" for _ in text_values)
            sub_clauses.append(f"(key = ? AND value_text IN ({placeholders}))")
            params.append(key)
            params.extend(text_values)
        if num_values:
            placeholders = ", ".join("?" for _ in num_values)
            sub_clauses.append(f"(key = ? AND value_num IN ({placeholders}))")
            params.append(key)
            params.extend(num_values)
        predicate = "NOT IN" if operator == "nin" else "IN"
        joined = " OR ".join(sub_clauses)
        return (
            f"r.id {predicate} (SELECT record_id FROM collection_meta WHERE {joined})",
            params,
        )
    if operator in _RANGE_OPERATORS:
        column, bind = _query_column(value)
        if column != "value_num":
            raise CollectionError(
                Messages.ERROR_COLLECTION_FILTER_RANGE_INVALID.format(operator=operator)
            )
        comparison = _RANGE_SQL[operator]
        return (
            "r.id IN (SELECT record_id FROM collection_meta "
            f"WHERE key = ? AND value_num IS NOT NULL AND value_num {comparison} ?)",
            [key, bind],
        )
    # eq / ne
    if value is None:
        clause = (
            "SELECT record_id FROM collection_meta "
            "WHERE key = ? AND value_text IS NULL AND value_num IS NULL"
        )
        params = [key]
    else:
        column, bind = _query_column(value)
        clause = f"SELECT record_id FROM collection_meta WHERE key = ? AND {column} = ?"
        params = [key, bind]
    predicate = "NOT IN" if operator in _NEGATIVE_OPERATORS else "IN"
    return (f"r.id {predicate} ({clause})", params)


def resolve_filter_ids(
    collection_id: int,
    filters: Mapping[str, object] | None,
) -> list[int]:
    """Return the record ids matching *filters*, resolved before any scoring.

    Filtering is strict and happens first: post-filtering a global top-k would
    return nothing for a single chat whose records never reach the global head.
    """

    clauses, params = _compile_filter(filters)
    conn = _open_readonly()
    if conn is None:
        return []
    try:
        sql = "SELECT r.id FROM collection_record AS r WHERE r.collection_id = ?"
        if clauses:
            sql += " AND " + " AND ".join(clauses)
        sql += " ORDER BY r.id"
        try:
            rows = conn.execute(sql, (int(collection_id), *params)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [int(row["id"]) for row in rows]
    finally:
        conn.close()


def load_vectors(
    collection_id: int,
    record_ids: Sequence[int],
    dimension: int,
) -> tuple[list[int], np.ndarray]:
    """Load embeddings for *record_ids*, dropping rows with no stored vector."""

    ids = [int(value) for value in record_ids]
    if not ids:
        return [], np.empty((0, dimension), dtype=np.float32)
    conn = _open_readonly()
    if conn is None:
        return [], np.empty((0, dimension), dtype=np.float32)
    try:
        vectors: dict[int, np.ndarray] = {}
        for batch in chunk_values(ids):
            placeholders = ", ".join("?" for _ in batch)
            try:
                rows = conn.execute(
                    f"""
                    SELECT e.record_id, e.vector_blob
                    FROM collection_embedding AS e
                    JOIN collection_record AS r ON r.id = e.record_id
                    WHERE r.collection_id = ? AND e.record_id IN ({placeholders})
                    """,
                    (int(collection_id), *batch),
                ).fetchall()
            except sqlite3.OperationalError:
                return [], np.empty((0, dimension), dtype=np.float32)
            for row in rows:
                blob = row["vector_blob"]
                if not blob:
                    continue
                vector = np.frombuffer(blob, dtype=np.float32)
                if vector.size != dimension:
                    continue
                vectors[int(row["record_id"])] = vector
        ordered_ids = [value for value in ids if value in vectors]
        if not ordered_ids:
            return [], np.empty((0, dimension), dtype=np.float32)
        matrix = np.vstack([vectors[value] for value in ordered_ids])
        return ordered_ids, matrix
    finally:
        conn.close()


def load_bm25_stats(
    collection_id: int,
    record_ids: Sequence[int],
) -> tuple[int, float]:
    """Return BM25 corpus statistics over *record_ids* only.

    The corpus is the filtered subset, not the whole collection, so idf and
    document-length normalization agree with what is actually being scored.
    """

    ids = [int(value) for value in record_ids]
    if not ids:
        return 0, 0.0
    conn = _open_readonly()
    if conn is None:
        return 0, 0.0
    try:
        total = 0
        length_sum = 0
        for batch in chunk_values(ids):
            placeholders = ", ".join("?" for _ in batch)
            try:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS doc_count, SUM(token_count) AS length_sum
                    FROM collection_bm25_doc
                    WHERE record_id IN ({placeholders})
                    """,
                    tuple(batch),
                ).fetchone()
            except sqlite3.OperationalError:
                return 0, 0.0
            if row is None:
                continue
            total += int(row["doc_count"] or 0)
            length_sum += int(row["length_sum"] or 0)
        if total <= 0:
            return 0, 0.0
        return total, length_sum / total
    finally:
        conn.close()


def load_bm25_postings(
    collection_id: int,
    record_ids: Sequence[int],
    terms: Sequence[str],
) -> dict[str, list[tuple[int, int, int]]]:
    """Load posting lists restricted to *record_ids*, shaped for ``bm25``."""

    ids = {int(value) for value in record_ids}
    unique_terms = [term for term in dict.fromkeys(terms) if term]
    if not ids or not unique_terms:
        return {}
    conn = _open_readonly()
    if conn is None:
        return {}
    try:
        results: dict[str, list[tuple[int, int, int]]] = {}
        for batch in chunk_values(unique_terms):
            placeholders = ", ".join("?" for _ in batch)
            try:
                rows = conn.execute(
                    f"""
                    SELECT p.term, p.record_id, p.tf, d.token_count
                    FROM collection_bm25_posting AS p
                    JOIN collection_bm25_doc AS d ON d.record_id = p.record_id
                    WHERE p.collection_id = ? AND p.term IN ({placeholders})
                    """,
                    (int(collection_id), *batch),
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
            for row in rows:
                record_id = int(row["record_id"])
                if record_id not in ids:
                    continue
                results.setdefault(str(row["term"]), []).append(
                    (record_id, int(row["tf"]), int(row["token_count"]))
                )
        return results
    finally:
        conn.close()


def fetch_by_ids(collection_id: int, record_ids: Sequence[int]) -> dict[int, StoredRecord]:
    """Return ``{record_id: StoredRecord}`` for the given internal ids."""

    ids = [int(value) for value in dict.fromkeys(record_ids)]
    if not ids:
        return {}
    conn = _open_readonly()
    if conn is None:
        return {}
    try:
        results: dict[int, StoredRecord] = {}
        for batch in chunk_values(ids):
            placeholders = ", ".join("?" for _ in batch)
            try:
                rows = conn.execute(
                    f"""
                    SELECT id, record_key, text, metadata_json
                    FROM collection_record
                    WHERE collection_id = ? AND id IN ({placeholders})
                    """,
                    (int(collection_id), *batch),
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
            for row in rows:
                results[int(row["id"])] = StoredRecord(
                    id=str(row["record_key"]),
                    text=str(row["text"]),
                    metadata=_decode_metadata(row["metadata_json"]),
                )
        return results
    finally:
        conn.close()
