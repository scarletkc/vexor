"""Index cache helpers for Vexor backed by SQLite."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Iterable, Mapping, Sequence

import numpy as np

from .utils import collect_files

DEFAULT_CACHE_DIR = Path(os.path.expanduser("~")) / ".vexor"
CACHE_DIR = DEFAULT_CACHE_DIR
_CACHE_DIR_OVERRIDE: ContextVar[Path | None] = ContextVar(
    "vexor_cache_dir_override",
    default=None,
)
CACHE_VERSION = 6
DB_FILENAME = "index.db"
EMBED_CACHE_TTL_DAYS = 30
EMBED_CACHE_MAX_ENTRIES = 50_000
EMBED_MEMORY_CACHE_MAX_ENTRIES = 2_048

_EMBED_MEMORY_CACHE: "OrderedDict[tuple[str, str], np.ndarray]" = OrderedDict()
_EMBED_MEMORY_LOCK = Lock()


@dataclass(slots=True)
class IndexedChunk:
    path: Path
    rel_path: str
    chunk_index: int
    preview: str
    embedding: Sequence[float]
    label_hash: str = ""
    size_bytes: int | None = None
    mtime: float | None = None
    start_line: int | None = None
    end_line: int | None = None


def _cache_key(
    root: Path,
    include_hidden: bool,
    respect_gitignore: bool,
    recursive: bool,
    mode: str,
    extensions: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> str:
    base = (
        f"{root.resolve()}|hidden={include_hidden}|gitignore={respect_gitignore}"
        f"|recursive={recursive}|mode={mode}"
    )
    ext_key = _serialize_extensions(extensions)
    if ext_key:
        base = f"{base}|ext={ext_key}"
    exclude_key = _serialize_exclude_patterns(exclude_patterns)
    if exclude_key:
        base = f"{base}|exclude={exclude_key}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return digest


def _normalize_model_for_query_cache(model: str) -> str:
    normalized = (model or "").strip()
    lowered = normalized.lower()
    for prefix in ("openai/", "gemini/", "custom/", "local/"):
        if lowered.startswith(prefix):
            return normalized.split("/", 1)[1]
    return normalized


def query_cache_key(query: str, model: str) -> str:
    """Return the stable cache hash for a semantic query embedding."""

    clean_query = (query or "").strip()
    clean_model = _normalize_model_for_query_cache(model)
    base = f"{clean_query}|model={clean_model}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def embedding_cache_key(text: str) -> str:
    """Return a stable hash for embedding cache lookups."""

    clean_text = text or ""
    return hashlib.sha1(clean_text.encode("utf-8")).hexdigest()


def _clear_embedding_memory_cache() -> None:
    if EMBED_MEMORY_CACHE_MAX_ENTRIES <= 0:
        return
    with _EMBED_MEMORY_LOCK:
        _EMBED_MEMORY_CACHE.clear()


def _load_embedding_memory_cache(
    model: str,
    text_hashes: Sequence[str],
) -> dict[str, np.ndarray]:
    if EMBED_MEMORY_CACHE_MAX_ENTRIES <= 0:
        return {}
    results: dict[str, np.ndarray] = {}
    with _EMBED_MEMORY_LOCK:
        for text_hash in text_hashes:
            if not text_hash:
                continue
            key = (model, text_hash)
            vector = _EMBED_MEMORY_CACHE.pop(key, None)
            if vector is None:
                continue
            _EMBED_MEMORY_CACHE[key] = vector
            results[text_hash] = vector
    return results


def _store_embedding_memory_cache(
    *,
    model: str,
    embeddings: Mapping[str, np.ndarray],
) -> None:
    if EMBED_MEMORY_CACHE_MAX_ENTRIES <= 0 or not embeddings:
        return
    with _EMBED_MEMORY_LOCK:
        for text_hash, vector in embeddings.items():
            if not text_hash:
                continue
            array = np.asarray(vector, dtype=np.float32)
            if array.size == 0:
                continue
            key = (model, text_hash)
            if key in _EMBED_MEMORY_CACHE:
                _EMBED_MEMORY_CACHE.pop(key, None)
            _EMBED_MEMORY_CACHE[key] = array
        while len(_EMBED_MEMORY_CACHE) > EMBED_MEMORY_CACHE_MAX_ENTRIES:
            _EMBED_MEMORY_CACHE.popitem(last=False)


def _serialize_extensions(extensions: Sequence[str] | None) -> str:
    if not extensions:
        return ""
    return ",".join(extensions)


def _serialize_exclude_patterns(patterns: Sequence[str] | None) -> str:
    if not patterns:
        return ""
    return "\n".join(patterns)


def _deserialize_extensions(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    parts = [part for part in value.split(",") if part]
    return tuple(parts)


def _deserialize_exclude_patterns(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    parts = [part for part in value.split("\n") if part]
    return tuple(parts)


def _chunk_values(values: Sequence[object], size: int) -> Iterable[Sequence[object]]:
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def _resolve_cache_dir() -> Path:
    override = _CACHE_DIR_OVERRIDE.get()
    return override if override is not None else CACHE_DIR


@contextmanager
def cache_dir_context(path: Path | str | None):
    """Temporarily override the cache directory for the current context."""

    if path is None:
        yield
        return
    dir_path = Path(path).expanduser().resolve()
    if dir_path.exists() and not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    token = _CACHE_DIR_OVERRIDE.set(dir_path)
    try:
        yield
    finally:
        _CACHE_DIR_OVERRIDE.reset(token)


def ensure_cache_dir() -> Path:
    cache_dir = _resolve_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def set_cache_dir(path: Path | str | None) -> None:
    global CACHE_DIR
    if path is None:
        CACHE_DIR = DEFAULT_CACHE_DIR
        return
    dir_path = Path(path).expanduser().resolve()
    if dir_path.exists() and not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    CACHE_DIR = dir_path


def cache_db_path() -> Path:
    """Return the absolute path to the shared SQLite cache database."""

    cache_dir = ensure_cache_dir()
    return cache_dir / DB_FILENAME


def cache_file(root: Path, model: str, include_hidden: bool) -> Path:  # pragma: no cover - kept for API parity
    """Return the on-disk cache artifact path (single SQLite DB)."""
    return cache_db_path()


def _connect(
    db_path: Path,
    *,
    readonly: bool = False,
    query_only: bool = False,
) -> sqlite3.Connection:
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


def _ensure_schema_readonly(
    conn: sqlite3.Connection,
    *,
    tables: Sequence[str],
) -> None:
    if _schema_needs_reset(conn):
        raise sqlite3.OperationalError("Schema reset required")
    for table in tables:
        if not _table_exists(conn, table):
            raise sqlite3.OperationalError(f"Missing table: {table}")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _schema_needs_reset(conn: sqlite3.Connection) -> bool:
    if _table_exists(conn, "indexed_chunk"):
        return False
    return any(
        _table_exists(conn, table)
        for table in ("index_metadata", "indexed_file", "file_embedding", "query_cache")
    )


def _reset_index_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.executescript(
        """
        DROP TABLE IF EXISTS query_cache;
        DROP TABLE IF EXISTS file_embedding;
        DROP TABLE IF EXISTS chunk_embedding;
        DROP TABLE IF EXISTS chunk_meta;
        DROP TABLE IF EXISTS indexed_chunk;
        DROP TABLE IF EXISTS indexed_file;
        DROP TABLE IF EXISTS index_metadata;
        """
    )
    conn.execute("PRAGMA foreign_keys = ON;")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    if _schema_needs_reset(conn):
        _reset_index_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS index_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key TEXT NOT NULL,
            root_path TEXT NOT NULL,
            model TEXT NOT NULL,
            include_hidden INTEGER NOT NULL,
            respect_gitignore INTEGER NOT NULL DEFAULT 1,
            recursive INTEGER NOT NULL DEFAULT 1,
            mode TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            version INTEGER NOT NULL,
            generated_at TEXT NOT NULL,
            exclude_patterns TEXT DEFAULT '',
            extensions TEXT DEFAULT '',
            UNIQUE(cache_key, model)
        );

        CREATE TABLE IF NOT EXISTS indexed_file (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id INTEGER NOT NULL REFERENCES index_metadata(id) ON DELETE CASCADE,
            rel_path TEXT NOT NULL,
            abs_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime REAL NOT NULL,
            UNIQUE(index_id, rel_path)
        );

        CREATE TABLE IF NOT EXISTS indexed_chunk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id INTEGER NOT NULL REFERENCES index_metadata(id) ON DELETE CASCADE,
            file_id INTEGER NOT NULL REFERENCES indexed_file(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            position INTEGER NOT NULL,
            UNIQUE(index_id, file_id, chunk_index)
        );

        CREATE TABLE IF NOT EXISTS chunk_embedding (
            chunk_id INTEGER PRIMARY KEY REFERENCES indexed_chunk(id) ON DELETE CASCADE,
            vector_blob BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunk_meta (
            chunk_id INTEGER PRIMARY KEY REFERENCES indexed_chunk(id) ON DELETE CASCADE,
            preview TEXT DEFAULT '',
            label_hash TEXT DEFAULT '',
            start_line INTEGER,
            end_line INTEGER
        );

        CREATE TABLE IF NOT EXISTS query_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id INTEGER NOT NULL REFERENCES index_metadata(id) ON DELETE CASCADE,
            query_hash TEXT NOT NULL,
            query_text TEXT NOT NULL,
            query_vector BLOB NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(index_id, query_hash)
        );

        CREATE TABLE IF NOT EXISTS embedding_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            vector_blob BLOB NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(model, text_hash)
        );

        CREATE INDEX IF NOT EXISTS idx_indexed_chunk_order
            ON indexed_chunk(index_id, position);

        CREATE INDEX IF NOT EXISTS idx_indexed_file_lookup
            ON indexed_file(index_id, rel_path);

        CREATE INDEX IF NOT EXISTS idx_query_cache_lookup
            ON query_cache(index_id, query_hash);

        CREATE INDEX IF NOT EXISTS idx_embedding_cache_lookup
            ON embedding_cache(model, text_hash);
        """
    )


def store_index(
    *,
    root: Path,
    model: str,
    include_hidden: bool,
    respect_gitignore: bool = True,
    mode: str,
    recursive: bool,
    entries: Sequence[IndexedChunk],
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
) -> Path:
    db_path = cache_file(root, model, include_hidden)
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(
            root,
            include_hidden,
            respect_gitignore,
            recursive,
            mode,
            extensions,
            exclude_patterns,
        )
        generated_at = datetime.now(timezone.utc).isoformat()
        dimension = int(len(entries[0].embedding) if entries else 0)
        include_flag = 1 if include_hidden else 0
        gitignore_flag = 1 if respect_gitignore else 0
        recursive_flag = 1 if recursive else 0
        extensions_value = _serialize_extensions(extensions)
        exclude_patterns_value = _serialize_exclude_patterns(exclude_patterns)

        with conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                "DELETE FROM index_metadata WHERE cache_key = ? AND model = ?",
                (key, model),
            )
            cursor = conn.execute(
                """
                INSERT INTO index_metadata (
                    cache_key,
                    root_path,
                    model,
                    include_hidden,
                    respect_gitignore,
                    recursive,
                    mode,
                    dimension,
                    version,
                    generated_at,
                    exclude_patterns,
                    extensions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    str(root),
                    model,
                    include_flag,
                    gitignore_flag,
                    recursive_flag,
                    mode,
                    dimension,
                    CACHE_VERSION,
                    generated_at,
                    exclude_patterns_value,
                    extensions_value,
                ),
            )
            index_id = cursor.lastrowid

            file_rows_by_rel: dict[str, tuple] = {}
            for entry in entries:
                if entry.rel_path in file_rows_by_rel:
                    continue
                size_bytes = entry.size_bytes
                mtime = entry.mtime
                if size_bytes is None or mtime is None:
                    stat = entry.path.stat()
                    size_bytes = stat.st_size
                    mtime = stat.st_mtime
                file_rows_by_rel[entry.rel_path] = (
                    index_id,
                    entry.rel_path,
                    str(entry.path),
                    size_bytes,
                    mtime,
                )

            conn.executemany(
                """
                INSERT INTO indexed_file (
                    index_id,
                    rel_path,
                    abs_path,
                    size_bytes,
                    mtime
                ) VALUES (?, ?, ?, ?, ?)
                """,
                list(file_rows_by_rel.values()),
            )

            file_id_map: dict[str, int] = {}
            rel_paths = list(file_rows_by_rel.keys())
            for chunk in _chunk_values(rel_paths, 900):
                placeholders = ", ".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT id, rel_path
                    FROM indexed_file
                    WHERE index_id = ? AND rel_path IN ({placeholders})
                    """,
                    (index_id, *chunk),
                ).fetchall()
                for row in rows:
                    file_id_map[row["rel_path"]] = int(row["id"])

            chunk_rows: list[tuple] = []
            vector_blobs: list[bytes] = []
            meta_rows: list[tuple] = []
            for position, entry in enumerate(entries):
                file_id = file_id_map.get(entry.rel_path)
                if file_id is None:
                    continue
                chunk_rows.append(
                    (index_id, file_id, entry.chunk_index, position)
                )
                vector_blobs.append(
                    np.asarray(entry.embedding, dtype=np.float32).tobytes()
                )
                meta_rows.append(
                    (
                        entry.preview or "",
                        entry.label_hash or "",
                        entry.start_line,
                        entry.end_line,
                    )
                )

            conn.executemany(
                """
                INSERT INTO indexed_chunk (
                    index_id,
                    file_id,
                    chunk_index,
                    position
                ) VALUES (?, ?, ?, ?)
                """,
                chunk_rows,
            )

            inserted_ids = conn.execute(
                "SELECT id FROM indexed_chunk WHERE index_id = ? ORDER BY position ASC",
                (index_id,),
            ).fetchall()
            conn.executemany(
                "INSERT OR REPLACE INTO chunk_embedding (chunk_id, vector_blob) VALUES (?, ?)",
                (
                    (row["id"], vector_blobs[idx])
                    for idx, row in enumerate(inserted_ids)
                ),
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_meta (
                    chunk_id,
                    preview,
                    label_hash,
                    start_line,
                    end_line
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (row["id"], *meta_rows[idx])
                    for idx, row in enumerate(inserted_ids)
                ),
            )

        return db_path
    finally:
        conn.close()


def apply_index_updates(
    *,
    root: Path,
    model: str,
    include_hidden: bool,
    respect_gitignore: bool = True,
    mode: str,
    recursive: bool,
    ordered_entries: Sequence[tuple[str, int]],
    changed_entries: Sequence[IndexedChunk],
    touched_entries: Sequence[
        tuple[str, int, int, float, str | None, int | None, int | None, str]
    ] = (),
    removed_rel_paths: Sequence[str],
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
) -> Path:
    """Apply incremental updates to an existing cached index."""

    db_path = cache_file(root, model, include_hidden)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(
            root,
            include_hidden,
            respect_gitignore,
            recursive,
            mode,
            extensions,
            exclude_patterns,
        )
        include_flag = 1 if include_hidden else 0
        gitignore_flag = 1 if respect_gitignore else 0
        recursive_flag = 1 if recursive else 0

        with conn:
            conn.execute("BEGIN IMMEDIATE;")
            meta = conn.execute(
                """
                SELECT id, dimension
                FROM index_metadata
                WHERE cache_key = ? AND model = ? AND include_hidden = ? AND respect_gitignore = ? AND recursive = ? AND mode = ?
                """,
                (key, model, include_flag, gitignore_flag, recursive_flag, mode),
            ).fetchone()
            if meta is None:
                raise FileNotFoundError(db_path)
            index_id = meta["id"]
            existing_dimension = int(meta["dimension"])

            if removed_rel_paths:
                conn.executemany(
                    "DELETE FROM indexed_file WHERE index_id = ? AND rel_path = ?",
                    ((index_id, rel) for rel in removed_rel_paths),
                )

            vector_dimension = None
            if changed_entries:
                chunk_map: dict[str, list[IndexedChunk]] = {}
                for entry in changed_entries:
                    chunk_map.setdefault(entry.rel_path, []).append(entry)

                for rel_path in chunk_map:
                    conn.execute(
                        "DELETE FROM indexed_file WHERE index_id = ? AND rel_path = ?",
                        (index_id, rel_path),
                    )

                file_rows_by_rel: dict[str, tuple] = {}
                for rel_path, chunk_list in chunk_map.items():
                    chunk_list.sort(key=lambda item: item.chunk_index)
                    sample = chunk_list[0]
                    size_bytes = sample.size_bytes
                    mtime = sample.mtime
                    if size_bytes is None or mtime is None:
                        stat = sample.path.stat()
                        size_bytes = stat.st_size
                        mtime = stat.st_mtime
                    file_rows_by_rel[rel_path] = (
                        index_id,
                        rel_path,
                        str(sample.path),
                        size_bytes,
                        mtime,
                    )

                if file_rows_by_rel:
                    conn.executemany(
                        """
                        INSERT INTO indexed_file (
                            index_id,
                            rel_path,
                            abs_path,
                            size_bytes,
                            mtime
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        list(file_rows_by_rel.values()),
                    )

                file_id_map: dict[str, int] = {}
                rel_paths = list(file_rows_by_rel.keys())
                for chunk in _chunk_values(rel_paths, 900):
                    placeholders = ", ".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"""
                        SELECT id, rel_path
                        FROM indexed_file
                        WHERE index_id = ? AND rel_path IN ({placeholders})
                        """,
                        (index_id, *chunk),
                    ).fetchall()
                    for row in rows:
                        file_id_map[row["rel_path"]] = int(row["id"])

                for rel_path, chunk_list in chunk_map.items():
                    file_id = file_id_map.get(rel_path)
                    if file_id is None:
                        continue
                    chunk_list.sort(key=lambda item: item.chunk_index)
                    chunk_rows: list[tuple] = []
                    vector_blobs: list[bytes] = []
                    meta_rows: list[tuple] = []
                    for chunk in chunk_list:
                        vector = np.asarray(chunk.embedding, dtype=np.float32)
                        if vector_dimension is None:
                            vector_dimension = vector.shape[0]
                        chunk_rows.append(
                            (index_id, file_id, chunk.chunk_index, 0)
                        )
                        vector_blobs.append(vector.tobytes())
                        meta_rows.append(
                            (
                                chunk.preview or "",
                                chunk.label_hash or "",
                                chunk.start_line,
                                chunk.end_line,
                            )
                        )

                    conn.executemany(
                        """
                        INSERT INTO indexed_chunk (
                            index_id,
                            file_id,
                            chunk_index,
                            position
                        ) VALUES (?, ?, ?, ?)
                        """,
                        chunk_rows,
                    )

                    inserted_ids = conn.execute(
                        """
                        SELECT id FROM indexed_chunk
                        WHERE index_id = ? AND file_id = ?
                        ORDER BY chunk_index ASC
                        """,
                        (index_id, file_id),
                    ).fetchall()
                    conn.executemany(
                        "INSERT OR REPLACE INTO chunk_embedding (chunk_id, vector_blob) VALUES (?, ?)",
                        (
                            (row["id"], vector_blobs[idx])
                            for idx, row in enumerate(inserted_ids)
                        ),
                    )
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO chunk_meta (
                            chunk_id,
                            preview,
                            label_hash,
                            start_line,
                            end_line
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            (row["id"], *meta_rows[idx])
                            for idx, row in enumerate(inserted_ids)
                        ),
                    )

            if touched_entries:
                file_updates: dict[str, tuple[int, float]] = {}
                for (
                    rel_path,
                    _chunk_index,
                    size_bytes,
                    mtime,
                    _preview,
                    _start_line,
                    _end_line,
                    _label_hash,
                ) in touched_entries:
                    if rel_path not in file_updates:
                        file_updates[rel_path] = (size_bytes, mtime)
                conn.executemany(
                    """
                    UPDATE indexed_file
                    SET size_bytes = ?, mtime = ?
                    WHERE index_id = ? AND rel_path = ?
                    """,
                    (
                        (size_bytes, mtime, index_id, rel_path)
                        for rel_path, (size_bytes, mtime) in file_updates.items()
                    ),
                )

            chunk_id_map: dict[tuple[str, int], int] = {}
            if ordered_entries or touched_entries:
                rows = conn.execute(
                    """
                    SELECT c.id, c.chunk_index, f.rel_path
                    FROM indexed_chunk AS c
                    JOIN indexed_file AS f ON f.id = c.file_id
                    WHERE c.index_id = ?
                    """,
                    (index_id,),
                ).fetchall()
                for row in rows:
                    chunk_id_map[(row["rel_path"], int(row["chunk_index"]))] = int(
                        row["id"]
                    )

            if touched_entries and chunk_id_map:
                meta_rows: list[tuple] = []
                for (
                    rel_path,
                    chunk_index,
                    _size_bytes,
                    _mtime,
                    preview,
                    start_line,
                    end_line,
                    label_hash,
                ) in touched_entries:
                    chunk_id = chunk_id_map.get((rel_path, chunk_index))
                    if chunk_id is None:
                        continue
                    meta_rows.append(
                        (
                            chunk_id,
                            preview or "",
                            label_hash or "",
                            start_line,
                            end_line,
                        )
                    )
                if meta_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO chunk_meta (
                            chunk_id,
                            preview,
                            label_hash,
                            start_line,
                            end_line
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        meta_rows,
                    )

            if ordered_entries and chunk_id_map:
                position_updates = []
                for position, (rel_path, chunk_index) in enumerate(ordered_entries):
                    chunk_id = chunk_id_map.get((rel_path, chunk_index))
                    if chunk_id is None:
                        continue
                    position_updates.append((position, chunk_id))
                if position_updates:
                    conn.executemany(
                        "UPDATE indexed_chunk SET position = ? WHERE id = ?",
                        position_updates,
                    )

            generated_at = datetime.now(timezone.utc).isoformat()
            new_dimension = vector_dimension or existing_dimension
            conn.execute(
                """
                UPDATE index_metadata
                SET generated_at = ?, dimension = ?, version = ?
                WHERE id = ?
                """,
                (generated_at, new_dimension, CACHE_VERSION, index_id),
            )

        return db_path
    finally:
        conn.close()


def backfill_chunk_lines(
    *,
    root: Path,
    model: str,
    include_hidden: bool,
    mode: str,
    recursive: bool,
    updates: Sequence[tuple[str, int, int | None, int | None]],
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
    respect_gitignore: bool = True,
) -> Path:
    """Backfill start/end line metadata for an existing cached index."""

    db_path = cache_file(root, model, include_hidden)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(
            root,
            include_hidden,
            respect_gitignore,
            recursive,
            mode,
            extensions,
            exclude_patterns,
        )
        include_flag = 1 if include_hidden else 0
        gitignore_flag = 1 if respect_gitignore else 0
        recursive_flag = 1 if recursive else 0
        meta = conn.execute(
            """
            SELECT id
            FROM index_metadata
            WHERE cache_key = ? AND model = ? AND include_hidden = ? AND respect_gitignore = ? AND recursive = ? AND mode = ?
            """,
            (key, model, include_flag, gitignore_flag, recursive_flag, mode),
        ).fetchone()
        if meta is None:
            raise FileNotFoundError(db_path)
        index_id = int(meta["id"])

        with conn:
            conn.execute("BEGIN IMMEDIATE;")
            update_rows: list[tuple[int | None, int | None, int]] = []
            insert_rows: list[tuple[int]] = []
            if updates:
                rel_paths = sorted({rel_path for rel_path, *_ in updates})
                chunk_id_map: dict[tuple[str, int], int] = {}
                for chunk in _chunk_values(rel_paths, 900):
                    placeholders = ", ".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"""
                        SELECT c.id, c.chunk_index, f.rel_path
                        FROM indexed_chunk AS c
                        JOIN indexed_file AS f ON f.id = c.file_id
                        WHERE c.index_id = ? AND f.rel_path IN ({placeholders})
                        """,
                        (index_id, *chunk),
                    ).fetchall()
                    for row in rows:
                        chunk_id_map[(row["rel_path"], int(row["chunk_index"]))] = int(
                            row["id"]
                        )
                for rel_path, chunk_index, start_line, end_line in updates:
                    chunk_id = chunk_id_map.get((rel_path, chunk_index))
                    if chunk_id is None:
                        continue
                    insert_rows.append((chunk_id,))
                    update_rows.append((start_line, end_line, chunk_id))
            if insert_rows:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO chunk_meta (
                        chunk_id,
                        preview,
                        label_hash
                    ) VALUES (?, '', '')
                    """,
                    insert_rows,
                )
            if update_rows:
                conn.executemany(
                    """
                    UPDATE chunk_meta
                    SET start_line = ?, end_line = ?
                    WHERE chunk_id = ?
                    """,
                    update_rows,
                )
            generated_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE index_metadata
                SET generated_at = ?, version = ?
                WHERE id = ?
                """,
                (generated_at, CACHE_VERSION, index_id),
            )

        return db_path
    finally:
        conn.close()


def load_index(
    root: Path,
    model: str,
    include_hidden: bool,
    mode: str,
    recursive: bool,
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
    *,
    respect_gitignore: bool = True,
) -> dict:
    db_path = cache_file(root, model, include_hidden)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = _connect(db_path, readonly=True)
    try:
        try:
            _ensure_schema_readonly(
                conn,
                tables=("index_metadata", "indexed_file", "indexed_chunk", "chunk_meta"),
            )
        except sqlite3.OperationalError:
            raise FileNotFoundError(db_path)
        key = _cache_key(
            root,
            include_hidden,
            respect_gitignore,
            recursive,
            mode,
            extensions,
            exclude_patterns,
        )
        include_flag = 1 if include_hidden else 0
        gitignore_flag = 1 if respect_gitignore else 0
        recursive_flag = 1 if recursive else 0
        meta = conn.execute(
            """
            SELECT id, root_path, model, include_hidden, respect_gitignore, recursive, mode, dimension, version, generated_at, exclude_patterns, extensions
            FROM index_metadata
            WHERE cache_key = ? AND model = ? AND include_hidden = ? AND respect_gitignore = ? AND recursive = ? AND mode = ?
            """,
            (key, model, include_flag, gitignore_flag, recursive_flag, mode),
        ).fetchone()
        if meta is None:
            raise FileNotFoundError(db_path)
        version = int(meta["version"] or 0)
        if version < CACHE_VERSION:
            raise FileNotFoundError(db_path)

        rows = conn.execute(
            """
            SELECT
                f.rel_path,
                f.abs_path,
                f.size_bytes,
                f.mtime,
                c.chunk_index,
                m.preview,
                m.label_hash,
                m.start_line,
                m.end_line
            FROM indexed_chunk AS c
            JOIN indexed_file AS f ON f.id = c.file_id
            LEFT JOIN chunk_meta AS m ON m.chunk_id = c.id
            WHERE c.index_id = ?
            ORDER BY c.position ASC
            """,
            (meta["id"],),
        ).fetchall()

        file_snapshot: dict[str, dict] = {}
        chunk_entries: list[dict] = []
        for row in rows:
            rel_path = row["rel_path"]
            chunk_index = int(row["chunk_index"])
            chunk_entries.append(
                {
                    "path": rel_path,
                    "absolute": row["abs_path"],
                    "mtime": row["mtime"],
                    "size": row["size_bytes"],
                    "preview": row["preview"],
                    "label_hash": row["label_hash"],
                    "chunk_index": chunk_index,
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                }
            )
            if rel_path not in file_snapshot:
                file_snapshot[rel_path] = {
                    "path": rel_path,
                    "absolute": row["abs_path"],
                    "mtime": row["mtime"],
                    "size": row["size_bytes"],
                }

        return {
            "version": meta["version"],
            "generated_at": meta["generated_at"],
            "root": meta["root_path"],
            "model": meta["model"],
            "include_hidden": bool(meta["include_hidden"]),
            "respect_gitignore": bool(meta["respect_gitignore"]),
            "recursive": bool(meta["recursive"]),
            "mode": meta["mode"],
            "dimension": meta["dimension"],
            "exclude_patterns": _deserialize_exclude_patterns(meta["exclude_patterns"]),
            "extensions": _deserialize_extensions(meta["extensions"]),
            "files": list(file_snapshot.values()),
            "chunks": chunk_entries,
        }
    finally:
        conn.close()


def load_index_vectors(
    root: Path,
    model: str,
    include_hidden: bool,
    mode: str,
    recursive: bool,
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
    *,
    respect_gitignore: bool = True,
):
    db_path = cache_file(root, model, include_hidden)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = _connect(db_path, readonly=True)
    try:
        try:
            _ensure_schema_readonly(
                conn,
                tables=(
                    "index_metadata",
                    "indexed_file",
                    "indexed_chunk",
                    "chunk_embedding",
                ),
            )
        except sqlite3.OperationalError:
            raise FileNotFoundError(db_path)
        key = _cache_key(
            root,
            include_hidden,
            respect_gitignore,
            recursive,
            mode,
            extensions,
            exclude_patterns,
        )
        include_flag = 1 if include_hidden else 0
        gitignore_flag = 1 if respect_gitignore else 0
        recursive_flag = 1 if recursive else 0
        meta = conn.execute(
            """
            SELECT id, root_path, model, include_hidden, respect_gitignore, recursive, mode, dimension, version, generated_at, exclude_patterns, extensions
            FROM index_metadata
            WHERE cache_key = ? AND model = ? AND include_hidden = ? AND respect_gitignore = ? AND recursive = ? AND mode = ?
            """,
            (key, model, include_flag, gitignore_flag, recursive_flag, mode),
        ).fetchone()
        if meta is None:
            raise FileNotFoundError(db_path)
        version = int(meta["version"] or 0)
        if version < CACHE_VERSION:
            raise FileNotFoundError(db_path)

        index_id = meta["id"]
        dimension = int(meta["dimension"])
        chunk_count = conn.execute(
            "SELECT COUNT(*) AS count FROM indexed_chunk WHERE index_id = ?",
            (index_id,),
        ).fetchone()["count"]
        chunk_total = int(chunk_count or 0)

        if chunk_total == 0 or dimension == 0:
            empty = np.empty((0, dimension), dtype=np.float32)
            metadata = {
                "index_id": int(index_id),
                "version": meta["version"],
                "generated_at": meta["generated_at"],
                "root": meta["root_path"],
                "model": meta["model"],
                "include_hidden": bool(meta["include_hidden"]),
                "respect_gitignore": bool(meta["respect_gitignore"]),
                "recursive": bool(meta["recursive"]),
                "mode": meta["mode"],
                "dimension": meta["dimension"],
                "exclude_patterns": _deserialize_exclude_patterns(meta["exclude_patterns"]),
                "extensions": _deserialize_extensions(meta["extensions"]),
                "files": [],
                "chunks": [],
                "chunk_ids": [],
            }
            return [], empty, metadata

        embeddings = np.empty((chunk_total, dimension), dtype=np.float32)
        paths: list[Path] = []
        chunk_ids: list[int] = []
        file_snapshot: list[dict] = []
        file_meta_by_rel: dict[str, dict] = {}

        file_rows = conn.execute(
            """
            SELECT rel_path, abs_path, size_bytes, mtime
            FROM indexed_file
            WHERE index_id = ?
            """,
            (index_id,),
        ).fetchall()
        for row in file_rows:
            file_meta_by_rel[row["rel_path"]] = {
                "path": row["rel_path"],
                "absolute": row["abs_path"],
                "mtime": row["mtime"],
                "size": row["size_bytes"],
            }
        seen_files: set[str] = set()

        cursor = conn.execute(
            """
            SELECT c.id AS chunk_id, f.rel_path, e.vector_blob
            FROM indexed_chunk AS c
            JOIN indexed_file AS f ON f.id = c.file_id
            JOIN chunk_embedding AS e ON e.chunk_id = c.id
            WHERE c.index_id = ?
            ORDER BY c.position ASC
            """,
            (index_id,),
        )

        for idx, row in enumerate(cursor):
            rel_path = row["rel_path"]
            chunk_id = int(row["chunk_id"])
            vector = np.frombuffer(row["vector_blob"], dtype=np.float32)
            if vector.size != dimension:
                raise RuntimeError(
                    f"Cached embedding dimension {vector.size} does not match index metadata {dimension}"
                )
            embeddings[idx] = vector
            paths.append(root / Path(rel_path))
            chunk_ids.append(chunk_id)
            if rel_path not in seen_files:
                meta_row = file_meta_by_rel.get(rel_path)
                if meta_row is not None:
                    file_snapshot.append(meta_row)
                seen_files.add(rel_path)

        metadata = {
            "index_id": int(index_id),
            "version": meta["version"],
            "generated_at": meta["generated_at"],
            "root": meta["root_path"],
            "model": meta["model"],
            "include_hidden": bool(meta["include_hidden"]),
            "respect_gitignore": bool(meta["respect_gitignore"]),
            "recursive": bool(meta["recursive"]),
            "mode": meta["mode"],
            "dimension": meta["dimension"],
            "exclude_patterns": _deserialize_exclude_patterns(meta["exclude_patterns"]),
            "extensions": _deserialize_extensions(meta["extensions"]),
            "files": file_snapshot,
            "chunks": [],
            "chunk_ids": chunk_ids,
        }
        return paths, embeddings, metadata
    finally:
        conn.close()


def load_chunk_metadata(
    chunk_ids: Sequence[int],
    conn: sqlite3.Connection | None = None,
) -> dict[int, dict]:
    """Load cached chunk metadata keyed by chunk_id."""

    if not chunk_ids:
        return {}
    unique_ids: list[int] = []
    seen: set[int] = set()
    for value in chunk_ids:
        try:
            chunk_id = int(value)
        except (TypeError, ValueError):
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique_ids.append(chunk_id)
    if not unique_ids:
        return {}
    db_path = cache_db_path()
    owns_connection = conn is None
    try:
        connection = conn or _connect(db_path, readonly=True)
    except sqlite3.OperationalError:
        return {}
    try:
        try:
            _ensure_schema_readonly(
                connection,
                tables=("indexed_chunk", "chunk_meta"),
            )
        except sqlite3.OperationalError:
            return {}
        results: dict[int, dict] = {}
        for chunk in _chunk_values(unique_ids, 900):
            placeholders = ", ".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                SELECT c.id AS chunk_id, c.chunk_index, m.preview, m.label_hash, m.start_line, m.end_line
                FROM indexed_chunk AS c
                LEFT JOIN chunk_meta AS m ON m.chunk_id = c.id
                WHERE c.id IN ({placeholders})
                """,
                tuple(chunk),
            ).fetchall()
            for row in rows:
                results[int(row["chunk_id"])] = {
                    "chunk_index": int(row["chunk_index"]),
                    "preview": row["preview"],
                    "label_hash": row["label_hash"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                }
        return results
    finally:
        if owns_connection:
            connection.close()


def load_query_vector(
    index_id: int,
    query_hash: str,
    conn: sqlite3.Connection | None = None,
) -> np.ndarray | None:
    """Load a cached query embedding vector for *index_id*."""

    db_path = cache_db_path()
    owns_connection = conn is None
    try:
        connection = conn or _connect(db_path, readonly=True)
    except sqlite3.OperationalError:
        return None
    try:
        try:
            _ensure_schema_readonly(connection, tables=("query_cache",))
        except sqlite3.OperationalError:
            return None
        row = connection.execute(
            """
            SELECT query_vector
            FROM query_cache
            WHERE index_id = ? AND query_hash = ?
            """,
            (int(index_id), query_hash),
        ).fetchone()
        if row is None:
            return None
        blob = row["query_vector"]
        if not blob:
            return None
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size == 0:
            return None
        return vector
    finally:
        if owns_connection:
            connection.close()


def store_query_vector(
    index_id: int,
    query_hash: str,
    query_text: str,
    query_vector: np.ndarray,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Store *query_vector* for *query_text* under *index_id*."""

    db_path = cache_db_path()
    owns_connection = conn is None
    connection = conn or _connect(db_path)
    try:
        _ensure_schema(connection)
        created_at = datetime.now(timezone.utc).isoformat()
        vector_blob = np.asarray(query_vector, dtype=np.float32).tobytes()
        with connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO query_cache (
                    index_id,
                    query_hash,
                    query_text,
                    query_vector,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (int(index_id), query_hash, query_text, vector_blob, created_at),
            )
    finally:
        if owns_connection:
            connection.close()


def load_embedding_cache(
    model: str,
    text_hashes: Sequence[str],
    conn: sqlite3.Connection | None = None,
) -> dict[str, np.ndarray]:
    """Load cached embeddings keyed by (model, text_hash)."""

    unique_hashes = list(dict.fromkeys([value for value in text_hashes if value]))
    if not unique_hashes:
        return {}
    results = _load_embedding_memory_cache(model, unique_hashes)
    missing = [value for value in unique_hashes if value not in results]
    if not missing:
        return results
    db_path = cache_db_path()
    owns_connection = conn is None
    try:
        connection = conn or _connect(db_path, readonly=True)
    except sqlite3.OperationalError:
        return results
    try:
        try:
            _ensure_schema_readonly(connection, tables=("embedding_cache",))
        except sqlite3.OperationalError:
            return results
        disk_results: dict[str, np.ndarray] = {}
        for chunk in _chunk_values(missing, 900):
            placeholders = ", ".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                SELECT text_hash, vector_blob
                FROM embedding_cache
                WHERE model = ? AND text_hash IN ({placeholders})
                """,
                (model, *chunk),
            ).fetchall()
            for row in rows:
                blob = row["vector_blob"]
                if not blob:
                    continue
                vector = np.frombuffer(blob, dtype=np.float32)
                if vector.size == 0:
                    continue
                disk_results[row["text_hash"]] = vector
        if disk_results:
            _store_embedding_memory_cache(model=model, embeddings=disk_results)
            results.update(disk_results)
        return results
    finally:
        if owns_connection:
            connection.close()


def store_embedding_cache(
    *,
    model: str,
    embeddings: Mapping[str, np.ndarray],
    conn: sqlite3.Connection | None = None,
) -> None:
    """Store embedding vectors keyed by (model, text_hash)."""

    if not embeddings:
        return
    _store_embedding_memory_cache(model=model, embeddings=embeddings)
    db_path = cache_db_path()
    owns_connection = conn is None
    connection = conn or _connect(db_path)
    try:
        _ensure_schema(connection)
        created_at = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                model,
                text_hash,
                np.asarray(vector, dtype=np.float32).tobytes(),
                created_at,
            )
            for text_hash, vector in embeddings.items()
        ]
        with connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (
                    model,
                    text_hash,
                    vector_blob,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            _prune_embedding_cache(
                connection,
                ttl_days=EMBED_CACHE_TTL_DAYS,
                max_entries=EMBED_CACHE_MAX_ENTRIES,
            )
    finally:
        if owns_connection:
            connection.close()


def _prune_embedding_cache(
    conn: sqlite3.Connection,
    *,
    ttl_days: int,
    max_entries: int,
    now: datetime | None = None,
) -> None:
    if ttl_days > 0:
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=ttl_days)
        conn.execute(
            "DELETE FROM embedding_cache WHERE created_at < ?",
            (cutoff.isoformat(),),
        )
    if max_entries > 0:
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM embedding_cache"
        ).fetchone()
        total = int(row["total"] if row is not None else 0)
        overflow = total - max_entries
        if overflow > 0:
            conn.execute(
                """
                DELETE FROM embedding_cache
                WHERE id IN (
                    SELECT id FROM embedding_cache
                    ORDER BY created_at ASC, id ASC
                    LIMIT ?
                )
                """,
                (overflow,),
            )


def clear_index(
    root: Path,
    include_hidden: bool,
    mode: str,
    recursive: bool,
    model: str | None = None,
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
    *,
    respect_gitignore: bool = True,
) -> int:
    """Remove cached index entries for *root* (optionally filtered by *model*)."""
    db_path = cache_file(root, model or "_", include_hidden)
    if not db_path.exists():
        return 0

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(
            root,
            include_hidden,
            respect_gitignore,
            recursive,
            mode,
            extensions,
            exclude_patterns,
        )
        # when model is None we still need a mode; reuse provided mode
        if model is None:
            query = "DELETE FROM index_metadata WHERE cache_key = ? AND mode = ?"
            params = (key, mode)
        else:
            query = "DELETE FROM index_metadata WHERE cache_key = ? AND model = ? AND mode = ?"
            params = (key, model, mode)
        with conn:
            cursor = conn.execute(query, params)
        return cursor.rowcount
    finally:
        conn.close()


def list_cache_entries() -> list[dict[str, object]]:
    """Return metadata for every cached index currently stored."""

    db_path = cache_db_path()
    if not db_path.exists():
        return []

    try:
        conn = _connect(db_path, readonly=True)
    except sqlite3.OperationalError:
        return []
    try:
        try:
            _ensure_schema_readonly(conn, tables=("index_metadata", "indexed_file"))
        except sqlite3.OperationalError:
            return []
        rows = conn.execute(
            """
            SELECT
                root_path,
                model,
                include_hidden,
                respect_gitignore,
                recursive,
                mode,
                dimension,
                version,
                generated_at,
                exclude_patterns,
                extensions,
                (
                    SELECT COUNT(*)
                    FROM indexed_file
                    WHERE index_id = index_metadata.id
                ) AS file_count
            FROM index_metadata
            ORDER BY generated_at DESC
            """
        ).fetchall()

        entries: list[dict[str, object]] = []
        for row in rows:
            entries.append(
                {
                    "root_path": row["root_path"],
                    "model": row["model"],
                    "include_hidden": bool(row["include_hidden"]),
                    "respect_gitignore": bool(row["respect_gitignore"]),
                    "recursive": bool(row["recursive"]),
                    "mode": row["mode"],
                    "dimension": row["dimension"],
                    "version": row["version"],
                    "generated_at": row["generated_at"],
                    "exclude_patterns": _deserialize_exclude_patterns(
                        row["exclude_patterns"]
                    ),
                    "extensions": _deserialize_extensions(row["extensions"]),
                    "file_count": int(row["file_count"] or 0),
                }
            )
        return entries
    finally:
        conn.close()


def clear_all_cache() -> int:
    """Remove the entire cache database, returning number of entries removed."""

    db_path = cache_db_path()
    if not db_path.exists():
        return 0

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        count_row = conn.execute("SELECT COUNT(*) AS total FROM index_metadata").fetchone()
        total = int(count_row["total"] if count_row is not None else 0)
    finally:
        conn.close()

    if db_path.exists():
        db_path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{db_path}{suffix}")
        if sidecar.exists():
            sidecar.unlink()

    return total


def compare_snapshot(
    root: Path,
    include_hidden: bool,
    cached_files: Sequence[dict],
    *,
    recursive: bool,
    exclude_patterns: Sequence[str] | None = None,
    extensions: Sequence[str] | None = None,
    current_files: Sequence[Path] | None = None,
    respect_gitignore: bool = True,
) -> bool:
    """Return True if the current filesystem matches the cached snapshot."""
    if current_files is None:
        current_files = collect_files(
            root,
            include_hidden=include_hidden,
            recursive=recursive,
            extensions=extensions,
            exclude_patterns=exclude_patterns,
            respect_gitignore=respect_gitignore,
        )
    if len(current_files) != len(cached_files):
        return False
    cached_map = {
        entry["path"]: (entry["mtime"], entry.get("size"))
        for entry in cached_files
    }
    for file in current_files:
        rel = _relative_path(file, root)
        data = cached_map.get(rel)
        if data is None:
            return False
        cached_mtime, cached_size = data
        stat = file.stat()
        current_mtime = stat.st_mtime
        current_size = stat.st_size
        # allow drift due to filesystem precision (approx 0.5s on some platforms)
        if abs(current_mtime - cached_mtime) > 5e-1:
            if cached_size is not None and cached_size == current_size:
                continue
            return False
        if cached_size is not None and cached_size != current_size:
            return False
    return True


def _relative_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return str(rel)
