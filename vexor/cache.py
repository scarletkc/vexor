"""Index cache helpers for Vexor backed by SQLite."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .utils import collect_files

CACHE_DIR = Path(os.path.expanduser("~")) / ".vexor"
CACHE_VERSION = 4
DB_FILENAME = "index.db"


@dataclass(slots=True)
class IndexedChunk:
    path: Path
    rel_path: str
    chunk_index: int
    preview: str
    embedding: Sequence[float]
    size_bytes: int | None = None
    mtime: float | None = None


def _cache_key(
    root: Path,
    include_hidden: bool,
    recursive: bool,
    mode: str,
    extensions: Sequence[str] | None = None,
) -> str:
    base = f"{root.resolve()}|hidden={include_hidden}|recursive={recursive}|mode={mode}"
    ext_key = _serialize_extensions(extensions)
    if ext_key:
        base = f"{base}|ext={ext_key}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return digest


def _serialize_extensions(extensions: Sequence[str] | None) -> str:
    if not extensions:
        return ""
    return ",".join(extensions)


def _deserialize_extensions(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    parts = [part for part in value.split(",") if part]
    return tuple(parts)


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def cache_db_path() -> Path:
    """Return the absolute path to the shared SQLite cache database."""

    ensure_cache_dir()
    return CACHE_DIR / DB_FILENAME


def cache_file(root: Path, model: str, include_hidden: bool) -> Path:  # pragma: no cover - kept for API parity
    """Return the on-disk cache artifact path (single SQLite DB)."""
    return cache_db_path()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS index_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key TEXT NOT NULL,
            root_path TEXT NOT NULL,
            model TEXT NOT NULL,
            include_hidden INTEGER NOT NULL,
            recursive INTEGER NOT NULL DEFAULT 1,
            mode TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            version INTEGER NOT NULL,
            generated_at TEXT NOT NULL,
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
            position INTEGER NOT NULL,
            preview TEXT DEFAULT '',
            chunk_index INTEGER NOT NULL DEFAULT 0,
            UNIQUE(index_id, rel_path, chunk_index)
        );

        CREATE TABLE IF NOT EXISTS file_embedding (
            file_id INTEGER PRIMARY KEY REFERENCES indexed_file(id) ON DELETE CASCADE,
            vector_blob BLOB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_indexed_file_order
            ON indexed_file(index_id, position);
        """
    )
    try:
        conn.execute(
            "ALTER TABLE index_metadata ADD COLUMN recursive INTEGER NOT NULL DEFAULT 1"
        )
    except sqlite3.OperationalError:
        # Column already exists; ignore error.
        pass
    try:
        conn.execute(
            "ALTER TABLE index_metadata ADD COLUMN mode TEXT NOT NULL DEFAULT 'name'"
        )
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute(
            "ALTER TABLE indexed_file ADD COLUMN preview TEXT DEFAULT ''"
        )
    except sqlite3.OperationalError:
        pass
    if not _table_has_column(conn, "indexed_file", "chunk_index"):
        _upgrade_indexed_file_with_chunk(conn)
    try:
        conn.execute(
            "ALTER TABLE index_metadata ADD COLUMN extensions TEXT DEFAULT ''"
        )
    except sqlite3.OperationalError:
        pass
    _cleanup_orphan_embeddings(conn)


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def _upgrade_indexed_file_with_chunk(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute("ALTER TABLE indexed_file RENAME TO indexed_file_legacy;")
    conn.executescript(
        """
        CREATE TABLE indexed_file (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id INTEGER NOT NULL REFERENCES index_metadata(id) ON DELETE CASCADE,
            rel_path TEXT NOT NULL,
            abs_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime REAL NOT NULL,
            position INTEGER NOT NULL,
            preview TEXT DEFAULT '',
            chunk_index INTEGER NOT NULL DEFAULT 0,
            UNIQUE(index_id, rel_path, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_indexed_file_order
            ON indexed_file(index_id, position);
        """
    )
    conn.execute(
        """
        INSERT INTO indexed_file (
            id,
            index_id,
            rel_path,
            abs_path,
            size_bytes,
            mtime,
            position,
            preview,
            chunk_index
        )
        SELECT
            id,
            index_id,
            rel_path,
            abs_path,
            size_bytes,
            mtime,
            position,
            preview,
            0
        FROM indexed_file_legacy;
        """
    )
    conn.execute("DROP TABLE indexed_file_legacy;")
    conn.execute("PRAGMA foreign_keys = ON;")


def _cleanup_orphan_embeddings(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            "DELETE FROM file_embedding WHERE file_id NOT IN (SELECT id FROM indexed_file)"
        )


def store_index(
    *,
    root: Path,
    model: str,
    include_hidden: bool,
    mode: str,
    recursive: bool,
    entries: Sequence[IndexedChunk],
    extensions: Sequence[str] | None = None,
) -> Path:
    db_path = cache_file(root, model, include_hidden)
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(root, include_hidden, recursive, mode, extensions)
        generated_at = datetime.now(timezone.utc).isoformat()
        dimension = int(len(entries[0].embedding) if entries else 0)
        include_flag = 1 if include_hidden else 0
        recursive_flag = 1 if recursive else 0
        extensions_value = _serialize_extensions(extensions)

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
                    recursive,
                    mode,
                    dimension,
                    version,
                    generated_at,
                    extensions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    str(root),
                    model,
                    include_flag,
                    recursive_flag,
                    mode,
                    dimension,
                    CACHE_VERSION,
                    generated_at,
                    extensions_value,
                ),
            )
            index_id = cursor.lastrowid

            file_rows: list[tuple] = []
            vector_blobs: list[bytes] = []
            for position, entry in enumerate(entries):
                size_bytes = entry.size_bytes
                mtime = entry.mtime
                if size_bytes is None or mtime is None:
                    stat = entry.path.stat()
                    size_bytes = stat.st_size
                    mtime = stat.st_mtime
                file_rows.append(
                    (
                        index_id,
                        entry.rel_path,
                        str(entry.path),
                        size_bytes,
                        mtime,
                        position,
                        entry.preview,
                        entry.chunk_index,
                    )
                )
                vector_blobs.append(
                    np.asarray(entry.embedding, dtype=np.float32).tobytes()
                )

            conn.executemany(
                """
                INSERT INTO indexed_file (
                    index_id,
                    rel_path,
                    abs_path,
                    size_bytes,
                    mtime,
                    position,
                    preview,
                    chunk_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                file_rows,
            )

            inserted_ids = conn.execute(
                "SELECT id FROM indexed_file WHERE index_id = ? ORDER BY position ASC",
                (index_id,),
            ).fetchall()
            conn.executemany(
                "INSERT OR REPLACE INTO file_embedding (file_id, vector_blob) VALUES (?, ?)",
                (
                    (row["id"], vector_blobs[idx])
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
    mode: str,
    recursive: bool,
    ordered_entries: Sequence[tuple[str, int]],
    changed_entries: Sequence[IndexedChunk],
    removed_rel_paths: Sequence[str],
    extensions: Sequence[str] | None = None,
) -> Path:
    """Apply incremental updates to an existing cached index."""

    db_path = cache_file(root, model, include_hidden)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(root, include_hidden, recursive, mode, extensions)
        include_flag = 1 if include_hidden else 0
        recursive_flag = 1 if recursive else 0

        with conn:
            conn.execute("BEGIN IMMEDIATE;")
            meta = conn.execute(
                """
                SELECT id, dimension
                FROM index_metadata
                WHERE cache_key = ? AND model = ? AND include_hidden = ? AND recursive = ? AND mode = ?
                """,
                (key, model, include_flag, recursive_flag, mode),
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
                    if entry.rel_path not in chunk_map:
                        chunk_map[entry.rel_path] = []
                    chunk_map[entry.rel_path].append(entry)

                for rel_path, chunk_list in chunk_map.items():
                    conn.execute(
                        "DELETE FROM indexed_file WHERE index_id = ? AND rel_path = ?",
                        (index_id, rel_path),
                    )
                    chunk_list.sort(key=lambda item: item.chunk_index)
                    file_rows: list[tuple] = []
                    vector_blobs: list[bytes] = []
                    for chunk in chunk_list:
                        vector = np.asarray(chunk.embedding, dtype=np.float32)
                        if vector_dimension is None:
                            vector_dimension = vector.shape[0]
                        size_bytes = chunk.size_bytes
                        mtime = chunk.mtime
                        if size_bytes is None or mtime is None:
                            stat = chunk.path.stat()
                            size_bytes = stat.st_size
                            mtime = stat.st_mtime
                        file_rows.append(
                            (
                                index_id,
                                rel_path,
                                str(chunk.path),
                                size_bytes,
                                mtime,
                                0,
                                chunk.preview,
                                chunk.chunk_index,
                            )
                        )
                        vector_blobs.append(vector.tobytes())

                    conn.executemany(
                        """
                        INSERT INTO indexed_file (
                            index_id,
                            rel_path,
                            abs_path,
                            size_bytes,
                            mtime,
                            position,
                            preview,
                            chunk_index
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        file_rows,
                    )

                    inserted_ids = conn.execute(
                        """
                        SELECT id FROM indexed_file
                        WHERE index_id = ? AND rel_path = ?
                        ORDER BY chunk_index ASC
                        """,
                        (index_id, rel_path),
                    ).fetchall()
                    conn.executemany(
                        "INSERT INTO file_embedding (file_id, vector_blob) VALUES (?, ?)",
                        (
                            (row["id"], vector_blobs[idx])
                            for idx, row in enumerate(inserted_ids)
                        ),
                    )

            for position, (rel_path, chunk_index) in enumerate(ordered_entries):
                conn.execute(
                    """
                    UPDATE indexed_file
                    SET position = ?
                    WHERE index_id = ? AND rel_path = ? AND chunk_index = ?
                    """,
                    (position, index_id, rel_path, chunk_index),
                )

            generated_at = datetime.now(timezone.utc).isoformat()
            new_dimension = vector_dimension or existing_dimension
            conn.execute(
                """
                UPDATE index_metadata
                SET generated_at = ?, dimension = ?
                WHERE id = ?
                """,
                (generated_at, new_dimension, index_id),
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
    extensions: Sequence[str] | None = None,
) -> dict:
    db_path = cache_file(root, model, include_hidden)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(root, include_hidden, recursive, mode, extensions)
        include_flag = 1 if include_hidden else 0
        recursive_flag = 1 if recursive else 0
        meta = conn.execute(
            """
            SELECT id, root_path, model, include_hidden, recursive, mode, dimension, version, generated_at, extensions
            FROM index_metadata
            WHERE cache_key = ? AND model = ? AND include_hidden = ? AND recursive = ? AND mode = ?
            """,
            (key, model, include_flag, recursive_flag, mode),
        ).fetchone()
        if meta is None:
            raise FileNotFoundError(db_path)

        rows = conn.execute(
            """
            SELECT rel_path, abs_path, size_bytes, mtime, preview, chunk_index
            FROM indexed_file
            WHERE index_id = ?
            ORDER BY position ASC
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
                    "chunk_index": chunk_index,
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
            "recursive": bool(meta["recursive"]),
            "mode": meta["mode"],
            "dimension": meta["dimension"],
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
    extensions: Sequence[str] | None = None,
):
    db_path = cache_file(root, model, include_hidden)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(root, include_hidden, recursive, mode, extensions)
        include_flag = 1 if include_hidden else 0
        recursive_flag = 1 if recursive else 0
        meta = conn.execute(
            """
            SELECT id, root_path, model, include_hidden, recursive, mode, dimension, version, generated_at, extensions
            FROM index_metadata
            WHERE cache_key = ? AND model = ? AND include_hidden = ? AND recursive = ? AND mode = ?
            """,
            (key, model, include_flag, recursive_flag, mode),
        ).fetchone()
        if meta is None:
            raise FileNotFoundError(db_path)

        index_id = meta["id"]
        dimension = int(meta["dimension"])
        chunk_count = conn.execute(
            "SELECT COUNT(*) AS count FROM indexed_file WHERE index_id = ?",
            (index_id,),
        ).fetchone()["count"]
        chunk_total = int(chunk_count or 0)

        if chunk_total == 0 or dimension == 0:
            empty = np.empty((0, dimension), dtype=np.float32)
            metadata = {
                "version": meta["version"],
                "generated_at": meta["generated_at"],
                "root": meta["root_path"],
                "model": meta["model"],
                "include_hidden": bool(meta["include_hidden"]),
                "recursive": bool(meta["recursive"]),
                "mode": meta["mode"],
                "dimension": meta["dimension"],
                "extensions": _deserialize_extensions(meta["extensions"]),
                "files": [],
                "chunks": [],
            }
            return [], empty, metadata

        embeddings = np.empty((chunk_total, dimension), dtype=np.float32)
        paths: list[Path] = []
        chunk_entries: list[dict] = []
        file_snapshot: dict[str, dict] = {}

        cursor = conn.execute(
            """
            SELECT f.rel_path, f.abs_path, f.size_bytes, f.mtime, f.preview, f.chunk_index, e.vector_blob
            FROM indexed_file AS f
            JOIN file_embedding AS e ON e.file_id = f.id
            WHERE f.index_id = ?
            ORDER BY f.position ASC
            """,
            (index_id,),
        )

        for idx, row in enumerate(cursor):
            rel_path = row["rel_path"]
            vector = np.frombuffer(row["vector_blob"], dtype=np.float32)
            if vector.size != dimension:
                raise RuntimeError(
                    f"Cached embedding dimension {vector.size} does not match index metadata {dimension}"
                )
            embeddings[idx] = vector
            paths.append(root / Path(rel_path))
            chunk_index = int(row["chunk_index"])
            chunk_entries.append(
                {
                    "path": rel_path,
                    "absolute": row["abs_path"],
                    "mtime": row["mtime"],
                    "size": row["size_bytes"],
                    "preview": row["preview"],
                    "chunk_index": chunk_index,
                }
            )
            if rel_path not in file_snapshot:
                file_snapshot[rel_path] = {
                    "path": rel_path,
                    "absolute": row["abs_path"],
                    "mtime": row["mtime"],
                    "size": row["size_bytes"],
                }

        metadata = {
            "version": meta["version"],
            "generated_at": meta["generated_at"],
            "root": meta["root_path"],
            "model": meta["model"],
            "include_hidden": bool(meta["include_hidden"]),
            "recursive": bool(meta["recursive"]),
            "mode": meta["mode"],
            "dimension": meta["dimension"],
            "extensions": _deserialize_extensions(meta["extensions"]),
            "files": list(file_snapshot.values()),
            "chunks": chunk_entries,
        }
        return paths, embeddings, metadata
    finally:
        conn.close()


def clear_index(
    root: Path,
    include_hidden: bool,
    mode: str,
    recursive: bool,
    model: str | None = None,
    extensions: Sequence[str] | None = None,
) -> int:
    """Remove cached index entries for *root* (optionally filtered by *model*)."""
    db_path = cache_file(root, model or "_", include_hidden)
    if not db_path.exists():
        return 0

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(root, include_hidden, recursive, mode, extensions)
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

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT
                root_path,
                model,
                include_hidden,
                recursive,
                mode,
                dimension,
                version,
                generated_at,
                extensions,
                (
                    SELECT COUNT(DISTINCT rel_path)
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
                    "recursive": bool(row["recursive"]),
                    "mode": row["mode"],
                    "dimension": row["dimension"],
                    "version": row["version"],
                    "generated_at": row["generated_at"],
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
    extensions: Sequence[str] | None = None,
    current_files: Sequence[Path] | None = None,
) -> bool:
    """Return True if the current filesystem matches the cached snapshot."""
    if current_files is None:
        current_files = collect_files(
            root,
            include_hidden=include_hidden,
            recursive=recursive,
            extensions=extensions,
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
