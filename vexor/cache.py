"""Index cache helpers for Vexor backed by SQLite."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .utils import collect_files

CACHE_DIR = Path(os.path.expanduser("~")) / ".vexor"
CACHE_VERSION = 2
DB_FILENAME = "index.db"


def _cache_key(root: Path, include_hidden: bool, recursive: bool) -> str:
    digest = hashlib.sha1(
        f"{root.resolve()}|hidden={include_hidden}|recursive={recursive}".encode("utf-8")
    ).hexdigest()
    return digest


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def cache_file(root: Path, model: str, include_hidden: bool) -> Path:  # pragma: no cover - kept for API parity
    """Return the on-disk cache artifact path (single SQLite DB)."""
    ensure_cache_dir()
    return CACHE_DIR / DB_FILENAME


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
            dimension INTEGER NOT NULL,
            version INTEGER NOT NULL,
            generated_at TEXT NOT NULL,
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
            UNIQUE(index_id, rel_path)
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


def store_index(
    *,
    root: Path,
    model: str,
    include_hidden: bool,
    recursive: bool,
    files: Sequence[Path],
    embeddings: np.ndarray,
) -> Path:
    db_path = cache_file(root, model, include_hidden)
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(root, include_hidden, recursive)
        generated_at = datetime.now(timezone.utc).isoformat()
        dimension = int(embeddings.shape[1] if embeddings.size else 0)
        include_flag = 1 if include_hidden else 0
        recursive_flag = 1 if recursive else 0

        with conn:
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
                    dimension,
                    version,
                    generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    str(root),
                    model,
                    include_flag,
                    recursive_flag,
                    dimension,
                    CACHE_VERSION,
                    generated_at,
                ),
            )
            index_id = cursor.lastrowid

            for position, file in enumerate(files):
                stat = file.stat()
                try:
                    rel_path = file.relative_to(root)
                except ValueError:
                    rel_path = file
                file_cursor = conn.execute(
                    """
                    INSERT INTO indexed_file (
                        index_id,
                        rel_path,
                        abs_path,
                        size_bytes,
                        mtime,
                        position
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        index_id,
                        str(rel_path),
                        str(file),
                        stat.st_size,
                        stat.st_mtime,
                        position,
                    ),
                )
                vector_blob = embeddings[position].astype(np.float32).tobytes()
                conn.execute(
                    "INSERT INTO file_embedding (file_id, vector_blob) VALUES (?, ?)",
                    (file_cursor.lastrowid, vector_blob),
                )

        return db_path
    finally:
        conn.close()


def load_index(root: Path, model: str, include_hidden: bool, recursive: bool) -> dict:
    db_path = cache_file(root, model, include_hidden)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(root, include_hidden, recursive)
        include_flag = 1 if include_hidden else 0
        recursive_flag = 1 if recursive else 0
        meta = conn.execute(
            """
            SELECT id, root_path, model, include_hidden, recursive, dimension, version, generated_at
            FROM index_metadata
            WHERE cache_key = ? AND model = ? AND include_hidden = ? AND recursive = ?
            """,
            (key, model, include_flag, recursive_flag),
        ).fetchone()
        if meta is None:
            raise FileNotFoundError(db_path)

        files = conn.execute(
            """
            SELECT f.rel_path, f.abs_path, f.size_bytes, f.mtime, e.vector_blob
            FROM indexed_file AS f
            JOIN file_embedding AS e ON e.file_id = f.id
            WHERE f.index_id = ?
            ORDER BY f.position ASC
            """,
            (meta["id"],),
        ).fetchall()

        serialized_files = []
        for row in files:
            vector = np.frombuffer(row["vector_blob"], dtype=np.float32)
            serialized_files.append(
                {
                    "path": row["rel_path"],
                    "absolute": row["abs_path"],
                    "mtime": row["mtime"],
                    "size": row["size_bytes"],
                    "embedding": vector.tolist(),
                }
            )

        return {
            "version": meta["version"],
            "generated_at": meta["generated_at"],
            "root": meta["root_path"],
            "model": meta["model"],
            "include_hidden": bool(meta["include_hidden"]),
            "recursive": bool(meta["recursive"]),
            "dimension": meta["dimension"],
            "files": serialized_files,
        }
    finally:
        conn.close()


def load_index_vectors(root: Path, model: str, include_hidden: bool, recursive: bool):
    data = load_index(root, model, include_hidden, recursive)
    files = data.get("files", [])
    paths = [root / Path(entry["path"]) for entry in files]
    embeddings = np.asarray([entry["embedding"] for entry in files], dtype=np.float32)
    return paths, embeddings, data


def clear_index(
    root: Path,
    include_hidden: bool,
    recursive: bool,
    model: str | None = None,
) -> int:
    """Remove cached index entries for *root* (optionally filtered by *model*)."""
    db_path = cache_file(root, model or "_", include_hidden)
    if not db_path.exists():
        return 0

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        key = _cache_key(root, include_hidden, recursive)
        if model is None:
            query = "DELETE FROM index_metadata WHERE cache_key = ?"
            params = (key,)
        else:
            query = "DELETE FROM index_metadata WHERE cache_key = ? AND model = ?"
            params = (key, model)
        with conn:
            cursor = conn.execute(query, params)
        return cursor.rowcount
    finally:
        conn.close()


def compare_snapshot(
    root: Path,
    include_hidden: bool,
    cached_files: Sequence[dict],
    *,
    recursive: bool,
    current_files: Sequence[Path] | None = None,
) -> bool:
    """Return True if the current filesystem matches the cached snapshot."""
    if current_files is None:
        current_files = collect_files(
            root,
            include_hidden=include_hidden,
            recursive=recursive,
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
