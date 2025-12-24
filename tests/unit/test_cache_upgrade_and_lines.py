from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import vexor.cache as cache


def test_ensure_schema_upgrades_legacy_indexed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    db_path = cache.cache_db_path()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE index_metadata (
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
                extensions TEXT DEFAULT ''
            );

            -- Legacy indexed_file schema without chunk_index/start/end lines.
            CREATE TABLE indexed_file (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_id INTEGER NOT NULL,
                rel_path TEXT NOT NULL,
                abs_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime REAL NOT NULL,
                position INTEGER NOT NULL,
                preview TEXT DEFAULT ''
            );

            CREATE TABLE file_embedding (
                file_id INTEGER PRIMARY KEY,
                vector_blob BLOB NOT NULL
            );
            """
        )
        cache._ensure_schema(conn)  # type: ignore[attr-defined]
        columns = [row[1] for row in conn.execute("PRAGMA table_info(indexed_file)").fetchall()]
        assert "position" not in columns
        assert "chunk_index" not in columns
        assert "preview" not in columns
        table_names = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "indexed_chunk" in table_names
        assert "chunk_meta" in table_names
    finally:
        conn.close()


def test_backfill_chunk_lines_updates_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    root = tmp_path / "project"
    root.mkdir()
    file_path = root / "a.txt"
    file_path.write_text("line1\nline2\n", encoding="utf-8")

    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    entry = cache.IndexedChunk(
        path=file_path,
        rel_path="a.txt",
        chunk_index=0,
        preview="p",
        embedding=embeddings[0],
        start_line=None,
        end_line=None,
    )
    cache.store_index(
        root=root,
        model="m",
        include_hidden=False,
        mode="full",
        recursive=True,
        entries=[entry],
    )

    cache.backfill_chunk_lines(
        root=root,
        model="m",
        include_hidden=False,
        mode="full",
        recursive=True,
        updates=[("a.txt", 0, 10, 20)],
    )

    meta = cache.load_index(
        root=root,
        model="m",
        include_hidden=False,
        mode="full",
        recursive=True,
    )
    assert meta["chunks"][0]["start_line"] == 10
    assert meta["chunks"][0]["end_line"] == 20


def test_backfill_chunk_lines_missing_db_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        cache.backfill_chunk_lines(
            root=root,
            model="m",
            include_hidden=False,
            mode="full",
            recursive=True,
            updates=[],
        )
