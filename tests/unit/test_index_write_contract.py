import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from vexor import cache


def entry(root: Path, name: str, chunk: int, term: str | None) -> cache.IndexedChunk:
    path = root / name
    stat = path.stat()
    return cache.IndexedChunk(
        path=path,
        rel_path=name,
        chunk_index=chunk,
        preview=term or "no terms",
        label_hash=f"{name}-{chunk}-{term}",
        embedding=[float(chunk), 1.0],
        size_bytes=stat.st_size,
        mtime=stat.st_mtime,
        start_line=chunk + 1,
        end_line=chunk + 2,
        bm25_terms={term: 2} if term else None,
        bm25_doc_len=2 if term else None,
    )


def snapshot(root: Path) -> tuple:
    paths, vectors, _ = cache.load_index_vectors(root, "model", False, "full", True)
    with closing(sqlite3.connect(cache.cache_db_path())) as conn, conn:
        chunks = conn.execute("""
            SELECT f.rel_path, c.chunk_index, c.position, m.preview, m.label_hash,
                   m.start_line, m.end_line, d.token_count
            FROM indexed_chunk c JOIN indexed_file f ON f.id = c.file_id
            JOIN chunk_meta m ON m.chunk_id = c.id
            LEFT JOIN bm25_doc d ON d.chunk_id = c.id ORDER BY c.position
        """).fetchall()
        postings = conn.execute("""
            SELECT f.rel_path, c.chunk_index, p.term, p.tf
            FROM bm25_posting p JOIN indexed_chunk c ON c.id = p.chunk_id
            JOIN indexed_file f ON f.id = c.file_id
            ORDER BY f.rel_path, c.chunk_index, p.term
        """).fetchall()
    return [path.name for path in paths], vectors.tolist(), chunks, postings


def test_incremental_rows_and_vectors_match_full_rebuild(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
        (root / name).write_text(name, encoding="utf-8")
    initial = [
        entry(root, "a.txt", 0, "old"),
        entry(root, "a.txt", 2, "old"),
        entry(root, "b.txt", 0, "removed"),
        entry(root, "c.txt", 0, None),
    ]
    a0, a2 = entry(root, "a.txt", 0, "new"), entry(root, "a.txt", 2, "new")
    d1 = entry(root, "d.txt", 1, "added")
    c0 = entry(root, "c.txt", 0, None)
    c0.preview, c0.start_line, c0.end_line = "moved", 10, 11
    final = [d1, a2, c0, a0]
    options = {
        "root": root,
        "model": "model",
        "include_hidden": False,
        "mode": "full",
        "recursive": True,
    }
    with cache.cache_dir_context(tmp_path / "incremental"):
        cache.store_index(**options, entries=initial)
        cache.apply_index_updates(
            **options,
            changed_entries=[a2, d1, a0],
            removed_rel_paths=["b.txt"],
            ordered_entries=[(e.rel_path, e.chunk_index) for e in final],
            touched_entries=[
                (
                    c0.rel_path,
                    c0.chunk_index,
                    c0.size_bytes,
                    c0.mtime,
                    c0.preview,
                    c0.start_line,
                    c0.end_line,
                    c0.label_hash,
                )
            ],
        )
        incremental = snapshot(root)
    with cache.cache_dir_context(tmp_path / "rebuilt"):
        cache.store_index(**options, entries=final)
        assert snapshot(root) == incremental


@pytest.mark.parametrize("incremental", [False, True])
def test_failed_posting_write_rolls_back_rows_and_removes_new_sidecar(
    tmp_path: Path,
    incremental: bool,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    options = {
        "root": root,
        "model": "model",
        "include_hidden": False,
        "mode": "full",
        "recursive": True,
    }
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**options, entries=[entry(root, "a.txt", 0, "original")])
        before = snapshot(root)
        sidecars = set((tmp_path / "cache" / "vectors").glob("*.npy"))
        with closing(sqlite3.connect(cache.cache_db_path())) as conn, conn:
            conn.execute("""
                CREATE TRIGGER reject_posting BEFORE INSERT ON bm25_posting
                WHEN NEW.term = 'reject'
                BEGIN SELECT RAISE(ABORT, 'rejected posting'); END
            """)
        rejected = entry(root, "a.txt", 0, "reject")
        # Fail after SQLite has consumed part of the posting stream.
        rejected.bm25_terms = {"accepted": 1, "reject": 2}
        rejected.bm25_doc_len = 3
        with pytest.raises(sqlite3.IntegrityError, match="rejected posting"):
            if incremental:
                cache.apply_index_updates(
                    **options,
                    changed_entries=[rejected],
                    removed_rel_paths=[],
                    ordered_entries=[("a.txt", 0)],
                )
            else:
                cache.store_index(**options, entries=[rejected])
        assert snapshot(root) == before
        assert set((tmp_path / "cache" / "vectors").glob("*.npy")) == sidecars


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("file_count", [1, 64, 1001])
def test_postings_are_consumed_without_buffering_the_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    incremental: bool,
    file_count: int,
) -> None:
    """Bound outstanding posting rows independently of corpus size or allocator."""
    produced = consumed = 0

    class StreamingTerms(dict[str, int]):
        def items(self) -> Iterator[tuple[str, int]]:
            nonlocal produced
            for term, frequency in super().items():
                # Each row must reach SQLite before the next one is produced.
                assert produced == consumed, "posting rows were buffered before insertion"
                produced += 1
                yield term, frequency

    def record_posting() -> int:
        nonlocal consumed
        consumed += 1
        return consumed

    connect = cache._connect

    def observed_connect(db_path: Path) -> sqlite3.Connection:
        conn = connect(db_path)
        conn.create_function("record_posting", 0, record_posting)
        conn.execute("""
            CREATE TEMP TRIGGER observe_posting AFTER INSERT ON bm25_posting
            BEGIN SELECT record_posting(); END
        """)
        return conn

    root = tmp_path / "source"
    root.mkdir()
    (root / "unchanged.txt").write_text("unchanged", encoding="utf-8")
    unchanged = entry(root, "unchanged.txt", 0, "unchanged")
    changed: list[cache.IndexedChunk] = []
    for index in range(file_count):
        name = f"{index}.txt"
        (root / name).write_text(name, encoding="utf-8")
        item = entry(root, name, index % 3, "changed")
        item.bm25_terms = StreamingTerms({f"term-{term}": 2 for term in range(300)})
        item.bm25_doc_len = 600
        changed.append(item)
    final = [*reversed(changed), unchanged]
    options = {
        "root": root,
        "model": "model",
        "include_hidden": False,
        "mode": "full",
        "recursive": True,
    }
    with cache.cache_dir_context(tmp_path / "cache"):
        initial = [
            entry(root, item.rel_path, item.chunk_index, "old") for item in changed
        ]
        cache.store_index(**options, entries=[*initial, unchanged])
        monkeypatch.setattr(cache, "_connect", observed_connect)
        if incremental:
            cache.apply_index_updates(
                **options,
                changed_entries=changed,
                removed_rel_paths=[],
                ordered_entries=[(item.rel_path, item.chunk_index) for item in final],
            )
        else:
            cache.store_index(**options, entries=final)
        assert produced == file_count * 300
        assert consumed == produced + (0 if incremental else 1)
        monkeypatch.setattr(cache, "_connect", connect)
        paths, vectors, chunks, postings = snapshot(root)
        assert paths == [item.rel_path for item in final]
        assert vectors == [item.embedding for item in final]
        assert len(postings) == file_count * 300 + 1
        assert chunks == [
            (
                item.rel_path, item.chunk_index, position, item.preview, item.label_hash,
                item.start_line, item.end_line, item.bm25_doc_len,
            )
            for position, item in enumerate(final)
        ]
