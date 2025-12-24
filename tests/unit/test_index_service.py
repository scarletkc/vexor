from pathlib import Path

import numpy as np

import vexor.cache as cache
from vexor.services import index_service
from vexor.services.index_service import IndexStatus, build_index

import sqlite3


class DummySearcher:
    calls = []

    def __init__(self, *args, **kwargs):
        self.device = "dummy"

    def embed_texts(self, texts):
        DummySearcher.calls.append(list(texts))
        length = len(texts)
        if not length:
            return np.zeros((0, 3), dtype=np.float32)
        data = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
        return data


def _patch_cache_dir(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)


def test_build_index_runs_incremental_update(tmp_path, monkeypatch):
    _patch_cache_dir(tmp_path, monkeypatch)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)
    DummySearcher.calls = []

    root = tmp_path / "project"
    root.mkdir()
    file_a = root / "a.txt"
    file_b = root / "b.txt"
    file_a.write_text("a")
    file_b.write_text("b")

    kwargs = dict(provider="gemini", base_url=None, api_key=None)
    first = build_index(
        root,
        include_hidden=False,
        mode="name",
        recursive=True,
        model_name="model",
        batch_size=0,
        **kwargs,
    )
    assert first.status == IndexStatus.STORED
    assert len(DummySearcher.calls) == 1
    assert len(DummySearcher.calls[0]) == 2

    DummySearcher.calls = []
    file_a.write_text("updated")

    second = build_index(
        root,
        include_hidden=False,
        mode="name",
        recursive=True,
        model_name="model",
        batch_size=0,
        **kwargs,
    )
    assert second.status == IndexStatus.STORED
    assert DummySearcher.calls == []

    DummySearcher.calls = []
    file_c = root / "c.txt"
    file_c.write_text("c")

    third = build_index(
        root,
        include_hidden=False,
        mode="name",
        recursive=True,
        model_name="model",
        batch_size=0,
        **kwargs,
    )
    assert third.status == IndexStatus.STORED
    assert len(DummySearcher.calls) == 1
    assert len(DummySearcher.calls[0]) == 1  # only c.txt embedded

    paths, _, _ = cache.load_index_vectors(root, "model", False, "name", True)
    assert sorted(p.name for p in paths) == ["a.txt", "b.txt", "c.txt"]


def test_embed_labels_with_cache_reuses_embeddings(tmp_path, monkeypatch):
    _patch_cache_dir(tmp_path, monkeypatch)
    DummySearcher.calls = []
    searcher = DummySearcher()
    labels = ["alpha", "beta"]

    first = index_service._embed_labels_with_cache(  # type: ignore[attr-defined]
        searcher=searcher,
        model_name="model",
        labels=labels,
    )

    assert len(DummySearcher.calls) == 1
    DummySearcher.calls = []

    second = index_service._embed_labels_with_cache(  # type: ignore[attr-defined]
        searcher=searcher,
        model_name="model",
        labels=labels,
    )

    assert DummySearcher.calls == []
    assert np.allclose(first, second)


def test_build_index_falls_back_to_full_rebuild(tmp_path, monkeypatch):
    _patch_cache_dir(tmp_path, monkeypatch)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)
    DummySearcher.calls = []

    root = tmp_path / "project"
    root.mkdir()
    files = []
    for name in ["a.txt", "b.txt", "c.txt", "d.txt"]:
        path = root / name
        path.write_text(name)
        files.append(path)

    build_index(root, include_hidden=False, mode="name", recursive=True, model_name="model", batch_size=0, provider="gemini", base_url=None, api_key=None)
    DummySearcher.calls = []

    for file in files[:3]:
        file.write_text(file.read_text() + "!")

    build_index(root, include_hidden=False, mode="name", recursive=True, model_name="model", batch_size=0, provider="gemini", base_url=None, api_key=None)
    assert DummySearcher.calls == []


def test_build_index_backfills_line_metadata_when_missing(tmp_path, monkeypatch):
    _patch_cache_dir(tmp_path, monkeypatch)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)
    DummySearcher.calls = []

    root = tmp_path / "project"
    root.mkdir()
    py_path = root / "sample.py"
    py_path.write_text(
        """\"\"\"Doc.\"\"\"\n\nX = 1\n\n\ndef foo():\n    return X\n""",
        encoding="utf-8",
    )

    kwargs = dict(provider="gemini", base_url=None, api_key=None)
    first = build_index(
        root,
        include_hidden=False,
        mode="code",
        recursive=True,
        model_name="model",
        batch_size=0,
        **kwargs,
    )
    assert first.status == IndexStatus.STORED

    # Simulate an old cache that lacks line metadata.
    db_path = cache.cache_db_path()
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        meta_row = conn.execute(
            """
            SELECT id FROM index_metadata
            WHERE root_path = ? AND model = ? AND include_hidden = 0 AND respect_gitignore = 1
              AND recursive = 1 AND mode = 'code'
            """,
            (str(root), "model"),
        ).fetchone()
        assert meta_row is not None
        index_id = int(meta_row["id"])
        conn.execute(
            """
            UPDATE chunk_meta
            SET start_line = NULL, end_line = NULL
            WHERE chunk_id IN (
                SELECT id FROM indexed_chunk WHERE index_id = ?
            )
            """,
            (index_id,),
        )
        conn.commit()
    finally:
        conn.close()

    second = build_index(
        root,
        include_hidden=False,
        mode="code",
        recursive=True,
        model_name="model",
        batch_size=0,
        **kwargs,
    )
    assert second.status == IndexStatus.STORED

    meta = cache.load_index(root, "model", False, "code", True)
    chunks = [entry for entry in meta["chunks"] if entry["path"] == "sample.py"]
    assert chunks
    assert any(entry["start_line"] is not None and entry["end_line"] is not None for entry in chunks)


def test_build_index_returns_empty_when_no_files(tmp_path, monkeypatch):
    _patch_cache_dir(tmp_path, monkeypatch)
    monkeypatch.setattr("vexor.utils.collect_files", lambda *_args, **_kwargs: [])

    result = build_index(
        tmp_path,
        include_hidden=False,
        mode="name",
        recursive=True,
        model_name="model",
        batch_size=0,
        provider="gemini",
        base_url=None,
        api_key=None,
    )
    assert result.status == IndexStatus.EMPTY


def test_build_index_returns_up_to_date_when_no_changes(tmp_path, monkeypatch):
    _patch_cache_dir(tmp_path, monkeypatch)
    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)

    root = tmp_path / "project"
    root.mkdir()
    file_a = root / "a.txt"
    file_a.write_text("a", encoding="utf-8")

    monkeypatch.setattr("vexor.utils.collect_files", lambda *_args, **_kwargs: [file_a])
    monkeypatch.setattr(index_service, "_diff_cached_files", lambda *_args, **_kwargs: index_service.FileDiff())

    class DummyStrategy:
        def payloads_for_files(self, files):
            raise AssertionError("payloads_for_files should not be called when no changes")

    monkeypatch.setattr(index_service, "get_strategy", lambda *_args, **_kwargs: DummyStrategy())

    monkeypatch.setattr(
        index_service,
        "load_index_metadata_safe",
        lambda *_args, **_kwargs: {
            "version": cache.CACHE_VERSION,
            "files": [{"path": "a.txt"}],
            "chunks": [{"path": "a.txt", "chunk_index": 0, "start_line": 1, "end_line": 1}],
        },
    )

    result = build_index(
        root,
        include_hidden=False,
        mode="name",
        recursive=True,
        model_name="model",
        batch_size=0,
        provider="gemini",
        base_url=None,
        api_key=None,
    )

    assert result.status == IndexStatus.UP_TO_DATE
    assert result.files_indexed == 1


def test_clear_index_entries_delegates_to_cache(tmp_path, monkeypatch):
    called = {}

    def fake_clear_index(*_args, **_kwargs):
        called["ok"] = True
        return 3

    monkeypatch.setattr("vexor.cache.clear_index", fake_clear_index)

    removed = index_service.clear_index_entries(
        tmp_path,
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=True,
        model=None,
    )
    assert called["ok"] is True
    assert removed == 3


def test_relative_to_root_handles_unrelated_path(tmp_path):
    rel = index_service._relative_to_root(Path("/tmp/elsewhere"), tmp_path)  # type: ignore[attr-defined]
    assert "elsewhere" in rel


def test_stat_for_path_without_cache(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("x", encoding="utf-8")
    stat = index_service._stat_for_path(file_path, cache=None)  # type: ignore[attr-defined]
    assert stat.st_size == 1
