import numpy as np

import vexor.cache as cache
from vexor.services.index_service import build_index, IndexStatus


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

    kwargs = dict(provider="gemini", base_url=None)
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
    assert len(DummySearcher.calls) == 1
    assert len(DummySearcher.calls[0]) == 1  # only a.txt re-embedded

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

    build_index(root, include_hidden=False, mode="name", recursive=True, model_name="model", batch_size=0, provider="gemini", base_url=None)
    DummySearcher.calls = []

    for file in files[:3]:
        file.write_text(file.read_text() + "!")

    build_index(root, include_hidden=False, mode="name", recursive=True, model_name="model", batch_size=0, provider="gemini", base_url=None)
    assert len(DummySearcher.calls) == 1
    assert len(DummySearcher.calls[0]) == 4  # full rebuild embeds every file
