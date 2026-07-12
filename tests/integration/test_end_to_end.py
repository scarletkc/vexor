import numpy as np
import string

import vexor.cache as cache
import vexor.search as search_module
from vexor.search import VexorSearcher
from vexor.services.index_service import IndexStatus, build_index
from vexor.utils import collect_files


ALPHABET = string.ascii_lowercase
ALPHABET_INDEX = {ch: idx for idx, ch in enumerate(ALPHABET)}


class DummyBackend:
    device = "dummy"

    def embed(self, texts):
        vectors = []
        for text in texts:
            vec = np.zeros(len(ALPHABET), dtype=np.float32)
            for char in text.lower():
                idx = ALPHABET_INDEX.get(char)
                if idx is not None:
                    vec[idx] += 1.0
            vectors.append(vec)
        return np.stack(vectors)

    def embed_texts(self, texts):
        return self.embed(texts)


def test_full_search_pipeline(tmp_path):
    files = [
        tmp_path / "config_loader.py",
        tmp_path / "utils_config.py",
        tmp_path / "unrelated.txt",
    ]
    for file in files:
        file.write_text("data")

    collected = collect_files(tmp_path)
    searcher = VexorSearcher(backend=DummyBackend())

    results = searcher.search("config", collected, top_k=2)

    assert len(results) == 2
    assert results[0].path.name in {"config_loader.py", "utils_config.py"}


def test_vexorignore_change_removes_file_from_cached_index(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(search_module, "VexorSearcher", lambda **_kwargs: DummyBackend())

    project = tmp_path / "project"
    project.mkdir()
    (project / "kept.txt").write_text("kept", encoding="utf-8")
    (project / "ignored.txt").write_text("ignored", encoding="utf-8")
    index_kwargs = {
        "include_hidden": False,
        "respect_gitignore": True,
        "mode": "name",
        "recursive": True,
        "model_name": "model",
        "batch_size": 0,
        "provider": "gemini",
        "base_url": None,
        "api_key": None,
    }

    first = build_index(project, **index_kwargs)
    assert first.status == IndexStatus.STORED
    paths, _, _ = cache.load_index_vectors(project, "model", False, "name", True)
    assert sorted(path.name for path in paths) == ["ignored.txt", "kept.txt"]

    (project / ".vexorignore").write_text("ignored.txt\n", encoding="utf-8")
    second = build_index(project, **index_kwargs)

    assert second.status == IndexStatus.STORED
    paths, _, _ = cache.load_index_vectors(project, "model", False, "name", True)
    assert [path.name for path in paths] == ["kept.txt"]
