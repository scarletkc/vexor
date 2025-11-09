import numpy as np
import string

from vexor.search import VexorSearcher
from vexor.utils import collect_files


ALPHABET = string.ascii_lowercase
ALPHABET_INDEX = {ch: idx for idx, ch in enumerate(ALPHABET)}


class DummyBackend:
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
