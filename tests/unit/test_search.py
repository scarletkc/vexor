from pathlib import Path
import numpy as np
import pytest
import string

from vexor.search import VexorSearcher


ALPHABET = string.ascii_lowercase
ALPHABET_INDEX = {ch: idx for idx, ch in enumerate(ALPHABET)}


def _text_to_vector(text: str) -> np.ndarray:
    vec = np.zeros(len(ALPHABET), dtype=np.float32)
    for char in text.lower():
        idx = ALPHABET_INDEX.get(char)
        if idx is not None:
            vec[idx] += 1.0
    return vec


class DummyBackend:
    def embed(self, texts):
        vectors = [_text_to_vector(text) for text in texts]
        return np.stack(vectors)


def test_search_returns_sorted_results(tmp_path):
    files = [tmp_path / name for name in ["alpha.txt", "beta.txt", "gamma.txt"]]
    for f in files:
        f.write_text("data")

    searcher = VexorSearcher(backend=DummyBackend())

    results = searcher.search("alpha", files, top_k=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].path.name == "alpha.txt"


def test_search_handles_no_files(tmp_path):
    searcher = VexorSearcher(backend=DummyBackend())

    assert searcher.search("query", [], top_k=5) == []


def test_search_validates_query(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("data")
    searcher = VexorSearcher(backend=DummyBackend())

    with pytest.raises(ValueError):
        searcher.search("   ", [file_path])
