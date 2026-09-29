import pytest

from vexor.text_chunks import chunk_text


@pytest.mark.parametrize("text,size,overlap,expected", [
    (" \r\n ", 3, 1, []),
    ("ab\r\ncd", 3, 1, ["ab", "cd"]),
    ("abcdef", 4, 1, ["abcd", "def"]),
    ("abcdef", 3, -1, ["abc", "def"]),
    ("ab", 0, 10, ["a", "b"]),
    ("你好世界", 3, 1, ["你好世", "世界"]),
])
def test_text_window_boundaries(text: str, size: int, overlap: int, expected: list[str]) -> None:
    assert chunk_text(text, chunk_size=size, overlap=overlap) == expected
