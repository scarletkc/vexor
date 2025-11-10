from pathlib import Path

from vexor.services.content_extract_service import extract_head, HEAD_CHAR_LIMIT


def test_extract_head_from_text(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Title\n\nLine one\nLine two\n")

    snippet = extract_head(text_file, char_limit=HEAD_CHAR_LIMIT)

    assert snippet == "Title Line one Line two"


def test_extract_head_unknown_extension(tmp_path):
    data_file = tmp_path / "data.bin"
    data_file.write_bytes(b"\x00\x01\x02")

    assert extract_head(data_file) is None
