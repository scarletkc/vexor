from pathlib import Path

from vexor.modes import HeadStrategy, ModePayload, NameStrategy


def test_name_strategy_payload():
    strategy = NameStrategy()
    payload = strategy.payload_for_file(Path("my_file-name.py"))
    assert isinstance(payload, ModePayload)
    assert payload.label == "my file-name.py"
    assert payload.preview == "my_file-name.py"


def test_head_strategy_uses_snippet(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Title\nBody\n")
    strategy = HeadStrategy()
    payload = strategy.payload_for_file(file_path)
    assert payload.label.startswith("sample.txt :: Title Body")
    assert payload.preview.startswith("Title Body")


def test_head_strategy_fallback(tmp_path):
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"\x00\x01")
    strategy = HeadStrategy()
    payload = strategy.payload_for_file(file_path)
    assert payload.label == file_path.name
    assert payload.preview == file_path.name
