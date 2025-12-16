from pathlib import Path

from docx import Document

from vexor.modes import (
    AutoStrategy,
    BriefStrategy,
    CodeStrategy,
    FullStrategy,
    HeadStrategy,
    ModePayload,
    NameStrategy,
    OutlineStrategy,
)


def test_name_strategy_payload():
    strategy = NameStrategy()
    payload = strategy.payload_for_file(Path("my_file-name.py"))
    assert isinstance(payload, ModePayload)
    assert payload.file.name == "my_file-name.py"
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


def test_full_strategy_chunks_text(tmp_path):
    file_path = tmp_path / "long.txt"
    file_path.write_text("abc" * 400)
    strategy = FullStrategy()
    payloads = strategy.payloads_for_files([file_path])
    assert len(payloads) >= 1
    assert payloads[0].chunk_index == 0
    if len(payloads) > 1:
        assert payloads[1].chunk_index == 1
    assert "[Chunk" not in payloads[0].preview
    assert "\n" not in payloads[0].preview


def test_full_strategy_fallback(tmp_path):
    file_path = tmp_path / "image.bin"
    file_path.write_bytes(b"\x00\x01")
    strategy = FullStrategy()
    payloads = strategy.payloads_for_files([file_path])
    assert len(payloads) == 1
    assert payloads[0].label == file_path.name


def test_full_strategy_supports_docx(tmp_path):
    doc_path = tmp_path / "sample.docx"
    document = Document()
    for idx in range(5):
        document.add_paragraph(f"Doc paragraph {idx} " + "text " * 10)
    document.save(doc_path)

    strategy = FullStrategy(chunk_size=50, overlap=0)
    payloads = strategy.payloads_for_files([doc_path])

    assert payloads
    assert payloads[0].label.startswith("sample.docx")
    assert "[Chunk" not in payloads[0].preview
    assert "\n" not in payloads[0].preview


def test_brief_strategy_extracts_keywords(tmp_path):
    file_path = tmp_path / "req.md"
    file_path.write_text(
        """# Login requirement\nUsers must login with MFA. The login flow should support backup codes."""
    )
    strategy = BriefStrategy()
    payload = strategy.payload_for_file(file_path)
    assert "login" in (payload.preview or "").lower()


def test_brief_strategy_handles_chinese(tmp_path):
    file_path = tmp_path / "需求.md"
    file_path.write_text("用户需要支持离线模式，离线同步，确保数据安全。")
    strategy = BriefStrategy()
    payload = strategy.payload_for_file(file_path)
    assert payload.preview is not None
    assert "离线" in payload.preview


def test_code_strategy_chunks_python(tmp_path):
    py_path = tmp_path / "sample.py"
    py_path.write_text(
        """\"\"\"Module docstring.\"\"\"

def foo(a, b):
    \"\"\"Foo does bar.\"\"\"
    return a + b


class Bar:
    VALUE = 42

    def method(self, x):
        return x * 2


TAIL_CONSTANT = "tail"
"""
    )

    strategy = CodeStrategy(chunk_size=80, overlap=0)
    payloads = strategy.payloads_for_files([py_path])

    assert payloads
    assert payloads[0].chunk_index == 0
    assert any("foo" in payload.label for payload in payloads)
    assert any("VALUE = 42" in payload.label for payload in payloads)
    assert any("Bar.method" in (payload.preview or "") for payload in payloads)
    assert any((payload.preview or "").startswith("module globals") for payload in payloads)


def test_code_strategy_falls_back_to_full_for_non_python(tmp_path):
    file_path = tmp_path / "long.txt"
    file_path.write_text("abc" * 400)

    strategy = CodeStrategy(chunk_size=50, overlap=0)
    payloads = strategy.payloads_for_files([file_path])

    assert payloads
    assert payloads[0].label.startswith("long.txt")
    assert payloads[0].preview is None or "\n" not in payloads[0].preview


def test_outline_strategy_chunks_markdown(tmp_path):
    md_path = tmp_path / "doc.md"
    md_path.write_text(
        """Intro before headings.

# Top
Top body.

## Child
Child body.
"""
    )

    strategy = OutlineStrategy(context_char_limit=200)
    payloads = strategy.payloads_for_files([md_path])

    assert payloads
    assert payloads[0].chunk_index == 0
    assert payloads[0].preview is not None
    assert payloads[0].preview.startswith("preamble")
    assert any("Top > Child" in (payload.preview or "") for payload in payloads)


def test_outline_strategy_falls_back_to_full_for_non_markdown(tmp_path):
    file_path = tmp_path / "long.txt"
    file_path.write_text("abc" * 400)

    strategy = OutlineStrategy(context_char_limit=200)
    payloads = strategy.payloads_for_files([file_path])

    assert payloads
    assert payloads[0].label.startswith("long.txt")


def test_auto_strategy_routes_python_to_code(tmp_path):
    py_path = tmp_path / "sample.py"
    py_path.write_text(
        """def foo(a, b):\n    return a + b\n\nTAIL_CONSTANT = \"tail\"\n"""
    )

    strategy = AutoStrategy()
    payloads = strategy.payloads_for_files([py_path])

    assert payloads
    assert any((payload.preview or "").startswith("module globals") for payload in payloads)
    assert any("def foo" in (payload.preview or "") for payload in payloads)


def test_auto_strategy_routes_markdown_to_outline(tmp_path):
    md_path = tmp_path / "doc.md"
    md_path.write_text(
        """Intro before headings.\n\n# Top\nTop body.\n\n## Child\nChild body.\n"""
    )

    strategy = AutoStrategy()
    payloads = strategy.payloads_for_files([md_path])

    assert payloads
    assert any((payload.preview or "").startswith("preamble") for payload in payloads)
    assert any("Top > Child" in (payload.preview or "") for payload in payloads)


def test_auto_strategy_routes_small_text_to_full(tmp_path):
    text_path = tmp_path / "note.txt"
    text_path.write_text("Hello world\nSecond line\n")

    strategy = AutoStrategy()
    payload = strategy.payload_for_file(text_path)

    assert payload.preview == "Hello world Second line"
    assert "[#1]" in payload.label


def test_auto_strategy_routes_large_text_to_head(tmp_path):
    text_path = tmp_path / "big.txt"
    text_path.write_text("a" * 20_000)

    strategy = AutoStrategy()
    payload = strategy.payload_for_file(text_path)

    assert payload.preview is not None
    assert "[#1]" not in payload.label
