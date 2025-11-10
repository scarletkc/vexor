from pathlib import Path

from docx import Document

import vexor.services.content_extract_service as ces
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


def test_extract_head_from_docx(tmp_path):
    doc_path = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("Docx Title")
    document.add_paragraph("First paragraph")
    document.save(doc_path)

    snippet = extract_head(doc_path)

    assert snippet.startswith("Docx Title First paragraph")


def test_extract_head_from_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class DummyPage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self):
            return self._text

    class DummyReader:
        def __init__(self, *_):
            self.pages = [DummyPage("PDF snippet one"), DummyPage("Second page text")]

    monkeypatch.setattr(ces, "PdfReader", lambda path: DummyReader(path))

    snippet = extract_head(pdf_path)

    assert snippet.startswith("PDF snippet one Second page text")
