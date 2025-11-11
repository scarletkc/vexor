from pathlib import Path

from docx import Document
from pptx import Presentation

import vexor.services.content_extract_service as ces
from vexor.services.content_extract_service import (
    extract_full_chunks,
    extract_head,
    HEAD_CHAR_LIMIT,
)


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


def test_extract_head_from_pptx(tmp_path):
    ppt_path = tmp_path / "sample.pptx"
    presentation = Presentation()
    slide_layout = presentation.slide_layouts[1]
    slide = presentation.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Slide Title"
    slide.placeholders[1].text = "Body paragraph"
    presentation.save(ppt_path)

    snippet = extract_head(ppt_path)

    assert snippet.startswith("Slide Title Body paragraph")


def test_extract_full_chunks_from_docx(tmp_path):
    doc_path = tmp_path / "long.docx"
    document = Document()
    for idx in range(10):
        document.add_paragraph(f"Paragraph {idx} " + "text " * 5)
    document.save(doc_path)

    chunks = extract_full_chunks(doc_path, chunk_size=50, overlap=0)

    assert chunks
    assert any("Paragraph" in chunk for chunk in chunks)


def test_extract_full_chunks_from_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class DummyPage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self):
            return self._text

    class DummyReader:
        def __init__(self, *_):
            self.pages = [DummyPage("One two three four five six seven eight nine ten" * 5)]

    monkeypatch.setattr(ces, "PdfReader", lambda path: DummyReader(path))

    chunks = extract_full_chunks(pdf_path, chunk_size=40, overlap=0)

    assert len(chunks) >= 2


def test_extract_full_chunks_from_pptx(tmp_path):
    ppt_path = tmp_path / "chunks.pptx"
    presentation = Presentation()
    slide_layout = presentation.slide_layouts[1]
    slide = presentation.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Chunk Title"
    slide.placeholders[1].text = "Paragraph one text"
    presentation.save(ppt_path)

    chunks = extract_full_chunks(ppt_path, chunk_size=20, overlap=0)

    assert chunks
    assert chunks[0].startswith("Chunk Title")
