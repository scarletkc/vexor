"""Helpers to extract head snippets from various file types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol

from charset_normalizer import from_path
from docx import Document
from pptx import Presentation
from pypdf import PdfReader

HEAD_CHAR_LIMIT = 1000
FULL_CHAR_LIMIT = 200_000
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100


class HeadExtractor(Protocol):
    """Protocol describing a file head extractor."""

    def __call__(self, path: Path, char_limit: int = HEAD_CHAR_LIMIT) -> str | None:
        ...


@dataclass(frozen=True)
class ExtractorEntry:
    extensions: tuple[str, ...]
    extractor: HeadExtractor


_registry: Dict[str, HeadExtractor] = {}

TEXT_EXTENSIONS = (
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".toml",
    ".csv",
    ".log",
    ".ini",
    ".cfg",
    ".rst",
    ".tex",
    ".xml",
    ".sh",
    ".bat",
    ".go",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".swift",
    ".rs",
    ".kt",
    ".dart",
    ".scala",
    ".pl",
    ".r",
    ".jl",
    ".hs",
    ".lua",
    ".vb",
    ".ps1",
    ".bash",
)


def register_extractor(entry: ExtractorEntry) -> None:
    for ext in entry.extensions:
        _registry[ext.lower()] = entry.extractor


def extract_head(path: Path, char_limit: int = HEAD_CHAR_LIMIT) -> str | None:
    """Return a text snippet representing the head of *path*."""

    extractor = _registry.get(path.suffix.lower())
    if extractor is None:
        return None
    return extractor(path, char_limit)


def extract_full_chunks(
    path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    char_limit: int = FULL_CHAR_LIMIT,
) -> list[str]:
    """Return sliding-window chunks for text-like files."""

    suffix = path.suffix.lower()
    text: str | None = None
    if suffix in TEXT_EXTENSIONS:
        text = _read_text_full(path, char_limit)
    elif suffix == ".pdf":
        text = _pdf_extractor(path, char_limit)
    elif suffix == ".docx":
        text = _docx_extractor(path, char_limit)
    elif suffix == ".pptx":
        text = _pptx_extractor(path, char_limit)
    else:
        return []
    if text is None:
        return []
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    size = max(int(chunk_size), 1)
    stride = max(size - max(int(overlap), 0), 1)
    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        window = normalized[start : start + size].strip()
        if window:
            chunks.append(window)
        if start + size >= length:
            break
        start += stride
    return chunks


# Placeholder extractors ----------------------------------------------------

def _read_text_head(path: Path, char_limit: int = HEAD_CHAR_LIMIT) -> str | None:
    """Return the first *char_limit* characters of a text-like file."""

    try:
        result = from_path(path)
    except Exception:
        return None
    if result is None or not len(result):
        return None
    best = result.best()
    if best is None:
        return None
    text = str(best)
    if not text:
        return None
    snippet = text[:char_limit]
    return _cleanup_snippet(snippet)


def _read_text_full(path: Path, char_limit: int = FULL_CHAR_LIMIT) -> str | None:
    """Return up to *char_limit* characters from a text-like file."""

    try:
        result = from_path(path)
    except Exception:
        return None
    if result is None or not len(result):
        return None
    best = result.best()
    if best is None:
        return None
    text = str(best)
    if not text:
        return None
    if char_limit > 0:
        return text[:char_limit]
    return text


def _pdf_extractor(path: Path, char_limit: int = HEAD_CHAR_LIMIT) -> str | None:
    try:
        reader = PdfReader(str(path))
    except Exception:
        return None
    buffer: list[str] = []
    total_chars = 0
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if not text:
            continue
        buffer.append(text)
        total_chars += len(text)
        if total_chars >= char_limit:
            break
    combined = "\n".join(buffer)
    if not combined:
        return None
    cleaned = _cleanup_snippet(combined)
    if not cleaned:
        return None
    return cleaned[:char_limit]


def _docx_extractor(path: Path, char_limit: int = HEAD_CHAR_LIMIT) -> str | None:
    try:
        document = Document(str(path))
    except Exception:
        return None
    buffer: list[str] = []
    total_chars = 0
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        buffer.append(text)
        total_chars += len(text)
        if total_chars >= char_limit:
            break
    combined = "\n".join(buffer)
    if not combined:
        return None
    cleaned = _cleanup_snippet(combined)
    if not cleaned:
        return None
    return cleaned[:char_limit]


def _pptx_extractor(path: Path, char_limit: int = HEAD_CHAR_LIMIT) -> str | None:
    try:
        presentation = Presentation(str(path))
    except Exception:
        return None
    buffer: list[str] = []
    total_chars = 0
    for slide in presentation.slides:
        for shape in slide.shapes:
            text = _extract_shape_text(shape)
            if not text:
                continue
            buffer.append(text)
            total_chars += len(text)
            if total_chars >= char_limit:
                break
        if total_chars >= char_limit:
            break
    combined = "\n".join(buffer)
    if not combined:
        return None
    cleaned = _cleanup_snippet(combined)
    if not cleaned:
        return None
    return cleaned[:char_limit]


def _extract_shape_text(shape) -> str | None:
    text_frame = getattr(shape, "text_frame", None)
    if text_frame is None:
        text = getattr(shape, "text", "")
        text = text.strip()
        return text or None
    paragraphs: list[str] = []
    for paragraph in text_frame.paragraphs:
        if getattr(paragraph, "runs", None):
            text = "".join(run.text for run in paragraph.runs)
        else:
            text = paragraph.text
        text = (text or "").strip()
        if text:
            paragraphs.append(text)
    if not paragraphs:
        return None
    return " ".join(paragraphs)


def _cleanup_snippet(snippet: str) -> str | None:
    lines = [line.strip() for line in snippet.splitlines() if line.strip()]
    joined = " ".join(lines)
    return joined or None


def _unimplemented_extractor(path: Path, char_limit: int = HEAD_CHAR_LIMIT) -> str | None:
    return None


register_extractor(
    ExtractorEntry(
        extensions=TEXT_EXTENSIONS,
        extractor=_read_text_head,
    )
)

register_extractor(
    ExtractorEntry((".pdf",), _pdf_extractor)
)

register_extractor(
    ExtractorEntry((".docx",), _docx_extractor)
)

register_extractor(
    ExtractorEntry((".pptx",), _pptx_extractor)
)
