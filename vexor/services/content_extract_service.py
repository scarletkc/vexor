"""Helpers to extract head snippets from various file types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol

from charset_normalizer import from_path
from docx import Document
from pypdf import PdfReader

HEAD_CHAR_LIMIT = 1000


class HeadExtractor(Protocol):
    """Protocol describing a file head extractor."""

    def __call__(self, path: Path, char_limit: int = HEAD_CHAR_LIMIT) -> str | None:
        ...


@dataclass(frozen=True)
class ExtractorEntry:
    extensions: tuple[str, ...]
    extractor: HeadExtractor


_registry: Dict[str, HeadExtractor] = {}


def register_extractor(entry: ExtractorEntry) -> None:
    for ext in entry.extensions:
        _registry[ext.lower()] = entry.extractor


def extract_head(path: Path, char_limit: int = HEAD_CHAR_LIMIT) -> str | None:
    """Return a text snippet representing the head of *path*."""

    extractor = _registry.get(path.suffix.lower())
    if extractor is None:
        return None
    return extractor(path, char_limit)


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


def _cleanup_snippet(snippet: str) -> str | None:
    lines = [line.strip() for line in snippet.splitlines() if line.strip()]
    joined = " ".join(lines)
    return joined or None


def _unimplemented_extractor(path: Path, char_limit: int = HEAD_CHAR_LIMIT) -> str | None:
    return None


register_extractor(
    ExtractorEntry(
        extensions=(
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
        ),
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
    ExtractorEntry((".pptx",), _unimplemented_extractor)
)
