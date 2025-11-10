"""Helpers to extract head snippets from various file types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol

from charset_normalizer import from_path

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
            ".json",
            ".yaml",
            ".yml",
        ),
        extractor=_read_text_head,
    )
)

register_extractor(
    ExtractorEntry(
        extensions=(".pdf", ".docx", ".pptx", ".html"),
        extractor=_unimplemented_extractor,
    )
)
