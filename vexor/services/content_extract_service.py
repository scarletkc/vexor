"""Helpers to extract head snippets from various file types."""

from __future__ import annotations

import ast
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


@dataclass(frozen=True, slots=True)
class CodeChunk:
    kind: str
    name: str
    display: str
    text: str
    start_line: int
    end_line: int


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


def extract_code_chunks(
    path: Path,
    *,
    char_limit: int = FULL_CHAR_LIMIT,
) -> list[CodeChunk]:
    """Return AST-aware code chunks for supported languages (Python only for now)."""

    if path.suffix.lower() != ".py":
        return []

    source = _read_text_full(path, char_limit)
    if not source:
        return []
    source = source.replace("\r\n", "\n")

    try:
        module = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines(keepends=True)
    max_line = len(lines)

    def clamp_line(value: int) -> int:
        if value < 1:
            return 1
        if value > max_line:
            return max_line
        return value

    def node_start_line(node) -> int:
        lineno = getattr(node, "lineno", None)
        start = int(lineno) if isinstance(lineno, int) else 1
        decorators = getattr(node, "decorator_list", None) or []
        for deco in decorators:
            deco_line = getattr(deco, "lineno", None)
            if isinstance(deco_line, int):
                start = min(start, deco_line)
        return clamp_line(start)

    def node_end_line(node) -> int:
        end_lineno = getattr(node, "end_lineno", None)
        if isinstance(end_lineno, int):
            return clamp_line(end_lineno)
        body = getattr(node, "body", None) or []
        if body:
            last = body[-1]
            last_end = getattr(last, "end_lineno", None)
            if isinstance(last_end, int):
                return clamp_line(last_end)
            last_line = getattr(last, "lineno", None)
            if isinstance(last_line, int):
                return clamp_line(last_line)
        lineno = getattr(node, "lineno", None)
        if isinstance(lineno, int):
            return clamp_line(lineno)
        return max_line

    def slice_lines(start: int, end: int) -> str:
        if not max_line:
            return ""
        start = clamp_line(start)
        end = clamp_line(end)
        if end < start:
            end = start
        return "".join(lines[start - 1 : end]).strip()

    def signature_line(node) -> str:
        lineno = getattr(node, "lineno", None)
        if not isinstance(lineno, int):
            return ""
        idx = lineno - 1
        if idx < 0 or idx >= max_line:
            return ""
        return lines[idx].strip()

    chunks: list[CodeChunk] = []

    symbols: list[tuple[int, int, object]] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append((node_start_line(node), node_end_line(node), node))
    symbols.sort(key=lambda item: item[0])

    def add_module_chunk(start: int, end: int, *, prelude: bool) -> None:
        text = slice_lines(start, end)
        if not text:
            return
        chunks.append(
            CodeChunk(
                kind="module",
                name="module" if prelude else "module_globals",
                display="module" if prelude else "module globals",
                text=text,
                start_line=start,
                end_line=end,
            )
        )

    if not symbols:
        add_module_chunk(1, max_line, prelude=True)
        return chunks

    cursor = 1
    seen_symbol = False

    for start, end, node in symbols:
        if cursor <= start - 1:
            add_module_chunk(cursor, start - 1, prelude=not seen_symbol)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            text = slice_lines(start, end)
            if text:
                chunks.append(
                    CodeChunk(
                        kind="function",
                        name=node.name,
                        display=signature_line(node) or f"def {node.name}",
                        text=text,
                        start_line=start,
                        end_line=end,
                    )
                )
            cursor = end + 1
            seen_symbol = True
            continue

        if isinstance(node, ast.ClassDef):
            class_display = signature_line(node) or f"class {node.name}"
            docstring = ast.get_docstring(node) or ""
            method_names: list[str] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_names.append(child.name)
            class_parts = [slice_lines(start, node.lineno)]
            if docstring.strip():
                class_parts.append(docstring.strip())
            if method_names:
                class_parts.append("Methods: " + ", ".join(method_names))
            class_text = "\n".join(part for part in class_parts if part).strip()
            if class_text:
                chunks.append(
                    CodeChunk(
                        kind="class",
                        name=node.name,
                        display=class_display,
                        text=class_text,
                        start_line=start,
                        end_line=end,
                    )
                )

            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                child_start = node_start_line(child)
                child_end = node_end_line(child)
                text = slice_lines(child_start, child_end)
                if not text:
                    continue
                raw_sig = signature_line(child)
                display = f"{node.name}.{child.name}"
                if raw_sig:
                    normalized_sig = raw_sig.strip()
                    if normalized_sig.startswith("async def "):
                        tail = normalized_sig[len("async def ") :].rstrip(":").strip()
                        display = f"async {node.name}.{tail}"
                    elif normalized_sig.startswith("def "):
                        tail = normalized_sig[len("def ") :].rstrip(":").strip()
                        display = f"{node.name}.{tail}"
                    else:
                        display = f"{node.name}.{normalized_sig.rstrip(':').strip()}"
                chunks.append(
                    CodeChunk(
                        kind="method",
                        name=f"{node.name}.{child.name}",
                        display=display,
                        text=text,
                        start_line=child_start,
                        end_line=child_end,
                    )
                )

            cursor = end + 1
            seen_symbol = True
            continue

        cursor = end + 1
        seen_symbol = True

    if cursor <= max_line:
        add_module_chunk(cursor, max_line, prelude=False)

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
