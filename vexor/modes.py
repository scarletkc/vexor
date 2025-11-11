"""Index mode registry and strategy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol, Sequence

from .services.content_extract_service import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    extract_full_chunks,
    extract_head,
)
from .services.keyword_service import (
    BRIEF_CHAR_LIMIT,
    BRIEF_KEYWORD_LIMIT,
    summarize_keywords,
)

PREVIEW_CHAR_LIMIT = 160
BRIEF_PREVIEW_LIMIT = 10


@dataclass(slots=True)
class ModePayload:
    file: Path
    label: str
    preview: str | None
    chunk_index: int = 0


class IndexModeStrategy(Protocol):
    name: str

    def payloads_for_files(self, files: Sequence[Path]) -> list[ModePayload]:
        raise NotImplementedError

    def payload_for_file(self, file: Path) -> ModePayload:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class NameStrategy(IndexModeStrategy):
    name: str = "name"

    def payloads_for_files(self, files: Sequence[Path]) -> list[ModePayload]:
        return [self.payload_for_file(file) for file in files]

    def payload_for_file(self, file: Path) -> ModePayload:
        label = file.name.replace("_", " ")
        preview = file.name
        return ModePayload(file=file, label=label, preview=preview)


@dataclass(frozen=True, slots=True)
class HeadStrategy(IndexModeStrategy):
    name: str = "head"
    fallback: NameStrategy = NameStrategy()

    def payloads_for_files(self, files: Sequence[Path]) -> list[ModePayload]:
        return [self.payload_for_file(file) for file in files]

    def payload_for_file(self, file: Path) -> ModePayload:
        snippet = extract_head(file)
        if snippet:
            label = f"{file.name} :: {snippet}"
            preview = _trim_preview(snippet)
            return ModePayload(file=file, label=label, preview=preview)
        return self.fallback.payload_for_file(file)


@dataclass(frozen=True, slots=True)
class FullStrategy(IndexModeStrategy):
    name: str = "full"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP
    fallback: NameStrategy = NameStrategy()

    def payloads_for_files(self, files: Sequence[Path]) -> list[ModePayload]:
        payloads: list[ModePayload] = []
        for file in files:
            payloads.extend(self._payloads_for_file(file))
        return payloads

    def payload_for_file(self, file: Path) -> ModePayload:
        chunks = self._payloads_for_file(file)
        if chunks:
            return chunks[0]
        return self.fallback.payload_for_file(file)

    def _payloads_for_file(self, file: Path) -> list[ModePayload]:
        chunks = extract_full_chunks(
            file,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        if not chunks:
            return [self.fallback.payload_for_file(file)]
        payloads: list[ModePayload] = []
        for index, chunk in enumerate(chunks):
            normalized = _normalize_preview_chunk(chunk)
            if not normalized:
                continue
            preview = _trim_preview(normalized)
            label = f"{file.name} [#{index + 1}] :: {normalized}"
            payloads.append(
                ModePayload(
                    file=file,
                    label=label,
                    preview=preview,
                    chunk_index=index,
                )
            )
        if not payloads:
            return [self.fallback.payload_for_file(file)]
        return payloads


@dataclass(frozen=True, slots=True)
class BriefStrategy(IndexModeStrategy):
    name: str = "brief"
    fallback: NameStrategy = NameStrategy()

    def payloads_for_files(self, files: Sequence[Path]) -> list[ModePayload]:
        return [self.payload_for_file(file) for file in files]

    def payload_for_file(self, file: Path) -> ModePayload:
        keywords = summarize_keywords(
            file,
            char_limit=BRIEF_CHAR_LIMIT,
            limit=BRIEF_KEYWORD_LIMIT,
        )
        if keywords:
            preview_tokens = keywords[:BRIEF_PREVIEW_LIMIT]
            preview = ", ".join(preview_tokens)
            label = f"{file.name} :: {' '.join(preview_tokens)}"
            return ModePayload(
                file=file,
                label=label,
                preview=preview,
                chunk_index=0,
            )
        return self.fallback.payload_for_file(file)


_STRATEGIES: Dict[str, IndexModeStrategy] = {
    "name": NameStrategy(),
    "head": HeadStrategy(),
    "brief": BriefStrategy(),
    "full": FullStrategy(),
}


def get_strategy(mode: str) -> IndexModeStrategy:
    try:
        return _STRATEGIES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode}") from exc


def available_modes() -> list[str]:
    return sorted(_STRATEGIES.keys())


def _trim_preview(text: str, limit: int = PREVIEW_CHAR_LIMIT) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1].rstrip() + "…"


def _normalize_preview_chunk(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return " ".join(lines)
    stripped = text.strip()
    return stripped or None
