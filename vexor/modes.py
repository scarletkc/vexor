"""Index mode registry and strategy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol, Sequence

from .services.content_extract_service import extract_head

PREVIEW_CHAR_LIMIT = 160


@dataclass(slots=True)
class ModePayload:
    label: str
    preview: str | None


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
        return ModePayload(label=label, preview=preview)


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
            return ModePayload(label=label, preview=preview)
        return self.fallback.payload_for_file(file)


_STRATEGIES: Dict[str, IndexModeStrategy] = {
    "name": NameStrategy(),
    "head": HeadStrategy(),
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
