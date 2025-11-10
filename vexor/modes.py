"""Index mode registry and strategy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol, Sequence


class IndexModeStrategy(Protocol):
    """Protocol for components that transform files into embedding labels."""

    name: str

    def labels_for_files(self, files: Sequence[Path]) -> list[str]:
        """Return human-readable labels used as embedding input."""
        raise NotImplementedError

    def label_for_file(self, file: Path) -> str:
        """Return the embedding label for *file*."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class NameStrategy(IndexModeStrategy):
    name: str = "name"

    def labels_for_files(self, files: Sequence[Path]) -> list[str]:
        return [self.label_for_file(file) for file in files]

    def label_for_file(self, file: Path) -> str:
        return file.name.replace("_", " ")


_STRATEGIES: Dict[str, IndexModeStrategy] = {
    "name": NameStrategy(),
}


def get_strategy(mode: str) -> IndexModeStrategy:
    try:
        return _STRATEGIES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode}") from exc


def available_modes() -> list[str]:
    return sorted(_STRATEGIES.keys())
