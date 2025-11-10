"""Logic helpers for the `vexor index` command."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .cache_service import is_cache_current, load_index_metadata_safe


class IndexStatus(str, Enum):
    EMPTY = "empty"
    UP_TO_DATE = "up_to_date"
    STORED = "stored"


@dataclass(slots=True)
class IndexResult:
    status: IndexStatus
    cache_path: Path | None = None
    files_indexed: int = 0


def build_index(
    directory: Path,
    *,
    include_hidden: bool,
    recursive: bool,
    model_name: str,
    batch_size: int,
) -> IndexResult:
    """Create or refresh the cached index for *directory*."""

    from ..search import VexorSearcher  # local import
    from ..utils import collect_files  # local import
    from ..cache import store_index  # local import

    files = collect_files(directory, include_hidden=include_hidden, recursive=recursive)
    if not files:
        return IndexResult(status=IndexStatus.EMPTY)

    existing_meta = load_index_metadata_safe(directory, model_name, include_hidden, recursive)
    if existing_meta:
        cached_files = existing_meta.get("files", [])
        if cached_files and is_cache_current(
            directory,
            include_hidden,
            cached_files,
            recursive=recursive,
            current_files=files,
        ):
            return IndexResult(status=IndexStatus.UP_TO_DATE)

    searcher = VexorSearcher(model_name=model_name, batch_size=batch_size)
    file_labels = [_label_for_path(file) for file in files]
    embeddings = searcher.embed_texts(file_labels)

    cache_path = store_index(
        root=directory,
        model=model_name,
        include_hidden=include_hidden,
        recursive=recursive,
        files=files,
        embeddings=embeddings,
    )
    return IndexResult(
        status=IndexStatus.STORED,
        cache_path=cache_path,
        files_indexed=len(files),
    )


def clear_index_entries(
    directory: Path,
    *,
    include_hidden: bool,
    recursive: bool,
    model: str | None = None,
) -> int:
    """Remove cached entries for *directory* and return number removed."""

    from ..cache import clear_index as clear_index_cache  # local import

    return clear_index_cache(
        root=directory,
        include_hidden=include_hidden,
        recursive=recursive,
        model=model,
    )


def _label_for_path(path: Path) -> str:
    return path.name.replace("_", " ")
