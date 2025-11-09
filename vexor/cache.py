"""Index cache helpers for Vexor."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .utils import collect_files

CACHE_DIR = Path(os.path.expanduser("~")) / ".vexor"
CACHE_VERSION = 1


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def _cache_key(root: Path, include_hidden: bool) -> str:
    digest = hashlib.sha1(f"{root.resolve()}|hidden={include_hidden}".encode("utf-8")).hexdigest()
    return digest


def cache_file(root: Path, model: str, include_hidden: bool) -> Path:
    key = _cache_key(root, include_hidden)
    safe_model = _safe_model_name(model)
    return CACHE_DIR / f"{key}-{safe_model}.json"


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def store_index(
    *,
    root: Path,
    model: str,
    include_hidden: bool,
    files: Sequence[Path],
    embeddings: np.ndarray,
) -> Path:
    ensure_cache_dir()
    payload = {
        "version": CACHE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "model": model,
        "include_hidden": include_hidden,
        "dimension": int(embeddings.shape[1] if embeddings.size else 0),
        "files": [],
    }
    for idx, file in enumerate(files):
        stat = file.stat()
        try:
            rel_path = file.relative_to(root)
        except ValueError:
            rel_path = file
        payload["files"].append(
            {
                "path": str(rel_path),
                "absolute": str(file),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "embedding": embeddings[idx].astype(float).tolist(),
            }
        )
    path = cache_file(root, model, include_hidden)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_index(root: Path, model: str, include_hidden: bool) -> dict:
    path = cache_file(root, model, include_hidden)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def load_index_vectors(root: Path, model: str, include_hidden: bool):
    data = load_index(root, model, include_hidden)
    files = data.get("files", [])
    paths = [root / Path(entry["path"]) for entry in files]
    embeddings = np.asarray([entry["embedding"] for entry in files], dtype=np.float32)
    return paths, embeddings, data


def compare_snapshot(
    root: Path,
    include_hidden: bool,
    cached_files: Sequence[dict],
    current_files: Sequence[Path] | None = None,
) -> bool:
    """Return True if the current filesystem matches the cached snapshot."""
    if current_files is None:
        current_files = collect_files(root, include_hidden=include_hidden)
    if len(current_files) != len(cached_files):
        return False
    cached_map = {
        entry["path"]: (entry["mtime"], entry.get("size"))
        for entry in cached_files
    }
    for file in current_files:
        rel = _relative_path(file, root)
        data = cached_map.get(rel)
        if data is None:
            return False
        cached_mtime, cached_size = data
        stat = file.stat()
        current_mtime = stat.st_mtime
        current_size = stat.st_size
        # allow drift due to filesystem precision (approx 0.5s on some platforms)
        if abs(current_mtime - cached_mtime) > 5e-1:
            if cached_size is not None and cached_size == current_size:
                continue
            return False
        if cached_size is not None and cached_size != current_size:
            return False
    return True


def _relative_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return str(rel)
