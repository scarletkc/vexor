from __future__ import annotations

from pathlib import Path

import pytest

from vexor.services import cache_service


def test_is_cache_current_false_when_empty_cached_files(tmp_path: Path) -> None:
    assert (
        cache_service.is_cache_current(
            tmp_path,
            include_hidden=False,
            respect_gitignore=True,
            cached_files=[],
            recursive=True,
        )
        is False
    )


def test_is_cache_current_delegates_to_compare_snapshot(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_compare_snapshot(
        root: Path,
        include_hidden: bool,
        cached_files,
        *,
        recursive: bool,
        exclude_patterns,
        extensions,
        current_files,
        respect_gitignore: bool,
    ) -> bool:
        calls["root"] = root
        calls["include_hidden"] = include_hidden
        calls["cached_files"] = cached_files
        calls["recursive"] = recursive
        calls["exclude_patterns"] = exclude_patterns
        calls["extensions"] = extensions
        calls["current_files"] = current_files
        calls["respect_gitignore"] = respect_gitignore
        return True

    monkeypatch.setattr("vexor.cache.compare_snapshot", fake_compare_snapshot)

    assert (
        cache_service.is_cache_current(
            tmp_path,
            include_hidden=True,
            respect_gitignore=False,
            cached_files=[{"path": "a.txt"}],
            recursive=False,
            extensions=(".py",),
            current_files=["a.txt"],
        )
        is True
    )
    assert calls["root"] == tmp_path
    assert calls["respect_gitignore"] is False


def test_load_index_metadata_safe_returns_none_when_missing(tmp_path: Path, monkeypatch) -> None:
    def fake_load_index(*_args, **_kwargs):
        raise FileNotFoundError("no cache")

    monkeypatch.setattr("vexor.cache.load_index", fake_load_index)

    assert (
        cache_service.load_index_metadata_safe(
            tmp_path,
            model="model",
            include_hidden=False,
            respect_gitignore=True,
            mode="name",
            recursive=True,
        )
        is None
    )


def test_load_index_metadata_safe_passes_flags(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_load_index(
        root: Path,
        model: str,
        include_hidden: bool,
        mode: str,
        recursive: bool,
        exclude_patterns,
        extensions,
        *,
        respect_gitignore: bool,
    ):
        calls["root"] = root
        calls["model"] = model
        calls["include_hidden"] = include_hidden
        calls["mode"] = mode
        calls["recursive"] = recursive
        calls["exclude_patterns"] = exclude_patterns
        calls["extensions"] = extensions
        calls["respect_gitignore"] = respect_gitignore
        return {"ok": True}

    monkeypatch.setattr("vexor.cache.load_index", fake_load_index)

    meta = cache_service.load_index_metadata_safe(
        tmp_path,
        model="demo",
        include_hidden=True,
        respect_gitignore=False,
        mode="auto",
        recursive=False,
        extensions=(".md",),
    )
    assert meta == {"ok": True}
    assert calls["mode"] == "auto"
    assert calls["respect_gitignore"] is False
