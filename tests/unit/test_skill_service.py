from __future__ import annotations

from pathlib import Path

import pytest

from vexor.services.skill_service import (
    SkillInstallStatus,
    install_bundled_skill,
    resolve_skill_roots,
    _trees_equal,
)
import vexor.services.skill_service as skill_service


def test_resolve_skill_roots_variants(tmp_path):
    home = tmp_path / "home"
    expected_claude = home / ".claude" / "skills"
    expected_codex = home / ".codex" / "skills"

    assert resolve_skill_roots("claude", home=home) == [expected_claude]
    assert resolve_skill_roots("auto", home=home) == [expected_claude, expected_codex]
    assert resolve_skill_roots("claude/codex", home=home) == [expected_claude, expected_codex]
    assert resolve_skill_roots("claude,codex,claude", home=home) == [expected_claude, expected_codex]

    custom = tmp_path / "custom-skills"
    assert resolve_skill_roots(str(custom), home=home) == [custom]


def test_resolve_skill_roots_validation(tmp_path):
    with pytest.raises(ValueError, match="Missing --skills target"):
        resolve_skill_roots("   ", home=tmp_path)
    with pytest.raises(ValueError, match="Unknown --skills target"):
        resolve_skill_roots("claude,unknown", home=tmp_path)


def test_trees_equal(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a.txt").write_text("hello", encoding="utf-8")
    (right / "a.txt").write_text("hello", encoding="utf-8")
    assert _trees_equal(left, right) is True

    (right / "b.txt").write_text("extra", encoding="utf-8")
    assert _trees_equal(left, right) is False

    (right / "b.txt").unlink()
    (right / "a.txt").write_text("different", encoding="utf-8")
    assert _trees_equal(left, right) is False

    assert _trees_equal(left / "a.txt", right) is False


def test_install_bundled_skill_installs_and_updates(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_dir = source_root / "vexor-cli"
    source_dir.mkdir(parents=True)
    (source_dir / "SKILL.md").write_text("hello skill", encoding="utf-8")

    monkeypatch.setattr(
        "vexor.services.skill_service._resolve_skill_source_dir",
        lambda name: source_root / name,
    )

    destination_root = tmp_path / "dest"
    result = install_bundled_skill(skill_name="vexor-cli", skills_dir=destination_root)
    assert result.status == SkillInstallStatus.installed
    assert (destination_root / "vexor-cli" / "SKILL.md").read_text(encoding="utf-8") == "hello skill"

    result2 = install_bundled_skill(skill_name="vexor-cli", skills_dir=destination_root)
    assert result2.status == SkillInstallStatus.up_to_date

    # Mutate destination and ensure we refuse without force.
    (destination_root / "vexor-cli" / "SKILL.md").write_text("changed", encoding="utf-8")
    with pytest.raises(FileExistsError):
        install_bundled_skill(skill_name="vexor-cli", skills_dir=destination_root, force=False)

    # Force should overwrite.
    result3 = install_bundled_skill(skill_name="vexor-cli", skills_dir=destination_root, force=True)
    assert result3.status == SkillInstallStatus.installed
    assert (destination_root / "vexor-cli" / "SKILL.md").read_text(encoding="utf-8") == "hello skill"


def test_resolve_skill_source_dir_uses_packaged_resources(tmp_path, monkeypatch):
    packaged_dir = tmp_path / "bundled" / "vexor-cli"
    packaged_dir.mkdir(parents=True)
    (packaged_dir / "SKILL.md").write_text("bundled", encoding="utf-8")

    monkeypatch.setattr(skill_service, "_repo_skill_dir", lambda _name: tmp_path / "missing")

    class DummyFiles:
        def joinpath(self, *_parts):
            return packaged_dir

    class DummyAsFile:
        def __init__(self, path: Path) -> None:
            self._path = path

        def __enter__(self):
            return self._path

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(skill_service.resources, "files", lambda _pkg: DummyFiles())
    monkeypatch.setattr(skill_service.resources, "as_file", lambda path: DummyAsFile(path))

    resolved = skill_service._resolve_skill_source_dir("vexor-cli")
    assert resolved == packaged_dir


def test_resolve_skill_source_dir_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_service, "_repo_skill_dir", lambda _name: tmp_path / "missing")
    monkeypatch.setattr(skill_service.resources, "files", lambda _pkg: tmp_path / "missing-traversable")

    def raise_not_found(_path):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(skill_service.resources, "as_file", raise_not_found)
    with pytest.raises(FileNotFoundError, match="Unable to locate bundled skill"):
        skill_service._resolve_skill_source_dir("vexor-cli")
