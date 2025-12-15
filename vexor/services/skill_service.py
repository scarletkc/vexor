"""Helpers for installing Agent Skills into agent skill folders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
import shutil


class SkillInstallStatus(str, Enum):
    installed = "installed"
    up_to_date = "up-to-date"


@dataclass(slots=True)
class SkillInstallResult:
    destination: Path
    status: SkillInstallStatus


DEFAULT_SKILL_NAME = "vexor-cli"
_SKILL_INSTALL_LOCATIONS = {
    "claude": (".claude", "skills"),
    "codex": (".codex", "skills"),
}


def resolve_skill_roots(targets: str, *, home: Path | None = None) -> list[Path]:
    """Resolve `--skills` targets into a list of skill root directories.

    Supported values:
    - `claude`, `codex`
    - `auto` / `all` (installs to both)
    - `claude/codex` or `claude,codex` (any order; duplicates ignored)
    - Any filesystem path (treated as a custom skill root directory)
    """

    raw = targets.strip()
    if not raw:
        raise ValueError("Missing --skills target.")

    lowered = raw.lower()
    if lowered in {"auto", "all"}:
        return [
            _default_skill_root("claude", home=home),
            _default_skill_root("codex", home=home),
        ]

    if lowered in _SKILL_INSTALL_LOCATIONS:
        return [_default_skill_root(lowered, home=home)]

    if "," in raw:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        return _resolve_target_list(parts, home=home)

    if "/" in raw:
        parts = [part.strip() for part in raw.split("/") if part.strip()]
        lowered_parts = [part.lower() for part in parts]
        if parts and all(part in _SKILL_INSTALL_LOCATIONS for part in lowered_parts):
            return _resolve_target_list(parts, home=home)

    return [Path(raw).expanduser()]


def install_bundled_skill(
    *,
    skill_name: str,
    skills_dir: Path,
    force: bool = False,
) -> SkillInstallResult:
    """Install *skill_name* into *skills_dir* (a directory containing skill folders)."""

    source_dir = _resolve_skill_source_dir(skill_name)
    destination = skills_dir / skill_name
    if destination.exists():
        if _trees_equal(source_dir, destination):
            return SkillInstallResult(destination=destination, status=SkillInstallStatus.up_to_date)
        if not force:
            raise FileExistsError(destination)
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, destination)
    return SkillInstallResult(destination=destination, status=SkillInstallStatus.installed)


def _resolve_target_list(parts: list[str], *, home: Path | None) -> list[Path]:
    seen: set[str] = set()
    roots: list[Path] = []
    for part in parts:
        lowered = part.lower()
        if lowered not in _SKILL_INSTALL_LOCATIONS:
            allowed = ", ".join(sorted(_SKILL_INSTALL_LOCATIONS))
            raise ValueError(f"Unknown --skills target '{part}'. Allowed: {allowed}.")
        if lowered in seen:
            continue
        seen.add(lowered)
        roots.append(_default_skill_root(lowered, home=home))
    if not roots:
        raise ValueError("Missing --skills target.")
    return roots


def _default_skill_root(target: str, *, home: Path | None) -> Path:
    base = home if home is not None else Path.home()
    prefix = _SKILL_INSTALL_LOCATIONS[target]
    return base.joinpath(*prefix)


def _resolve_skill_source_dir(skill_name: str) -> Path:
    repo_candidate = _repo_skill_dir(skill_name)
    if repo_candidate.exists():
        return repo_candidate

    packaged = resources.files("vexor").joinpath("_bundled_skills", skill_name)
    try:
        with resources.as_file(packaged) as packaged_path:
            packaged_dir = Path(packaged_path)
            if packaged_dir.exists():
                return packaged_dir
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        f"Unable to locate bundled skill '{skill_name}'. "
        "Reinstall Vexor from PyPI or run from the source repository."
    )


def _repo_skill_dir(skill_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "plugins" / "vexor" / "skills" / skill_name


def _trees_equal(left: Path, right: Path) -> bool:
    """Return True when two directory trees have the same set of files and bytes."""

    if not left.is_dir() or not right.is_dir():
        return False

    left_files = {path.relative_to(left) for path in left.rglob("*") if path.is_file()}
    right_files = {path.relative_to(right) for path in right.rglob("*") if path.is_file()}
    if left_files != right_files:
        return False

    for relpath in left_files:
        if (left / relpath).read_bytes() != (right / relpath).read_bytes():
            return False
    return True
