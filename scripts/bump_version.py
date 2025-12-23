#!/usr/bin/env python3
"""Bump Vexor versions in one command.

Usage:
  python scripts/bump_version.py 0.6.4
  python scripts/bump_version.py v0.6.4
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[0-9A-Za-z.+-]+)?$")
_SEMVER_PATTERN = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in {"-h", "--help"}:
        print(__doc__.strip())
        return 2

    raw = argv[1].strip()
    if raw.startswith("v"):
        raw = raw[1:]
    if not raw or not _VERSION_PATTERN.fullmatch(raw):
        raise SystemExit(f"Invalid version '{argv[1]}'. Expected like 0.6.4")

    repo_root = Path(__file__).resolve().parents[1]
    package_init = repo_root / "vexor" / "__init__.py"
    plugin_manifest = repo_root / "plugins" / "vexor" / ".claude-plugin" / "plugin.json"
    gui_package_json = repo_root / "gui" / "package.json"
    gui_package_lock = repo_root / "gui" / "package-lock.json"

    gui_version = _to_gui_semver(raw)
    _set_python_version(package_init, raw)
    _set_plugin_version(plugin_manifest, raw)
    _set_gui_version(gui_package_json, gui_version)
    _set_gui_lock_version(gui_package_lock, gui_version)

    print(f"Updated version to {raw}")
    if gui_version != raw:
        print(f"GUI version normalized to {gui_version}")
    print(f"- {package_init}")
    print(f"- {plugin_manifest}")
    print(f"- {gui_package_json}")
    if gui_package_lock.exists():
        print(f"- {gui_package_lock}")
    return 0


def _set_python_version(path: Path, version: str) -> None:
    content = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^__version__\s*=\s*"[^"]+"$',
        f'__version__ = "{version}"',
        content,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Expected exactly one __version__ assignment in {path}")
    path.write_text(updated, encoding="utf-8")


def _set_plugin_version(path: Path, version: str) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["version"] = version
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _set_gui_version(path: Path, version: str) -> None:
    package_json = json.loads(path.read_text(encoding="utf-8"))
    package_json["version"] = version
    path.write_text(json.dumps(package_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _set_gui_lock_version(path: Path, version: str) -> None:
    if not path.exists():
        return

    lock = json.loads(path.read_text(encoding="utf-8"))

    # npm lockfile typically stores the root package version in two places:
    # - top-level "version"
    # - packages[""]{"version"}
    if isinstance(lock, dict):
        lock["version"] = version
        packages = lock.get("packages")
        if isinstance(packages, dict):
            root_pkg = packages.get("")
            if isinstance(root_pkg, dict):
                root_pkg["version"] = version

    path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _to_gui_semver(version: str) -> str:
    """Normalize PEP 440-like versions into SemVer for the GUI."""
    if _SEMVER_PATTERN.fullmatch(version):
        return version
    match = re.match(r"^(?P<base>[0-9]+\.[0-9]+\.[0-9]+)(?P<suffix>.*)$", version)
    if not match:
        return version
    base = match.group("base")
    suffix = match.group("suffix")
    if not suffix:
        return base
    if suffix.startswith(("-", "+")):
        return f"{base}{suffix}"
    if suffix.startswith("."):
        suffix = suffix[1:]
    return f"{base}-{suffix}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

