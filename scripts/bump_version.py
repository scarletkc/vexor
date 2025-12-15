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

    _set_python_version(package_init, raw)
    _set_plugin_version(plugin_manifest, raw)

    print(f"Updated version to {raw}")
    print(f"- {package_init}")
    print(f"- {plugin_manifest}")
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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

