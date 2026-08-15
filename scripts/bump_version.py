#!/usr/bin/env python3
"""Bump Vexor versions in one command.

Updates the Python package, plugin manifest, and MCP server manifest.

Usage:
    python scripts/bump_version.py 0.6.4
    python scripts/bump_version.py v0.6.4
    python scripts/bump_version.py 0.6.4 --note "Reranker now reads chunk text"

--note starts docs/release-notes/<version>.md, an optional hand-written section
the release workflow publishes above the generated changelog. Fill in the body
before opening the bump PR; a file with only a heading fails the release.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[0-9A-Za-z.+-]+)?$")


def main(argv: list[str]) -> int:
    if any(arg in {"-h", "--help"} for arg in argv[1:]):
        print(__doc__.strip())
        return 2

    version, note_title = _parse_args(argv)
    return _run(
        version=version,
        repo_root=Path(__file__).resolve().parents[1],
        note_title=note_title,
    )


def _parse_args(argv: list[str]) -> tuple[str, str | None]:
    """Parse CLI args.

    Accepted forms:
      bump_version.py 0.1.2
      bump_version.py v0.1.2
      bump_version.py 0.1.2 --note "Release note title"
      bump_version.py 0.1.2 --note="Release note title"
    """
    positional: list[str] = []
    note_title: str | None = None
    pending_note = False
    for arg in argv[1:]:
        if pending_note:
            note_title = arg
            pending_note = False
            continue
        if arg == "--note":
            pending_note = True
        elif arg.startswith("--note="):
            note_title = arg[len("--note=") :]
        elif arg.startswith("-"):
            raise SystemExit(f"Unknown option '{arg}'. Use --help for usage.")
        else:
            positional.append(arg)

    if pending_note:
        raise SystemExit("Option '--note' requires a title. Use --help for usage.")
    if note_title is not None and not note_title.strip():
        raise SystemExit("Option '--note' requires a non-empty title.")

    if len(positional) != 1:
        print(__doc__.strip())
        raise SystemExit(2)

    raw_input = positional[0]
    raw = raw_input.strip()
    if raw.startswith("v"):
        raw = raw[1:]
    if not raw or not _VERSION_PATTERN.fullmatch(raw):
        raise SystemExit(f"Invalid version '{raw_input}'. Expected like 0.6.4")
    return raw, note_title.strip() if note_title is not None else None


def _run(*, version: str, repo_root: Path, note_title: str | None = None) -> int:
    package_init = repo_root / "vexor" / "__init__.py"
    plugin_manifest = repo_root / "plugins" / "vexor" / ".claude-plugin" / "plugin.json"
    mcp_server_manifest = repo_root / "server.json"

    _set_python_version(package_init, version)
    _set_plugin_version(plugin_manifest, version)

    print(f"Updated version to {version}")
    print(f"- {package_init}")
    print(f"- {plugin_manifest}")

    if mcp_server_manifest.exists():
        _set_mcp_server_version(mcp_server_manifest, version)
        print(f"- {mcp_server_manifest}")

    if note_title is not None:
        note_path = _start_release_note(repo_root, version=version, title=note_title)
        print(f"- {note_path} (write the body before opening the bump PR)")

    return 0


def _start_release_note(repo_root: Path, *, version: str, title: str) -> Path:
    """Create the hand-written release note stub the publish workflow reads."""
    path = repo_root / "docs" / "release-notes" / f"{version}.md"
    if path.exists():
        raise SystemExit(f"{path} already exists; edit it instead of re-running --note.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"## {title}\n\n", encoding="utf-8")
    return path


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


def _set_mcp_server_version(path: Path, version: str) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["version"] = version
    for package in manifest.get("packages", []):
        package["version"] = version
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
