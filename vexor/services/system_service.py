"""Logic helpers for diagnostics, editors, and update checks."""

from __future__ import annotations

import os
import re
import shlex
import shutil
from typing import Optional, Sequence
from urllib import error, request

EDITOR_FALLBACKS = ("nano", "vi", "notepad", "notepad.exe")


def version_tuple(raw: str) -> tuple[int, int, int, int]:
    """Parse a version string into a comparable tuple."""

    raw = raw.strip()
    release_parts: list[int] = []
    suffix_number = 0

    for piece in raw.split('.'):
        match = re.match(r"^(\d+)", piece)
        if not match:
            break
        release_parts.append(int(match.group(1)))
        remainder = piece[match.end():]
        if remainder:
            suffix_match = re.match(r"[A-Za-z]+(\d+)", remainder)
            if suffix_match:
                suffix_number = int(suffix_match.group(1))
            break
        if len(release_parts) >= 4:
            break

    while len(release_parts) < 4:
        release_parts.append(0)

    if suffix_number:
        release_parts[3] = suffix_number

    return tuple(release_parts[:4])


def fetch_remote_version(url: str, *, timeout: float = 10.0) -> str:
    """Fetch the latest version string from *url*."""

    try:
        with request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            text = response.read().decode("utf-8")
    except error.URLError as exc:  # pragma: no cover - network error
        raise RuntimeError(str(exc)) from exc

    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not match:
        raise RuntimeError("Version string not found")
    return match.group(1)


def find_command_on_path(command: str) -> Optional[str]:
    """Return the resolved path for *command* if present on PATH."""

    return shutil.which(command)


def resolve_editor_command() -> Optional[Sequence[str]]:
    """Return the preferred editor command as a tokenized sequence."""

    for env_var in ("VISUAL", "EDITOR"):
        value = os.environ.get(env_var)
        if value:
            return tuple(shlex.split(value))

    for candidate in EDITOR_FALLBACKS:
        path = shutil.which(candidate)
        if path:
            return (path,)

    return None
