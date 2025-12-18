"""Logic helpers for diagnostics, editors, and update checks."""

from __future__ import annotations

import os
import re
import shlex
import shutil
from dataclasses import dataclass
from typing import Optional, Sequence
from urllib import error, request

from ..text import Messages

EDITOR_FALLBACKS = ("nano", "vi", "notepad", "notepad.exe")


@dataclass
class DoctorCheckResult:
    """Result of a single doctor check."""

    name: str
    passed: bool
    message: str
    detail: str | None = None


def check_command_on_path() -> DoctorCheckResult:
    """Check if vexor command is available on PATH."""
    path = find_command_on_path("vexor")
    if path:
        return DoctorCheckResult(
            name="Command",
            passed=True,
            message=Messages.DOCTOR_CMD_FOUND.format(path=path),
        )
    return DoctorCheckResult(
        name="Command",
        passed=False,
        message=Messages.DOCTOR_CMD_MISSING,
        detail=Messages.DOCTOR_CMD_MISSING_DETAIL,
    )


def check_config_exists() -> DoctorCheckResult:
    """Check if config file exists."""
    from ..config import CONFIG_FILE

    if CONFIG_FILE.exists():
        return DoctorCheckResult(
            name="Config",
            passed=True,
            message=Messages.DOCTOR_CONFIG_EXISTS.format(path=CONFIG_FILE),
        )
    return DoctorCheckResult(
        name="Config",
        passed=True,
        message=Messages.DOCTOR_CONFIG_DEFAULT,
        detail=str(CONFIG_FILE),
    )


def check_api_key_configured(provider: str, api_key: str | None) -> DoctorCheckResult:
    """Check if API key is configured."""
    from ..config import resolve_api_key

    resolved = resolve_api_key(api_key, provider)
    if resolved:
        masked = resolved[:4] + "..." + resolved[-4:] if len(resolved) > 12 else "****"
        return DoctorCheckResult(
            name="API Key",
            passed=True,
            message=Messages.DOCTOR_API_KEY_CONFIGURED.format(masked=masked),
        )
    return DoctorCheckResult(
        name="API Key",
        passed=False,
        message=Messages.DOCTOR_API_KEY_MISSING,
        detail=Messages.DOCTOR_API_KEY_MISSING_DETAIL,
    )


def check_api_connectivity(
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> DoctorCheckResult:
    """Test API connectivity with a minimal embedding request."""
    from ..config import resolve_api_key

    resolved_key = resolve_api_key(api_key, provider)
    if not resolved_key:
        return DoctorCheckResult(
            name="API Test",
            passed=False,
            message=Messages.DOCTOR_API_SKIPPED,
        )

    try:
        if provider == "gemini":
            from ..providers.gemini import GeminiEmbeddingBackend

            backend = GeminiEmbeddingBackend(
                model_name=model,
                api_key=resolved_key,
                base_url=base_url,
            )
        else:
            from ..providers.openai import OpenAIEmbeddingBackend

            backend = OpenAIEmbeddingBackend(
                model_name=model,
                api_key=resolved_key,
                base_url=base_url,
            )
        # Minimal test embedding
        result = backend.embed(["test"])
        if result.shape[0] == 1 and result.shape[1] > 0:
            return DoctorCheckResult(
                name="API Test",
                passed=True,
                message=Messages.DOCTOR_API_REACHABLE.format(model=model, dim=result.shape[1]),
            )
        return DoctorCheckResult(
            name="API Test",
            passed=False,
            message=Messages.DOCTOR_API_UNEXPECTED,
        )
    except Exception as exc:
        return DoctorCheckResult(
            name="API Test",
            passed=False,
            message=Messages.DOCTOR_API_FAILED,
            detail=str(exc),
        )


def check_cache_directory() -> DoctorCheckResult:
    """Check if cache directory exists and is writable."""
    from ..config import CONFIG_DIR

    if not CONFIG_DIR.exists():
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            return DoctorCheckResult(
                name="Cache Dir",
                passed=True,
                message=Messages.DOCTOR_CACHE_CREATED.format(path=CONFIG_DIR),
            )
        except OSError as exc:
            return DoctorCheckResult(
                name="Cache Dir",
                passed=False,
                message=Messages.DOCTOR_CACHE_CANNOT_CREATE.format(path=CONFIG_DIR),
                detail=str(exc),
            )

    # Check writable
    test_file = CONFIG_DIR / ".doctor_test"
    try:
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        return DoctorCheckResult(
            name="Cache Dir",
            passed=True,
            message=Messages.DOCTOR_CACHE_WRITABLE.format(path=CONFIG_DIR),
        )
    except OSError as exc:
        return DoctorCheckResult(
            name="Cache Dir",
            passed=False,
            message=Messages.DOCTOR_CACHE_NOT_WRITABLE.format(path=CONFIG_DIR),
            detail=str(exc),
        )


def run_all_doctor_checks(
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    *,
    skip_api_test: bool = False,
) -> list[DoctorCheckResult]:
    """Run all doctor checks and return results."""
    results = [
        check_command_on_path(),
        check_config_exists(),
        check_cache_directory(),
        check_api_key_configured(provider, api_key),
    ]
    if not skip_api_test:
        results.append(check_api_connectivity(provider, model, api_key, base_url))
    return results


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
