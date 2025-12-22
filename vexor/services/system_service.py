"""Logic helpers for diagnostics, editors, and update checks."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse
from urllib import error, request

from ..config import (
    DEFAULT_FLASHRANK_MODEL,
    DEFAULT_RERANK,
    SUPPORTED_RERANKERS,
    RemoteRerankConfig,
    normalize_remote_rerank_url,
    resolve_remote_rerank_api_key,
)
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

    if (provider or "").lower() == "local":
        return DoctorCheckResult(
            name="API Key",
            passed=True,
            message=Messages.DOCTOR_API_KEY_NOT_REQUIRED,
        )
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
    local_cuda: bool = False,
) -> DoctorCheckResult:
    """Test API connectivity with a minimal embedding request."""
    from ..config import resolve_api_key

    normalized = (provider or "").lower()
    if normalized == "local":
        try:
            from ..providers.local import LocalEmbeddingBackend

            if local_cuda:
                try:
                    import onnxruntime as ort
                except Exception as exc:
                    return DoctorCheckResult(
                        name="Local Model",
                        passed=False,
                        message=Messages.DOCTOR_LOCAL_CUDA_IMPORT_FAILED,
                        detail=Messages.DOCTOR_LOCAL_CUDA_IMPORT_DETAIL.format(reason=str(exc)),
                    )
                try:
                    providers = ort.get_available_providers()
                except Exception as exc:
                    return DoctorCheckResult(
                        name="Local Model",
                        passed=False,
                        message=Messages.DOCTOR_LOCAL_CUDA_MISSING,
                        detail=Messages.DOCTOR_LOCAL_CUDA_IMPORT_DETAIL.format(reason=str(exc)),
                    )
                if "CUDAExecutionProvider" not in providers:
                    return DoctorCheckResult(
                        name="Local Model",
                        passed=False,
                        message=Messages.DOCTOR_LOCAL_CUDA_MISSING,
                        detail=Messages.DOCTOR_LOCAL_CUDA_MISSING_DETAIL.format(
                            providers=", ".join(providers) if providers else "none"
                        ),
                    )

            backend = LocalEmbeddingBackend(model_name=model, cuda=local_cuda)
            result = backend.embed(["test"])
            if result.shape[0] == 1 and result.shape[1] > 0:
                return DoctorCheckResult(
                    name="Local Model",
                    passed=True,
                    message=Messages.DOCTOR_LOCAL_READY.format(model=model, dim=result.shape[1]),
                )
            return DoctorCheckResult(
                name="Local Model",
                passed=False,
                message=Messages.DOCTOR_LOCAL_UNEXPECTED,
            )
        except Exception as exc:
            return DoctorCheckResult(
                name="Local Model",
                passed=False,
                message=Messages.DOCTOR_LOCAL_FAILED,
                detail=str(exc),
            )

    if normalized == "custom":
        if not (base_url and base_url.strip()):
            return DoctorCheckResult(
                name="API Test",
                passed=False,
                message=Messages.ERROR_CUSTOM_BASE_URL_REQUIRED,
            )
        if not (model and model.strip()):
            return DoctorCheckResult(
                name="API Test",
                passed=False,
                message=Messages.ERROR_CUSTOM_MODEL_REQUIRED,
            )

    resolved_key = resolve_api_key(api_key, normalized)
    if not resolved_key:
        return DoctorCheckResult(
            name="API Test",
            passed=False,
            message=Messages.DOCTOR_API_SKIPPED,
        )

    try:
        if normalized == "gemini":
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


def check_rerank_configured(
    rerank: str | None,
    *,
    flashrank_model: str | None,
    remote_rerank: RemoteRerankConfig | None,
    skip_api_test: bool,
) -> DoctorCheckResult | None:
    """Check whether the configured reranker is available and configured."""

    normalized = (rerank or DEFAULT_RERANK).strip().lower()
    if normalized in {"", "off", DEFAULT_RERANK}:
        return None
    if normalized not in SUPPORTED_RERANKERS:
        return DoctorCheckResult(
            name="Rerank",
            passed=False,
            message=Messages.ERROR_RERANK_INVALID.format(
                value=rerank, allowed=", ".join(SUPPORTED_RERANKERS)
            ),
        )
    if normalized == "bm25":
        if importlib.util.find_spec("rank_bm25") is None:
            return DoctorCheckResult(
                name="Rerank",
                passed=False,
                message=Messages.DOCTOR_RERANK_BM25_MISSING,
            )
        return DoctorCheckResult(
            name="Rerank",
            passed=True,
            message=Messages.DOCTOR_RERANK_BM25_READY,
        )
    if normalized == "flashrank":
        if importlib.util.find_spec("flashrank") is None:
            return DoctorCheckResult(
                name="Rerank",
                passed=False,
                message=Messages.DOCTOR_RERANK_FLASHRANK_MISSING,
            )
        model_label = flashrank_model or DEFAULT_FLASHRANK_MODEL
        return DoctorCheckResult(
            name="Rerank",
            passed=True,
            message=Messages.DOCTOR_RERANK_FLASHRANK_READY.format(model=model_label),
        )

    # Remote rerank
    if remote_rerank is None:
        return DoctorCheckResult(
            name="Rerank",
            passed=False,
            message=Messages.DOCTOR_RERANK_REMOTE_INCOMPLETE,
        )
    base_url = normalize_remote_rerank_url(remote_rerank.base_url)
    api_key = resolve_remote_rerank_api_key(remote_rerank.api_key)
    model = (remote_rerank.model or "").strip()
    if not (base_url and api_key and model):
        return DoctorCheckResult(
            name="Rerank",
            passed=False,
            message=Messages.DOCTOR_RERANK_REMOTE_INCOMPLETE,
        )
    if skip_api_test:
        return DoctorCheckResult(
            name="Rerank",
            passed=True,
            message=Messages.DOCTOR_RERANK_REMOTE_SKIPPED.format(model=model),
        )
    try:
        from .search_service import _remote_rerank_request

        _remote_rerank_request(
            config=RemoteRerankConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
            ),
            query="test",
            documents=["test"],
        )
    except RuntimeError as exc:
        return DoctorCheckResult(
            name="Rerank",
            passed=False,
            message=Messages.DOCTOR_RERANK_REMOTE_FAILED,
            detail=str(exc),
        )
    return DoctorCheckResult(
        name="Rerank",
        passed=True,
        message=Messages.DOCTOR_RERANK_REMOTE_READY.format(model=model),
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
    local_cuda: bool = False,
    rerank: str | None = None,
    flashrank_model: str | None = None,
    remote_rerank: RemoteRerankConfig | None = None,
) -> list[DoctorCheckResult]:
    """Run all doctor checks and return results."""
    results = [
        check_command_on_path(),
        check_config_exists(),
        check_cache_directory(),
        check_api_key_configured(provider, api_key),
    ]
    rerank_result = check_rerank_configured(
        rerank,
        flashrank_model=flashrank_model,
        remote_rerank=remote_rerank,
        skip_api_test=skip_api_test,
    )
    if rerank_result is not None:
        results.append(rerank_result)
    if not skip_api_test:
        results.append(
            check_api_connectivity(
                provider,
                model,
                api_key,
                base_url,
                local_cuda=local_cuda,
            )
        )
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


@dataclass(frozen=True, order=True, slots=True)
class ParsedVersion:
    release: tuple[int, int, int, int]
    stage: int
    stage_num: int
    raw: str
    is_prerelease: bool


def parse_version(raw: str) -> ParsedVersion | None:
    """Parse a semver-ish version string with optional a/b/rc suffix.

    Ordering follows PEP 440 semantics for pre-releases:
    a < b < rc < final for the same release segment.
    """

    text = (raw or "").strip()
    if not text:
        return None

    release_parts: list[int] = []
    stage = 3  # final
    stage_num = 0
    is_prerelease = False

    for piece in text.split("."):
        match = re.match(r"^(\d+)", piece)
        if not match:
            break
        release_parts.append(int(match.group(1)))
        remainder = piece[match.end() :]
        if remainder:
            suffix = remainder.lower()
            suffix_match = re.match(r"^(a|b|rc)(\d+)?", suffix)
            if suffix_match:
                label = suffix_match.group(1)
                stage_num = int(suffix_match.group(2) or 0)
                stage = {"a": 0, "b": 1, "rc": 2}.get(label, 0)
                is_prerelease = True
            break
        if len(release_parts) >= 4:
            break

    if not release_parts:
        return None

    while len(release_parts) < 4:
        release_parts.append(0)

    return ParsedVersion(
        release=tuple(release_parts[:4]),
        stage=stage,
        stage_num=stage_num,
        raw=text,
        is_prerelease=is_prerelease,
    )


def fetch_pypi_versions(package: str, *, timeout: float = 10.0) -> list[str]:
    """Fetch published versions from PyPI."""

    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            payload = response.read().decode("utf-8")
    except error.URLError as exc:  # pragma: no cover - network error
        raise RuntimeError(str(exc)) from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid PyPI response") from exc

    releases = data.get("releases", {})
    versions: list[str] = []
    for version, files in releases.items():
        if files:
            versions.append(version)
    return versions


def select_latest_version(versions: Sequence[str], *, include_prerelease: bool) -> str:
    parsed_versions: list[ParsedVersion] = []
    for version in versions:
        parsed = parse_version(version)
        if parsed is None:
            continue
        if not include_prerelease and parsed.is_prerelease:
            continue
        parsed_versions.append(parsed)
    if not parsed_versions:
        raise RuntimeError("No matching versions found")
    parsed_versions.sort()
    return parsed_versions[-1].raw


def fetch_latest_pypi_version(
    package: str,
    *,
    include_prerelease: bool = False,
    timeout: float = 10.0,
) -> str:
    versions = fetch_pypi_versions(package, timeout=timeout)
    return select_latest_version(versions, include_prerelease=include_prerelease)


class InstallMethod(str, Enum):
    PIP_USER = "pip-user"
    PIP_SYSTEM = "pip-system"
    PIP_VENV = "pip-venv"
    PIPX = "pipx"
    UV = "uv"
    GIT_EDITABLE = "git-editable"
    STANDALONE = "standalone"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class InstallInfo:
    method: InstallMethod
    executable: Path | None
    editable_root: Path | None
    dist_location: Path | None
    requires_admin: bool = False


def detect_install_method() -> InstallInfo:
    """Detect how the current Vexor process is installed."""

    if getattr(sys, "frozen", False):
        return InstallInfo(
            method=InstallMethod.STANDALONE,
            executable=Path(sys.executable).resolve(),
            editable_root=None,
            dist_location=None,
            requires_admin=False,
        )

    dist_location = None
    editable_root = None
    try:
        from importlib import metadata

        dist = metadata.distribution("vexor")
        dist_location = Path(dist.locate_file("")).resolve()
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            try:
                direct_data = json.loads(direct_url)
            except json.JSONDecodeError:
                direct_data = None
            if direct_data:
                dir_info = direct_data.get("dir_info") or {}
                if bool(dir_info.get("editable")):
                    parsed = urlparse(direct_data.get("url") or "")
                    if parsed.scheme == "file":
                        editable_root = Path(request.url2pathname(parsed.path)).resolve()
    except Exception:
        dist_location = None
        editable_root = None

    vexor_exe = find_command_on_path("vexor")
    executable = Path(vexor_exe).resolve() if vexor_exe else None

    if editable_root and (editable_root / ".git").exists():
        return InstallInfo(
            method=InstallMethod.GIT_EDITABLE,
            executable=executable,
            editable_root=editable_root,
            dist_location=dist_location,
            requires_admin=False,
        )

    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix)).resolve()

    if "pipx" in prefix.parts and "venvs" in prefix.parts:
        return InstallInfo(
            method=InstallMethod.PIPX,
            executable=executable,
            editable_root=None,
            dist_location=dist_location,
            requires_admin=False,
        )

    if "uv" in prefix.parts and "tools" in prefix.parts:
        return InstallInfo(
            method=InstallMethod.UV,
            executable=executable,
            editable_root=None,
            dist_location=dist_location,
            requires_admin=False,
        )

    if prefix != base_prefix:
        return InstallInfo(
            method=InstallMethod.PIP_VENV,
            executable=executable,
            editable_root=None,
            dist_location=dist_location,
            requires_admin=False,
        )

    user_site = None
    try:
        import site

        user_site = Path(site.getusersitepackages()).resolve()
    except Exception:  # pragma: no cover - platform specific
        user_site = None

    if dist_location and user_site and dist_location.is_relative_to(user_site):
        return InstallInfo(
            method=InstallMethod.PIP_USER,
            executable=executable,
            editable_root=None,
            dist_location=dist_location,
            requires_admin=False,
        )

    if dist_location:
        return InstallInfo(
            method=InstallMethod.PIP_SYSTEM,
            executable=executable,
            editable_root=None,
            dist_location=dist_location,
            requires_admin=True,
        )

    return InstallInfo(
        method=InstallMethod.UNKNOWN,
        executable=executable,
        editable_root=None,
        dist_location=None,
        requires_admin=False,
    )


def build_upgrade_commands(
    install_info: InstallInfo,
    *,
    include_prerelease: bool = False,
) -> list[list[str]]:
    """Return the command(s) used to upgrade based on *install_info*."""

    if install_info.method == InstallMethod.GIT_EDITABLE and install_info.editable_root:
        return [
            ["git", "-C", str(install_info.editable_root), "pull", "--ff-only"],
            [sys.executable, "-m", "pip", "install", "-e", str(install_info.editable_root)],
        ]

    if install_info.method == InstallMethod.PIPX:
        cmd = ["pipx", "upgrade", "vexor"]
        if include_prerelease:
            cmd.extend(["--pip-args", "--pre"])
        return [cmd]

    if install_info.method == InstallMethod.UV:
        cmd = ["uv", "tool", "upgrade", "vexor"]
        if include_prerelease:
            cmd.extend(["--prerelease", "allow"])
        return [cmd]

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if include_prerelease:
        cmd.append("--pre")
    if install_info.method == InstallMethod.PIP_USER:
        cmd.append("--user")
    cmd.append("vexor")
    return [cmd]


def build_standalone_download_url(version: str) -> tuple[str | None, str]:
    """Return a (asset_name, url) tuple for the standalone binary download."""

    system = platform.system().lower()
    asset_suffix = None
    if system.startswith("windows"):
        asset_suffix = "windows.exe"
    elif system.startswith("linux"):
        asset_suffix = "linux"

    tag = f"v{version}"
    base = f"https://github.com/scarletkc/vexor/releases/tag/{tag}"
    if asset_suffix is None:
        return None, base

    asset_name = f"vexor-{version}-{asset_suffix}"
    download = f"https://github.com/scarletkc/vexor/releases/download/{tag}/{asset_name}"
    return asset_name, download


def run_upgrade_commands(commands: Sequence[Sequence[str]], *, timeout: float = 300.0) -> int:
    """Run upgrade commands, returning the first non-zero exit code if any."""

    for command in commands:
        try:
            completed = subprocess.run(list(command), check=False, timeout=timeout)
        except FileNotFoundError:
            return 127
        except subprocess.TimeoutExpired:
            return 124
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def git_worktree_is_dirty(repo: Path, *, timeout: float = 10.0) -> bool:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return False
    return bool(completed.stdout.strip())


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
