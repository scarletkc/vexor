from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from vexor.services import system_service
from vexor.config import RemoteRerankConfig


class DummyResponse:
    def __init__(self, *, status: int, body: str) -> None:
        self.status = status
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_version_tuple_parses_release_and_suffix():
    assert system_service.version_tuple("1.2.3") == (1, 2, 3, 0)
    assert system_service.version_tuple("1.2") == (1, 2, 0, 0)
    assert system_service.version_tuple("1") == (1, 0, 0, 0)
    assert system_service.version_tuple("1.2.3rc4") == (1, 2, 3, 4)
    assert system_service.version_tuple("  2.0.0b12 ") == (2, 0, 0, 12)
    assert system_service.version_tuple("nonsense") == (0, 0, 0, 0)
    assert system_service.version_tuple("1.2.3.4.5") == (1, 2, 3, 4)


def test_parse_version_orders_prereleases_before_final():
    assert system_service.parse_version("1.0.0a1") < system_service.parse_version("1.0.0b1")
    assert system_service.parse_version("1.0.0b1") < system_service.parse_version("1.0.0rc1")
    assert system_service.parse_version("1.0.0rc1") < system_service.parse_version("1.0.0")


def test_select_latest_version_respects_prerelease_flag():
    versions = ["0.9.2", "0.10.0a1"]
    assert system_service.select_latest_version(versions, include_prerelease=False) == "0.9.2"
    assert system_service.select_latest_version(versions, include_prerelease=True) == "0.10.0a1"


def test_fetch_remote_version_success(monkeypatch):
    def fake_urlopen(url, timeout=10.0):
        assert "example.com" in url
        assert timeout == 0.5
        return DummyResponse(status=200, body="__version__ = '9.9.9'\\n")

    monkeypatch.setattr(system_service.request, "urlopen", fake_urlopen)

    assert system_service.fetch_remote_version("https://example.com/x.py", timeout=0.5) == "9.9.9"


def test_fetch_remote_version_raises_on_non_200(monkeypatch):
    monkeypatch.setattr(
        system_service.request,
        "urlopen",
        lambda *_args, **_kwargs: DummyResponse(status=404, body="not found"),
    )
    with pytest.raises(RuntimeError, match=r"HTTP 404"):
        system_service.fetch_remote_version("https://example.com/x.py")


def test_fetch_remote_version_raises_when_missing_version(monkeypatch):
    monkeypatch.setattr(
        system_service.request,
        "urlopen",
        lambda *_args, **_kwargs: DummyResponse(status=200, body="print('no version')"),
    )
    with pytest.raises(RuntimeError, match="Version string not found"):
        system_service.fetch_remote_version("https://example.com/x.py")


def test_resolve_editor_command_from_env(monkeypatch):
    monkeypatch.setenv("VISUAL", "code --wait")
    assert system_service.resolve_editor_command() == ("code", "--wait")


def test_resolve_editor_command_uses_fallback(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)

    def fake_which(cmd: str):
        if cmd == "nano":
            return "/usr/bin/nano"
        return None

    monkeypatch.setattr(system_service.shutil, "which", fake_which)
    assert system_service.resolve_editor_command() == ("/usr/bin/nano",)


def test_resolve_editor_command_none_when_missing(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setattr(system_service.shutil, "which", lambda *_: None)
    assert system_service.resolve_editor_command() is None


def test_find_command_on_path(monkeypatch):
    monkeypatch.setattr(system_service.shutil, "which", lambda cmd: f"/bin/{cmd}")
    assert system_service.find_command_on_path("vexor") == "/bin/vexor"


def test_check_rerank_bm25_missing(monkeypatch):
    def fake_find_spec(name):
        if name == "rank_bm25":
            return None
        return object()

    monkeypatch.setattr(system_service.importlib.util, "find_spec", fake_find_spec)

    result = system_service.check_rerank_configured(
        "bm25",
        flashrank_model=None,
        remote_rerank=None,
        skip_api_test=False,
    )

    assert result is not None
    assert result.passed is False


def test_check_rerank_flashrank_ready(monkeypatch):
    monkeypatch.setattr(system_service.importlib.util, "find_spec", lambda _name: object())

    result = system_service.check_rerank_configured(
        "flashrank",
        flashrank_model=None,
        remote_rerank=None,
        skip_api_test=False,
    )

    assert result is not None
    assert result.passed is True


def test_check_rerank_hybrid_ready_and_degraded(monkeypatch):
    monkeypatch.setattr(system_service.importlib.util, "find_spec", lambda _name: object())
    ready = system_service.check_rerank_configured(
        "hybrid",
        flashrank_model=None,
        remote_rerank=None,
        skip_api_test=False,
    )
    assert ready is not None
    assert ready.passed is True

    monkeypatch.setattr(system_service.importlib.util, "find_spec", lambda _name: None)
    degraded = system_service.check_rerank_configured(
        "hybrid",
        flashrank_model=None,
        remote_rerank=None,
        skip_api_test=False,
    )
    assert degraded is not None
    assert degraded.passed is True
    assert "degraded" in degraded.message.lower()


def test_check_rerank_remote_incomplete():
    result = system_service.check_rerank_configured(
        "remote",
        flashrank_model=None,
        remote_rerank=None,
        skip_api_test=False,
    )

    assert result is not None
    assert result.passed is False


def test_check_rerank_remote_skipped():
    remote = RemoteRerankConfig(
        base_url="https://example.com/rerank",
        api_key="secret",
        model="model-x",
    )
    result = system_service.check_rerank_configured(
        "remote",
        flashrank_model=None,
        remote_rerank=remote,
        skip_api_test=True,
    )

    assert result is not None
    assert result.passed is True


def test_command_config_api_key_and_cache_checks(monkeypatch, tmp_path):
    monkeypatch.setattr(system_service, "find_command_on_path", lambda _cmd: "C:/bin/vexor.exe")
    assert system_service.check_command_on_path().passed is True

    monkeypatch.setattr(system_service, "find_command_on_path", lambda _cmd: None)
    missing = system_service.check_command_on_path()
    assert missing.passed is False
    assert missing.detail

    config_file = tmp_path / "config.json"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("vexor.config.CONFIG_FILE", config_file)
    assert system_service.check_config_exists().passed is True
    config_file.unlink()
    assert system_service.check_config_exists().passed is True

    assert system_service.check_api_key_configured("local", None).passed is True
    assert system_service.check_api_key_configured("openai", None).passed is False
    assert system_service.check_api_key_configured("openai", "abcd1234wxyz9999").passed is True

    config_dir = tmp_path / "cache"
    monkeypatch.setattr("vexor.config.CONFIG_DIR", config_dir)
    created = system_service.check_cache_directory()
    assert created.passed is True
    writable = system_service.check_cache_directory()
    assert writable.passed is True


def test_check_config_exists_includes_resolution_origins(tmp_path):
    from vexor.config import (
        CONFIG_FIELD_NAMES,
        Config,
        ConfigOrigin,
        ConfigResolution,
    )

    global_file = tmp_path / "global" / "config.json"
    global_file.parent.mkdir()
    global_file.write_text("{}", encoding="utf-8")
    project_file = tmp_path / "project" / ".vexor" / "config.json"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("{}", encoding="utf-8")
    origins = {field: ConfigOrigin.DEFAULT for field in CONFIG_FIELD_NAMES}
    origins["provider"] = ConfigOrigin.GLOBAL
    origins["model"] = ConfigOrigin.PROJECT
    resolution = ConfigResolution(
        config=Config(),
        origins=origins,
        global_file=global_file,
        project_file=project_file,
    )

    result = system_service.check_config_exists(resolution)

    assert result.passed is True
    assert result.detail is not None
    assert f"Project config: {project_file}" in result.detail
    assert "provider: global" in result.detail
    assert "model: project" in result.detail
    assert "base_url: default" in result.detail

    global_file.unlink()
    project_only = system_service.check_config_exists(resolution)
    assert "Project config file exists" in project_only.message


def test_check_cache_directory_reports_create_and_write_failures(monkeypatch, tmp_path):
    config_dir = tmp_path / "cache"
    monkeypatch.setattr("vexor.config.CONFIG_DIR", config_dir)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("nope")))
        created = system_service.check_cache_directory()

    assert created.passed is False

    config_dir.mkdir()

    def fail_write(self, *_args, **_kwargs):
        raise OSError("readonly")

    monkeypatch.setattr(Path, "write_text", fail_write)
    writable = system_service.check_cache_directory()
    assert writable.passed is False


def test_api_connectivity_local_success_unexpected_and_errors(monkeypatch):
    class ReadyLocalBackend:
        def __init__(self, *, model_name, cuda):
            assert model_name == "local-model"
            assert cuda is False

        def embed(self, _texts):
            import numpy as np

            return np.ones((1, 3), dtype=np.float32)

    monkeypatch.setattr("vexor.providers.local.LocalEmbeddingBackend", ReadyLocalBackend)
    ready = system_service.check_api_connectivity("local", "local-model", None, None)
    assert ready.passed is True

    class EmptyLocalBackend:
        def __init__(self, **_kwargs):
            pass

        def embed(self, _texts):
            import numpy as np

            return np.empty((0, 0), dtype=np.float32)

    monkeypatch.setattr("vexor.providers.local.LocalEmbeddingBackend", EmptyLocalBackend)
    unexpected = system_service.check_api_connectivity("local", "local-model", None, None)
    assert unexpected.passed is False

    class BrokenLocalBackend:
        def __init__(self, **_kwargs):
            raise RuntimeError("cannot load")

    monkeypatch.setattr("vexor.providers.local.LocalEmbeddingBackend", BrokenLocalBackend)
    failed = system_service.check_api_connectivity("local", "local-model", None, None)
    assert failed.passed is False
    assert failed.detail == "cannot load"


def test_api_connectivity_local_cuda_validation(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    missing = system_service.check_api_connectivity("local", "m", None, None, local_cuda=True)
    assert missing.passed is False

    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    no_cuda = system_service.check_api_connectivity("local", "m", None, None, local_cuda=True)
    assert no_cuda.passed is False


def test_api_connectivity_remote_validation_and_success(monkeypatch):
    assert system_service.check_api_connectivity("custom", "m", "key", None).passed is False
    assert system_service.check_api_connectivity("custom", "", "key", "https://x").passed is False
    assert system_service.check_api_connectivity("openai", "m", None, None).passed is False

    class ReadyBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed(self, _texts):
            import numpy as np

            return np.ones((1, 2), dtype=np.float32)

    monkeypatch.setattr("vexor.providers.openai.OpenAIEmbeddingBackend", ReadyBackend)
    reachable = system_service.check_api_connectivity("openai", "m", "key", "https://x")
    assert reachable.passed is True

    class BrokenBackend:
        def __init__(self, **_kwargs):
            pass

        def embed(self, _texts):
            raise RuntimeError("network down")

    monkeypatch.setattr("vexor.providers.gemini.GeminiEmbeddingBackend", BrokenBackend)
    failed = system_service.check_api_connectivity("gemini", "m", "key", None)
    assert failed.passed is False
    assert failed.detail == "network down"


def test_rerank_variants(monkeypatch):
    invalid = system_service.check_rerank_configured(
        "bogus",
        flashrank_model=None,
        remote_rerank=None,
        skip_api_test=False,
    )
    assert invalid is not None and invalid.passed is False

    monkeypatch.setattr(system_service.importlib.util, "find_spec", lambda name: object())
    bm25 = system_service.check_rerank_configured(
        "bm25",
        flashrank_model=None,
        remote_rerank=None,
        skip_api_test=False,
    )
    assert bm25 is not None and bm25.passed is True

    monkeypatch.setattr(system_service.importlib.util, "find_spec", lambda name: None)
    flashrank = system_service.check_rerank_configured(
        "flashrank",
        flashrank_model=None,
        remote_rerank=None,
        skip_api_test=False,
    )
    assert flashrank is not None and flashrank.passed is False


def test_remote_rerank_api_failure_and_success(monkeypatch):
    remote = RemoteRerankConfig(
        base_url="https://example.com/rerank",
        api_key="secret",
        model="model-x",
    )

    def fail_request(**_kwargs):
        raise RuntimeError("bad gateway")

    monkeypatch.setattr("vexor.services.search_service._remote_rerank_request", fail_request)
    failed = system_service.check_rerank_configured(
        "remote",
        flashrank_model=None,
        remote_rerank=remote,
        skip_api_test=False,
    )
    assert failed is not None and failed.passed is False

    monkeypatch.setattr(
        "vexor.services.search_service._remote_rerank_request",
        lambda **_kwargs: {"results": [{"index": 0}]},
    )
    ready = system_service.check_rerank_configured(
        "remote",
        flashrank_model=None,
        remote_rerank=remote,
        skip_api_test=False,
    )
    assert ready is not None and ready.passed is True


def test_run_all_doctor_checks_respects_skip_api(monkeypatch):
    monkeypatch.setattr(system_service, "check_command_on_path", lambda: system_service.DoctorCheckResult("cmd", True, "ok"))
    monkeypatch.setattr(system_service, "check_config_exists", lambda: system_service.DoctorCheckResult("cfg", True, "ok"))
    monkeypatch.setattr(system_service, "check_cache_directory", lambda: system_service.DoctorCheckResult("cache", True, "ok"))
    monkeypatch.setattr(system_service, "check_api_key_configured", lambda *_args: system_service.DoctorCheckResult("key", True, "ok"))
    monkeypatch.setattr(system_service, "check_api_connectivity", lambda *_args, **_kwargs: pytest.fail("should skip"))

    results = system_service.run_all_doctor_checks(
        provider="openai",
        model="m",
        api_key=None,
        base_url=None,
        skip_api_test=True,
        rerank="off",
    )

    assert [result.name for result in results] == ["cmd", "cfg", "cache", "key"]


def test_fetch_pypi_versions_and_latest(monkeypatch):
    payload = json.dumps(
        {
            "releases": {
                "1.0.0": [{"filename": "pkg.whl"}],
                "1.1.0a1": [{"filename": "pkg.whl"}],
                "broken": [{"filename": "pkg.whl"}],
                "0.1.0": [],
            }
        }
    )
    monkeypatch.setattr(
        system_service.request,
        "urlopen",
        lambda *_args, **_kwargs: DummyResponse(status=200, body=payload),
    )

    versions = system_service.fetch_pypi_versions("vexor", timeout=0.1)

    assert versions == ["1.0.0", "1.1.0a1", "broken"]
    assert system_service.fetch_latest_pypi_version("vexor") == "1.0.0"
    assert system_service.fetch_latest_pypi_version("vexor", include_prerelease=True) == "1.1.0a1"


def test_fetch_pypi_versions_rejects_bad_http_and_json(monkeypatch):
    monkeypatch.setattr(
        system_service.request,
        "urlopen",
        lambda *_args, **_kwargs: DummyResponse(status=500, body="nope"),
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        system_service.fetch_pypi_versions("vexor")

    monkeypatch.setattr(
        system_service.request,
        "urlopen",
        lambda *_args, **_kwargs: DummyResponse(status=200, body="{"),
    )
    with pytest.raises(RuntimeError, match="Invalid PyPI response"):
        system_service.fetch_pypi_versions("vexor")

    with pytest.raises(RuntimeError, match="No matching versions"):
        system_service.select_latest_version(["bad"], include_prerelease=False)


def test_detect_install_method_variants(monkeypatch, tmp_path):
    monkeypatch.setattr(system_service.sys, "frozen", True, raising=False)
    frozen = system_service.detect_install_method()
    assert frozen.method == system_service.InstallMethod.STANDALONE

    monkeypatch.setattr(system_service.sys, "frozen", False, raising=False)
    monkeypatch.setattr(
        "importlib.metadata.distribution",
        lambda _name: (_ for _ in ()).throw(RuntimeError("none")),
    )
    monkeypatch.setattr(system_service, "find_command_on_path", lambda _cmd: None)
    monkeypatch.setattr(system_service.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(system_service.sys, "base_prefix", str(tmp_path / "base"), raising=False)
    venv = system_service.detect_install_method()
    assert venv.method == system_service.InstallMethod.PIP_VENV


def test_detect_install_method_editable_user_system_and_unknown(monkeypatch, tmp_path):
    class DummyDist:
        def __init__(self, location: Path, direct_url: str | None):
            self.location = location
            self.direct_url = direct_url

        def locate_file(self, _name):
            return self.location

        def read_text(self, name):
            assert name == "direct_url.json"
            return self.direct_url

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    direct_url = json.dumps(
        {"url": repo.as_uri(), "dir_info": {"editable": True}}
    )
    monkeypatch.setattr(system_service.sys, "frozen", False, raising=False)
    monkeypatch.setattr(system_service.sys, "prefix", str(tmp_path / "base"))
    monkeypatch.setattr(system_service.sys, "base_prefix", str(tmp_path / "base"), raising=False)
    monkeypatch.setattr(system_service, "find_command_on_path", lambda _cmd: str(tmp_path / "vexor.exe"))
    monkeypatch.setattr("importlib.metadata.distribution", lambda _name: DummyDist(repo, direct_url))
    editable = system_service.detect_install_method()
    assert editable.method == system_service.InstallMethod.GIT_EDITABLE

    site_dir = tmp_path / "site"
    dist_dir = site_dir / "vexor"
    dist_dir.mkdir(parents=True)
    monkeypatch.setattr("importlib.metadata.distribution", lambda _name: DummyDist(dist_dir, None))
    monkeypatch.setattr("site.getusersitepackages", lambda: str(site_dir))
    user = system_service.detect_install_method()
    assert user.method == system_service.InstallMethod.PIP_USER

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.setattr("importlib.metadata.distribution", lambda _name: DummyDist(other_dir, None))
    system = system_service.detect_install_method()
    assert system.method == system_service.InstallMethod.PIP_SYSTEM

    def raise_distribution(_name):
        raise RuntimeError("missing")

    monkeypatch.setattr("importlib.metadata.distribution", raise_distribution)
    unknown = system_service.detect_install_method()
    assert unknown.method == system_service.InstallMethod.UNKNOWN


def test_detect_install_method_pipx_and_uv(monkeypatch, tmp_path):
    monkeypatch.setattr(system_service.sys, "frozen", False, raising=False)
    monkeypatch.setattr("importlib.metadata.distribution", lambda _name: (_ for _ in ()).throw(RuntimeError("none")))
    monkeypatch.setattr(system_service, "find_command_on_path", lambda _cmd: None)
    monkeypatch.setattr(system_service.sys, "base_prefix", str(tmp_path / "base"), raising=False)

    pipx_prefix = tmp_path / "pipx" / "venvs" / "vexor"
    monkeypatch.setattr(system_service.sys, "prefix", str(pipx_prefix))
    assert system_service.detect_install_method().method == system_service.InstallMethod.PIPX

    uv_prefix = tmp_path / "uv" / "tools" / "vexor"
    monkeypatch.setattr(system_service.sys, "prefix", str(uv_prefix))
    assert system_service.detect_install_method().method == system_service.InstallMethod.UV


def test_upgrade_commands_download_urls_and_runners(monkeypatch, tmp_path):
    editable = system_service.InstallInfo(
        method=system_service.InstallMethod.GIT_EDITABLE,
        executable=None,
        editable_root=tmp_path,
        dist_location=None,
    )
    assert system_service.build_upgrade_commands(editable)[0][:3] == ["git", "-C", str(tmp_path)]

    pipx = system_service.InstallInfo(system_service.InstallMethod.PIPX, None, None, None)
    assert "--pre" in system_service.build_upgrade_commands(pipx, include_prerelease=True)[0]

    uv = system_service.InstallInfo(system_service.InstallMethod.UV, None, None, None)
    assert "allow" in system_service.build_upgrade_commands(uv, include_prerelease=True)[0]

    user = system_service.InstallInfo(system_service.InstallMethod.PIP_USER, None, None, None)
    assert "--user" in system_service.build_upgrade_commands(user)[0]

    monkeypatch.setattr(system_service.platform, "system", lambda: "Windows")
    asset, url = system_service.build_standalone_download_url("1.2.3")
    assert asset == "vexor-1.2.3-windows.exe"
    assert url.endswith(asset)

    monkeypatch.setattr(system_service.platform, "system", lambda: "Darwin")
    asset, url = system_service.build_standalone_download_url("1.2.3")
    assert asset is None
    assert url.endswith("/v1.2.3")

    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(system_service.subprocess, "run", fake_run)
    assert system_service.run_upgrade_commands([["ok"], ["also-ok"]]) == 0
    assert calls == [["ok"], ["also-ok"]]

    monkeypatch.setattr(system_service.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=7))
    assert system_service.run_upgrade_commands([["bad"]]) == 7

    monkeypatch.setattr(system_service.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    assert system_service.run_upgrade_commands([["missing"]]) == 127

    monkeypatch.setattr(system_service.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 1)))
    assert system_service.run_upgrade_commands([["slow"]]) == 124


def test_git_worktree_dirty_handles_success_clean_and_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(
        system_service.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=" M file.py\n"),
    )
    assert system_service.git_worktree_is_dirty(tmp_path) is True

    monkeypatch.setattr(
        system_service.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=""),
    )
    assert system_service.git_worktree_is_dirty(tmp_path) is False

    monkeypatch.setattr(
        system_service.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("git missing")),
    )
    assert system_service.git_worktree_is_dirty(tmp_path) is False


def test_check_for_update_returns_newer_and_caches(tmp_path, monkeypatch):
    from vexor.services import system_service

    calls = []

    def fake_fetch(package, *, timeout=5.0, **kw):
        calls.append(package)
        return "9.9.9"

    monkeypatch.setattr(system_service, "fetch_latest_pypi_version", fake_fetch)
    state_file = tmp_path / "update_check.json"

    latest = system_service.check_for_update("0.1.0", state_file=state_file)
    assert latest == "9.9.9"
    assert state_file.exists()

    # Second call within the TTL is served from the cache.
    latest = system_service.check_for_update("0.1.0", state_file=state_file)
    assert latest == "9.9.9"
    assert len(calls) == 1


def test_check_for_update_returns_none_when_current(tmp_path, monkeypatch):
    from vexor.services import system_service

    monkeypatch.setattr(
        system_service, "fetch_latest_pypi_version", lambda *a, **kw: "0.1.0"
    )
    state_file = tmp_path / "update_check.json"
    assert system_service.check_for_update("0.1.0", state_file=state_file) is None
    assert system_service.check_for_update("9.9.9", state_file=state_file) is None


def test_check_for_update_expired_ttl_refetches(tmp_path, monkeypatch):
    import json as json_module

    from vexor.services import system_service

    state_file = tmp_path / "update_check.json"
    state_file.write_text(
        json_module.dumps({"checked_at": 0, "latest": "0.0.1"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        system_service, "fetch_latest_pypi_version", lambda *a, **kw: "9.9.9"
    )
    latest = system_service.check_for_update(
        "0.1.0", state_file=state_file, ttl_seconds=60
    )
    assert latest == "9.9.9"


def test_check_for_update_never_raises(tmp_path, monkeypatch):
    from vexor.services import system_service

    def _boom(*a, **kw):
        raise RuntimeError("offline")

    monkeypatch.setattr(system_service, "fetch_latest_pypi_version", _boom)
    state_file = tmp_path / "update_check.json"
    assert system_service.check_for_update("0.1.0", state_file=state_file) is None


def test_check_for_update_cache_only_mode(tmp_path, monkeypatch):
    from vexor.services import system_service

    def _no_network(*a, **kw):
        raise AssertionError("cache-only mode must not touch the network")

    monkeypatch.setattr(system_service, "fetch_latest_pypi_version", _no_network)
    state_file = tmp_path / "update_check.json"

    # Empty cache: returns None without fetching.
    assert (
        system_service.check_for_update(
            "0.1.0", state_file=state_file, allow_network=False
        )
        is None
    )

    system_service.write_update_cache("9.9.9", state_file=state_file)
    assert (
        system_service.check_for_update(
            "0.1.0", state_file=state_file, allow_network=False
        )
        == "9.9.9"
    )


def test_update_check_enabled_env_and_config(monkeypatch):
    from vexor.config import Config, ENV_NO_UPDATE_CHECK
    from vexor.services import system_service

    monkeypatch.delenv(ENV_NO_UPDATE_CHECK, raising=False)
    assert system_service.update_check_enabled(Config()) is True
    assert system_service.update_check_enabled(Config(update_check=False)) is False

    monkeypatch.setenv(ENV_NO_UPDATE_CHECK, "1")
    assert system_service.update_check_enabled(Config()) is False
