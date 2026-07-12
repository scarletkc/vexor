from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vexor.config import DEFAULT_LOCAL_MODEL, RemoteRerankConfig
from vexor.services import init_service
from vexor.services.skill_service import SkillInstallResult, SkillInstallStatus
from vexor.services.system_service import DoctorCheckResult, InstallInfo, InstallMethod


def _prompt_sequence(monkeypatch, values):
    iterator = iter(values)
    monkeypatch.setattr(init_service.typer, "prompt", lambda *_args, **_kwargs: next(iterator))


def _confirm_sequence(monkeypatch, values):
    iterator = iter(values)
    monkeypatch.setattr(init_service.typer, "confirm", lambda *_args, **_kwargs: next(iterator))


def test_prompt_choice_required_and_secret_loops(monkeypatch):
    _prompt_sequence(monkeypatch, ["bad", "", "Yes"])
    assert init_service._prompt_choice(
        "Pick",
        {"yes": "selected"},
        default="yes",
        allowed="yes",
    ) == "selected"

    _prompt_sequence(monkeypatch, ["", " value "])
    assert init_service._prompt_required("Required") == "value"

    monkeypatch.setattr(init_service, "resolve_remote_rerank_api_key", lambda _value: "env-key")
    _prompt_sequence(monkeypatch, [""])
    assert init_service._prompt_required_secret("Secret") is None

    monkeypatch.setattr(init_service, "resolve_remote_rerank_api_key", lambda _value: None)
    _prompt_sequence(monkeypatch, ["", "typed-key"])
    assert init_service._prompt_required_secret("Secret") == "typed-key"


def test_prompt_api_key_typed_env_and_skip(monkeypatch):
    _prompt_sequence(monkeypatch, ["typed-key"])
    assert init_service._prompt_api_key("API", "openai") == "typed-key"

    monkeypatch.setattr(init_service, "resolve_api_key", lambda _value, _provider: "env-key")
    _prompt_sequence(monkeypatch, [""])
    assert init_service._prompt_api_key("API", "openai") is None

    monkeypatch.setattr(init_service, "resolve_api_key", lambda _value, _provider: None)
    _prompt_sequence(monkeypatch, [""])
    _confirm_sequence(monkeypatch, [True])
    assert init_service._prompt_api_key("API", "openai") is None


def test_collect_remote_settings_variants(monkeypatch):
    _prompt_sequence(monkeypatch, ["custom", "https://proxy.example.com", "embed-model", "api-key"])
    custom = init_service._collect_remote_settings()
    assert custom == {
        "provider": "custom",
        "base_url": "https://proxy.example.com",
        "model": "embed-model",
        "api_key": "api-key",
    }

    _prompt_sequence(monkeypatch, ["gemini", "gemini-key"])
    gemini = init_service._collect_remote_settings()
    assert gemini == {"provider": "gemini", "api_key": "gemini-key"}

    _prompt_sequence(monkeypatch, ["voyage", "voyage-key"])
    voyage = init_service._collect_remote_settings()
    assert voyage == {"provider": "voyageai", "api_key": "voyage-key"}

    _prompt_sequence(monkeypatch, ["", "openai-key"])
    openai = init_service._collect_remote_settings()
    assert openai == {"provider": "openai", "api_key": "openai-key"}


def test_collect_provider_settings_local_and_fallback(monkeypatch):
    monkeypatch.setattr(init_service, "_collect_local_settings", lambda dry_run: {"provider": "local"})
    _prompt_sequence(monkeypatch, ["local"])
    assert init_service._collect_provider_settings(dry_run=False) == {"provider": "local"}

    monkeypatch.setattr(init_service, "_collect_local_settings", lambda dry_run: None)
    monkeypatch.setattr(init_service, "_collect_remote_settings", lambda: {"provider": "openai"})
    _prompt_sequence(monkeypatch, ["local"])
    assert init_service._collect_provider_settings(dry_run=False) == {"provider": "openai"}


def test_collect_local_settings_fastembed_missing_install_paths(monkeypatch):
    _prompt_sequence(monkeypatch, ["cpu"])
    _confirm_sequence(monkeypatch, [True, True, True])
    monkeypatch.setattr(init_service, "_is_fastembed_available", lambda: False)
    monkeypatch.setattr(init_service, "_install_extras", lambda extras, dry_run: False)
    assert init_service._collect_local_settings(dry_run=False) is None

    _prompt_sequence(monkeypatch, ["cpu"])
    _confirm_sequence(monkeypatch, [False, False])
    monkeypatch.setattr(init_service, "_is_fastembed_available", lambda: False)
    assert init_service._collect_local_settings(dry_run=False) is None

    _prompt_sequence(monkeypatch, ["cpu"])
    _confirm_sequence(monkeypatch, [False, True])
    monkeypatch.setattr(init_service, "_is_fastembed_available", lambda: False)
    result = init_service._collect_local_settings(dry_run=False)
    assert result == {
        "provider": "local",
        "model": DEFAULT_LOCAL_MODEL,
        "local_cuda": False,
    }


def test_collect_local_settings_cuda_and_prepare_paths(monkeypatch):
    _prompt_sequence(monkeypatch, ["cuda"])
    _confirm_sequence(monkeypatch, [True, True])
    monkeypatch.setattr(init_service, "_ensure_cuda_available", lambda: False)
    monkeypatch.setattr(init_service, "_is_fastembed_available", lambda: True)
    monkeypatch.setattr(init_service, "_prepare_local_model", lambda *_args, **_kwargs: True)
    result = init_service._collect_local_settings(dry_run=False)
    assert result["local_cuda"] is False

    _prompt_sequence(monkeypatch, ["cpu"])
    _confirm_sequence(monkeypatch, [True, True])
    monkeypatch.setattr(init_service, "_is_fastembed_available", lambda: True)
    monkeypatch.setattr(init_service, "_prepare_local_model", lambda *_args, **_kwargs: False)
    assert init_service._collect_local_settings(dry_run=False) is None


def test_collect_rerank_settings_variants(monkeypatch):
    _prompt_sequence(monkeypatch, ["bm25"])
    assert init_service._collect_rerank_settings(dry_run=False) == {"rerank": "bm25"}

    _prompt_sequence(monkeypatch, ["flashrank", "bm25"])
    monkeypatch.setattr(init_service, "_ensure_flashrank_available", lambda dry_run: False)
    assert init_service._collect_rerank_settings(dry_run=False) == {"rerank": "bm25"}

    prepared = {"called": False}
    _prompt_sequence(monkeypatch, ["flashrank"])
    monkeypatch.setattr(init_service, "_ensure_flashrank_available", lambda dry_run: True)
    monkeypatch.setattr(init_service, "_maybe_prepare_flashrank_model", lambda dry_run: prepared.update(called=True))
    assert init_service._collect_rerank_settings(dry_run=False) == {"rerank": "flashrank"}
    assert prepared["called"] is True

    _prompt_sequence(monkeypatch, ["remote", "", "https://rerank.example.com", "model-x", "secret"])
    remote = init_service._collect_rerank_settings(dry_run=False)
    assert remote == {
        "rerank": "remote",
        "remote_rerank_url": "https://rerank.example.com/rerank",
        "remote_rerank_model": "model-x",
        "remote_rerank_api_key": "secret",
    }

    _prompt_sequence(monkeypatch, ["hybrid"])
    assert init_service._collect_rerank_settings(dry_run=False) == {
        "rerank": "hybrid"
    }


def test_alias_helpers_and_prompt_alias_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("SHELL", "/bin/zsh")
    assert init_service._detect_shell_name() == "zsh"
    monkeypatch.setenv("SHELL", "/usr/bin/fish")
    assert init_service._detect_shell_name() == "fish"
    monkeypatch.setenv("SHELL", "/bin/unknown")
    monkeypatch.setattr(init_service.os, "name", "posix", raising=False)
    assert init_service._detect_shell_name() is None

    assert "vexor" in init_service._resolve_alias_command("fish")
    assert "Set-Alias" in init_service._resolve_alias_command("powershell")
    assert init_service._resolve_alias_command("bash").startswith("alias vx=")

    profile = tmp_path / ".bashrc"
    monkeypatch.setattr(init_service, "_detect_shell_name", lambda: "bash")
    monkeypatch.setattr(init_service, "_resolve_alias_profile", lambda _shell: profile)
    _confirm_sequence(monkeypatch, [True])
    init_service._prompt_alias_setup(dry_run=False)
    assert init_service._resolve_alias_command("bash") in profile.read_text(encoding="utf-8")

    _confirm_sequence(monkeypatch, [True])
    init_service._prompt_alias_setup(dry_run=False)
    assert profile.read_text(encoding="utf-8").count("alias vx=") == 1

    _confirm_sequence(monkeypatch, [True])
    monkeypatch.setattr(init_service, "_resolve_alias_profile", lambda _shell: None)
    init_service._prompt_alias_setup(dry_run=False)

    _confirm_sequence(monkeypatch, [True])
    init_service._prompt_alias_setup(dry_run=True)

    _confirm_sequence(monkeypatch, [False])
    init_service._prompt_alias_setup(dry_run=False)


def test_prompt_skill_install_and_doctor(monkeypatch):
    _confirm_sequence(monkeypatch, [False])
    init_service._prompt_skill_install(dry_run=False)

    _confirm_sequence(monkeypatch, [True])
    init_service._prompt_skill_install(dry_run=True)

    installed = {"target": None}
    _confirm_sequence(monkeypatch, [True])
    _prompt_sequence(monkeypatch, ["D", "/tmp/skills"])
    monkeypatch.setattr(init_service, "_install_skills", lambda target: installed.update(target=target))
    init_service._prompt_skill_install(dry_run=False)
    assert installed["target"] == "/tmp/skills"

    _confirm_sequence(monkeypatch, [False])
    init_service._prompt_doctor_check(dry_run=False)

    _confirm_sequence(monkeypatch, [True])
    init_service._prompt_doctor_check(dry_run=True)

    called = {"doctor": False}
    _confirm_sequence(monkeypatch, [True])
    monkeypatch.setattr(init_service, "_run_doctor_checks", lambda: called.update(doctor=True))
    init_service._prompt_doctor_check(dry_run=False)
    assert called["doctor"] is True


def test_run_doctor_checks_handles_invalid_and_success_config(monkeypatch):
    def bad_load_config():
        raise json.JSONDecodeError("bad", "{", 0)

    monkeypatch.setattr(init_service, "load_config", bad_load_config)
    monkeypatch.setattr(
        init_service,
        "run_all_doctor_checks",
        lambda **_kwargs: [DoctorCheckResult("API", True, "ok")],
    )
    init_service._run_doctor_checks()

    config = SimpleNamespace(
        provider="openai",
        model=None,
        api_key="key",
        base_url=None,
        local_cuda=False,
        rerank="remote",
        flashrank_model=None,
        remote_rerank=RemoteRerankConfig(
            base_url="https://rerank.example.com",
            api_key="key",
            model="model",
        ),
    )
    monkeypatch.setattr(init_service, "load_config", lambda: config)
    monkeypatch.setattr(
        init_service,
        "run_all_doctor_checks",
        lambda **_kwargs: [DoctorCheckResult("API", False, "failed", "detail")],
    )
    init_service._run_doctor_checks()


def test_install_skills_status_and_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(init_service, "resolve_skill_roots", lambda _target: [tmp_path / "skills"])
    monkeypatch.setattr(
        init_service,
        "install_bundled_skill",
        lambda **_kwargs: SkillInstallResult(
            status=SkillInstallStatus.installed,
            destination=tmp_path / "skills" / "vexor-cli",
        ),
    )
    init_service._install_skills("auto")

    monkeypatch.setattr(
        init_service,
        "install_bundled_skill",
        lambda **_kwargs: SkillInstallResult(
            status=SkillInstallStatus.up_to_date,
            destination=tmp_path / "skills" / "vexor-cli",
        ),
    )
    init_service._install_skills("auto")

    monkeypatch.setattr(init_service, "resolve_skill_roots", lambda _target: (_ for _ in ()).throw(ValueError("bad target")))
    init_service._install_skills("bad")

    monkeypatch.setattr(init_service, "resolve_skill_roots", lambda _target: [tmp_path / "skills"])
    monkeypatch.setattr(
        init_service,
        "install_bundled_skill",
        lambda **_kwargs: (_ for _ in ()).throw(FileExistsError(tmp_path / "skills" / "vexor-cli")),
    )
    init_service._install_skills("auto")

    monkeypatch.setattr(
        init_service,
        "install_bundled_skill",
        lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing bundled skill")),
    )
    init_service._install_skills("auto")


def test_cuda_fastembed_flashrank_and_model_prepare(monkeypatch):
    monkeypatch.setattr(init_service.importlib.util, "find_spec", lambda name: object() if name == "fastembed" else None)
    assert init_service._is_fastembed_available() is True
    assert init_service._is_flashrank_available() is False

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert init_service._ensure_cuda_available() is False

    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    assert init_service._ensure_cuda_available() is False

    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    assert init_service._ensure_cuda_available() is True

    class ReadyLocalBackend:
        def __init__(self, *, model_name, cuda):
            self.model_name = model_name
            self.cuda = cuda

        def embed(self, _texts):
            return np.ones((1, 2), dtype=np.float32)

    monkeypatch.setattr(init_service, "LocalEmbeddingBackend", ReadyLocalBackend)
    assert init_service._prepare_local_model("model", False, dry_run=False) is True
    assert init_service._prepare_local_model("model", False, dry_run=True) is True

    class EmptyLocalBackend(ReadyLocalBackend):
        def embed(self, _texts):
            return np.empty((0, 0), dtype=np.float32)

    monkeypatch.setattr(init_service, "LocalEmbeddingBackend", EmptyLocalBackend)
    assert init_service._prepare_local_model("model", False, dry_run=False) is False

    class BrokenLocalBackend(ReadyLocalBackend):
        def __init__(self, **_kwargs):
            raise RuntimeError("broken")

    monkeypatch.setattr(init_service, "LocalEmbeddingBackend", BrokenLocalBackend)
    assert init_service._prepare_local_model("model", False, dry_run=False) is False


def test_flashrank_prepare_and_install_extras(monkeypatch, tmp_path):
    monkeypatch.setattr(init_service, "_is_flashrank_available", lambda: True)
    assert init_service._ensure_flashrank_available(dry_run=False) is True

    monkeypatch.setattr(init_service, "_is_flashrank_available", lambda: False)
    assert init_service._ensure_flashrank_available(dry_run=True) is True

    _confirm_sequence(monkeypatch, [False])
    assert init_service._ensure_flashrank_available(dry_run=False) is False

    _confirm_sequence(monkeypatch, [True])
    monkeypatch.setattr(init_service, "_install_extras", lambda extras, dry_run: False)
    assert init_service._ensure_flashrank_available(dry_run=False) is False

    _confirm_sequence(monkeypatch, [False])
    init_service._maybe_prepare_flashrank_model(dry_run=False)

    _confirm_sequence(monkeypatch, [True])
    init_service._maybe_prepare_flashrank_model(dry_run=True)

    with monkeypatch.context() as patch:
        _confirm_sequence(patch, [True])
        patch.setattr(init_service, "_prepare_flashrank_model", lambda _model: (_ for _ in ()).throw(RuntimeError("bad model")))
        init_service._maybe_prepare_flashrank_model(dry_run=False)

    flashrank_module = ModuleType("flashrank")

    class DummyRanker:
        kwargs = None

        def __init__(self, **kwargs):
            DummyRanker.kwargs = kwargs

    flashrank_module.Ranker = DummyRanker
    monkeypatch.setitem(sys.modules, "flashrank", flashrank_module)
    monkeypatch.setattr(init_service, "flashrank_cache_dir", lambda: tmp_path)
    init_service._prepare_flashrank_model(None)
    assert DummyRanker.kwargs["cache_dir"] == str(tmp_path)

    monkeypatch.setitem(sys.modules, "flashrank", ModuleType("flashrank"))
    with pytest.raises(RuntimeError):
        init_service._prepare_flashrank_model("model")


def test_install_extras_and_build_commands(monkeypatch, tmp_path):
    assert init_service._install_extras("local", dry_run=True) is True

    standalone = InstallInfo(InstallMethod.STANDALONE, None, None, None)
    monkeypatch.setattr(init_service, "detect_install_method", lambda: standalone)
    assert init_service._install_extras("local", dry_run=False) is False
    assert init_service._build_extras_install_command(standalone, "local") is None

    editable = InstallInfo(InstallMethod.GIT_EDITABLE, None, tmp_path, None)
    command = init_service._build_extras_install_command(editable, "local")
    assert command[-2:] == ["-e", f"{tmp_path}[local]"]

    pip_user = InstallInfo(InstallMethod.PIP_USER, None, None, None)
    command = init_service._build_extras_install_command(pip_user, "flashrank")
    assert "--user" in command
    assert command[-1] == "vexor[flashrank]"

    pip_system = InstallInfo(InstallMethod.PIP_SYSTEM, None, None, None, requires_admin=True)
    monkeypatch.setattr(init_service, "detect_install_method", lambda: pip_system)
    monkeypatch.setattr(init_service.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1))
    assert init_service._install_extras("local", dry_run=False) is False

    monkeypatch.setattr(init_service.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    assert init_service._install_extras("local", dry_run=False) is True

    formatted = init_service._format_command(["python", "path with spaces"])
    assert "path with spaces" in formatted


def test_run_init_wizard_dry_run(monkeypatch):
    calls = {"apply": 0}
    monkeypatch.setattr(init_service, "_collect_provider_settings", lambda dry_run: {"provider": "openai"})
    monkeypatch.setattr(init_service, "_collect_rerank_settings", lambda dry_run: {"rerank": "off"})
    monkeypatch.setattr(init_service, "_prompt_alias_setup", lambda dry_run: None)
    monkeypatch.setattr(init_service, "_prompt_skill_install", lambda dry_run: None)
    monkeypatch.setattr(init_service, "_prompt_doctor_check", lambda dry_run: None)
    monkeypatch.setattr(init_service, "apply_config_updates", lambda **_kwargs: calls.update(apply=calls["apply"] + 1))

    init_service.run_init_wizard(dry_run=True)

    assert calls["apply"] == 0
