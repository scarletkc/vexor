from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vexor import cli
import typer


def test_format_lines_variants():
    assert cli._format_lines(None, None) == "-"  # type: ignore[attr-defined]
    assert cli._format_lines(5, None) == "L5"  # type: ignore[attr-defined]
    assert cli._format_lines(5, 4) == "L5"  # type: ignore[attr-defined]
    assert cli._format_lines(5, 5) == "L5"  # type: ignore[attr-defined]
    assert cli._format_lines(5, 8) == "L5-8"  # type: ignore[attr-defined]


def test_format_extensions_display():
    assert cli._format_extensions_display(None) == "all"  # type: ignore[attr-defined]
    assert cli._format_extensions_display((".py", ".md")) == ".py, .md"  # type: ignore[attr-defined]


def test_validate_mode_rejects_invalid():
    assert cli._validate_mode("auto") == "auto"  # type: ignore[attr-defined]
    with pytest.raises(typer.BadParameter):
        cli._validate_mode("nope")  # type: ignore[attr-defined]


def test_cli_small_helpers_and_version_callback(capsys):
    assert cli._parse_boolean("yes") is True
    assert cli._parse_boolean("OFF") is False
    with pytest.raises(ValueError):
        cli._parse_boolean("maybe")

    assert cli._format_patterns_display(None) == "none"
    assert cli._format_patterns_display(("*.py", "build/")) == "*.py, build/"
    assert cli._format_preview(None) == "-"
    assert cli._format_preview(" short ") == "short"
    assert cli._format_preview("abcdef", limit=4) == "abc\u2026"
    assert cli._styled("text", "red") == "[red]text[/red]"
    assert cli._format_command(["vexor", "two words"]) == "vexor 'two words'"

    with pytest.raises(typer.Exit):
        cli._version_callback(True)
    assert "Vexor v" in capsys.readouterr().out
    cli._version_callback(False)


def test_cli_flashrank_prepare_success_and_errors(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError):
        cli._prepare_flashrank_model(None)

    flashrank_module = ModuleType("flashrank")

    class Ranker:
        kwargs = None

        def __init__(self, **kwargs):
            Ranker.kwargs = kwargs

    flashrank_module.Ranker = Ranker
    monkeypatch.setitem(sys.modules, "flashrank", flashrank_module)
    monkeypatch.setattr(cli, "flashrank_cache_dir", lambda: tmp_path)
    cli._prepare_flashrank_model("ranker-model")
    assert Ranker.kwargs["model_name"] == "ranker-model"

    class BrokenRanker:
        def __init__(self, **_kwargs):
            raise RuntimeError("broken")

    flashrank_module.Ranker = BrokenRanker
    with pytest.raises(RuntimeError, match="broken"):
        cli._prepare_flashrank_model(None)


def test_cli_snapshot_filters(tmp_path):
    entries = [
        {"path": "pkg/a.py"},
        {"path": "pkg/nested/b.py"},
        {"path": "docs/readme.md"},
    ]
    assert cli._filter_snapshot_by_extensions(entries, ()) == entries
    assert cli._filter_snapshot_by_extensions(entries, (".py",)) == entries[:2]

    filtered = cli._filter_snapshot_by_directory(entries, Path("pkg"), recursive=False)
    assert filtered == [{"path": "a.py"}]
    filtered_recursive = cli._filter_snapshot_by_directory(entries, Path("pkg"), recursive=True)
    assert filtered_recursive == [{"path": "a.py"}, {"path": "nested/b.py"}]

    spec = SimpleNamespace(
        check_file=lambda path: SimpleNamespace(include=path.endswith("nested/b.py"))
    )
    assert cli._filter_snapshot_by_exclude_patterns(entries, None) == entries
    assert cli._filter_snapshot_by_exclude_patterns(entries, spec) == [
        {"path": "pkg/a.py"},
        {"path": "docs/readme.md"},
    ]


def test_cli_should_index_before_search_direct_and_superset(monkeypatch, tmp_path):
    request = cli.SearchRequest(
        query="q",
        directory=tmp_path / "pkg",
        include_hidden=False,
        respect_gitignore=True,
        mode="name",
        recursive=False,
        top_k=1,
        model_name="model",
        batch_size=0,
        provider="openai",
        base_url=None,
        api_key="key",
        local_cuda=False,
        exclude_patterns=(),
        extensions=(".py",),
    )

    monkeypatch.setattr(cli, "load_index_metadata_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "list_cache_entries", lambda: [])
    assert cli._should_index_before_search(request) is True

    root = tmp_path
    metadata = {
        "files": [
            {"path": "pkg/a.py", "mtime": 1.0, "size": 1},
            {"path": "pkg/nested/b.py", "mtime": 1.0, "size": 1},
            {"path": "pkg/c.md", "mtime": 1.0, "size": 1},
        ]
    }

    def fake_load(root_arg, *_args, **_kwargs):
        if Path(root_arg) == request.directory:
            return None
        return metadata

    monkeypatch.setattr(cli, "load_index_metadata_safe", fake_load)
    monkeypatch.setattr(
        cli,
        "list_cache_entries",
        lambda: [
            {
                "root_path": str(root),
                "model": "model",
                "include_hidden": False,
                "respect_gitignore": True,
                "recursive": True,
                "mode": "name",
                "exclude_patterns": (),
                "extensions": (),
                "file_count": 3,
            }
        ],
    )
    monkeypatch.setattr(cli, "is_cache_current", lambda *_args, **_kwargs: True)
    assert cli._should_index_before_search(request) is False

    monkeypatch.setattr(cli, "is_cache_current", lambda *_args, **_kwargs: False)
    assert cli._should_index_before_search(request) is True


def test_cli_alias_profile_helpers(monkeypatch, tmp_path):
    monkeypatch.setenv("SHELL", "/bin/bash")
    assert cli._detect_shell_name() == "bash"
    monkeypatch.setenv("SHELL", "/bin/fish")
    assert cli._detect_shell_name() == "fish"

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ps7 = tmp_path / "Documents" / "PowerShell"
    ps7.mkdir(parents=True)
    assert cli._resolve_powershell_profile() == ps7 / "Microsoft.PowerShell_profile.ps1"
    ps7.rmdir()
    ps5 = tmp_path / "Documents" / "WindowsPowerShell"
    ps5.mkdir()
    assert cli._resolve_powershell_profile() == ps5 / "Microsoft.PowerShell_profile.ps1"

    assert cli._resolve_alias_profile("bash") == Path("~/.bashrc").expanduser()
    assert cli._resolve_alias_profile("zsh") == Path("~/.zshrc").expanduser()
    assert cli._resolve_alias_profile("fish") == Path("~/.config/fish/config.fish").expanduser()
    assert cli._resolve_alias_profile(None) is None
    assert "vexor" in cli._resolve_alias_command("fish")
    assert "Set-Alias" in cli._resolve_alias_command("powershell")
    assert cli._resolve_alias_command("bash").startswith("alias vx=")
