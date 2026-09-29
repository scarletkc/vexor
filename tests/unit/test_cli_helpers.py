from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import typer

from vexor import cli
from vexor.services import model_service, shell_service


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
    # See test_search_service_extra: assert the missing-extra path deterministically
    # rather than relying on ``flashrank`` being absent from the environment.
    monkeypatch.setitem(sys.modules, "flashrank", None)
    with pytest.raises(RuntimeError):
        model_service.prepare_flashrank_model(None)

    flashrank_module = ModuleType("flashrank")

    class Ranker:
        kwargs = None

        def __init__(self, **kwargs):
            Ranker.kwargs = kwargs

    flashrank_module.Ranker = Ranker
    monkeypatch.setitem(sys.modules, "flashrank", flashrank_module)
    monkeypatch.setattr(model_service, "flashrank_cache_dir", lambda: tmp_path)
    model_service.prepare_flashrank_model("ranker-model")
    assert Ranker.kwargs["model_name"] == "ranker-model"

    class BrokenRanker:
        def __init__(self, **_kwargs):
            raise RuntimeError("broken")

    flashrank_module.Ranker = BrokenRanker
    with pytest.raises(RuntimeError, match="broken"):
        model_service.prepare_flashrank_model(None)


def test_cli_alias_profile_helpers(monkeypatch, tmp_path):
    monkeypatch.setenv("SHELL", "/bin/bash")
    assert shell_service.detect_shell_name() == "bash"
    monkeypatch.setenv("SHELL", "/bin/fish")
    assert shell_service.detect_shell_name() == "fish"

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ps7 = tmp_path / "Documents" / "PowerShell"
    ps7.mkdir(parents=True)
    assert shell_service.resolve_powershell_profile() == ps7 / "Microsoft.PowerShell_profile.ps1"
    ps7.rmdir()
    ps5 = tmp_path / "Documents" / "WindowsPowerShell"
    ps5.mkdir()
    assert shell_service.resolve_powershell_profile() == ps5 / "Microsoft.PowerShell_profile.ps1"

    assert shell_service.resolve_alias_profile("bash") == Path("~/.bashrc").expanduser()
    assert shell_service.resolve_alias_profile("zsh") == Path("~/.zshrc").expanduser()
    assert (
        shell_service.resolve_alias_profile("fish")
        == Path("~/.config/fish/config.fish").expanduser()
    )
    assert shell_service.resolve_alias_profile(None) is None
    assert "vexor" in shell_service.resolve_alias_command("fish")
    assert "Set-Alias" in shell_service.resolve_alias_command("powershell")
    assert shell_service.resolve_alias_command("bash").startswith("alias vx=")


def test_should_offer_update_notice_gating(monkeypatch):
    from vexor import cli as cli_module

    monkeypatch.setattr(cli_module.sys.stderr, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli_module, "update_check_enabled", lambda: True)

    assert cli_module.should_offer_update_notice(["search", "q"]) is True
    assert cli_module.should_offer_update_notice([]) is False
    assert cli_module.should_offer_update_notice(["mcp"]) is False
    assert cli_module.should_offer_update_notice(["update"]) is False
    assert cli_module.should_offer_update_notice(["init"]) is False
    assert cli_module.should_offer_update_notice(["search", "--help"]) is False

    monkeypatch.setattr(cli_module, "update_check_enabled", lambda: False)
    assert cli_module.should_offer_update_notice(["search", "q"]) is False

    monkeypatch.setattr(cli_module, "update_check_enabled", lambda: True)
    monkeypatch.setattr(cli_module.sys.stderr, "isatty", lambda: False, raising=False)
    assert cli_module.should_offer_update_notice(["search", "q"]) is False


def test_run_dispatches_config_without_starting_init_wizard(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    original_should_auto_run_init = cli.should_auto_run_init

    def should_auto_run_init_interactively(args, *, config_path):
        return original_should_auto_run_init(
            args,
            config_path=config_path,
            is_tty=True,
        )

    dispatched: list[list[str]] = []
    monkeypatch.setattr(cli.config_module, "CONFIG_FILE", config_path)
    monkeypatch.setattr(cli, "should_auto_run_init", should_auto_run_init_interactively)
    monkeypatch.setattr(
        cli,
        "run_init_wizard",
        lambda: (_ for _ in ()).throw(AssertionError("init wizard must not run")),
    )
    monkeypatch.setattr(cli, "should_offer_update_notice", lambda _args: False)
    monkeypatch.setattr(cli, "app", lambda *, args: dispatched.append(args))

    cli.run(["config", "--show"])

    assert dispatched == [["config", "--show"]]


def test_print_update_notice_reads_cache_only(monkeypatch):
    from rich.console import Console

    from vexor import cli as cli_module

    captured = {}

    def fake_check(current, *, allow_network=True, **kw):
        captured["allow_network"] = allow_network
        return "9.9.9"

    monkeypatch.setattr(cli_module, "check_for_update", fake_check)
    import io as io_module

    buffer = io_module.StringIO()
    cli_module.print_update_notice_if_cached(Console(file=buffer, width=200))

    assert captured["allow_network"] is False
    assert "9.9.9" in buffer.getvalue()
    assert "vexor update --upgrade" in buffer.getvalue()


def test_print_update_notice_silent_without_cache(monkeypatch):
    from rich.console import Console

    from vexor import cli as cli_module

    monkeypatch.setattr(cli_module, "check_for_update", lambda current, **kw: None)
    import io as io_module

    buffer = io_module.StringIO()
    cli_module.print_update_notice_if_cached(Console(file=buffer, width=200))
    assert buffer.getvalue() == ""
