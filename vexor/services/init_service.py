"""Interactive first-run setup wizard for Vexor."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .. import __version__, config as config_module
from ..config import (
    DEFAULT_FLASHRANK_MAX_LENGTH,
    DEFAULT_FLASHRANK_MODEL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_PROVIDER,
    flashrank_cache_dir,
    load_config,
    normalize_remote_rerank_url,
    resolve_api_key,
    resolve_default_model,
    resolve_remote_rerank_api_key,
)
from ..providers.local import LocalEmbeddingBackend
from ..services.config_service import apply_config_updates
from ..services.skill_service import (
    DEFAULT_SKILL_NAME,
    SkillInstallStatus,
    install_bundled_skill,
    resolve_skill_roots,
)
from ..services.system_service import (
    DoctorCheckResult,
    InstallMethod,
    detect_install_method,
    run_all_doctor_checks,
)
from ..output import format_status_icon, supports_unicode_output
from ..text import Messages, Styles

console = Console()


def should_auto_run_init(
    args: Sequence[str] | None,
    *,
    config_path: Path = config_module.CONFIG_FILE,
    is_tty: bool | None = None,
) -> bool:
    """Return True when the init wizard should auto-run on first invocation."""
    if config_path.exists():
        return False
    if is_tty is None:
        is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    if not is_tty:
        return False
    tokens = list(args or [])
    if tokens and tokens[0] == "init":
        return False
    skip_flags = {
        "-h",
        "--help",
        "-v",
        "--version",
        "--install-completion",
        "--show-completion",
    }
    if any(token in skip_flags for token in tokens):
        return False
    return True


def run_init_wizard(*, dry_run: bool = False) -> None:
    """Run the interactive onboarding flow and persist configuration."""
    _print_welcome_banner()
    console.print()

    provider_updates = _collect_provider_settings(dry_run=dry_run)
    rerank_updates = _collect_rerank_settings(dry_run=dry_run)

    if not dry_run:
        apply_config_updates(**provider_updates, **rerank_updates)

    _prompt_alias_setup(dry_run=dry_run)
    _prompt_skill_install(dry_run=dry_run)
    _prompt_doctor_check(dry_run=dry_run)
    if dry_run:
        console.print(_styled(Messages.INIT_DRY_RUN_NOTICE, Styles.INFO))
    console.print(_styled(Messages.INIT_CONFIG_HINT, Styles.INFO))
    _print_next_steps()


def _print_welcome_banner() -> None:
    """Print a styled welcome banner for the init wizard."""
    title = Text()
    icon = "\u2699 " if supports_unicode_output(console) else "* "
    title.append(icon, style="bold")
    title.append(Messages.INIT_TITLE, style="bold cyan")
    title.append(f"  v{__version__}", style="dim")

    intro = Text(Messages.INIT_INTRO, style="white")

    banner_content = Text()
    banner_content.append_text(title)
    banner_content.append("\n")
    banner_content.append_text(intro)

    console.print(
        Panel(
            banner_content,
            border_style="cyan",
            padding=(0, 1),
        )
    )


def _print_step_header(step_num: str, title: str) -> None:
    """Print a styled step header."""
    console.print(f"[bold cyan]Step {step_num}:[/bold cyan] [bold]{title}[/bold]")


def _print_option(key: str, name: str, desc: str) -> None:
    """Print a styled option line."""
    console.print(f"  [cyan]{key})[/cyan] [bold]{name}[/bold] [dim]- {desc}[/dim]")


def _collect_provider_settings(*, dry_run: bool) -> dict[str, object]:
    _print_step_header("1", Messages.INIT_STEP_RUN_MODE)
    _print_option("A", Messages.INIT_OPTION_LOCAL, Messages.INIT_OPTION_LOCAL_DESC)
    _print_option("B", Messages.INIT_OPTION_REMOTE, Messages.INIT_OPTION_REMOTE_DESC)
    console.print()
    run_mode = _prompt_choice(
        Messages.INIT_PROMPT_RUN_MODE,
        {
            "a": "local",
            "local": "local",
            "b": "remote",
            "remote": "remote",
        },
        default="B",
        allowed="A/B",
    )
    console.print()
    if run_mode == "local":
        local_updates = _collect_local_settings(dry_run=dry_run)
        if local_updates is not None:
            return local_updates
        # Fall back to remote if requested.
    return _collect_remote_settings()


def _collect_local_settings(*, dry_run: bool) -> dict[str, object] | None:
    _print_step_header("1a", Messages.INIT_STEP_LOCAL_HARDWARE)
    _print_option("A", Messages.INIT_OPTION_CPU, Messages.INIT_OPTION_CPU_DESC)
    _print_option("B", Messages.INIT_OPTION_CUDA, Messages.INIT_OPTION_CUDA_DESC)
    console.print()
    hardware = _prompt_choice(
        Messages.INIT_PROMPT_LOCAL_HARDWARE,
        {
            "a": "cpu",
            "cpu": "cpu",
            "b": "cuda",
            "gpu": "cuda",
            "cuda": "cuda",
        },
        default="A",
        allowed="A/B",
    )
    use_cuda = hardware == "cuda"
    if use_cuda and not _ensure_cuda_available():
        if typer.confirm(Messages.INIT_CONFIRM_FALLBACK_CPU, default=True):
            use_cuda = False
    console.print()

    if not _is_fastembed_available():
        if typer.confirm(
            Messages.INIT_CONFIRM_INSTALL_LOCAL_CUDA
            if use_cuda
            else Messages.INIT_CONFIRM_INSTALL_LOCAL,
            default=True,
        ):
            extras = "local-cuda" if use_cuda else "local"
            if not _install_extras(extras, dry_run=dry_run):
                if typer.confirm(Messages.INIT_CONFIRM_SWITCH_REMOTE, default=True):
                    return None
        else:
            if not typer.confirm(
                Messages.INIT_CONFIRM_CONTINUE_WITHOUT_LOCAL,
                default=False,
            ):
                return None

    if _is_fastembed_available() and typer.confirm(
        Messages.INIT_CONFIRM_RUN_LOCAL_SETUP,
        default=True,
    ):
        if not _prepare_local_model(
            DEFAULT_LOCAL_MODEL,
            use_cuda,
            dry_run=dry_run,
        ):
            if typer.confirm(Messages.INIT_CONFIRM_SWITCH_REMOTE, default=True):
                return None

    return {
        "provider": "local",
        "model": DEFAULT_LOCAL_MODEL,
        "local_cuda": use_cuda,
    }


def _collect_remote_settings() -> dict[str, object]:
    _print_step_header("1b", Messages.INIT_STEP_PROVIDER)
    _print_option(
        "A",
        Messages.INIT_OPTION_PROVIDER_OPENAI,
        Messages.INIT_OPTION_PROVIDER_OPENAI_DESC,
    )
    _print_option(
        "B",
        Messages.INIT_OPTION_PROVIDER_GEMINI,
        Messages.INIT_OPTION_PROVIDER_GEMINI_DESC,
    )
    _print_option(
        "C",
        Messages.INIT_OPTION_PROVIDER_CUSTOM,
        Messages.INIT_OPTION_PROVIDER_CUSTOM_DESC,
    )
    console.print()
    provider = _prompt_choice(
        Messages.INIT_PROMPT_PROVIDER,
        {
            "a": "openai",
            "openai": "openai",
            "b": "gemini",
            "gemini": "gemini",
            "c": "custom",
            "custom": "custom",
        },
        default="A",
        allowed="A/B/C",
    )
    console.print()

    updates: dict[str, object] = {"provider": provider}
    if provider == "custom":
        base_url = _prompt_required(Messages.INIT_PROMPT_CUSTOM_BASE_URL)
        model = _prompt_required(Messages.INIT_PROMPT_CUSTOM_MODEL)
        api_key = _prompt_api_key(Messages.INIT_PROMPT_API_KEY_CUSTOM, provider)
        updates.update(
            {
                "base_url": base_url,
                "model": model,
                "api_key": api_key,
            }
        )
        return updates

    if provider == "gemini":
        api_key = _prompt_api_key(Messages.INIT_PROMPT_API_KEY_GEMINI, provider)
    else:
        api_key = _prompt_api_key(Messages.INIT_PROMPT_API_KEY_OPENAI, provider)
    updates["api_key"] = api_key
    return updates


def _collect_rerank_settings(*, dry_run: bool) -> dict[str, object]:
    _print_step_header("2", Messages.INIT_STEP_RERANK)
    _print_option("1", Messages.INIT_OPTION_RERANK_OFF, Messages.INIT_OPTION_RERANK_OFF_DESC)
    _print_option("2", Messages.INIT_OPTION_RERANK_BM25, Messages.INIT_OPTION_RERANK_BM25_DESC)
    _print_option(
        "3",
        Messages.INIT_OPTION_RERANK_FLASHRANK,
        Messages.INIT_OPTION_RERANK_FLASHRANK_DESC,
    )
    _print_option("4", Messages.INIT_OPTION_RERANK_REMOTE, Messages.INIT_OPTION_RERANK_REMOTE_DESC)
    console.print()
    rerank_choice = _prompt_choice(
        Messages.INIT_PROMPT_RERANK,
        {
            "1": "off",
            "off": "off",
            "none": "off",
            "2": "bm25",
            "bm25": "bm25",
            "3": "flashrank",
            "flashrank": "flashrank",
            "4": "remote",
            "remote": "remote",
        },
        default="1",
        allowed="1/2/3/4",
    )
    console.print()

    if rerank_choice == "flashrank":
        if not _ensure_flashrank_available(dry_run=dry_run):
            fallback = _prompt_choice(
                Messages.INIT_PROMPT_RERANK_FALLBACK,
                {
                    "b": "bm25",
                    "bm25": "bm25",
                    "o": "off",
                    "off": "off",
                },
                default="O",
                allowed="bm25/off",
            )
            return {"rerank": fallback}
        _maybe_prepare_flashrank_model(dry_run=dry_run)
        return {"rerank": "flashrank"}

    if rerank_choice == "remote":
        normalized_url = None
        while not normalized_url:
            base_url = _prompt_required(Messages.INIT_PROMPT_REMOTE_RERANK_URL)
            normalized_url = normalize_remote_rerank_url(base_url)
            if not normalized_url:
                console.print(
                    _styled(Messages.ERROR_REMOTE_RERANK_URL_EMPTY, Styles.ERROR)
                )
        model = _prompt_required(Messages.INIT_PROMPT_REMOTE_RERANK_MODEL)
        api_key = _prompt_required_secret(Messages.INIT_PROMPT_REMOTE_RERANK_API_KEY)
        return {
            "rerank": "remote",
            "remote_rerank_url": normalized_url,
            "remote_rerank_model": model,
            "remote_rerank_api_key": api_key,
        }

    return {"rerank": rerank_choice}


def _prompt_alias_setup(*, dry_run: bool) -> None:
    _print_step_header("3", Messages.INIT_STEP_ALIAS)
    if not typer.confirm(Messages.INIT_CONFIRM_ALIAS, default=False):
        console.print()
        return
    if dry_run:
        _note_dry_run("writing shell alias")
        console.print()
        return
    shell_name = _detect_shell_name()
    alias_command = _resolve_alias_command(shell_name)
    console.print(alias_command)
    profile_path = _resolve_alias_profile(shell_name)
    if profile_path is None:
        console.print(_styled(Messages.WARNING_ALIAS_PROFILE_MISSING, Styles.WARNING))
        console.print()
        return
    try:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        existing = (
            profile_path.read_text(encoding="utf-8") if profile_path.exists() else ""
        )
        if alias_command in existing:
            console.print(
                _styled(
                    Messages.INFO_ALIAS_ALREADY_SET.format(path=profile_path),
                    Styles.INFO,
                )
            )
            console.print()
            return
        with profile_path.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(alias_command + "\n")
        console.print(
            _styled(
                Messages.INFO_ALIAS_APPLIED.format(path=profile_path),
                Styles.SUCCESS,
            )
        )
    except OSError as exc:
        console.print(
            _styled(
                Messages.ERROR_ALIAS_WRITE.format(path=profile_path, reason=str(exc)),
                Styles.ERROR,
            )
        )
    console.print()


def _prompt_skill_install(*, dry_run: bool) -> None:
    _print_step_header("4", Messages.INIT_STEP_SKILLS)
    if not typer.confirm(Messages.INIT_CONFIRM_SKILLS_INSTALL, default=False):
        console.print()
        return
    if dry_run:
        _note_dry_run("installing skills")
        console.print()
        return
    console.print(f"  [bold]{Messages.INIT_STEP_SKILLS_TARGET}[/bold]")
    _print_option("A", Messages.INIT_OPTION_SKILLS_AUTO, Messages.INIT_OPTION_SKILLS_AUTO_DESC)
    _print_option("B", Messages.INIT_OPTION_SKILLS_CLAUDE, Messages.INIT_OPTION_SKILLS_CLAUDE_DESC)
    _print_option("C", Messages.INIT_OPTION_SKILLS_CODEX, Messages.INIT_OPTION_SKILLS_CODEX_DESC)
    _print_option("D", Messages.INIT_OPTION_SKILLS_CUSTOM, Messages.INIT_OPTION_SKILLS_CUSTOM_DESC)
    console.print()
    target = _prompt_choice(
        Messages.INIT_PROMPT_SKILLS_TARGET,
        {
            "a": "auto",
            "auto": "auto",
            "b": "claude",
            "claude": "claude",
            "c": "codex",
            "codex": "codex",
            "d": "custom",
        },
        default="A",
        allowed="A/B/C/D",
    )
    if target == "custom":
        target = _prompt_required(Messages.INIT_PROMPT_SKILLS_PATH)
    _install_skills(target)
    console.print()


def _prompt_doctor_check(*, dry_run: bool) -> None:
    _print_step_header("5", Messages.INIT_STEP_DOCTOR)
    if not typer.confirm(Messages.INIT_CONFIRM_DOCTOR, default=True):
        console.print()
        return
    if dry_run:
        _note_dry_run("running doctor checks")
        console.print()
        return
    _run_doctor_checks()
    console.print()


def _run_doctor_checks() -> None:
    console.print(_styled(Messages.DOCTOR_TITLE.format(version=__version__), Styles.TITLE))
    console.print()

    config_load_error: DoctorCheckResult | None = None
    try:
        config = load_config()
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        config = config_module.Config()
        config_load_error = DoctorCheckResult(
            name="Config JSON",
            passed=False,
            message=Messages.DOCTOR_CONFIG_INVALID.format(path=config_module.CONFIG_FILE),
            detail=str(exc),
        )

    provider = (config.provider or DEFAULT_PROVIDER).lower()
    model = resolve_default_model(provider, config.model)

    results: list[DoctorCheckResult] = []
    if config_load_error is not None:
        results.append(config_load_error)

    results.extend(
        run_all_doctor_checks(
            provider=provider,
            model=model,
            api_key=config.api_key,
            base_url=config.base_url,
            skip_api_test=False,
            local_cuda=bool(config.local_cuda),
            rerank=config.rerank,
            flashrank_model=config.flashrank_model,
            remote_rerank=config.remote_rerank,
        )
    )

    has_failure = False
    for result in results:
        icon = format_status_icon(result.passed, console=console)
        if not result.passed:
            has_failure = True

        console.print(f"  {icon} [bold]{result.name}:[/bold] {result.message}")
        if result.detail:
            console.print(f"      [dim]{result.detail}[/dim]")

    console.print()
    if has_failure:
        console.print(_styled(Messages.DOCTOR_SOME_FAILED, Styles.WARNING))
        return
    console.print(_styled(Messages.DOCTOR_ALL_PASSED, Styles.SUCCESS))


def _install_skills(target: str) -> None:
    try:
        roots = resolve_skill_roots(target)
    except ValueError as exc:
        console.print(_styled(str(exc), Styles.ERROR))
        return

    for root in roots:
        destination_root = root.expanduser()
        try:
            result = install_bundled_skill(
                skill_name=DEFAULT_SKILL_NAME,
                skills_dir=destination_root,
                force=False,
            )
        except FileExistsError as exc:
            console.print(
                _styled(
                    Messages.ERROR_INSTALL_SKILL_EXISTS.format(path=str(exc.args[0])),
                    Styles.ERROR,
                )
            )
            continue
        except FileNotFoundError as exc:
            console.print(
                _styled(
                    Messages.ERROR_INSTALL_SKILL_SOURCE.format(reason=str(exc)), Styles.ERROR
                )
            )
            return

        if result.status == SkillInstallStatus.up_to_date:
            console.print(
                _styled(
                    Messages.INFO_INSTALL_SKILL_UP_TO_DATE.format(path=result.destination),
                    Styles.INFO,
                )
            )
        else:
            console.print(
                _styled(
                    Messages.INFO_INSTALL_SKILL_DONE.format(path=result.destination),
                    Styles.SUCCESS,
                )
            )


def _prompt_choice(
    prompt: str,
    options: Mapping[str, str],
    *,
    default: str,
    allowed: str,
) -> str:
    while True:
        value = typer.prompt(prompt, default=default)
        cleaned = (value or "").strip().lower()
        if not cleaned:
            cleaned = default.lower()
        selection = options.get(cleaned)
        if selection:
            return selection
        console.print(
            _styled(
                Messages.INIT_ERROR_INVALID_CHOICE.format(value=value, allowed=allowed),
                Styles.WARNING,
            )
        )


def _prompt_required(prompt: str) -> str:
    while True:
        value = typer.prompt(prompt)
        cleaned = value.strip()
        if cleaned:
            return cleaned
        console.print(_styled(Messages.INIT_ERROR_REQUIRED, Styles.WARNING))


def _prompt_api_key(prompt: str, provider: str) -> str | None:
    while True:
        value = typer.prompt(
            prompt,
            default="",
            show_default=False,
            hide_input=True,
        )
        cleaned = value.strip()
        if cleaned:
            return cleaned
        if resolve_api_key(None, provider):
            console.print(_styled(Messages.INIT_USING_ENV_API_KEY, Styles.INFO))
            return None
        if typer.confirm(Messages.INIT_CONFIRM_SKIP_API_KEY, default=False):
            return None


def _prompt_required_secret(prompt: str) -> str | None:
    while True:
        value = typer.prompt(
            prompt,
            default="",
            show_default=False,
            hide_input=True,
        )
        cleaned = value.strip()
        if cleaned:
            return cleaned
        if resolve_remote_rerank_api_key(None):
            console.print(_styled(Messages.INIT_USING_ENV_API_KEY, Styles.INFO))
            return None
        console.print(_styled(Messages.INIT_ERROR_REMOTE_RERANK_KEY, Styles.WARNING))


def _ensure_cuda_available() -> bool:
    try:
        import onnxruntime as ort
    except Exception as exc:
        console.print(_styled(Messages.DOCTOR_LOCAL_CUDA_IMPORT_FAILED, Styles.ERROR))
        console.print(
            _styled(
                Messages.DOCTOR_LOCAL_CUDA_IMPORT_DETAIL.format(reason=str(exc)),
                Styles.ERROR,
            )
        )
        return False
    try:
        providers = ort.get_available_providers()
    except Exception as exc:
        console.print(_styled(Messages.DOCTOR_LOCAL_CUDA_MISSING, Styles.ERROR))
        console.print(
            _styled(
                Messages.DOCTOR_LOCAL_CUDA_IMPORT_DETAIL.format(reason=str(exc)),
                Styles.ERROR,
            )
        )
        return False
    if "CUDAExecutionProvider" not in providers:
        console.print(_styled(Messages.DOCTOR_LOCAL_CUDA_MISSING, Styles.ERROR))
        console.print(
            _styled(
                Messages.DOCTOR_LOCAL_CUDA_MISSING_DETAIL.format(
                    providers=", ".join(providers) if providers else "none"
                ),
                Styles.ERROR,
            )
        )
        return False
    return True


def _is_fastembed_available() -> bool:
    return importlib.util.find_spec("fastembed") is not None


def _is_flashrank_available() -> bool:
    return importlib.util.find_spec("flashrank") is not None


def _prepare_local_model(model: str, use_cuda: bool, *, dry_run: bool) -> bool:
    if dry_run:
        _note_dry_run("downloading local model")
        return True
    console.print(
        _styled(Messages.INFO_LOCAL_SETUP_START.format(model=model), Styles.INFO)
    )
    try:
        backend = LocalEmbeddingBackend(model_name=model, cuda=use_cuda)
        vectors = backend.embed(["test"])
    except RuntimeError as exc:
        console.print(_styled(str(exc), Styles.ERROR))
        return False
    if vectors.size == 0:
        console.print(_styled(Messages.ERROR_NO_EMBEDDINGS, Styles.ERROR))
        return False
    console.print(
        _styled(Messages.INFO_LOCAL_SETUP_DONE.format(model=model), Styles.SUCCESS)
    )
    return True


def _ensure_flashrank_available(*, dry_run: bool) -> bool:
    if _is_flashrank_available():
        return True
    console.print(_styled(Messages.INIT_FLASHRANK_MISSING, Styles.WARNING))
    if dry_run:
        _note_dry_run("installing extras (flashrank)")
        return True
    if not typer.confirm(Messages.INIT_CONFIRM_INSTALL_FLASHRANK, default=True):
        return False
    if not _install_extras("flashrank", dry_run=dry_run):
        return False
    return _is_flashrank_available()


def _maybe_prepare_flashrank_model(*, dry_run: bool) -> None:
    if not typer.confirm(Messages.INIT_CONFIRM_FLASHRANK_DOWNLOAD, default=True):
        return
    if dry_run:
        _note_dry_run("downloading FlashRank model")
        return
    console.print(_styled(Messages.INFO_FLASHRANK_SETUP_START, Styles.INFO))
    try:
        _prepare_flashrank_model(DEFAULT_FLASHRANK_MODEL)
    except RuntimeError as exc:
        console.print(_styled(str(exc), Styles.ERROR))
        return
    console.print(_styled(Messages.INFO_FLASHRANK_SETUP_DONE, Styles.SUCCESS))


def _prepare_flashrank_model(model_name: str | None) -> None:
    try:
        from flashrank import Ranker
    except ImportError as exc:
        raise RuntimeError(Messages.ERROR_FLASHRANK_MISSING) from exc
    cache_dir = flashrank_cache_dir()
    try:
        effective_model = model_name or DEFAULT_FLASHRANK_MODEL
        kwargs = {
            "max_length": DEFAULT_FLASHRANK_MAX_LENGTH,
            "cache_dir": str(cache_dir),
            "model_name": effective_model,
        }
        Ranker(**kwargs)
    except Exception as exc:
        raise RuntimeError(Messages.ERROR_FLASHRANK_SETUP.format(reason=str(exc))) from exc


def _install_extras(extras: str, *, dry_run: bool) -> bool:
    if dry_run:
        _note_dry_run(f"installing extras ({extras})")
        return True
    install_info = detect_install_method()
    if install_info.method == InstallMethod.STANDALONE:
        console.print(_styled(Messages.INIT_INSTALL_STANDALONE, Styles.WARNING))
        return False

    cmd = _build_extras_install_command(install_info, extras)
    if not cmd:
        console.print(_styled(Messages.INIT_INSTALL_UNSUPPORTED, Styles.WARNING))
        return False

    if install_info.requires_admin:
        console.print(_styled(Messages.INIT_INSTALL_REQUIRES_ADMIN, Styles.WARNING))

    console.print(_styled(Messages.INIT_INSTALL_START.format(extra=extras), Styles.INFO))
    console.print(
        _styled(
            Messages.INIT_INSTALL_COMMAND.format(cmd=_format_command(cmd)),
            Styles.INFO,
        )
    )
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        console.print(
            _styled(
                Messages.INIT_INSTALL_FAILED.format(code=completed.returncode),
                Styles.ERROR,
            )
        )
        return False
    console.print(_styled(Messages.INIT_INSTALL_DONE.format(extra=extras), Styles.SUCCESS))
    return True


def _build_extras_install_command(
    install_info,
    extras: str,
) -> list[str] | None:
    if install_info.method == InstallMethod.GIT_EDITABLE and install_info.editable_root:
        target = f"{install_info.editable_root}[{extras}]"
        return [sys.executable, "-m", "pip", "install", "-e", target]

    if install_info.method == InstallMethod.STANDALONE:
        return None

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if install_info.method == InstallMethod.PIP_USER:
        cmd.append("--user")
    cmd.append(f"vexor[{extras}]")
    return cmd


def _format_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _print_next_steps() -> None:
    next_steps = Text()
    next_steps.append(" ", style="bold")
    next_steps.append(Messages.INIT_NEXT_STEPS_TITLE, style="bold cyan")

    console.print(
        Panel(
            next_steps,
            border_style="green",
            padding=(0, 1),
        )
    )
    console.print(f"  [bold green]$[/bold green] {Messages.INIT_NEXT_STEP_SEARCH}")


def _note_dry_run(action: str) -> None:
    console.print(
        _styled(Messages.INIT_DRY_RUN_SKIPPED.format(action=action), Styles.INFO)
    )


def _detect_shell_name() -> str | None:
    shell_env = os.environ.get("SHELL", "")
    if shell_env:
        name = Path(shell_env).name.lower()
        if name in {"bash", "zsh", "fish"}:
            return name
    if os.name == "nt":
        return "powershell"
    return None


def _resolve_powershell_profile() -> Path:
    home = Path.home()
    ps7_dir = home / "Documents" / "PowerShell"
    ps5_dir = home / "Documents" / "WindowsPowerShell"
    if ps7_dir.exists():
        return ps7_dir / "Microsoft.PowerShell_profile.ps1"
    if ps5_dir.exists():
        return ps5_dir / "Microsoft.PowerShell_profile.ps1"
    return ps7_dir / "Microsoft.PowerShell_profile.ps1"


def _resolve_alias_profile(shell_name: str | None) -> Path | None:
    if shell_name == "bash":
        return Path("~/.bashrc").expanduser()
    if shell_name == "zsh":
        return Path("~/.zshrc").expanduser()
    if shell_name == "fish":
        return Path("~/.config/fish/config.fish").expanduser()
    if shell_name == "powershell":
        return _resolve_powershell_profile()
    return None


def _resolve_alias_command(shell_name: str | None) -> str:
    if shell_name == "fish":
        return Messages.INFO_ALIAS_FISH
    if shell_name == "powershell":
        return Messages.INFO_ALIAS_POWERSHELL
    return Messages.INFO_ALIAS_VX


def _styled(text: str, style: str) -> str:
    return f"[{style}]{text}[/{style}]"
