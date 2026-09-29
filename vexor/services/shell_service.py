"""Shell and profile resolution shared by setup and alias commands."""

import os
from pathlib import Path

from ..text import Messages


def detect_shell_name() -> str | None:
    shell_env = os.environ.get("SHELL", "")
    if shell_env:
        name = Path(shell_env).name.lower()
        if name in {"bash", "zsh", "fish"}:
            return name
    if os.name == "nt":
        return "powershell"
    return None


def resolve_powershell_profile() -> Path:
    home = Path.home()
    ps7_dir = home / "Documents" / "PowerShell"
    ps5_dir = home / "Documents" / "WindowsPowerShell"
    if ps7_dir.exists():
        return ps7_dir / "Microsoft.PowerShell_profile.ps1"
    if ps5_dir.exists():
        return ps5_dir / "Microsoft.PowerShell_profile.ps1"
    return ps7_dir / "Microsoft.PowerShell_profile.ps1"


def resolve_alias_profile(shell_name: str | None) -> Path | None:
    if shell_name == "bash":
        return Path("~/.bashrc").expanduser()
    if shell_name == "zsh":
        return Path("~/.zshrc").expanduser()
    if shell_name == "fish":
        return Path("~/.config/fish/config.fish").expanduser()
    if shell_name == "powershell":
        return resolve_powershell_profile()
    return None


def resolve_alias_command(shell_name: str | None) -> str:
    if shell_name == "fish":
        return Messages.INFO_ALIAS_FISH
    if shell_name == "powershell":
        return Messages.INFO_ALIAS_POWERSHELL
    return Messages.INFO_ALIAS_VX
