# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building the Vexor CLI executable."""

from pathlib import Path
import re
import sys

from PyInstaller.utils.hooks import collect_submodules


IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo,
        StringFileInfo,
        StringStruct,
        StringTable,
        VarFileInfo,
        VarStruct,
        VSVersionInfo,
    )

try:
    ROOT_DIR = Path(__file__).resolve().parent
except NameError:  # pragma: no cover - when spec executed without __file__
    ROOT_DIR = Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vexor import __version__ as VEXOR_VERSION


MAIN_SCRIPT = str(ROOT_DIR / "vexor" / "__main__.py")


def _version_tuple(raw: str):
    raw = raw.strip()
    release_parts = []
    suffix_number = 0

    for piece in raw.split("."):
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


def _string_version(raw: str):
    return ".".join(str(piece) for piece in _version_tuple(raw))


if IS_WINDOWS:
    FILE_VERSION = _string_version(VEXOR_VERSION)
    VERSION_INFO = VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=_version_tuple(VEXOR_VERSION),
            prodvers=_version_tuple(VEXOR_VERSION),
            mask=0x3F,
            flags=0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
            date=(0, 0),
        ),
        kids=[
            StringFileInfo(
                [
                    StringTable(
                        "040904B0",
                        [
                            StringStruct("CompanyName", "ScarletKc"),
                            StringStruct(
                                "FileDescription",
                                "Vexor",
                            ),
                            StringStruct("FileVersion", FILE_VERSION),
                            StringStruct(
                                "LegalCopyright",
                                "Copyright (C) ScarletKc",
                            ),
                            StringStruct("InternalName", "vexor"),
                            StringStruct("OriginalFilename", "vexor.exe"),
                            StringStruct("ProductName", "Vexor"),
                            StringStruct("ProductVersion", VEXOR_VERSION),
                        ],
                    )
                ]
            ),
            VarFileInfo([VarStruct("Translation", [1033, 1200])]),
        ],
    )
else:
    VERSION_INFO = None


hiddenimports = []
hiddenimports += collect_submodules("google.genai")
hiddenimports += collect_submodules("sklearn")

datas = []
bundled_skills_source = ROOT_DIR / "plugins" / "vexor" / "skills"
if bundled_skills_source.exists():
    bundled_skills_dest = Path("vexor") / "_bundled_skills"
    for skill_file in bundled_skills_source.rglob("*"):
        if not skill_file.is_file():
            continue
        relative_parent = skill_file.parent.relative_to(bundled_skills_source)
        datas.append((str(skill_file), str(bundled_skills_dest / relative_parent)))

icon_candidate = ROOT_DIR / "assets" / "vexor.ico"
ICON_PATH = str(icon_candidate) if IS_WINDOWS and icon_candidate.exists() else None


a = Analysis(
    [MAIN_SCRIPT],
    pathex=[str(ROOT_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="vexor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=VERSION_INFO,
    icon=ICON_PATH,
)
