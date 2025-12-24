import importlib.util
import json
from pathlib import Path


def _load_bump_version_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "bump_version.py"
    spec = importlib.util.spec_from_file_location("bump_version", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bump_version_does_not_update_gui_by_default(tmp_path: Path):
    bump = _load_bump_version_module()

    (tmp_path / "vexor").mkdir(parents=True)
    (tmp_path / "plugins" / "vexor" / ".claude-plugin").mkdir(parents=True)
    (tmp_path / "gui").mkdir(parents=True)

    package_init = tmp_path / "vexor" / "__init__.py"
    package_init.write_text('__version__ = "0.1.0"\n', encoding="utf-8")

    plugin_manifest = tmp_path / "plugins" / "vexor" / ".claude-plugin" / "plugin.json"
    plugin_manifest.write_text(json.dumps({"version": "0.1.0"}, indent=2) + "\n", encoding="utf-8")

    gui_package_json = tmp_path / "gui" / "package.json"
    gui_package_json.write_text(json.dumps({"name": "gui", "version": "0.1.0"}, indent=2) + "\n", encoding="utf-8")

    gui_lock = tmp_path / "gui" / "package-lock.json"
    gui_lock.write_text(
        json.dumps({"name": "gui", "version": "0.1.0", "packages": {"": {"version": "0.1.0"}}}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    bump._run(version="1.2.3", update_gui=False, repo_root=tmp_path)

    assert package_init.read_text(encoding="utf-8") == '__version__ = "1.2.3"\n'
    assert json.loads(plugin_manifest.read_text(encoding="utf-8"))["version"] == "1.2.3"

    # GUI files should remain untouched
    assert json.loads(gui_package_json.read_text(encoding="utf-8"))["version"] == "0.1.0"
    lock = json.loads(gui_lock.read_text(encoding="utf-8"))
    assert lock["version"] == "0.1.0"
    assert lock["packages"][""]["version"] == "0.1.0"


def test_bump_version_updates_gui_with_normalized_semver(tmp_path: Path):
    bump = _load_bump_version_module()

    (tmp_path / "vexor").mkdir(parents=True)
    (tmp_path / "plugins" / "vexor" / ".claude-plugin").mkdir(parents=True)
    (tmp_path / "gui").mkdir(parents=True)

    (tmp_path / "vexor" / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    (tmp_path / "plugins" / "vexor" / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"version": "0.1.0"}, indent=2) + "\n", encoding="utf-8"
    )
    (tmp_path / "gui" / "package.json").write_text(
        json.dumps({"name": "gui", "version": "0.1.0"}, indent=2) + "\n", encoding="utf-8"
    )
    (tmp_path / "gui" / "package-lock.json").write_text(
        json.dumps({"name": "gui", "version": "0.1.0", "packages": {"": {"version": "0.1.0"}}}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    bump._run(version="1.2.3rc1", update_gui=True, repo_root=tmp_path)

    # Python/plugin should keep the original version string.
    assert (tmp_path / "vexor" / "__init__.py").read_text(encoding="utf-8") == '__version__ = "1.2.3rc1"\n'
    assert json.loads((tmp_path / "plugins" / "vexor" / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))[
        "version"
    ] == "1.2.3rc1"

    # GUI should be normalized to SemVer (1.2.3-rc1)
    assert json.loads((tmp_path / "gui" / "package.json").read_text(encoding="utf-8"))["version"] == "1.2.3-rc1"
    lock = json.loads((tmp_path / "gui" / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["version"] == "1.2.3-rc1"
    assert lock["packages"][""]["version"] == "1.2.3-rc1"
