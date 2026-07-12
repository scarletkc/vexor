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


def test_bump_version_updates_package_and_plugin(tmp_path: Path):
    bump = _load_bump_version_module()

    (tmp_path / "vexor").mkdir(parents=True)
    (tmp_path / "plugins" / "vexor" / ".claude-plugin").mkdir(parents=True)

    package_init = tmp_path / "vexor" / "__init__.py"
    package_init.write_text('__version__ = "0.1.0"\n', encoding="utf-8")

    plugin_manifest = tmp_path / "plugins" / "vexor" / ".claude-plugin" / "plugin.json"
    plugin_manifest.write_text(json.dumps({"version": "0.1.0"}, indent=2) + "\n", encoding="utf-8")

    bump._run(version="1.2.3", repo_root=tmp_path)

    assert package_init.read_text(encoding="utf-8") == '__version__ = "1.2.3"\n'
    assert json.loads(plugin_manifest.read_text(encoding="utf-8"))["version"] == "1.2.3"


def test_bump_version_rejects_unknown_options():
    bump = _load_bump_version_module()

    try:
        bump._parse_args(["bump_version.py", "--gui", "1.2.3"])
    except SystemExit as exc:
        assert "Unknown option" in str(exc)
    else:
        raise AssertionError("expected SystemExit for unknown option")


def test_bump_version_syncs_mcp_server_manifest(tmp_path: Path):
    bump = _load_bump_version_module()

    (tmp_path / "vexor").mkdir(parents=True)
    (tmp_path / "plugins" / "vexor" / ".claude-plugin").mkdir(parents=True)

    package_init = tmp_path / "vexor" / "__init__.py"
    package_init.write_text('__version__ = "0.1.0"\n', encoding="utf-8")

    plugin_manifest = tmp_path / "plugins" / "vexor" / ".claude-plugin" / "plugin.json"
    plugin_manifest.write_text(json.dumps({"version": "0.1.0"}, indent=2) + "\n", encoding="utf-8")

    server_manifest = tmp_path / "server.json"
    server_manifest.write_text(
        json.dumps(
            {"version": "0.1.0", "packages": [{"identifier": "vexor", "version": "0.1.0"}]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    bump._run(version="1.2.3", repo_root=tmp_path)

    synced = json.loads(server_manifest.read_text(encoding="utf-8"))
    assert synced["version"] == "1.2.3"
    assert synced["packages"][0]["version"] == "1.2.3"


def test_bump_version_tolerates_missing_mcp_server_manifest(tmp_path: Path):
    bump = _load_bump_version_module()

    (tmp_path / "vexor").mkdir(parents=True)
    (tmp_path / "plugins" / "vexor" / ".claude-plugin").mkdir(parents=True)

    package_init = tmp_path / "vexor" / "__init__.py"
    package_init.write_text('__version__ = "0.1.0"\n', encoding="utf-8")

    plugin_manifest = tmp_path / "plugins" / "vexor" / ".claude-plugin" / "plugin.json"
    plugin_manifest.write_text(json.dumps({"version": "0.1.0"}, indent=2) + "\n", encoding="utf-8")

    bump._run(version="1.2.3", repo_root=tmp_path)

    assert '__version__ = "1.2.3"' in package_init.read_text(encoding="utf-8")
