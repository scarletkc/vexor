import json
from pathlib import Path

from vexor import __version__


def test_plugin_manifest_version_matches_package():
    manifest_path = Path("plugins/vexor/.claude-plugin/plugin.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == __version__

