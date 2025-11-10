import json
import re

import numpy as np
import pytest
from typer.testing import CliRunner

from vexor import __version__
from vexor.cli import app
from vexor.config import DEFAULT_MODEL
import vexor.cache as cache
from vexor.search import SearchResult
from vexor.services.index_service import IndexResult, IndexStatus
from vexor.services.search_service import SearchResponse
from vexor.text import Messages


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


@pytest.fixture(autouse=True)
def temp_config_home(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("vexor.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("vexor.config.CONFIG_FILE", config_file)
    return config_file


def test_search_outputs_table(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")
    captured = {}

    def fake_perform_search(request):
        captured["recursive"] = request.recursive
        return SearchResponse(
            base_path=tmp_path,
            backend="fake-backend",
            results=[SearchResult(path=sample_file, score=0.99, preview="alpha")],
            is_stale=False,
            index_empty=False,
        )

    monkeypatch.setattr("vexor.cli.perform_search", fake_perform_search)

    result = runner.invoke(
        app,
        [
            "search",
            "alpha",
            "--path",
            str(tmp_path),
            "--top",
            "1",
            "--mode",
            "name",
        ],
    )

    assert result.exit_code == 0
    assert "alpha.txt" in result.stdout
    assert "Similarity" in result.stdout
    assert "Preview" in result.stdout
    assert captured["recursive"] is True


def test_search_respects_no_recursive_flag(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")
    captured = {}

    def fake_perform_search(request):
        captured["recursive"] = request.recursive
        return SearchResponse(
            base_path=tmp_path,
            backend="fake",
            results=[SearchResult(path=sample_file, score=0.5)],
            is_stale=False,
            index_empty=False,
        )

    monkeypatch.setattr("vexor.cli.perform_search", fake_perform_search)

    result = runner.invoke(
        app,
        [
            "search",
            "alpha",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
            "--no-recursive",
        ],
    )

    assert result.exit_code == 0
    assert captured["recursive"] is False


def test_search_missing_index_prompts_user(tmp_path, monkeypatch):
    runner = CliRunner()

    def missing_cache(request):
        raise FileNotFoundError

    monkeypatch.setattr("vexor.cli.perform_search", missing_cache)

    result = runner.invoke(
        app,
        [
            "search",
            "query",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
        ],
    )

    assert result.exit_code == 1
    assert "No cached index" in result.stdout


def test_index_handles_empty_directory(tmp_path, monkeypatch):
    runner = CliRunner()

    def fake_build_index(*args, **kwargs):
        return IndexResult(status=IndexStatus.EMPTY)

    monkeypatch.setattr("vexor.cli.build_index", fake_build_index)

    result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
        ],
    )

    assert result.exit_code == 0
    assert "No files found" in result.stdout


def test_index_writes_cache(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")

    cache_file = tmp_path / "cache.json"
    captured = {}

    def fake_build_index(*args, **kwargs):
        captured["recursive"] = kwargs.get("recursive")
        return IndexResult(status=IndexStatus.STORED, cache_path=cache_file, files_indexed=1)

    monkeypatch.setattr("vexor.cli.build_index", fake_build_index)

    result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
        ],
    )

    assert result.exit_code == 0
    assert "Index saved" in result.stdout
    assert captured["recursive"] is True


def test_index_skips_when_up_to_date(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")
    captured = {}

    def fake_build_index(*args, **kwargs):
        captured["recursive"] = kwargs.get("recursive")
        return IndexResult(status=IndexStatus.UP_TO_DATE)

    monkeypatch.setattr("vexor.cli.build_index", fake_build_index)

    result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
        ],
    )

    assert result.exit_code == 0
    assert "matches the current directory" in result.stdout
    assert captured["recursive"] is True


def test_index_no_recursive_flag(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")
    captured = {}

    def fake_build_index(*args, **kwargs):
        captured["recursive"] = kwargs.get("recursive")
        return IndexResult(status=IndexStatus.STORED, cache_path=None, files_indexed=1)

    monkeypatch.setattr("vexor.cli.build_index", fake_build_index)

    result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
            "--no-recursive",
        ],
    )

    assert result.exit_code == 0
    assert captured["recursive"] is False


def test_index_show_displays_metadata(tmp_path, monkeypatch):
    runner = CliRunner()
    metadata = {
        "mode": "name",
        "model": "model-x",
        "include_hidden": False,
        "recursive": True,
        "dimension": 256,
        "version": 3,
        "generated_at": "2024-01-02T00:00:00Z",
        "files": [{}, {}],
    }
    monkeypatch.setattr("vexor.cli.load_index_metadata_safe", lambda *args, **kwargs: metadata)

    result = runner.invoke(
        app,
        [
            "index",
            "--show",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
        ],
    )

    assert result.exit_code == 0
    assert "Cached index details" in result.stdout
    assert "model-x" in result.stdout
    assert "Files: 2" in result.stdout


def test_index_show_handles_missing_cache(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.load_index_metadata_safe", lambda *args, **kwargs: None)

    result = runner.invoke(
        app,
        [
            "index",
            "--show",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
        ],
    )

    assert result.exit_code == 0
    assert "No cached index" in result.stdout


def test_index_show_conflicts_with_clear(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "index",
            "--show",
            "--clear",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
        ],
    )

    assert result.exit_code != 0
    combined = strip_ansi(result.output)
    assert Messages.ERROR_INDEX_SHOW_CONFLICT in combined


def test_index_clear_option(tmp_path, monkeypatch):
    runner = CliRunner()
    called = {}

    def fake_clear(root, include_hidden, mode, recursive, model=None):
        called["root"] = root
        called["include_hidden"] = include_hidden
        called["mode"] = mode
        called["recursive"] = recursive
        return 1

    monkeypatch.setattr("vexor.cli.clear_index_entries", fake_clear)

    result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
            "--clear",
            "--include-hidden",
        ],
    )

    assert result.exit_code == 0
    assert "Removed" in result.stdout
    assert called["root"] == tmp_path
    assert called["include_hidden"] is True
    assert called["mode"] == "name"
    assert called["recursive"] is True


def test_index_clear_honors_no_recursive(tmp_path, monkeypatch):
    runner = CliRunner()
    called = {}

    def fake_clear(root, include_hidden, mode, recursive, model=None):
        called["recursive"] = recursive
        called["mode"] = mode
        return 0

    monkeypatch.setattr("vexor.cli.clear_index_entries", fake_clear)

    result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
            "--clear",
            "--no-recursive",
        ],
    )

    assert result.exit_code == 0
    assert called["recursive"] is False
    assert called["mode"] == "name"


def test_search_warns_when_stale(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")

    def fake_perform_search(request):
        return SearchResponse(
            base_path=tmp_path,
            backend="fake-backend",
            results=[SearchResult(path=sample_file, score=0.9)],
            is_stale=True,
            index_empty=False,
        )

    monkeypatch.setattr("vexor.cli.perform_search", fake_perform_search)

    result = runner.invoke(
        app,
        [
            "search",
            "alpha",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
        ],
    )

    assert result.exit_code == 0
    assert "appears outdated" in result.stdout


def test_config_set_and_show(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "config",
            "--set-api-key",
            "abc123",
            "--set-model",
            "custom-model",
            "--set-batch-size",
            "42",
        ],
    )

    assert result.exit_code == 0
    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["api_key"] == "abc123"
    assert data["model"] == "custom-model"
    assert data["batch_size"] == 42

    result_show = runner.invoke(app, ["config", "--show"])
    assert "custom-model" in result_show.stdout


def test_config_clear_api_key(tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "config" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"api_key": "secret"}), encoding="utf-8")

    result = runner.invoke(app, ["config", "--clear-api-key", "--show"])

    assert result.exit_code == 0
    data = json.loads(config_path.read_text())
    assert "api_key" not in data


def test_config_without_args_opens_editor(tmp_path, monkeypatch):
    runner = CliRunner()
    captured = {}

    monkeypatch.setattr("vexor.cli.resolve_editor_command", lambda: ("editor",))

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr("vexor.cli.subprocess.run", fake_run)

    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "Opening config file" in result.stdout
    config_path = tmp_path / "config" / "config.json"
    assert captured["cmd"][-1] == str(config_path)


def test_config_without_editor_reports_error(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.resolve_editor_command", lambda: None)

    result = runner.invoke(app, ["config"])

    assert result.exit_code == 1
    assert Messages.ERROR_CONFIG_EDITOR_NOT_FOUND in result.stdout


def test_config_show_index_all(tmp_path, monkeypatch):
    runner = CliRunner()
    root = tmp_path / "proj"
    entries = [
        {
            "root_path": str(root),
            "mode": "name",
            "model": "model-x",
            "include_hidden": False,
            "recursive": True,
            "file_count": 3,
            "generated_at": "2024-01-01T00:00:00Z",
        }
    ]
    monkeypatch.setattr("vexor.cli.list_cache_entries", lambda: entries)

    result = runner.invoke(app, ["config", "--show-index-all"])

    assert result.exit_code == 0
    assert "Cached index overview" in result.stdout
    assert "model-x" in result.stdout


def test_config_show_index_all_empty(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.list_cache_entries", lambda: [])

    result = runner.invoke(app, ["config", "--show-index-all"])

    assert result.exit_code == 0
    assert "No cached indexes" in result.stdout


def test_config_clear_index_all(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.clear_all_cache", lambda: 5)

    result = runner.invoke(app, ["config", "--clear-index-all"])

    assert result.exit_code == 0
    assert "Removed 5 cached index" in result.stdout


def test_config_clear_index_all_noop(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.clear_all_cache", lambda: 0)

    result = runner.invoke(app, ["config", "--clear-index-all"])

    assert result.exit_code == 0
    assert "Cache already empty" in result.stdout


def test_doctor_reports_success(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.find_command_on_path", lambda cmd: "/usr/local/bin/vexor")

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "available" in result.stdout


def test_doctor_reports_failure(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.find_command_on_path", lambda cmd: None)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "not on PATH" in result.stdout


def test_update_detects_newer_version(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_remote_version", lambda url: "9.9.9")

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "New version available" in result.stdout


def test_update_reports_up_to_date(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_remote_version", lambda url: __version__)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "latest version" in result.stdout


def test_update_handles_fetch_error(monkeypatch):
    runner = CliRunner()

    def boom(url):
        raise RuntimeError("network down")

    monkeypatch.setattr("vexor.cli.fetch_remote_version", boom)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    assert "Unable to fetch" in result.stdout

def test_head_mode_end_to_end(tmp_path, monkeypatch):
    class DummySearcher:
        def __init__(self, *args, **kwargs):
            self.device = "dummy"

        def embed_texts(self, texts):
            if not texts:
                return np.zeros((0, 3), dtype=np.float32)
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr("vexor.search.VexorSearcher", DummySearcher)
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr("vexor.cache.CACHE_DIR", cache_dir)
    runner = CliRunner()
    project = tmp_path / "proj"
    project.mkdir()
    file_path = project / "sample.txt"
    file_path.write_text("Intro line\nMore content\n")
    from vexor.services.content_extract_service import extract_head
    assert extract_head(file_path).startswith("Intro line")

    index_result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(project),
            "--mode",
            "head",
        ],
    )
    assert index_result.exit_code == 0
    data = cache.load_index(
        project,
        DEFAULT_MODEL,
        include_hidden=False,
        mode="head",
        recursive=True,
    )
    previews = [entry.get("preview") for entry in data.get("files", [])]
    assert previews and previews[0].startswith("Intro line")

    search_result = runner.invoke(
        app,
        [
            "search",
            "intro",
            "--path",
            str(project),
            "--mode",
            "head",
        ],
    )
    assert search_result.exit_code == 0
    assert "Preview" in search_result.stdout
    assert "Intro line" in search_result.stdout
