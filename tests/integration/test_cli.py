import json
import pytest
from typer.testing import CliRunner

from vexor import __version__
from vexor.cli import app
from vexor.search import SearchResult
from vexor.services.index_service import IndexResult, IndexStatus
from vexor.services.search_service import SearchResponse


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
            results=[SearchResult(path=sample_file, score=0.99)],
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
        ],
    )

    assert result.exit_code == 0
    assert "alpha.txt" in result.stdout
    assert "Similarity" in result.stdout
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

    result = runner.invoke(app, ["search", "query", "--path", str(tmp_path)])

    assert result.exit_code == 1
    assert "No cached index" in result.stdout


def test_index_handles_empty_directory(tmp_path, monkeypatch):
    runner = CliRunner()

    def fake_build_index(*args, **kwargs):
        return IndexResult(status=IndexStatus.EMPTY)

    monkeypatch.setattr("vexor.cli.build_index", fake_build_index)

    result = runner.invoke(app, ["index", "--path", str(tmp_path)])

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

    result = runner.invoke(app, ["index", "--path", str(tmp_path)])

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

    result = runner.invoke(app, ["index", "--path", str(tmp_path)])

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
            "--no-recursive",
        ],
    )

    assert result.exit_code == 0
    assert captured["recursive"] is False


def test_index_clear_option(tmp_path, monkeypatch):
    runner = CliRunner()
    called = {}

    def fake_clear(root, include_hidden, recursive, model=None):
        called["root"] = root
        called["include_hidden"] = include_hidden
        called["recursive"] = recursive
        return 1

    monkeypatch.setattr("vexor.cli.clear_index_entries", fake_clear)

    result = runner.invoke(app, ["index", "--path", str(tmp_path), "--clear", "--include-hidden"])

    assert result.exit_code == 0
    assert "Removed" in result.stdout
    assert called["root"] == tmp_path
    assert called["include_hidden"] is True
    assert called["recursive"] is True


def test_index_clear_honors_no_recursive(tmp_path, monkeypatch):
    runner = CliRunner()
    called = {}

    def fake_clear(root, include_hidden, recursive, model=None):
        called["recursive"] = recursive
        return 0

    monkeypatch.setattr("vexor.cli.clear_index_entries", fake_clear)

    result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(tmp_path),
            "--clear",
            "--no-recursive",
        ],
    )

    assert result.exit_code == 0
    assert called["recursive"] is False


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
