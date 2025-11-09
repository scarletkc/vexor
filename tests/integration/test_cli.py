import json
import numpy as np
import pytest
from typer.testing import CliRunner

from vexor.cli import app


@pytest.fixture(autouse=True)
def temp_config_home(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("vexor.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("vexor.config.CONFIG_FILE", config_file)
    return config_file


class FakeSearcher:
    def __init__(self, *args, **kwargs):
        self.device = "fake-backend"

    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            vectors.append([float(len(text)), 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_search_outputs_table(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")

    def fake_load_index(root, model, include_hidden):
        return [sample_file], np.asarray([[1.0, 1.0]], dtype=np.float32), {"files": []}

    monkeypatch.setattr("vexor.cli._load_index", fake_load_index)
    monkeypatch.setattr("vexor.cli._create_searcher", lambda **kwargs: FakeSearcher())

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


def test_search_missing_index_prompts_user(tmp_path, monkeypatch):
    runner = CliRunner()

    def missing_cache(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("vexor.cli._load_index", missing_cache)

    result = runner.invoke(app, ["search", "query", "--path", str(tmp_path)])

    assert result.exit_code == 1
    assert "No cached index" in result.stdout


def test_index_handles_empty_directory(tmp_path, monkeypatch):
    runner = CliRunner()

    def fake_collect(root, include_hidden=False):
        return []

    monkeypatch.setattr("vexor.cli.collect_files", fake_collect)

    result = runner.invoke(app, ["index", "--path", str(tmp_path)])

    assert result.exit_code == 0
    assert "No files found" in result.stdout


def test_index_writes_cache(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")

    def fake_collect(root, include_hidden=False):
        return [sample_file]

    monkeypatch.setattr("vexor.cli.collect_files", fake_collect)
    monkeypatch.setattr("vexor.cli._create_searcher", lambda **kwargs: FakeSearcher())
    monkeypatch.setattr("vexor.cli._load_index_metadata_safe", lambda *args, **kwargs: None)

    stored = {}

    def fake_store_index(**kwargs):
        stored.update(kwargs)
        return tmp_path / "cache.json"

    monkeypatch.setattr("vexor.cli._store_index", fake_store_index)

    result = runner.invoke(app, ["index", "--path", str(tmp_path)])

    assert result.exit_code == 0
    assert "Index saved" in result.stdout
    assert stored["files"] == [sample_file]


def test_index_skips_when_up_to_date(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")

    def fake_collect(root, include_hidden=False):
        return [sample_file]

    monkeypatch.setattr("vexor.cli.collect_files", fake_collect)
    monkeypatch.setattr("vexor.cli._create_searcher", lambda **kwargs: FakeSearcher())
    meta = {
        "files": [
            {
                "path": "alpha.txt",
                "mtime": sample_file.stat().st_mtime,
                "size": sample_file.stat().st_size,
            }
        ]
    }
    monkeypatch.setattr("vexor.cli._load_index_metadata_safe", lambda *args, **kwargs: meta)
    monkeypatch.setattr("vexor.cli._is_cache_current", lambda *args, **kwargs: True)

    result = runner.invoke(app, ["index", "--path", str(tmp_path)])

    assert result.exit_code == 0
    assert "matches the current directory" in result.stdout


def test_search_warns_when_stale(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")

    def fake_load_index(root, model, include_hidden):
        meta = {
            "files": [
                {
                    "path": "alpha.txt",
                    "mtime": sample_file.stat().st_mtime,
                    "size": sample_file.stat().st_size,
                }
            ]
        }
        return [sample_file], np.asarray([[1.0, 1.0]], dtype=np.float32), meta

    monkeypatch.setattr("vexor.cli._load_index", fake_load_index)
    monkeypatch.setattr("vexor.cli._create_searcher", lambda **kwargs: FakeSearcher())
    monkeypatch.setattr("vexor.cli._is_cache_current", lambda *args, **kwargs: False)

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
