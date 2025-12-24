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
        captured["mode"] = request.mode
        return SearchResponse(
            base_path=tmp_path,
            backend="fake-backend",
            results=[
                SearchResult(
                    path=sample_file,
                    score=0.99,
                    preview="alpha",
                    start_line=12,
                    end_line=34,
                )
            ],
            is_stale=False,
            index_empty=False,
            reranker="bm25",
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
    assert "Lines" in result.stdout
    assert "Preview" in result.stdout
    assert "L12-34" in result.stdout
    assert "Reranker: bm25" in result.stdout
    assert captured["recursive"] is True
    assert captured["mode"] == "auto"


def test_search_no_respect_gitignore_flag_sets_false(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")
    captured = {}

    def fake_perform_search(request):
        captured["respect_gitignore"] = request.respect_gitignore
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
            "--no-respect-gitignore",
            "--format",
            "porcelain",
        ],
    )

    assert result.exit_code == 0
    assert captured["respect_gitignore"] is False


def test_search_no_cache_flag_sets_true(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")
    captured = {}

    def fake_perform_search(request):
        captured["no_cache"] = request.no_cache
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
            "--no-cache",
            "--format",
            "porcelain",
        ],
    )

    assert result.exit_code == 0
    assert captured["no_cache"] is True


def test_search_prints_index_message_when_auto_index_missing(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")

    monkeypatch.setattr("vexor.cli.load_index_metadata_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("vexor.cli.list_cache_entries", lambda: [])

    def fake_perform_search(request):
        return SearchResponse(
            base_path=tmp_path,
            backend="fake-backend",
            results=[SearchResult(path=sample_file, score=0.99)],
            is_stale=False,
            index_empty=False,
        )

    monkeypatch.setattr("vexor.cli.perform_search", fake_perform_search)

    result = runner.invoke(app, ["search", "alpha", "--path", str(tmp_path), "--top", "1"])

    assert result.exit_code == 0
    output = strip_ansi(result.stdout)
    assert "Indexing files under" in output
    assert str(tmp_path) in output
    assert "Searching cached index under" not in output


def test_search_prints_index_message_when_auto_index_stale(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")

    monkeypatch.setattr(
        "vexor.cli.load_index_metadata_safe",
        lambda *_args, **_kwargs: {"files": [{"path": "alpha.txt", "mtime": 0.0, "size": 1}]},
    )
    monkeypatch.setattr("vexor.cli.is_cache_current", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("vexor.cli.list_cache_entries", lambda: [])

    def fake_perform_search(request):
        return SearchResponse(
            base_path=tmp_path,
            backend="fake-backend",
            results=[SearchResult(path=sample_file, score=0.99)],
            is_stale=False,
            index_empty=False,
        )

    monkeypatch.setattr("vexor.cli.perform_search", fake_perform_search)

    result = runner.invoke(app, ["search", "alpha", "--path", str(tmp_path), "--top", "1"])

    assert result.exit_code == 0
    output = strip_ansi(result.stdout)
    assert "Indexing files under" in output
    assert str(tmp_path) in output
    assert "Searching cached index under" not in output


def test_search_outputs_porcelain(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")
    captured = {}

    def fake_perform_search(request):
        captured["recursive"] = request.recursive
        return SearchResponse(
            base_path=tmp_path,
            backend="fake-backend",
            results=[
                SearchResult(
                    path=sample_file,
                    score=0.99,
                    preview="alpha\tbeta\ncharlie",
                    start_line=5,
                    end_line=6,
                )
            ],
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
            "--format",
            "porcelain",
        ],
    )

    assert result.exit_code == 0
    assert "1\t0.990\t./alpha.txt\t0\t5\t6\talpha\\tbeta\\ncharlie\n" in result.stdout
    assert "Similarity" not in result.stdout
    assert captured["recursive"] is True


def test_default_search_invokes_search(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")
    captured = {}

    def fake_perform_search(request):
        captured["query"] = request.query
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
            "alpha",
            "--path",
            str(tmp_path),
            "--top",
            "1",
            "--format",
            "porcelain",
        ],
    )

    assert result.exit_code == 0
    assert captured["query"] == "alpha"
    assert "./alpha.txt" in result.stdout


def test_search_outputs_porcelain_z(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")

    def fake_perform_search(request):
        return SearchResponse(
            base_path=tmp_path,
            backend="fake-backend",
            results=[
                SearchResult(
                    path=sample_file,
                    score=0.99,
                    preview="alpha\tbeta",
                    start_line=7,
                    end_line=7,
                )
            ],
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
            "--format",
            "porcelain-z",
        ],
    )

    assert result.exit_code == 0
    assert "1\0" in result.stdout
    fields = result.stdout.split("\0")
    # Trailing delimiter yields an empty field at end.
    fields = [field for field in fields if field]
    assert fields[-7:] == ["1", "0.990", "./alpha.txt", "0", "7", "7", "alpha\tbeta"]


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


def test_search_parses_multiple_extensions_in_single_flag(tmp_path, monkeypatch):
    runner = CliRunner()
    captured = {}

    def fake_perform_search(request):
        captured["extensions"] = request.extensions
        return SearchResponse(
            base_path=tmp_path,
            backend="fake",
            results=[],
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
            "--ext",
            ".py,.md",
        ],
    )

    assert result.exit_code == 0
    assert captured["extensions"] == (".md", ".py")


def test_search_parses_space_separated_extensions_in_single_flag(tmp_path, monkeypatch):
    runner = CliRunner()
    captured = {}

    def fake_perform_search(request):
        captured["extensions"] = request.extensions
        return SearchResponse(
            base_path=tmp_path,
            backend="fake",
            results=[],
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
            "--ext",
            ".py .md",
        ],
    )

    assert result.exit_code == 0
    assert captured["extensions"] == (".md", ".py")


def test_install_skills_to_custom_path(tmp_path):
    runner = CliRunner()
    destination = tmp_path / "skills-root"

    result = runner.invoke(app, ["install", "--skills", str(destination)])

    assert result.exit_code == 0
    assert (destination / "vexor-cli" / "SKILL.md").exists()


def test_install_skills_presets_claude_and_codex(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("HOME", str(tmp_path))

    result = runner.invoke(app, ["install", "--skills", "claude/codex"])

    assert result.exit_code == 0
    assert (tmp_path / ".claude" / "skills" / "vexor-cli" / "SKILL.md").exists()
    assert (tmp_path / ".codex" / "skills" / "vexor-cli" / "SKILL.md").exists()


def test_install_skills_requires_force_when_destination_differs(tmp_path):
    runner = CliRunner()
    destination = tmp_path / "skills-root"
    skill_dir = destination / "vexor-cli"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("not the real skill")

    result = runner.invoke(app, ["install", "--skills", str(destination)])
    assert result.exit_code == 1

    forced = runner.invoke(app, ["install", "--skills", str(destination), "--force"])
    assert forced.exit_code == 0
    assert "Vexor CLI" in (skill_dir / "SKILL.md").read_text()


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


def test_index_parses_multiple_extensions_in_single_flag(tmp_path, monkeypatch):
    runner = CliRunner()
    captured = {}

    def fake_build_index(*_args, **kwargs):
        captured["extensions"] = kwargs.get("extensions")
        return IndexResult(status=IndexStatus.EMPTY)

    monkeypatch.setattr("vexor.cli.build_index", fake_build_index)

    result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(tmp_path),
            "--ext",
            ".py,.md",
        ],
    )

    assert result.exit_code == 0
    assert captured["extensions"] == (".md", ".py")


def test_index_writes_cache(tmp_path, monkeypatch):
    runner = CliRunner()
    sample_file = tmp_path / "alpha.txt"
    sample_file.write_text("data")

    cache_file = tmp_path / "cache.json"
    captured = {}

    def fake_build_index(*args, **kwargs):
        captured["recursive"] = kwargs.get("recursive")
        captured["mode"] = kwargs.get("mode")
        return IndexResult(status=IndexStatus.STORED, cache_path=cache_file, files_indexed=1)

    monkeypatch.setattr("vexor.cli.build_index", fake_build_index)

    result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Index saved" in result.stdout
    assert captured["recursive"] is True
    assert captured["mode"] == "auto"


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

    def fake_clear(
        root,
        include_hidden,
        respect_gitignore,
        mode,
        recursive,
        exclude_patterns=None,
        model=None,
        extensions=None,
    ):
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

    def fake_clear(
        root,
        include_hidden,
        respect_gitignore,
        mode,
        recursive,
        exclude_patterns=None,
        model=None,
        extensions=None,
    ):
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
            "--set-embed-concurrency",
            "3",
            "--set-provider",
            "gemini",
            "--set-base-url",
            "https://proxy.example.com",
            "--rerank",
            "bm25",
            "--set-flashrank-model",
            "ms-marco-MultiBERT-L-12",
        ],
    )

    assert result.exit_code == 0
    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["api_key"] == "abc123"
    assert data["model"] == "custom-model"
    assert data["batch_size"] == 42
    assert data["embed_concurrency"] == 3
    assert data["provider"] == "gemini"
    assert data["base_url"] == "https://proxy.example.com"
    assert data["auto_index"] is True
    assert data["local_cuda"] is False
    assert data["rerank"] == "bm25"
    assert data["flashrank_model"] == "ms-marco-MultiBERT-L-12"

    result_show = runner.invoke(app, ["config", "--show"])
    assert "custom-model" in result_show.stdout
    assert "Embedding concurrency: 3" in strip_ansi(result_show.stdout)
    assert "Rerank: bm25" in strip_ansi(result_show.stdout)
    assert "FlashRank model" not in strip_ansi(result_show.stdout)


def test_config_set_flashrank_model_empty_resets_to_default(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["config", "--set-flashrank-model", "ms-marco-MultiBERT-L-12"])
    assert result.exit_code == 0
    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["flashrank_model"] == "ms-marco-MultiBERT-L-12"

    result_reset = runner.invoke(app, ["config", "--set-flashrank-model"])
    assert result_reset.exit_code == 0
    data = json.loads(config_path.read_text())
    assert "flashrank_model" not in data

    result = runner.invoke(app, ["config", "--set-flashrank-model", "ms-marco-MultiBERT-L-12"])
    assert result.exit_code == 0
    data = json.loads(config_path.read_text())
    assert data["flashrank_model"] == "ms-marco-MultiBERT-L-12"

    result_reset = runner.invoke(app, ["config", "--set-flashrank-model", ""])
    assert result_reset.exit_code == 0
    data = json.loads(config_path.read_text())
    assert "flashrank_model" not in data


def test_config_sets_remote_rerank_and_shows_summary(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "config",
            "--rerank",
            "remote",
            "--set-remote-rerank-url",
            "https://api.example.test/v1/rerank",
            "--set-remote-rerank-model",
            "rerank-model",
            "--set-remote-rerank-api-key",
            "remote-key",
        ],
    )
    assert result.exit_code == 0
    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["rerank"] == "remote"
    assert data["remote_rerank"]["base_url"] == "https://api.example.test/v1/rerank"
    assert data["remote_rerank"]["model"] == "rerank-model"
    assert data["remote_rerank"]["api_key"] == "remote-key"

    result_show = runner.invoke(app, ["config", "--show"])
    output = strip_ansi(result_show.stdout)
    assert "Rerank: remote" in output
    assert "Remote rerank" in output
    assert "api.example.test" in output


def test_config_rejects_remote_rerank_missing_fields(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["config", "--rerank", "remote"])
    assert result.exit_code != 0
    assert "Remote rerank requires" in result.stderr


def test_config_allows_remote_rerank_api_key_env(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("VEXOR_REMOTE_RERANK_API_KEY", "env-remote-key")

    result = runner.invoke(
        app,
        [
            "config",
            "--rerank",
            "remote",
            "--set-remote-rerank-url",
            "https://api.example.test/v1/rerank",
            "--set-remote-rerank-model",
            "rerank-model",
        ],
    )
    assert result.exit_code == 0
    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["rerank"] == "remote"
    assert data["remote_rerank"]["base_url"] == "https://api.example.test/v1/rerank"
    assert data["remote_rerank"]["model"] == "rerank-model"
    assert "api_key" not in data["remote_rerank"]


def test_config_custom_requires_model_and_base_url(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["config", "--set-provider", "custom"])
    assert result.exit_code != 0
    assert "model name" in result.stderr

    result = runner.invoke(
        app,
        ["config", "--set-provider", "custom", "--set-model", "embed-model"],
    )
    assert result.exit_code != 0
    assert "base URL" in result.stderr

    result = runner.invoke(
        app,
        [
            "config",
            "--set-provider",
            "custom",
            "--set-model",
            "embed-model",
            "--set-base-url",
            "https://example.com",
        ],
    )
    assert result.exit_code == 0
    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["provider"] == "custom"
    assert data["model"] == "embed-model"
    assert data["base_url"] == "https://example.com"


def test_local_setup_updates_config(tmp_path, monkeypatch):
    runner = CliRunner()

    class DummyLocalBackend:
        def __init__(self, model_name: str, cuda: bool = False) -> None:
            self.model_name = model_name
            self.cuda = cuda

        def embed(self, texts):
            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr("vexor.cli.LocalEmbeddingBackend", DummyLocalBackend)
    def _cache_dir(create: bool = True):
        path = tmp_path / "models"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr("vexor.cli.resolve_fastembed_cache_dir", _cache_dir)

    result = runner.invoke(app, ["local", "--setup", "--model", "local-model"])

    assert result.exit_code == 0
    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["provider"] == "local"
    assert data["model"] == "local-model"


def test_local_cleanup_removes_cache(tmp_path, monkeypatch):
    runner = CliRunner()
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()
    (cache_dir / "dummy.txt").write_text("data", encoding="utf-8")

    monkeypatch.setattr(
        "vexor.cli.resolve_fastembed_cache_dir",
        lambda create=False: cache_dir,
    )

    result = runner.invoke(app, ["local", "--clean-up"])

    assert result.exit_code == 0
    assert not cache_dir.exists()


def test_local_cuda_toggle_updates_config(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["local", "--cuda"])
    assert result.exit_code == 0

    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["local_cuda"] is True

    result = runner.invoke(app, ["local", "--cpu"])
    assert result.exit_code == 0

    data = json.loads(config_path.read_text())
    assert data["local_cuda"] is False


def test_init_wizard_configures_remote(tmp_path):
    runner = CliRunner()
    user_input = "\n".join(
        [
            "B",  # remote
            "A",  # openai
            "sk-test",  # api key
            "1",  # rerank off
            "n",  # alias
            "n",  # skills
            "n",  # doctor
            "",
        ]
    )

    result = runner.invoke(app, ["init"], input=user_input)

    assert result.exit_code == 0
    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["provider"] == "openai"
    assert data["api_key"] == "sk-test"
    assert data["rerank"] == "off"
    output = strip_ansi(result.stdout)
    assert "vexor \"retry logic handled\"" in output


def test_config_set_auto_index(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["config", "--set-auto-index", "false"])

    assert result.exit_code == 0
    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["auto_index"] is False

    result_show = runner.invoke(app, ["config", "--show"])
    assert "Auto index: no" in strip_ansi(result_show.stdout)


def test_config_clear_api_key(tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "config" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"api_key": "secret"}), encoding="utf-8")

    result = runner.invoke(app, ["config", "--clear-api-key", "--show"])

    assert result.exit_code == 0
    data = json.loads(config_path.read_text())
    assert "api_key" not in data


def test_config_clear_base_url(tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "config" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"base_url": "https://proxy.example.com"}), encoding="utf-8"
    )

    result = runner.invoke(app, ["config", "--clear-base-url", "--show"])

    assert result.exit_code == 0
    data = json.loads(config_path.read_text())
    assert "base_url" not in data


def test_config_clear_flashrank_cache(tmp_path):
    runner = CliRunner()
    cache_dir = tmp_path / "config" / "flashrank"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "model.onnx").write_text("data", encoding="utf-8")

    result = runner.invoke(app, ["config", "--clear-flashrank"])

    assert result.exit_code == 0
    assert not cache_dir.exists()


def test_config_clear_flashrank_conflict(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["config", "--clear-flashrank", "--show"])

    assert result.exit_code != 0
    assert "clear-flashrank" in strip_ansi(result.stderr)


def test_config_rejects_unknown_provider(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["config", "--set-provider", "unknown"])

    assert result.exit_code != 0
    assert "Unsupported provider" in result.stderr


def test_config_rejects_flashrank_when_missing(tmp_path, monkeypatch):
    runner = CliRunner()
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "flashrank":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    result = runner.invoke(app, ["config", "--rerank", "flashrank"])

    assert result.exit_code != 0
    assert "FlashRank reranker is not installed" in result.stderr


def test_config_sets_flashrank_and_prefetches(tmp_path, monkeypatch):
    runner = CliRunner()
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "flashrank":
            return object()
        return real_find_spec(name, *args, **kwargs)

    called = {"ok": False}

    def fake_prepare_flashrank_model(_model_name=None):
        called["ok"] = True

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr("vexor.cli._prepare_flashrank_model", fake_prepare_flashrank_model)

    result = runner.invoke(app, ["config", "--rerank", "flashrank"])

    assert result.exit_code == 0
    assert called["ok"] is True
    config_path = tmp_path / "config" / "config.json"
    data = json.loads(config_path.read_text())
    assert data["rerank"] == "flashrank"

    result_show = runner.invoke(app, ["config", "--show"])
    assert "Rerank: flashrank" in strip_ansi(result_show.stdout)
    assert (
        "FlashRank model: default (ms-marco-TinyBERT-L-2-v2)"
        in strip_ansi(result_show.stdout)
    )


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
    # Mock all doctor checks to pass
    from vexor.services.system_service import DoctorCheckResult
    mock_results = [
        DoctorCheckResult(name="Command", passed=True, message="`vexor` found at /usr/local/bin/vexor"),
        DoctorCheckResult(name="Config", passed=True, message="Config exists"),
        DoctorCheckResult(name="Cache Dir", passed=True, message="Cache writable"),
        DoctorCheckResult(name="API Key", passed=True, message="API key configured"),
    ]
    monkeypatch.setattr("vexor.cli.run_all_doctor_checks", lambda **kw: mock_results)

    result = runner.invoke(app, ["doctor", "--skip-api-test"])

    assert result.exit_code == 0
    assert "passed" in result.stdout.lower()


def test_doctor_reports_failure(monkeypatch):
    runner = CliRunner()
    # Mock doctor checks with failures
    from vexor.services.system_service import DoctorCheckResult
    mock_results = [
        DoctorCheckResult(name="Command", passed=False, message="`vexor` not found on PATH"),
        DoctorCheckResult(name="Config", passed=True, message="Config exists"),
        DoctorCheckResult(name="Cache Dir", passed=True, message="Cache writable"),
        DoctorCheckResult(name="API Key", passed=False, message="API key not configured"),
    ]
    monkeypatch.setattr("vexor.cli.run_all_doctor_checks", lambda **kw: mock_results)

    result = runner.invoke(app, ["doctor", "--skip-api-test"])

    assert result.exit_code == 1
    assert "failed" in result.stdout.lower()


def test_doctor_handles_malformed_config(monkeypatch, temp_config_home):
    runner = CliRunner()
    temp_config_home.parent.mkdir(parents=True, exist_ok=True)
    temp_config_home.write_text("{invalid", encoding="utf-8")

    from vexor.services.system_service import DoctorCheckResult

    mock_results = [
        DoctorCheckResult(name="Command", passed=True, message="`vexor` found at /usr/local/bin/vexor"),
        DoctorCheckResult(name="Config", passed=True, message="Config exists"),
        DoctorCheckResult(name="Cache Dir", passed=True, message="Cache writable"),
        DoctorCheckResult(name="API Key", passed=True, message="API key configured"),
    ]
    monkeypatch.setattr("vexor.cli.run_all_doctor_checks", lambda **kw: mock_results)

    result = runner.invoke(app, ["doctor", "--skip-api-test"])

    assert result.exit_code == 1
    assert "invalid json" in result.stdout.lower()


def test_update_detects_newer_version(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_latest_pypi_version", lambda *_args, **_kwargs: "9.9.9")

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "New version available" in result.stdout


def test_update_reports_up_to_date(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_latest_pypi_version", lambda *_args, **_kwargs: __version__)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "latest version" in result.stdout


def test_update_handles_fetch_error(monkeypatch):
    runner = CliRunner()

    def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("vexor.cli.fetch_latest_pypi_version", boom)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    assert "Unable to fetch" in result.stdout


def test_update_upgrade_runs_commands(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_latest_pypi_version", lambda *_a, **_k: "9.9.9")

    from vexor.services.system_service import InstallInfo, InstallMethod

    monkeypatch.setattr(
        "vexor.cli.detect_install_method",
        lambda: InstallInfo(
            method=InstallMethod.PIP_VENV,
            executable=None,
            editable_root=None,
            dist_location=None,
            requires_admin=False,
        ),
    )

    captured = {}

    def fake_run_upgrade_commands(commands, *args, **kwargs):
        captured["commands"] = commands
        return 0

    monkeypatch.setattr("vexor.cli.run_upgrade_commands", fake_run_upgrade_commands)

    result = runner.invoke(app, ["update", "--upgrade"], input="y\n")

    assert result.exit_code == 0
    assert "Upgrading Vexor" in result.stdout
    assert captured["commands"][0][-2:] == ["--upgrade", "vexor"]


def test_update_upgrade_pre_adds_flag(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_latest_pypi_version", lambda *_a, **_k: "9.9.9")

    from vexor.services.system_service import InstallInfo, InstallMethod

    monkeypatch.setattr(
        "vexor.cli.detect_install_method",
        lambda: InstallInfo(
            method=InstallMethod.PIP_VENV,
            executable=None,
            editable_root=None,
            dist_location=None,
            requires_admin=False,
        ),
    )

    captured = {}

    def fake_run_upgrade_commands(commands, *args, **kwargs):
        captured["commands"] = commands
        return 0

    monkeypatch.setattr("vexor.cli.run_upgrade_commands", fake_run_upgrade_commands)

    result = runner.invoke(app, ["update", "--upgrade", "--pre"], input="y\n")

    assert result.exit_code == 0
    assert "--pre" in captured["commands"][0]


def test_update_upgrade_standalone_shows_download_url(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vexor.cli.fetch_latest_pypi_version", lambda *_a, **_k: "9.9.9")

    from vexor.services.system_service import InstallInfo, InstallMethod

    monkeypatch.setattr(
        "vexor.cli.detect_install_method",
        lambda: InstallInfo(
            method=InstallMethod.STANDALONE,
            executable=None,
            editable_root=None,
            dist_location=None,
            requires_admin=False,
        ),
    )

    result = runner.invoke(app, ["update", "--upgrade"])

    assert result.exit_code == 0
    assert "Download:" in result.stdout

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
        respect_gitignore=True,
        mode="head",
        recursive=True,
    )
    previews = [entry.get("preview") for entry in data.get("chunks", [])]
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


def test_index_no_respect_gitignore_flag_sets_false(tmp_path, monkeypatch):
    runner = CliRunner()
    captured = {}

    def fake_build_index(directory, **kwargs):
        captured["respect_gitignore"] = kwargs.get("respect_gitignore")
        return IndexResult(status=IndexStatus.UP_TO_DATE, files_indexed=0)

    monkeypatch.setattr("vexor.cli.build_index", fake_build_index)

    result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(tmp_path),
            "--mode",
            "name",
            "--no-respect-gitignore",
        ],
    )

    assert result.exit_code == 0
    assert captured["respect_gitignore"] is False


def test_full_mode_chunked_previews(tmp_path, monkeypatch):
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
    project = tmp_path / "proj-full"
    project.mkdir()
    file_path = project / "long.txt"
    file_path.write_text("paragraph " * 200)

    index_result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(project),
            "--mode",
            "full",
        ],
    )
    assert index_result.exit_code == 0
    data = cache.load_index(
        project,
        DEFAULT_MODEL,
        include_hidden=False,
        respect_gitignore=True,
        mode="full",
        recursive=True,
    )
    chunks = data.get("chunks", [])
    assert len(chunks) >= 2
    assert max(entry.get("chunk_index", 0) for entry in chunks) >= 1

    search_result = runner.invoke(
        app,
        [
            "search",
            "paragraph",
            "--path",
            str(project),
            "--mode",
            "full",
        ],
    )
    assert search_result.exit_code == 0
    assert "paragraph" in search_result.stdout
    assert "[Chunk" not in search_result.stdout


def test_brief_mode_keywords(tmp_path, monkeypatch):
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
    project = tmp_path / "proj-brief"
    project.mkdir()
    file_path = project / "prd.md"
    file_path.write_text(
        """# Messaging
The chat module must support offline messaging. Offline send, offline sync, and offline drafts are required."""
    )

    index_result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(project),
            "--mode",
            "brief",
        ],
    )
    assert index_result.exit_code == 0

    search_result = runner.invoke(
        app,
        [
            "search",
            "offline",
            "--path",
            str(project),
            "--mode",
            "brief",
        ],
    )
    assert search_result.exit_code == 0
    assert "offline" in search_result.stdout


def test_outline_mode_headings(tmp_path, monkeypatch):
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
    project = tmp_path / "proj-outline"
    project.mkdir()
    doc_path = project / "doc.md"
    doc_path.write_text(
        """Intro before headings.

# Top
Top body.

## Child
Child body.
"""
    )

    index_result = runner.invoke(
        app,
        [
            "index",
            "--path",
            str(project),
            "--mode",
            "outline",
        ],
    )
    assert index_result.exit_code == 0

    data = cache.load_index(
        project,
        DEFAULT_MODEL,
        include_hidden=False,
        respect_gitignore=True,
        mode="outline",
        recursive=True,
    )
    previews = [entry.get("preview") for entry in data.get("chunks", [])]
    assert any((preview or "").startswith("preamble") for preview in previews)
    assert any("Top > Child" in (preview or "") for preview in previews)

    search_result = runner.invoke(
        app,
        [
            "search",
            "child",
            "--path",
            str(project),
            "--mode",
            "outline",
        ],
    )
    assert search_result.exit_code == 0
    assert "Top > Child" in search_result.stdout
