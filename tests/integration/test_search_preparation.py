"""Exercise CLI progress against the same real cache preparation used by search."""

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
import pytest
from typer.testing import CliRunner

from vexor import cache, cli
from vexor.config import Config
from vexor.search import VexorSearcher
from vexor.services import index_service, search_service
from vexor.services.search_service import SearchPhase, SearchRequest


class Backend:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, 0 if "alpha" in text else 1] = 1.0
        return vectors


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Config]:
    root = tmp_path / "sources"
    (root / "pkg" / "nested").mkdir(parents=True)
    for relative in ("pkg/alpha.py", "pkg/skip.py", "pkg/beta.md", "pkg/nested/deep.py"):
        (root / relative).write_text("original text", encoding="utf-8")
    config = Config(model="text-embedding-3-small", embedding_dimensions=256)

    def create_backend(searcher: VexorSearcher) -> Backend:
        searcher._device = "test-backend"
        return Backend(searcher.embedding_dimensions or 256)

    monkeypatch.setattr(VexorSearcher, "_create_backend", create_backend)
    monkeypatch.setattr(cli, "_load_config_or_exit", lambda *_args: config)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    cache._clear_embedding_memory_cache()
    return root, config


def build_index(
    root: Path,
    config: Config,
    *,
    recursive: bool = True,
    extensions: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = (),
) -> None:
    index_service.build_index(
        root, include_hidden=False, respect_gitignore=True, mode="name",
        recursive=recursive, model_name=config.model, batch_size=0,
        provider=config.provider, base_url=None, api_key=None,
        embedding_dimensions=config.embedding_dimensions,
        extensions=extensions, exclude_patterns=exclude_patterns,
    )


@pytest.mark.parametrize("output_format", ["rich", "porcelain", "porcelain-z", "json"])
@pytest.mark.parametrize("superset", [False, True], ids=["direct", "parent-superset"])
def test_fresh_search_scans_once_and_preserves_filters(
    corpus: tuple[Path, Config], monkeypatch: pytest.MonkeyPatch,
    output_format: str, superset: bool,
) -> None:
    root, config = corpus
    directory = root / "pkg"
    if superset:
        build_index(root, config)
    else:
        build_index(directory, config, recursive=False, extensions=(".py",),
                    exclude_patterns=("skip.py",))
    scan = Mock(wraps=cache.collect_files)
    build = Mock(wraps=index_service.build_index)
    monkeypatch.setattr(cache, "collect_files", scan)
    monkeypatch.setattr(index_service, "build_index", build)

    result = CliRunner().invoke(cli.app, [
        "search", "alpha", "--path", str(directory), "--mode", "name",
        "--no-recursive", "--ext", ".py", "--exclude-pattern", "skip.py",
        "--format", output_format,
    ])

    assert result.exit_code == 0, result.output
    assert scan.call_count == 1
    build.assert_not_called()
    assert "alpha.py" in result.stdout
    assert all(name not in result.stdout for name in ("skip.py", "beta.md", "deep.py"))
    assert "Indexing files under" not in result.output
    assert ("Searching cached index under" in result.stdout) == (output_format == "rich")
    assert "Searching" not in result.stderr


@pytest.mark.parametrize("cache_state", ["missing", "stale", "dimension"])
@pytest.mark.parametrize("output_format", ["rich", "porcelain", "porcelain-z", "json"])
def test_rebuild_progress_matches_service_decision(
    corpus: tuple[Path, Config], monkeypatch: pytest.MonkeyPatch, cache_state: str,
    output_format: str,
) -> None:
    root, config = corpus
    if cache_state != "missing":
        build_index(root, config)
    if cache_state == "stale":
        (root / "pkg" / "alpha.py").write_text("changed and longer text", encoding="utf-8")
    elif cache_state == "dimension":
        config.embedding_dimensions = 512
    build = Mock(wraps=index_service.build_index)
    progress = Mock(wraps=cli._render_search_progress)
    scan = Mock(wraps=cache.collect_files)
    monkeypatch.setattr(index_service, "build_index", build)
    monkeypatch.setattr(cli, "_render_search_progress", progress)
    monkeypatch.setattr(cache, "collect_files", scan)

    result = CliRunner().invoke(cli.app, ["search", "alpha", "--path", str(root),
                                         "--mode", "name", "--format", output_format])

    assert result.exit_code == 0, result.output
    assert build.call_count == 1
    if output_format == "rich":
        assert progress.call_args_list == [
            call(SearchPhase.INDEXING, root), call(SearchPhase.SEARCHING, root),
        ]
        assert result.stdout.index("Indexing files under") < result.stdout.index("Searching")
    else:
        progress.assert_not_called()
        assert "Indexing files under" not in result.output
        assert "Searching" not in result.output
    # A rebuild still validates the newly loaded snapshot before searching it.
    assert scan.call_count == (2 if cache_state == "stale" else 1)
    _, vectors, _ = cache.load_index_vectors(root, config.model, False, "name", True)
    assert vectors.shape[1] == config.embedding_dimensions


def test_stale_superset_reports_rebuilt_parent_and_searched_child(
    corpus: tuple[Path, Config], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = corpus
    build_index(root, config)
    directory = root / "pkg"
    (directory / "alpha.py").write_text("changed and longer text", encoding="utf-8")
    progress = Mock(wraps=cli._render_search_progress)
    build = Mock(wraps=index_service.build_index)
    monkeypatch.setattr(cli, "_render_search_progress", progress)
    monkeypatch.setattr(index_service, "build_index", build)

    result = CliRunner().invoke(cli.app, [
        "search", "alpha", "--path", str(directory), "--mode", "name",
        "--no-recursive", "--ext", ".py", "--exclude-pattern", "skip.py",
    ])

    assert result.exit_code == 0, result.output
    assert build.call_count == 1
    assert build.call_args.args[0] == root
    assert progress.call_args_list == [
        call(SearchPhase.INDEXING, root), call(SearchPhase.SEARCHING, directory),
    ]
    assert "alpha.py" in result.stdout
    assert all(name not in result.stdout for name in ("skip.py", "beta.md", "deep.py"))


@pytest.mark.parametrize("cache_state", ["missing", "stale"])
def test_auto_index_disabled_never_reports_or_performs_rebuild(
    corpus: tuple[Path, Config], monkeypatch: pytest.MonkeyPatch, cache_state: str,
) -> None:
    root, config = corpus
    config.auto_index = False
    if cache_state == "stale":
        build_index(root, config)
        (root / "pkg" / "alpha.py").write_text("changed and longer text", encoding="utf-8")
    build = Mock(wraps=index_service.build_index)
    monkeypatch.setattr(index_service, "build_index", build)

    result = CliRunner().invoke(cli.app, ["search", "alpha", "--path", str(root),
                                         "--mode", "name"])

    assert result.exit_code == (1 if cache_state == "missing" else 0), result.output
    build.assert_not_called()
    assert "Indexing files under" not in result.output
    if cache_state == "stale":
        assert "Searching cached index under" in result.stdout
        assert "appears outdated" in " ".join(result.stdout.split())
    else:
        assert "Searching cached index under" not in result.stdout


def test_failed_rebuild_reports_indexing_before_the_error(
    corpus: tuple[Path, Config], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = corpus
    monkeypatch.setattr(index_service, "build_index", Mock(side_effect=RuntimeError("failed")))

    result = CliRunner().invoke(cli.app, ["search", "alpha", "--path", str(root)])

    assert result.exit_code == 1
    assert result.stdout.index("Indexing files under") < result.stdout.index("failed")
    assert "Searching cached index under" not in result.output


def test_empty_rebuild_does_not_report_searching(corpus: tuple[Path, Config]) -> None:
    root, _ = corpus
    directory = root / "empty"
    directory.mkdir()

    result = CliRunner().invoke(cli.app, ["search", "alpha", "--path", str(directory)])

    assert result.exit_code == 0, result.output
    assert "Indexing files under" in result.stdout
    assert "Searching" not in result.output


def test_no_cache_progress_leaves_index_storage_untouched(
    corpus: tuple[Path, Config], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = corpus
    scan = Mock(wraps=cache.collect_files)
    monkeypatch.setattr(cache, "collect_files", scan)

    result = CliRunner().invoke(cli.app, ["search", "alpha", "--path", str(root),
                                         "--mode", "name", "--no-cache"])

    assert result.exit_code == 0, result.output
    assert "Searching in-memory index under" in result.stdout
    assert "Indexing files under" not in result.output
    scan.assert_not_called()
    assert not cache.cache_db_path().exists()


@pytest.mark.parametrize("temporary", [False, True])
def test_batch_progress_is_per_operation_and_empty_batch_has_no_side_effects(
    corpus: tuple[Path, Config], temporary: bool,
) -> None:
    root, config = corpus
    events: list[tuple[SearchPhase, Path]] = []
    request = SearchRequest(
        query="alpha", directory=root, include_hidden=False, respect_gitignore=True,
        mode="name", recursive=True, top_k=1, model_name=config.model, batch_size=0,
        provider=config.provider, base_url=None, api_key=None, local_cuda=False,
        exclude_patterns=(), extensions=(), embedding_dimensions=config.embedding_dimensions,
        temporary_index=temporary, on_progress=lambda phase, directory: events.append(
            (phase, directory)
        ),
    )
    assert search_service.perform_search_many(request, []) == []
    assert events == []
    assert not cache.cache_db_path().exists()

    responses = search_service.perform_search_many(request, ["alpha", "beta", "alpha"])

    assert len(responses) == 3
    assert responses[0] == responses[2]
    assert responses[0].results[0] is not responses[2].results[0]
    assert events == (
        [(SearchPhase.SEARCHING_IN_MEMORY, root)] if temporary else
        [(SearchPhase.INDEXING, root), (SearchPhase.SEARCHING, root)]
    )
