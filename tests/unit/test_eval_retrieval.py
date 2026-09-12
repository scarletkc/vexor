import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval_retrieval as evaluation
from vexor.config import Config, RemoteRerankConfig
from vexor.search import SearchResult


def make_suite(tmp_path: Path) -> tuple[Path, dict]:
    (tmp_path / "source.md").write_text("# Context\nfirst fact\nsecond fact\n", encoding="utf-8")
    payload = {
        "version": 1,
        "corpora": [{"id": "docs", "mode": "outline", "files": ["source.md"]}],
        "queries": [
            {
                "id": "q1",
                "corpus": "docs",
                "language": "en",
                "query": "What happened?",
                "evidence": [{"path": "source.md", "anchor": "first fact"}],
            }
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_bundled_suite_has_balanced_valid_sources():
    suite = evaluation.load_suite(evaluation.DEFAULT_SUITE, evaluation.ROOT)
    assert len(suite["queries"]) == 36
    for corpus in suite["corpora"]:
        assert sum(row["corpus"] == corpus for row in suite["queries"]) == 12
    assert any(len(row["spans"]) > 1 for row in suite["queries"])


@pytest.mark.parametrize(
    "invalid",
    [
        "../source.md",
        "a/../source.md",
        "C:/source.md",
        "a\\source.md",
        "/source.md",
        ".env",
        ".vexor/x.md",
    ],
)
def test_reject_unsafe_paths(tmp_path, invalid):
    with pytest.raises(ValueError):
        evaluation.source_path(tmp_path, invalid)


@pytest.mark.parametrize(
    "change",
    [
        "missing-anchor",
        "ambiguous-anchor",
        "duplicate-id",
        "outside-corpus",
        "empty-evidence",
        "unknown-corpus",
    ],
)
def test_invalid_judgments_fail_before_execution(tmp_path, change):
    path, payload = make_suite(tmp_path)
    query = payload["queries"][0]
    if change == "missing-anchor":
        query["evidence"][0]["anchor"] = "absent"
    elif change == "ambiguous-anchor":
        (tmp_path / "source.md").write_text("first fact first fact", encoding="utf-8")
    elif change == "duplicate-id":
        payload["queries"].append(query.copy())
    elif change == "outside-corpus":
        query["evidence"][0]["path"] = "other.md"
    elif change == "empty-evidence":
        query["evidence"] = []
    else:
        query["corpus"] = "unknown"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.load_suite(path, tmp_path)


def hit(body, line=2, **kwargs):
    return {"path": "source.md", "content": body, "content_start_line": line, **kwargs}


def score(tmp_path, results, second=False):
    path, payload = make_suite(tmp_path)
    if second:
        payload["queries"][0]["evidence"].append({"path": "source.md", "anchor": "second fact"})
        path.write_text(json.dumps(payload), encoding="utf-8")
    suite = evaluation.load_suite(path, tmp_path)
    return evaluation.score_results(suite["queries"][0], results, suite["corpora"]["docs"]["texts"])


def test_correct_file_with_truncated_evidence_is_a_miss(tmp_path):
    row = score(tmp_path, [hit("first", content_truncated=True)])
    assert row["file_rank"] == 1
    assert row["evidence_rank"] is None
    assert row["evidence_recall"] == 0
    assert row["returned_chars"] == 5
    assert row["truncated_count"] == 1


def test_multiple_evidence_and_overlap_are_not_double_counted(tmp_path):
    row = score(tmp_path, [hit("first fact"), hit("first fact\nsecond fact")], second=True)
    assert row["evidence_rank"] == 1
    assert row["all_evidence_rank"] == 2
    assert row["evidence_recall"] == 1
    assert row["overlap_chars"] == len("first fact")
    assert row["overlap_fraction"] == pytest.approx(10 / 32)


def test_missing_content_and_empty_results_are_not_dropped(tmp_path):
    row = score(tmp_path, [hit(None)])
    assert row["unavailable_count"] == 1
    assert row["evidence_rank"] is None
    empty = score(tmp_path, [])
    assert empty["evidence_recall"] == 0
    assert empty["overlap_fraction"] == 0


def test_content_must_match_snapshot_and_have_source_location(tmp_path):
    with pytest.raises(ValueError, match="snapshot"):
        score(tmp_path, [hit("invented fact")])
    with pytest.raises(ValueError, match="source line"):
        score(tmp_path, [hit("first fact", line=None)])


def test_metric_denominators_include_misses_and_small_top(tmp_path):
    correct = {**score(tmp_path, [hit("first fact")]), "id": "q1", "latency_ms": [10]}
    missing = {**score(tmp_path, []), "id": "q2", "latency_ms": [20]}
    summary = evaluation.summarize([correct, missing], 3)
    assert summary["query_count"] == 2
    assert summary["evidence_mrr@3"] == 0.5
    assert summary["all_evidence_hit@3"] == 0.5
    assert "evidence_hit@5" not in summary
    assert summary["warm_latency_p95_ms"] == 20


def test_validate_does_not_load_config_or_call_provider(tmp_path, monkeypatch, capsys):
    path, _ = make_suite(tmp_path)

    def forbidden():
        pytest.fail("Validation must not load provider configuration")

    monkeypatch.setattr(evaluation, "load_config", forbidden)
    assert (
        evaluation.main(["--suite", str(path), "--source-root", str(tmp_path), "--validate"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["queries"] == 1


def test_report_config_excludes_credentials_and_endpoints():
    cfg = Config(
        api_key="private-key",
        base_url="https://private.test",
        remote_rerank=RemoteRerankConfig(
            api_key="private-rerank", base_url="https://private-rerank.test"
        ),
    )
    assert "private" not in json.dumps(evaluation.public_config(cfg))


def test_runner_isolates_sources_warms_every_arm_and_records_repeats(tmp_path, monkeypatch):
    path, _ = make_suite(tmp_path)
    suite = evaluation.load_suite(path, tmp_path)
    roots = []
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"use_config": False}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def index(self, **kwargs):
            root = kwargs["path"]
            roots.append(root)
            assert (root / ".vexor").is_dir()
            assert sorted(p.name for p in root.iterdir()) == [".vexor", "source.md"]

        def search(self, query, **kwargs):
            calls.append(kwargs["config"].rerank)
            assert query == "What happened?"
            assert kwargs["auto_index"] is False
            assert kwargs["include_content"] is True
            return SimpleNamespace(
                is_stale=False,
                index_empty=False,
                results=[
                    SearchResult(
                        path=kwargs["path"] / "source.md",
                        score=0.9,
                        content="first fact",
                        content_start_line=2,
                        content_end_line=2,
                    )
                ],
            )

    monkeypatch.setattr(evaluation, "VexorClient", FakeClient)
    args = argparse.Namespace(
        arms=["off", "hybrid"],
        top=3,
        repeats=2,
        content_chars_per_result=100,
        content_chars_total=200,
    )
    report = evaluation.run_suite(suite, Config(), args)
    assert calls == ["off"] * 3 + ["hybrid"] * 3
    assert all(not root.exists() for root in roots)
    assert str(tmp_path) not in json.dumps(report)
    for arm in args.arms:
        assert report["arms"][arm]["overall"]["query_count"] == 1
        assert report["arms"][arm]["overall"]["sample_count"] == 2
        assert report["arms"][arm]["overall"]["evidence_mrr@3"] == 1


def test_provider_error_propagates_without_success_report(tmp_path, monkeypatch):
    path, _ = make_suite(tmp_path)
    output = tmp_path / "result.json"

    def fail(*_args):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(evaluation, "load_config", Config)
    monkeypatch.setattr(evaluation, "run_suite", fail)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        evaluation.main(
            ["--suite", str(path), "--source-root", str(tmp_path), "--output", str(output)]
        )
    assert not output.exists()


def test_existing_report_is_not_overwritten(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("previous", encoding="utf-8")
    with pytest.raises(SystemExit):
        evaluation.main(["--output", str(output)])
    assert output.read_text(encoding="utf-8") == "previous"


@pytest.mark.parametrize("arm", ["flashrank", "remote"])
def test_optional_reranker_failure_is_not_reported_as_dense(tmp_path, monkeypatch, arm):
    path, _ = make_suite(tmp_path)
    suite = evaluation.load_suite(path, tmp_path)

    class FailedRanker:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def index(self, **_kwargs):
            pass

        def search(self, _query, **kwargs):
            assert kwargs["config"].rerank == arm
            raise RuntimeError("reranker failed")

    monkeypatch.setattr(evaluation, "VexorClient", FailedRanker)
    args = argparse.Namespace(
        arms=[arm], top=3, repeats=1, content_chars_per_result=100, content_chars_total=200
    )
    with pytest.raises(RuntimeError, match="reranker failed"):
        evaluation.run_suite(suite, Config(), args)


@pytest.mark.parametrize("state", ["stale", "empty", "outside", "nonfinite"])
def test_invalid_search_response_fails(tmp_path, state):
    root = tmp_path / "corpus"
    root.mkdir()
    response = SimpleNamespace(
        is_stale=state == "stale",
        index_empty=state == "empty",
        results=[
            SearchResult(
                path=(tmp_path if state == "outside" else root) / "source.md",
                score=float("nan") if state == "nonfinite" else 0.5,
            )
        ],
    )
    with pytest.raises(ValueError):
        evaluation.serialize_results(response, root, 3)


def test_provider_override_does_not_reuse_previous_endpoint_or_key(tmp_path, monkeypatch):
    path, _ = make_suite(tmp_path)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        evaluation,
        "load_config",
        lambda: Config(
            provider="custom",
            model="old-model",
            api_key="old-key",
            base_url="https://private.test",
            embedding_dimensions=512,
        ),
    )

    def capture(_suite, config, _args):
        assert config.provider == "local"
        assert config.model == "intfloat/multilingual-e5-small"
        assert config.base_url is None
        assert config.api_key is None
        assert config.embedding_dimensions is None
        return {"ok": True}

    monkeypatch.setattr(evaluation, "run_suite", capture)
    assert (
        evaluation.main(
            [
                "--suite",
                str(path),
                "--source-root",
                str(tmp_path),
                "--provider",
                "local",
                "--model",
                "intfloat/multilingual-e5-small",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
