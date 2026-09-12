"""Run an isolated, evidence-annotated retrieval benchmark (see docs/evaluation.md)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from vexor import VexorClient, __version__
from vexor.config import (
    DEFAULT_FLASHRANK_MODEL,
    SUPPORTED_RERANKERS,
    Config,
    load_config,
)
from vexor.providers.capabilities import resolve_default_model
from vexor.services.search_service import (
    DEFAULT_CONTENT_CHARS_PER_RESULT,
    DEFAULT_CONTENT_CHARS_TOTAL,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "benchmarks" / "retrieval" / "suite.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_path(root: Path, value: object) -> Path:
    """Reject traversal, platform-specific paths, and sources outside the corpus root."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid corpus path: {value!r}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part.startswith(".") for part in relative.parts)
        or relative.as_posix() != value
        or relative.suffix not in {".py", ".md", ".txt"}
    ):
        raise ValueError(f"Invalid corpus path: {value!r}")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping corpus source: {value}")
    return path


def nonempty_list(value: object, label: str) -> list:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    return value


def identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def load_suite(path: Path, source_root: Path) -> dict[str, Any]:
    """Resolve unique exact anchors before indexing or making provider calls."""
    raw = path.read_bytes()
    suite = json.loads(raw)
    if not isinstance(suite, dict) or suite.get("version") != 1:
        raise ValueError("Expected a version 1 retrieval suite")
    corpora: dict[str, dict] = {}
    for corpus in nonempty_list(suite.get("corpora"), "corpora"):
        if not isinstance(corpus, dict):
            raise ValueError("Each corpus must be an object")
        name = identifier(corpus.get("id"))
        if name in corpora or corpus.get("mode") not in {"code", "outline", "full", "auto"}:
            raise ValueError(f"Duplicate corpus or invalid mode: {name}")
        texts: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for filename in nonempty_list(corpus.get("files"), f"{name}.files"):
            source = source_path(source_root, filename)
            if filename in texts:
                raise ValueError(f"Duplicate corpus file: {filename}")
            text = source.read_text(encoding="utf-8")
            texts[filename] = text
            hashes[filename] = digest(text.encode("utf-8"))
        corpora[name] = {**corpus, "texts": texts, "sha256": hashes}
    queries: list[dict] = []
    query_ids: set[str] = set()
    for query in nonempty_list(suite.get("queries"), "queries"):
        if not isinstance(query, dict):
            raise ValueError("Each query must be an object")
        name = identifier(query.get("id"))
        if name in query_ids or query.get("corpus") not in corpora:
            raise ValueError(f"Duplicate query or unknown corpus: {name}")
        query_ids.add(name)
        if any(
            not isinstance(query.get(key), str) or not query[key].strip()
            for key in ("query", "language")
        ):
            raise ValueError(f"Missing query text or language: {name}")
        evidence: list[dict] = []
        texts = corpora[query["corpus"]]["texts"]
        for item in nonempty_list(query.get("evidence"), f"{name}.evidence"):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid evidence: {name}")
            filename, anchor = item.get("path"), item.get("anchor")
            if not isinstance(filename, str) or filename not in texts:
                raise ValueError(f"Evidence outside corpus: {name}")
            if not isinstance(anchor, str) or not anchor.strip():
                raise ValueError(f"Empty evidence anchor: {name}")
            text = texts[filename]
            start = text.find(anchor)
            if start < 0 or text.find(anchor, start + 1) >= 0:
                raise ValueError(f"Evidence anchor must occur exactly once: {name}: {filename}")
            span = {"path": filename, "start": start, "end": start + len(anchor)}
            if span in evidence:
                raise ValueError(f"Duplicate evidence: {name}")
            evidence.append(span)
        queries.append({**query, "spans": evidence})
    if set(corpora) != {query["corpus"] for query in queries}:
        raise ValueError("Every corpus must have queries")
    return {"corpora": corpora, "queries": queries, "suite_sha256": digest(raw)}


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def score_results(query: dict, results: list[dict], texts: dict[str, str]) -> dict:
    """Score only source characters actually returned, including content truncation."""
    covered: dict[str, list[tuple[int, int]]] = defaultdict(list)
    first_rank = None
    all_rank = None
    total_chars = 0
    evidence = query["spans"]
    for rank, result in enumerate(results, 1):
        filename = result["path"]
        if filename not in texts:
            raise ValueError(f"Search returned a file outside the corpus: {filename}")
        body = result.get("content") or ""
        total_chars += len(body)
        if body:
            line = result.get("content_start_line")
            if type(line) is not int or line < 1:
                raise ValueError("Returned content has no valid source line")
            text = texts[filename]
            lines = text.splitlines(keepends=True)
            if line > len(lines):
                raise ValueError("Returned content starts beyond the source")
            start = sum(map(len, lines[: line - 1]))
            if not text[start:].startswith(body):
                raise ValueError("Returned content does not match the corpus snapshot")
            covered[filename] = merge_intervals([*covered[filename], (start, start + len(body))])
        matched = sum(
            any(
                start <= span["start"] and end >= span["end"]
                for start, end in covered[span["path"]]
            )
            for span in evidence
        )
        if matched and first_rank is None:
            first_rank = rank
        if matched == len(evidence) and all_rank is None:
            all_rank = rank
    matched = sum(
        any(start <= span["start"] and end >= span["end"] for start, end in covered[span["path"]])
        for span in evidence
    )
    unique_chars = sum(end - start for spans in covered.values() for start, end in spans)
    expected_paths = {span["path"] for span in evidence}
    file_rank = next(
        (i for i, result in enumerate(results, 1) if result["path"] in expected_paths), None
    )
    return {
        "file_rank": file_rank,
        "evidence_rank": first_rank,
        "all_evidence_rank": all_rank,
        "evidence_recall": matched / len(evidence),
        "returned_chars": total_chars,
        "overlap_chars": total_chars - unique_chars,
        "overlap_fraction": (total_chars - unique_chars) / total_chars if total_chars else 0.0,
        "result_count": len(results),
        "unavailable_count": sum(not result.get("content") for result in results),
        "truncated_count": sum(bool(result.get("content_truncated")) for result in results),
    }


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(details: list[dict], top: int) -> dict:
    summary = {
        "query_count": len({row["id"] for row in details}),
        "sample_count": len(details),
        f"file_mrr@{top}": mean(1 / row["file_rank"] if row["file_rank"] else 0 for row in details),
        f"evidence_mrr@{top}": mean(
            1 / row["evidence_rank"] if row["evidence_rank"] else 0 for row in details
        ),
        f"evidence_recall@{top}": mean(row["evidence_recall"] for row in details),
        f"all_evidence_hit@{top}": mean(row["all_evidence_rank"] is not None for row in details),
        "mean_returned_chars": mean(row["returned_chars"] for row in details),
        "mean_overlap_fraction": mean(row["overlap_fraction"] for row in details),
        "content_unavailable_count": sum(row["unavailable_count"] for row in details),
        "content_truncated_count": sum(row["truncated_count"] for row in details),
    }
    for cutoff in sorted({1, min(5, top), top}):
        summary[f"evidence_hit@{cutoff}"] = mean(
            row["evidence_rank"] is not None and row["evidence_rank"] <= cutoff for row in details
        )
    timings = [sample for row in details for sample in row["latency_ms"]]
    summary["warm_latency_p50_ms"] = percentile(timings, 0.5)
    summary["warm_latency_p95_ms"] = percentile(timings, 0.95)
    return summary


def serialize_results(response: Any, root: Path, top: int) -> list[dict]:
    if response.is_stale or response.index_empty:
        raise ValueError("Benchmark search returned a stale or empty index")
    if len(response.results) > top:
        raise ValueError("Search returned more results than requested")
    results = []
    for hit in response.results:
        score = float(hit.score)
        if not math.isfinite(score):
            raise ValueError("Search returned a non-finite score")
        results.append(
            {
                "path": hit.path.resolve().relative_to(root.resolve()).as_posix(),
                "score": score,
                "start_line": hit.start_line,
                "end_line": hit.end_line,
                "content": hit.content,
                "content_start_line": hit.content_start_line,
                "content_end_line": hit.content_end_line,
                "content_truncated": hit.content_truncated,
                "content_unavailable": hit.content_unavailable,
            }
        )
    return results


def public_config(config: Config) -> dict:
    """Use an allowlist: reports must not copy keys or private endpoints."""
    return {
        key: getattr(config, key)
        for key in (
            "provider",
            "model",
            "embedding_dimensions",
            "batch_size",
            "embed_concurrency",
            "extract_concurrency",
            "extract_backend",
            "local_cuda",
            "flashrank_model",
        )
    } | {"remote_rerank_model": config.remote_rerank.model if config.remote_rerank else None}


def run_suite(suite: dict, config: Config, args: argparse.Namespace) -> dict:
    by_arm: dict[str, list[dict]] = {arm: [] for arm in args.arms}
    index_ms: dict[str, float] = {}
    with TemporaryDirectory(prefix="vexor-retrieval-") as temporary:
        workspace = Path(temporary)
        for name, corpus in suite["corpora"].items():
            root = workspace / name
            root.mkdir()
            # A local marker isolates this run from global and ancestor index caches.
            (root / ".vexor").mkdir()
            for filename, text in corpus["texts"].items():
                target = root / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8", newline="\n")
            print(f"Indexing {name} ({len(corpus['texts'])} files)", file=sys.stderr)
            common = {"path": root, "mode": corpus["mode"], "model": config.model}
            with VexorClient(use_config=False) as client:
                started = perf_counter()
                client.index(config=replace(config, rerank="off"), **common)
                index_ms[name] = (perf_counter() - started) * 1000
                queries = [row for row in suite["queries"] if row["corpus"] == name]
                for number, query in enumerate(queries):
                    # Rotate order to avoid consistently giving one arm the first position.
                    offset = number % len(args.arms)
                    for arm in args.arms[offset:] + args.arms[:offset]:
                        search_args = dict(
                            **common,
                            config=replace(config, rerank=arm),
                            top=args.top,
                            auto_index=False,
                            include_content=True,
                            content_chars_per_result=args.content_chars_per_result,
                            content_chars_total=args.content_chars_total,
                        )
                        # Warm every query/arm explicitly; these timings are not cold-start costs.
                        warmup = client.search(query["query"], **search_args)
                        serialize_results(warmup, root, args.top)
                        samples: list[float] = []
                        for repetition in range(args.repeats):
                            started = perf_counter()
                            response = client.search(query["query"], **search_args)
                            samples.append((perf_counter() - started) * 1000)
                            results = serialize_results(response, root, args.top)
                            detail = score_results(query, results, corpus["texts"])
                            by_arm[arm].append(
                                {
                                    "id": query["id"],
                                    "corpus": name,
                                    "language": query["language"],
                                    "repetition": repetition + 1,
                                    "query": query["query"],
                                    "evidence": query["evidence"],
                                    **detail,
                                    "latency_ms": [samples[-1]],
                                    "results": results,
                                }
                            )
                    print(f"  {name}: {number + 1}/{len(queries)}", file=sys.stderr)
    packages = {}
    for package in ("numpy", "openai", "google-genai", "fastembed", "flashrank", "tokenizers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_sha256": suite["suite_sha256"],
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "runtime_sha256": digest(
            json.dumps(
                {
                    path.relative_to(ROOT).as_posix(): digest(path.read_bytes())
                    for path in sorted((ROOT / "vexor").rglob("*.py"))
                },
                sort_keys=True,
            ).encode("utf-8")
        ),
        "vexor_version": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": packages,
        "config": public_config(config),
        "top": args.top,
        "repeats": args.repeats,
        "content_chars_per_result": args.content_chars_per_result,
        "content_chars_total": args.content_chars_total,
        "timing_protocol": "one untimed warmup per query/arm; shared query cache; rotated arms",
        "index_ms": index_ms,
        "corpora": {
            name: {
                "mode": corpus["mode"],
                "sha256": corpus["sha256"],
                "provenance": corpus.get("provenance"),
            }
            for name, corpus in suite["corpora"].items()
        },
        "arms": {
            arm: {
                "overall": summarize(details, args.top),
                "by_corpus": {
                    name: summarize([row for row in details if row["corpus"] == name], args.top)
                    for name in suite["corpora"]
                },
                "observations": details,
            }
            for arm, details in by_arm.items()
        },
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--validate", action="store_true", help="Validate sources; no provider calls"
    )
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument(
        "--arms", nargs="+", choices=SUPPORTED_RERANKERS, default=["off", "bm25", "hybrid"]
    )
    parser.add_argument("--top", type=positive_int, default=10)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument(
        "--content-chars-per-result", type=positive_int, default=DEFAULT_CONTENT_CHARS_PER_RESULT
    )
    parser.add_argument(
        "--content-chars-total", type=positive_int, default=DEFAULT_CONTENT_CHARS_TOTAL
    )
    parser.add_argument("--output", type=Path, help="Write JSON to a new file; default: stdout")
    args = parser.parse_args(argv)
    if len(set(args.arms)) != len(args.arms):
        parser.error("Duplicate arms are not allowed")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new report path")
    suite = load_suite(args.suite, args.source_root)
    if args.validate:
        print(
            json.dumps(
                {
                    "corpora": len(suite["corpora"]),
                    "queries": len(suite["queries"]),
                    "suite_sha256": suite["suite_sha256"],
                }
            )
        )
        return 0
    config = load_config()
    if args.provider and args.provider != config.provider:
        # Do not carry a different provider's endpoint, dimensions, or key into the override.
        config = replace(
            config,
            provider=args.provider,
            model="",
            base_url=None,
            api_key=None,
            embedding_dimensions=None,
        )
    config = replace(
        config,
        model=resolve_default_model(config.provider, args.model or config.model),
        flashrank_model=config.flashrank_model or DEFAULT_FLASHRANK_MODEL,
    )
    report = run_suite(suite, config, args)
    payload = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        print(f"Report written: {args.output}", file=sys.stderr)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
