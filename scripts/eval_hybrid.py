"""Evaluate dense, legacy BM25 reranking, and full-corpus hybrid retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from vexor import api
from vexor.config import DEFAULT_LOCAL_MODEL, load_config

ARMS = ("off", "bm25", "hybrid")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("."), help="Repository root")
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(__file__).with_name("eval_queries.jsonl"),
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_queries(path: Path) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("query"), str) or not isinstance(
                item.get("expected"), str
            ):
                raise ValueError(f"Invalid query record on line {line_number}")
            queries.append(item)
    return queries


def _relative_result_paths(response, root: Path) -> list[str]:
    paths: list[str] = []
    for result in response.results:
        try:
            relative = result.path.resolve().relative_to(root)
        except ValueError:
            relative = result.path
        paths.append(relative.as_posix())
    return paths


def _metrics(details: Sequence[dict[str, object]]) -> dict[str, float]:
    count = len(details)
    if not count:
        return {"mrr_at_10": 0.0, "hit_at_1": 0.0, "hit_at_5": 0.0}
    reciprocal_sum = 0.0
    hit_one = 0
    hit_five = 0
    for detail in details:
        rank = detail["rank"]
        if isinstance(rank, int) and rank <= 10:
            reciprocal_sum += 1.0 / rank
        hit_one += int(rank == 1)
        hit_five += int(isinstance(rank, int) and rank <= 5)
    return {
        "mrr_at_10": reciprocal_sum / count,
        "hit_at_1": hit_one / count,
        "hit_at_5": hit_five / count,
    }


def main() -> int:
    args = _parse_args()
    root = args.path.expanduser().resolve()
    config = load_config()
    provider = args.provider or config.provider
    model = args.model or config.model
    if provider == "local" and args.model is None:
        model = DEFAULT_LOCAL_MODEL
    common = {
        "path": root,
        "mode": args.mode,
        "provider": provider,
        "model": model,
        "use_config": True,
    }
    api.index(**common)

    queries = _load_queries(args.queries)
    by_arm: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
    for item in queries:
        for arm in ARMS:
            response = api.search(
                item["query"],
                top=args.top,
                auto_index=False,
                config={"rerank": arm},
                **common,
            )
            returned = _relative_result_paths(response, root)
            try:
                rank: int | None = returned.index(item["expected"]) + 1
            except ValueError:
                rank = None
            by_arm[arm].append(
                {
                    "query": item["query"],
                    "expected": item["expected"],
                    "rank": rank,
                    "results": returned,
                }
            )

    output = {
        "query_count": len(queries),
        "top": args.top,
        "arms": {
            arm: {"metrics": _metrics(details), "queries": details}
            for arm, details in by_arm.items()
        },
    }
    if args.as_json:
        print(json.dumps(output, indent=2))
        return 0

    print("| Rerank | MRR@10 | Hit@1 | Hit@5 |")
    print("|---|---:|---:|---:|")
    for arm in ARMS:
        metrics = output["arms"][arm]["metrics"]
        print(
            f"| {arm} | {metrics['mrr_at_10']:.3f} | "
            f"{metrics['hit_at_1']:.3f} | {metrics['hit_at_5']:.3f} |"
        )
    if args.verbose:
        for item in queries:
            ranks = []
            for arm in ARMS:
                detail = next(
                    row for row in by_arm[arm] if row["query"] == item["query"]
                )
                ranks.append(f"{arm}={detail['rank'] or '-'}")
            print(f"- {item['query']} -> {item['expected']} ({', '.join(ranks)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
