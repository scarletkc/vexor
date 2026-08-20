"""Shared helpers for the retrieval evaluation scripts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

DEFAULT_QUERIES = Path(__file__).with_name("eval_queries.jsonl")


def load_queries(path: Path) -> list[dict[str, str]]:
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


def relative_result_paths(response, root: Path) -> list[str]:
    paths: list[str] = []
    for result in response.results:
        try:
            relative = result.path.resolve().relative_to(root)
        except ValueError:
            relative = result.path
        paths.append(relative.as_posix())
    return paths


def metrics(details: Sequence[dict[str, object]]) -> dict[str, float]:
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
