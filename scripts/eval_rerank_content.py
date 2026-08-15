"""Measure whether the rerankers do better on chunk source text than on previews.

Every reranker scores a document built from the filename, the path, and the stored
160-character preview. Chunk text became readable at search time in 0.27.0, so the
roadmap asked whether feeding that text to the rerankers instead lifts ranking
quality; result ordering shifts either way, so the question needed a measurement.

Each arm reruns the same query set with a different per-document character cap.
``--chars 0`` is the shipped preview behavior; any other value patches the reranker's
document builder to read the chunk back out of the file, the way ``search --content``
does, and score that instead.

Recorded result on this repository (30 queries, top 10, MRR@10):

    | Rerank    | Doc chars | bge-m3 | e5-small |
    |-----------|-----------|--------|----------|
    | off       | -         | 0.628  | 0.611    |
    | remote    | preview   | 0.686  | 0.634    |
    | remote    | 1000      | 0.591  | 0.550    |
    | flashrank | preview   | 0.669  | 0.623    |
    | flashrank | 1000      | 0.611  | 0.563    |
    | bm25      | preview   | 0.605  | 0.509    |
    | bm25      | 1000      | 0.520  | 0.514    |

No cap between 300 and 2000 characters beat the preview on any reranker, so
the change was not shipped. See docs/roadmap.md.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from _eval_common import DEFAULT_QUERIES, load_queries, metrics, relative_result_paths

from vexor import api
from vexor.config import DEFAULT_LOCAL_MODEL, load_config
from vexor.services import search_service
from vexor.services.content_extract_service import read_chunk_content


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("."), help="Repository root")
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument(
        "--rerank",
        nargs="+",
        default=["bm25"],
        choices=["bm25", "flashrank", "remote"],
        help="Rerankers to evaluate",
    )
    parser.add_argument(
        "--chars",
        nargs="+",
        type=int,
        default=[0, 1000],
        help="Per-document chunk text caps to compare (0 = stored preview)",
    )
    parser.add_argument(
        "--unverified",
        action="store_true",
        help="Skip the preview check on re-read text, so no candidate falls back",
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _chunk_text(result, *, chars: int, verify: bool) -> str | None:
    if result.start_line is None or result.end_line is None:
        return None
    try:
        chunk = read_chunk_content(
            result.path,
            result.start_line,
            result.end_line,
            max_chars=chars,
            anchor=search_service._chunk_window_anchor(result.preview),
        )
    except OSError:
        return None
    if chunk is None:
        return None
    if verify and not search_service._content_matches_preview(chunk.text, result.preview):
        return None
    return chunk.text


def _content_document(result, content: str) -> str:
    """The shipped document shape, with the chunk body in place of the preview.

    The preview is a 160-character prefix of the same text, so only what it adds is
    kept: ``code`` and ``outline`` previews carry a ``display :: snippet`` prefix
    naming the symbol or heading, which the file text itself does not repeat.
    """

    label = ""
    if result.preview:
        head, separator, _ = result.preview.rpartition(" :: ")
        label = head.strip() if separator else ""
    identity = f"{result.path.name} {result.path.as_posix()} {label}".strip()
    return f"{identity}\n{content}"


class _Meter:
    """Count the characters each arm hands to its reranker."""

    def __init__(self) -> None:
        self.documents = 0
        self.characters = 0

    def record(self, documents: Sequence[str]) -> Sequence[str]:
        self.documents += len(documents)
        self.characters += sum(len(document) for document in documents)
        return documents

    @property
    def chars_per_document(self) -> float:
        return self.characters / max(self.documents, 1)


@contextmanager
def _rerank_documents(*, chars: int, verify: bool) -> Iterator[_Meter]:
    """Swap in chunk-text documents for the duration of one arm."""

    meter = _Meter()
    shipped = search_service._build_rerank_documents

    def patched(results):
        if chars <= 0:
            return meter.record(shipped(results))
        documents = []
        for result in results:
            content = _chunk_text(result, chars=chars, verify=verify)
            documents.append(
                _content_document(result, content)
                if content
                else search_service._build_rerank_document(result)
            )
        return meter.record(documents)

    search_service._build_rerank_documents = patched
    try:
        yield meter
    finally:
        search_service._build_rerank_documents = shipped


def _run_arm(
    queries: Sequence[dict[str, str]],
    *,
    rerank: str,
    chars: int,
    verify: bool,
    top: int,
    common: dict[str, object],
    root: Path,
) -> dict[str, object]:
    details: list[dict[str, object]] = []
    started = time.perf_counter()
    with _rerank_documents(chars=chars, verify=verify) as meter:
        for item in queries:
            response = api.search(
                item["query"],
                top=top,
                auto_index=False,
                config={"rerank": rerank},
                **common,
            )
            returned = relative_result_paths(response, root)
            try:
                rank: int | None = returned.index(item["expected"]) + 1
            except ValueError:
                rank = None
            details.append(
                {
                    "query": item["query"],
                    "expected": item["expected"],
                    "rank": rank,
                    "results": returned,
                }
            )
    elapsed = time.perf_counter() - started
    return {
        "rerank": rerank,
        "chars": chars,
        "metrics": metrics(details),
        "seconds_per_query": elapsed / max(len(queries), 1),
        "chars_per_document": meter.chars_per_document,
        "queries": details,
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
    queries = load_queries(args.queries)

    arms = [
        _run_arm(
            queries,
            rerank="off",
            chars=0,
            verify=True,
            top=args.top,
            common=common,
            root=root,
        )
    ]
    for rerank in args.rerank:
        for chars in args.chars:
            arms.append(
                _run_arm(
                    queries,
                    rerank=rerank,
                    chars=chars,
                    verify=not args.unverified,
                    top=args.top,
                    common=common,
                    root=root,
                )
            )

    output = {
        "query_count": len(queries),
        "top": args.top,
        "provider": provider,
        "model": model,
        "verified": not args.unverified,
        "arms": arms,
    }
    if args.as_json:
        print(json.dumps(output, indent=2))
        return 0

    print(f"provider={provider} model={model} queries={len(queries)} top={args.top}")
    print("| Rerank | Doc chars | MRR@10 | Hit@1 | Hit@5 | Avg doc | s/query |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        summary = arm["metrics"]
        if arm["rerank"] == "off":
            label, cap = "off (dense)", "-"
        else:
            label = arm["rerank"]
            cap = "preview" if not arm["chars"] else str(arm["chars"])
        print(
            f"| {label} | {cap} | {summary['mrr_at_10']:.3f} | "
            f"{summary['hit_at_1']:.3f} | {summary['hit_at_5']:.3f} | "
            f"{arm['chars_per_document']:.0f} | {arm['seconds_per_query']:.2f} |"
        )
    if args.verbose:
        for index, item in enumerate(queries):
            ranks = []
            for arm in arms:
                cap = "preview" if not arm["chars"] else str(arm["chars"])
                name = "off" if arm["rerank"] == "off" else f"{arm['rerank']}/{cap}"
                ranks.append(f"{name}={arm['queries'][index]['rank'] or '-'}")
            print(f"- {item['query']} -> {item['expected']} ({', '.join(ranks)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
