# Retrieval evaluation

Use `scripts/eval_retrieval.py` to compare ranking changes against source evidence,
not just filenames. The runner copies an explicit corpus allowlist into temporary
directories, builds isolated file indexes, and checks the text actually returned
by the Python search API. Query text and evidence labels are never copied into
those directories. Existing project indexes are not modified.

The [bundled suite](../benchmarks/retrieval/suite.json) is a development seed with
authored queries across three corpus types:

| Corpus | Sources | Index mode | Query languages |
| --- | --- | --- | --- |
| `code` | Vexor runtime modules | `code` | English and Chinese |
| `docs` | Canonical Vexor documentation | `outline` | English and Chinese |
| `zh-records` | Authored Chinese paraphrases of cited Vexor sources | `outline` | Chinese |

The Chinese records are synthetic regression material. They are searched as files;
this runner does not evaluate the Collections API. Some queries require multiple
evidence spans. All queries have an answer in their corpus; abstention on
unanswerable queries is not measured. The labels are seed judgments and have not
been independently reviewed for all possible alternative answers.

See the [initial baseline results](evaluation-baseline.md) for measured outcomes
and their limitations.

## Run the baseline

Use the repository environment described in [Development](development.md).

```bash
# Validate all sources and exact evidence anchors without provider calls.
uv run python scripts/eval_retrieval.py --validate

# Use the effective global/environment provider configuration.
uv run python scripts/eval_retrieval.py --output .cache/evaluation/baseline.json

# Compare a second embedding model with the same suite and content budgets.
# This installs the local extra if needed; model assets must be available.
uv run --extra local python scripts/eval_retrieval.py --provider local \
  --model intfloat/multilingual-e5-small --output .cache/evaluation/local.json

# Optional rerankers require their normal dependencies/configuration.
uv run python scripts/eval_retrieval.py --arms off bm25 hybrid flashrank remote \
  --output .cache/evaluation/all-rerankers.json
```

Use `--help` for ranking arms, repetition counts, and result limits. Content
limits default to the Python API's character budgets; override them with
`--content-chars-per-result` and `--content-chars-total`.
Remote embeddings and remote reranking make provider
calls and may incur charges. A remote reranker is invoked for warmups too.

Configuration is loaded once from global settings and environment variables;
source-project overlays do not affect the copied corpora. `--provider` selects a
different provider without carrying over the previous provider's endpoint, key,
model, or vector dimension. Use provider environment variables for its credentials.
Other settings, including reranker configuration, remain in effect. `--model`
selects an explicit model. To override endpoints or dimensions, use the existing
Vexor configuration mechanisms.

Without `--output`, stdout contains only the JSON report; progress goes to stderr.
Output files are created only after a successful run and existing reports are not
overwritten. Invalid labels fail before indexing. Provider failures, stale or empty
indexes, unexpected source content, and invalid responses fail the run instead of
being converted into zero-scored queries. A valid search with no hits is a miss.

## Read the report

Each arm includes `overall`, `by_corpus`, and individual `observations`. Every
measured repetition records relative result paths, returned source text, evidence,
scores, truncation/unavailability flags, and search duration. Metrics average all
observations; each query has the same number of repetitions, so each query has
equal weight. `query_count` counts distinct queries and `sample_count` counts
measured observations. Inspect per-corpus results before using the overall mean.

Evidence anchors must occur exactly once in the specified source file. They are
resolved to character intervals before indexing. A hit only covers an anchor when
its actual returned content covers the complete interval. Adjacent or overlapping
returned fragments can jointly cover it. Finding the right file, mentioning the
same phrase elsewhere, or returning a line range whose body was truncated is not
enough. Newlines are normalized to LF in the corpus snapshots.

| Metric | Meaning |
| --- | --- |
| `file_mrr@k` | Reciprocal rank of the first result in any annotated file; misses contribute zero |
| `evidence_mrr@k` | Reciprocal rank of the earliest result prefix covering at least one complete evidence anchor |
| `evidence_hit@1`, `@5`, `@k` | Fraction covering any complete evidence within that prefix; cutoffs never exceed requested top k |
| `evidence_recall@k` | Mean fraction of the query's evidence anchors completely covered |
| `all_evidence_hit@k` | Fraction covering every annotated evidence anchor |
| `mean_returned_chars` | Mean number of returned source characters, including repeated content |
| `mean_overlap_fraction` | Mean fraction of returned characters repeating an already returned interval of the same source file |
| `content_unavailable_count` | Total results with no returned body across measured observations |
| `content_truncated_count` | Total results marked truncated across measured observations |
| `warm_latency_p50_ms`, `p95_ms` | Nearest-rank percentiles across measured searches |

Overlap measures repeated source intervals, not semantic duplication across files.
Returned characters are not tokens, billed usage, or the full serialized tool
response size. Missing content is counted and can cause evidence misses. Unannotated
results are not automatically classified as irrelevant: labels are not exhaustive,
so the report does not claim precision or an irrelevant-token fraction.

Every query/arm gets one untimed warmup. Arms rotate their order between queries;
measured repetitions use a shared query cache and a persistent client for each
corpus. Repeated outputs are recorded separately, including variation from remote
rerankers. These durations include content retrieval, but exclude report scoring,
indexing, and warmups. `index_ms` separately records index construction. This is
warm in-process search latency, not CLI startup, cold query latency, or an estimate
of an agent's total task time.

Reports contain suite, runner, runtime-source, and normalized corpus SHA-256
fingerprints, plus package versions, ranking model names, content budgets, and
safe configuration fields. Keys, endpoints, absolute source paths, and machine
names are excluded. Provider-side model revisions and unpublished backend changes
cannot be identified automatically: retain that context privately when comparing
remote runs. Source text and queries are present in the report, so review reports
before sharing when using your own corpus.

## Extend the suite

Pass `--suite PATH --source-root PATH` for a custom version 1 JSON suite. A corpus
declares a unique `id`, `mode`, and explicit relative `files`. Supported source
extensions are `.py`, `.md`, and `.txt`; traversal, hidden path components,
Windows drive paths, and sources resolving outside the source root are rejected.
A query declares `id`, `corpus`, `language`, `query`, and a non-empty `evidence`
list, with a `path` and unique exact `anchor` in each evidence object. See the
[bundled suite](../benchmarks/retrieval/suite.json) for complete examples.

Add real distractor documents as well as relevant sources. Annotate enough text
to establish the answer; a function name alone rarely does that. When a source
changes and an anchor stops resolving uniquely, review the judgment rather than
automatically moving it to any matching line. Keep the suite fixed across arms.
Version changed labels as a new comparison baseline, and retain the original JSON
reports. Do not tune ranking on a query set and present gains on that same set as
independent validation. Add held-out projects and human-reviewed labels before
using these results to change the default ranking or make public quality claims.

For task success, token usage, and end-to-end time, use the separate
[agent evaluation protocol](agent-evaluation.md).
