# Development

Use uv to create and keep the repository environment in sync with the committed
lockfile:

```bash
uv sync
uv run vexor
uv run pytest
uv run ruff check .
```

Tests rely on fake embedding backends, so no network access is required after
the environment is synced.

Runtime copy lives in [`Messages`](../vexor/text.py).
[`test_text_contract.py`](../tests/unit/test_text_contract.py) checks exception messages,
CLI output and help, prompts, diagnostic labels, and MCP descriptions. It also scans
for prose assigned before output; this is a static guard, not a complete data-flow
analysis. SQL, protocol values, indexed source snippets, and external-error match
patterns remain runtime data rather than message templates.

Ruff is the lint gate, configured under `[tool.ruff]` in `pyproject.toml` and
run by the `ruff` job in `.github/workflows/publish.yml`, which a release now
depends on. Its version is pinned in the `dev` dependency group and captured in
`uv.lock`, so bumping the pin and refreshing the lockfile moves CI and local
development together. `uv run ruff check --fix .` applies the safe fixes; read
the diff before reaching for `--unsafe-fixes`.

Global cache files and configuration live in `~/.vexor`. A project with a
`.vexor/` directory keeps its index there and may add the restricted
`config.json` overlay documented in `docs/configuration.md`. The text embedded
for each chunk is built by the mode strategies in `vexor/modes.py`; adjust the
strategy `label` construction there if you need to encode additional context.

Index metadata, paths, BM25 postings, and query caches live in `index.db`.
Dense vectors use generation-specific `vectors/*.npy` sidecars so searches can
open them through NumPy mmap instead of reconstructing a matrix from SQLite
BLOB rows. `CACHE_VERSION` changes invalidate older layouts and trigger a
normal rebuild; do not add a silent compatibility path for corrupt sidecars.

Run the local performance harness without provider credentials:

```bash
uv run python scripts/benchmark_search_cache.py
uv run python scripts/benchmark_search_cache.py --vector-count 30000 --file-count 10000
```

The harness reports first and process-cached vector loads, full snapshot
validation, and event-backed freshness checks. Timing is diagnostic evidence,
not a fixed CI threshold.

Compare batch and repeated single-query retrieval with a real local model:

```bash
uv sync --extra local
uv run --extra local python scripts/benchmark_batch_search.py
```

The benchmark checks result equivalence on synthetic documents: files and filtered
collections use `off` and `hybrid`; in-memory search uses `hybrid`. It reports
embedding calls and elapsed time with cold query caches, excluding initial
indexing and model warmup. Per-search model initialization remains included.
Use `--model` to select a model; download it before setting `HF_HUB_OFFLINE=1`
for an offline run. Timing is diagnostic, not a CI or retrieval-quality gate.

Ranking changes are argued with numbers, not intuition. Two scripts score the
same 30-query set in `scripts/eval_queries.jsonl` against whatever provider the
config resolves to, and report MRR@10, Hit@1, and Hit@5:

```bash
uv run python scripts/eval_hybrid.py --path .            # dense vs BM25 rerank vs hybrid
uv run python scripts/eval_rerank_content.py --path . --rerank remote --chars 0 1000
```

`eval_rerank_content.py` compares what each reranker scores: the stored preview
against the chunk's source text, at a per-document character cap. Both scripts
index first, so point `--path` at the repository you want measured and expect
provider calls when the model is remote. Record the table in the PR, and keep
the query set fixed while comparing arms — 30 queries is small enough that one
rank change moves MRR@10 by about 0.03.

For retrieval comparisons scored against returned source evidence, see
[Retrieval evaluation](evaluation.md).

## Runtime boundaries

- Remote batch scheduling and retry classification live in
  [embed_batches](../vexor/providers/batching.py) and
  [should_retry_error](../vexor/providers/retry.py). Each provider adapter owns
  its SDK requests, response decoding, and exception handling.
- [rerank_candidates](../vexor/services/ranking_service.py) ranks prepared
  documents from file search and collections. Each source service owns its
  candidate loading, filtering, and hybrid corpus statistics.
- [_insert_indexed_chunks](../vexor/cache.py) writes chunk metadata and lexical
  postings within the full or incremental writer's transaction. Postings stream
  directly into SQLite: do not buffer all term tuples for a rebuild or update.
  `tests/unit/test_index_write_contract.py` enforces this by checking that SQLite
  stores each posting before the next one is produced.
- [search_response_payload](../vexor/services/result_serialization.py) owns
  shared search fields; CLI and MCP choose their transport-specific envelope
  and fields.

## Releases

Bump the version on a branch and land it through a PR; merging to `main`
publishes the release. `.github/workflows/publish.yml` builds the release body
from the commit subjects between the previous tag and the new one.

To put a hand-written section above that generated changelog, add
`docs/release-notes/<version>.md`. `--note` starts the file for you:

```bash
uv run python scripts/bump_version.py 0.28.0 --note "Reranker now reads chunk text"
```

The file must open with its own `## <title>` line so the section sits beside
`## Changelog`, and it must have a body — the publish job fails on a missing
heading or an empty note rather than shipping a malformed release. Leave the
file out entirely when a release needs no note. Because the note is committed
with the bump, a `force_release` re-run publishes the same text.
