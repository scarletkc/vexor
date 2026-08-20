# Development
Run development tasks inside the project virtualenv, creating one only when the
repository has none:
```bash
python -m venv .venv          # only if .venv is missing
python -m pip install -e .[dev]
python -m vexor
python -m pytest
python -m ruff check .
```
Tests rely on fake embedding backends, so no network access is required.

Ruff is the lint gate, configured under `[tool.ruff]` in `pyproject.toml` and
run by the `ruff` job in `.github/workflows/publish.yml`, which a release now
depends on. The version is pinned in the `dev` extra and CI installs that exact
pin, so bumping it there is what moves CI. `ruff check --fix` applies the safe
fixes; read the diff before reaching for `--unsafe-fixes`.

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
python scripts/benchmark_search_cache.py
python scripts/benchmark_search_cache.py --vector-count 30000 --file-count 10000
```

The harness reports first and process-cached vector loads, full snapshot
validation, and event-backed freshness checks. Timing is diagnostic evidence,
not a fixed CI threshold.

Ranking changes are argued with numbers, not intuition. Two scripts score the
same 30-query set in `scripts/eval_queries.jsonl` against whatever provider the
config resolves to, and report MRR@10, Hit@1, and Hit@5:

```bash
python scripts/eval_hybrid.py --path .            # dense vs BM25 rerank vs hybrid
python scripts/eval_rerank_content.py --path . --rerank remote --chars 0 1000
```

`eval_rerank_content.py` compares what each reranker scores: the stored preview
against the chunk's source text, at a per-document character cap. Both scripts
index first, so point `--path` at the repository you want measured and expect
provider calls when the model is remote. Record the table in the PR, and keep
the query set fixed while comparing arms — 30 queries is small enough that one
rank change moves MRR@10 by about 0.03.

## Releases

Bump the version on a branch and land it through a PR; merging to `main`
publishes the release. `.github/workflows/publish.yml` builds the release body
from the commit subjects between the previous tag and the new one.

To put a hand-written section above that generated changelog, add
`docs/release-notes/<version>.md`. `--note` starts the file for you:

```bash
python scripts/bump_version.py 0.28.0 --note "Reranker now reads chunk text"
```

The file must open with its own `## <title>` line so the section sits beside
`## Changelog`, and it must have a body — the publish job fails on a missing
heading or an empty note rather than shipping a malformed release. Leave the
file out entirely when a release needs no note. Because the note is committed
with the bump, a `force_release` re-run publishes the same text.
