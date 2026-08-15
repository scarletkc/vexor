# Development
Run development tasks inside the project virtualenv, creating one only when the
repository has none:
```bash
python -m venv .venv          # only if .venv is missing
python -m pip install -e .[dev]
python -m vexor
python -m pytest
```
Tests rely on fake embedding backends, so no network access is required.

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
