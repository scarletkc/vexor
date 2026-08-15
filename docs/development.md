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
