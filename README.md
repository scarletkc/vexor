<div align="center">

<img src="https://raw.githubusercontent.com/scarletkc/vexor/refs/heads/main/assets/vexor.svg" alt="Vexor" width="50%" height="auto">

# Vexor

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/vexor.svg)](https://pypi.org/project/vexor/)
[![CI](https://img.shields.io/github/actions/workflow/status/scarletkc/vexor/publish.yml?branch=main)](https://github.com/scarletkc/vexor/actions/workflows/publish.yml)
[![Codecov](https://img.shields.io/codecov/c/github/scarletkc/vexor/main)](https://codecov.io/github/scarletkc/vexor)
[![License](https://img.shields.io/github/license/scarletkc/vexor.svg)](https://github.com/scarletkc/vexor/blob/main/LICENSE)

</div>

---

Vexor is a vector-powered CLI that searches files semantically. It uses Google GenAI's `gemini-embedding-001` model to embed files and queries, then shows matches with cosine similarity.

## Install
Download from [releases](https://github.com/scarletkc/vexor/releases) without python, or with:
```bash
pip install vexor # or use pipx, uv
```
The CLI entry point is `vexor`.

## Configure
Set the Gemini API key once and reuse it everywhere:
```bash
vexor config --set-api-key "YOUR_KEY"
```
Optional defaults:
```bash
vexor config --set-model gemini-embedding-001
vexor config --set-batch-size 0   # 0 = single request
```
Configuration is stored in `~/.vexor/config.json`.

Inspect or reset every cached index:
```bash
vexor config --show-index-all
vexor config --clear-index-all
```

## Workflow
1. **Index** the project root (includes every subdirectory):
   ```bash
   vexor index --path ~/projects/demo --mode name --include-hidden
   ```
2. **Search** from anywhere, pointing to the same path:
   ```bash
   vexor search "api client config" --path ~/projects/demo --mode name --top 5
   ```
   Output example:
   ```
   Vexor semantic file search results
   ──────────────────────────────────
   #   Similarity   File path                      Preview
   1   0.923        ./src/config_loader.py        config loader entrypoint
   2   0.871        ./src/utils/config_parse.py   parse config helpers
   3   0.809        ./tests/test_config_loader.py tests for config loader
   ```

Tips:
- Keep one index per project root; subdirectories need separate indexes only if you explicitly run `vexor index` on them.
- Toggle `--no-recursive` (or `-n`) on both `index` and `search` when you only care about the current directory; recursive and non-recursive caches are stored separately.
- Hidden files are included only if both `index` and `search` use `--include-hidden`.
- Re-running `vexor index` only re-embeds files whose names changed (or were added/removed); if more than half the files differ, it automatically falls back to a full rebuild for consistency.
- Specify the indexing mode with `--mode`; currently `name` (file names only) and `head` (first chunk of supported text/code/PDF/DOCX/etc. files) are available, each with its own cache.

## Commands
| Command | Description |
| ------- | ----------- |
| `vexor index --path PATH --mode MODE [--include-hidden] [--no-recursive] [--clear/--show]` | Scans `PATH` (recursively by default), embeds content according to `MODE` (`name` or `head`), and writes a cache under `~/.vexor`. |
| `vexor search QUERY --path PATH --mode MODE [--top K] [--include-hidden] [--no-recursive]` | Loads the cached embeddings for `PATH` (matching the chosen mode/recursion/hidden settings), shows matches for `QUERY`. |
| `vexor doctor` | Checks whether the `vexor` command is available on the current `PATH`. |
| `vexor update` | Fetches the latest release version and shows links to update via GitHub or PyPI. |
| `vexor config --set-api-key/--clear-api-key` | Manage the stored Gemini API key. |
| `vexor config --set-model/--set-batch-size/--show` | Manage default model and batch size. |
| `vexor config --show-index-all/--clear-index-all` | Inspect or delete every cached index regardless of path/mode. |
