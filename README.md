# Vexor

Vexor is a vector-powered CLI that searches file names semantically. It uses Google GenAI's `gemini-embedding-001` model to embed file names and queries, then ranks matches with cosine similarity.

## Install
```bash
pip install -e .
```
The CLI entry point is `vexor` (or `python -m vexor`).

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

## Workflow
1. **Index** the project root (includes every subdirectory):
   ```bash
   vexor index --path ~/projects/demo --include-hidden
   ```
2. **Search** from anywhere, pointing to the same path:
   ```bash
   vexor search "api client config" --path ~/projects/demo --top 5
   ```
   Output example:
   ```
   Vexor semantic file search results
   ──────────────────────────────────
   1   0.923   ./src/config_loader.py
   2   0.871   ./src/utils/config_parse.py
   3   0.809   ./tests/test_config_loader.py
   ```

Tips:
- Keep one index per project root; subdirectories need separate indexes only if you explicitly run `vexor index` on them.
- Hidden files are included only if both `index` and `search` use `--include-hidden`.

## Commands
| Command | Description |
| ------- | ----------- |
| `vexor index --path PATH [--include-hidden] [--clear]` | Recursively scans `PATH`, embeds file names, and writes a cache under `~/.vexor`. |
| `vexor search QUERY --path PATH [--top K] [--include-hidden]` | Loads the cached embeddings for `PATH` and ranks matches for `QUERY`. |
| `vexor config --set-api-key/--clear-api-key` | Manage the stored Gemini API key. |
| `vexor config --set-model/--set-batch-size/--show` | Manage default model and batch size. |

## Development
Run tests with:
```bash
pip install -e .[dev]
pytest
```
Tests rely on fake embedding backends, so no network access is required.

Cache files and configuration live in `~/.vexor`. Adjust `_label_for_path` or `VexorSearcher._prepare_text` if you need to encode additional context (e.g., relative paths).
