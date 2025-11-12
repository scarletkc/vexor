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

Vexor is a vector-powered CLI for semantic file search. It supports configurable remote embedding models and ranks results by cosine similarity.

## Usage
Vexor is designed to work seamlessly with both humans and AI coding assistants through the terminal, enabling semantic file search in autonomous agent workflows.

When you remember what a file does but forget its name or location, Vexor's semantic search finds it instantly—no grep patterns or directory traversal needed.

## Install
Download from [releases](https://github.com/scarletkc/vexor/releases) without python, or with:
```bash
pip install vexor # or use pipx, uv
```
The CLI entry point is `vexor`.

## Configure
Set the API key once and reuse it everywhere:
```bash
vexor config --set-api-key "YOUR_KEY"
```
Optional defaults:
```bash
vexor config --set-model gemini-embedding-001
vexor config --set-batch-size 0   # 0 = single request
vexor config --set-provider gemini
vexor config --set-base-url https://proxy.example.com  # optional proxy support for local LM Studio and similar tools; use --clear-base-url to reset
```
Provider defaults to `gemini`, so you only need to override it when switching to other backends (e.g., `openai`). Base URLs are optional and let you route requests through a custom proxy; run `vexor config --clear-base-url` to return to the official endpoint.

Environment/API keys can be supplied via `vexor config --set-api-key`, `VEXOR_API_KEY`, or provider-specific variables (`GOOGLE_GENAI_API_KEY`, `OPENAI_API_KEY`). Example OpenAI setup:
```bash
vexor config --set-provider openai
vexor config --set-model text-embedding-3-small
export OPENAI_API_KEY="sk-..."   # or use vexor config --set-api-key
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

**Tips:**
- Keep one index per project root; subdirectories need separate indexes only if you explicitly run `vexor index` on them.
- Toggle `--no-recursive` (or `-n`) on both `index` and `search` when you only care about the current directory; recursive and non-recursive caches are stored separately.
- Hidden files are included only if both `index` and `search` use `--include-hidden` (or `-i`).
- Use `--ext`/`-e` (repeatable) on both `index` and `search` to limit indexing and search results to specific extensions, e.g. `--ext .py --ext .md`.
- Re-running `vexor index` only re-embeds files whose names/contents changed (or were added/removed); if more than half the files differ, it automatically falls back to a full rebuild for consistency.
- Specify the indexing mode with `--mode`:
  - `name`: embed only the file name (fastest, zero content reads).
  - `head`: grab the first snippet of supported text/code/PDF/DOCX/PPTX files for lightweight semantic context.
  - `brief`: summarize PRDs/high-frequency keywords (English/Chinese) in requirements documents enable quick location of key requirements.
  - `full`: chunk the entire contents of supported text/code/PDF/DOCX/PPTX files into windows so long documents stay searchable end-to-end.
- Switch embedding providers (Gemini by default, OpenAI format supported) via `vexor config --set-provider PROVIDER` and pick a matching embedding model.

## Commands
| Command | Description |
| ------- | ----------- |
| `vexor index --path PATH --mode MODE [--include-hidden] [--no-recursive] [--ext EXT ...] [--clear/--show]` | Scans `PATH` (recursively by default), embeds content according to `MODE` (`name`, `head`, or `full`), and writes a cache under `~/.vexor`. |
| `vexor search QUERY --path PATH --mode MODE [--top K] [--include-hidden] [--no-recursive] [--ext EXT ...]` | Loads the cached embeddings for `PATH` (matching the chosen mode/recursion/hidden settings), shows matches for `QUERY`. |
| `vexor doctor` | Checks whether the `vexor` command is available on the current `PATH`. |
| `vexor update` | Fetches the latest release version and shows links to update via GitHub or PyPI. |
| `vexor config --set-api-key/--clear-api-key` | Manage the stored API key (Gemini by default). |
| `vexor config --set-model/--set-batch-size/--show` | Manage default model, batch size, and inspect current settings. |
| `vexor config --set-provider/--set-base-url/--clear-base-url` | Switch embedding providers and optionally override the remote base URL. |
| `vexor config --show-index-all/--clear-index-all` | Inspect or delete every cached index regardless of path/mode. |

## Documentation
See the [docs](https://github.com/scarletkc/vexor/tree/main/docs) for more details.

Contributions, issues, and PRs are all welcome!

Star this repo if you find it helpful!

## License
This project is licensed under the [MIT](http://github.com/scarletkc/vexor/blob/main/LICENSE) License.
