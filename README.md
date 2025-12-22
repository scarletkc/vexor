<div align="center">

<img src="https://raw.githubusercontent.com/scarletkc/vexor/refs/heads/main/assets/vexor.svg" alt="Vexor" width="35%" height="auto">

# Vexor

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/vexor.svg)](https://pypi.org/project/vexor/)
[![CI](https://img.shields.io/github/actions/workflow/status/scarletkc/vexor/publish.yml?branch=main)](https://github.com/scarletkc/vexor/actions/workflows/publish.yml)
[![Codecov](https://img.shields.io/codecov/c/github/scarletkc/vexor/main)](https://codecov.io/github/scarletkc/vexor)
[![License](https://img.shields.io/github/license/scarletkc/vexor.svg)](https://github.com/scarletkc/vexor/blob/main/LICENSE)

</div>

---

**Vexor** is a vector-powered CLI for semantic file search. It uses configurable embedding models and ranks results by cosine similarity.

<video src="https://github.com/user-attachments/assets/4d53eefd-ab35-4232-98a7-f8dc005983a9" controls="controls" style="max-width: 600px;">
      Vexor Demo Video
    </video>

## Why Vexor?

When you remember what a file *does* but forget its name or location, Vexor finds it instantly—no grep patterns or directory traversal needed.

Designed for both humans and AI coding assistants, enabling semantic file discovery in autonomous agent workflows.

## Install

Download standalone binary from [releases](https://github.com/scarletkc/vexor/releases) (no Python required), or:
```bash
pip install vexor  # also works with pipx, uv
```

## Quick Start

### 1. Configure API Key
```bash
vexor config --set-api-key "YOUR_KEY"
```
Or via environment: `VEXOR_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_GENAI_API_KEY`.

### 2. Search
```bash
vexor "api client config"  # defaults to search
vexor search "api client config"  # searches current directory
# or explicit path:
vexor search "api client config" --path ~/projects/demo --top 5
```

Vexor auto-indexes on first search. Example output:
```
Vexor semantic file search results
──────────────────────────────────
#   Similarity   File path                       Lines   Preview
1   0.923        ./src/config_loader.py          -       config loader entrypoint
2   0.871        ./src/utils/config_parse.py     -       parse config helpers
3   0.809        ./tests/test_config_loader.py   -       tests for config loader
```

### 3. Explicit Index (Optional)
```bash
vexor index  # indexes current directory
# or explicit path:
vexor index --path ~/projects/demo --mode code
```
Useful for CI warmup or when `auto_index` is disabled.

## Python API

Vexor can also be imported and used directly from Python:

```python
from vexor import index, search

index(path=".", mode="head")
response = search("config loader", path=".", mode="name")

for hit in response.results:
    print(hit.path, hit.score)
```

By default it reads `~/.vexor/config.json`. To ignore config and pass everything explicitly,
set `use_config=False`.

## Configuration

```bash
vexor config --set-provider openai          # default; also supports gemini/custom/local
vexor config --set-model text-embedding-3-small
vexor config --set-batch-size 0             # 0 = single request
vexor config --set-embed-concurrency 2       # parallel embedding requests
vexor config --set-auto-index true          # auto-index before search (default)
vexor config --rerank bm25                  # optional BM25 rerank for top-k results
vexor config --rerank flashrank             # FlashRank rerank (requires optional extra)
vexor config --clear-flashrank              # remove cached FlashRank models
vexor config --set-base-url https://proxy.example.com  # optional proxy
vexor config --clear-base-url               # reset to official endpoint
vexor config --show                         # view current settings
```

Rerank defaults to `off`.
FlashRank requires `pip install "vexor[flashrank]"` and caches models under `~/.vexor/flashrank`.

Config stored in `~/.vexor/config.json`.

### Rerank

Rerank reorders the semantic results with a secondary ranker. It uses 2x the requested
`--top` as candidates (e.g., top 10 reranked to show 5).

Recommended defaults:
- Keep `off` unless you want extra precision.
- Use `bm25` for lightweight lexical boosts; it is fast and dependency-free.
- Use `flashrank` for stronger reranking (requires `pip install "vexor[flashrank]"` and
  downloads a model to `~/.vexor/flashrank`).

Thanks to the FlashRank project: https://github.com/PrithivirajDamodaran/FlashRank

### Providers: Remote vs Local

Vexor supports both remote API providers (`openai`, `gemini`, `custom`) and a local provider (`local`):
- Remote providers use `api_key` and optional `base_url`.
- `custom` is OpenAI-compatible and requires both `model` and `base_url`.
- Local provider ignores `api_key/base_url` and only uses `model` plus `local_cuda` (CPU/GPU switch).

### Local Model (Offline)

Install the lightweight local backend:
```bash
pip install "vexor[local]"
```

GPU backend (requires CUDA drivers):
```bash
pip install "vexor[local-cuda]"
```

Download a local embedding model and auto-configure Vexor:
```bash
vexor local --setup --model intfloat/multilingual-e5-small
```

Then use `vexor search` / `vexor index` as usual.

Local models are stored in `~/.vexor/models` (clear with `vexor local --clean-up`).

GPU (optional): install `onnxruntime-gpu` (or `vexor[local-cuda]`) and use `vexor local --setup --cuda` (or `vexor local --cuda`).
Switch back with `vexor local --cpu`.

## Index Modes

Control embedding granularity with `--mode`:

| Mode | Description |
|------|-------------|
| `auto` | **Default.** Smart routing: Python/JS/TS → `code`, Markdown → `outline`, small files → `full`, large files → `head` |
| `name` | Embed filename only (fastest, zero content reads) |
| `head` | Extract first snippet for lightweight semantic context |
| `brief` | Extract high-frequency keywords from PRDs/requirements docs |
| `full` | Chunk entire content; long documents searchable end-to-end |
| `code` | AST-aware chunking by module/class/function boundaries for Python and JavaScript/TypeScript; other files fall back to `full` |
| `outline` | Chunk Markdown by heading hierarchy with breadcrumbs; non-`.md` falls back to `full` |

## Cache Behavior

Index cache keys derive from: `--path`, `--mode`, `--include-hidden`, `--no-recursive`, `--no-respect-gitignore`, `--ext`, `--exclude-pattern`.

Keep flags consistent to reuse cache; changing flags creates a separate index.

```bash
vexor config --show-index-all    # list all cached indexes
vexor config --clear-index-all   # clear all cached indexes
vexor index --path . --clear     # clear index for specific path
```

Re-running `vexor index` only re-embeds changed files; >50% changes trigger full rebuild.

## Command Reference

| Command | Description |
|---------|-------------|
| `vexor QUERY` | Shortcut for `vexor search QUERY` |
| `vexor search QUERY --path PATH` | Semantic search (auto-indexes if needed) |
| `vexor index --path PATH` | Build/refresh index manually |
| `vexor config --show` | Display current configuration |
| `vexor config --clear-flashrank` | Remove cached FlashRank models under `~/.vexor/flashrank` |
| `vexor local --setup [--model MODEL]` | Download a local model and set provider to `local` |
| `vexor local --clean-up` | Remove local model cache under `~/.vexor/models` |
| `vexor local --cuda` | Enable CUDA for local embeddings (requires `onnxruntime-gpu`) |
| `vexor local --cpu` | Disable CUDA and use CPU for local embeddings |
| `vexor install --skills claude` | Install Agent Skill for Claude Code |
| `vexor install --skills codex` | Install Agent Skill for Codex |
| `vexor doctor` | Run diagnostic checks (command, config, cache, API key, API connectivity) |
| `vexor update [--upgrade] [--pre]` | Check for new version (optionally upgrade; `--pre` includes pre-releases) |
| `vexor feedback` | Open GitHub issue form (or use `gh`) |
| `vexor alias` | Print a shell alias for `vx` and optionally apply it |

### Common Flags

| Flag | Description |
|------|-------------|
| `--path PATH` | Target directory (default: current working directory) |
| `--mode MODE` | Index mode (`auto`/`name`/`head`/`brief`/`full`/`code`/`outline`) |
| `--top K` / `-k` | Number of results (default: 5) |
| `--ext .py,.md` / `-e` | Filter by extension (repeatable) |
| `--exclude-pattern PATTERN` | Exclude paths by gitignore-style pattern (repeatable; `.js` treated as `**/*.js`) |
| `--include-hidden` / `-i` | Include hidden files |
| `--no-recursive` / `-n` | Don't recurse into subdirectories |
| `--no-respect-gitignore` | Include gitignored files |
| `--format porcelain` | Script-friendly TSV output |
| `--format porcelain-z` | NUL-delimited output |

Porcelain output fields: `rank`, `similarity`, `path`, `chunk_index`, `start_line`, `end_line`, `preview` (line fields are `-` when unavailable).

## AI Agent Skill

This repo includes a skill for AI agents to use Vexor effectively:

```bash
vexor install --skills claude  # Claude Code
vexor install --skills codex   # Codex
```

Skill source: [`plugins/vexor/skills/vexor-cli`](https://github.com/scarletkc/vexor/raw/refs/heads/main/plugins/vexor/skills/vexor-cli/SKILL.md)

## Documentation

See [docs](https://github.com/scarletkc/vexor/tree/main/docs) for more details.

Contributions, issues, and PRs welcome! Star if you find it helpful.

## License

[MIT](http://github.com/scarletkc/vexor/blob/main/LICENSE)
