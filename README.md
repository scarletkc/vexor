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

**Vexor** is a vector-powered CLI for semantic file search. It uses configurable embedding models and ranks results by cosine similarity.

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

## Configuration

```bash
vexor config --set-provider openai          # default; also supports gemini
vexor config --set-model text-embedding-3-small
vexor config --set-batch-size 0             # 0 = single request
vexor config --set-auto-index true          # auto-index before search (default)
vexor config --set-base-url https://proxy.example.com  # optional proxy
vexor config --clear-base-url               # reset to official endpoint
vexor config --show                         # view current settings
```

Config stored in `~/.vexor/config.json`.

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

Index cache keys derive from: `--path`, `--mode`, `--include-hidden`, `--no-recursive`, `--no-respect-gitignore`, `--ext`.

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
| `vexor search QUERY --path PATH` | Semantic search (auto-indexes if needed) |
| `vexor index --path PATH` | Build/refresh index manually |
| `vexor config --show` | Display current configuration |
| `vexor install --skills claude` | Install Agent Skill for Claude Code |
| `vexor install --skills codex` | Install Agent Skill for Codex |
| `vexor doctor` | Run diagnostic checks (command, config, cache, API key, API connectivity) |
| `vexor update` | Check for new version |

### Common Flags

| Flag | Description |
|------|-------------|
| `--path PATH` | Target directory (default: current working directory) |
| `--mode MODE` | Index mode (`auto`/`name`/`head`/`brief`/`full`/`code`/`outline`) |
| `--top K` / `-k` | Number of results (default: 5) |
| `--ext .py,.md` / `-e` | Filter by extension (repeatable) |
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
