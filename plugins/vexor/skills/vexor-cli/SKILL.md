---
name: vexor-cli
description: Semantic file discovery via `vexor` for complex queries and large codebases (accurate intent-based file lookup).
license: MIT
---

# Vexor CLI Skill

## Purpose

Search codebases: find files by what they do, not exact text.

## When to Use

- Use for intent-based file discovery (e.g., "Where is configuration loaded/validated?").
- Disfavor use for exact string matching; use `rg` instead.

## How to Use

```bash
vexor search "<QUERY>" [--path <ROOT>] [--mode <MODE>] [--ext .py,.md] [--top 5] [--format rich|porcelain|porcelain-z]
```

If `vexor` is missing: `pip install vexor`.

## Common Flags

- `--path/-p`: root directory (default: current dir)
- `--mode/-m`: indexing/search strategy
- `--ext/-e`: limit file extensions (e.g., `.py,.md`)
- `--top/-k`: number of results
- `--include-hidden`: include dotfiles
- `--no-respect-gitignore`: include ignored files
- `--no-recursive`: only the top directory
- `--format`: `rich` (default) or `porcelain`/`porcelain-z` for scripts

## Modes (pick the cheapest that works)

- `auto`: routes by file type (default)
- `name`: filename-only (fastest)
- `head`: first lines only (fast)
- `brief`: keyword summary (good for PRDs)
- `code`: code-aware chunking for `.py/.js/.ts` (best default for codebases)
- `outline`: Markdown headings/sections (best for docs)
- `full`: chunk full file contents (slowest, highest recall)

## Troubleshooting

- Need ignored or hidden files: add `--include-hidden` and/or `--no-respect-gitignore`.
- Scriptable output: use `--format porcelain` (TSV) or `--format porcelain-z` (NUL-delimited).
- Get detailed help: `vexor --help` or `vexor search --help`.
- Config issues: `vexor doctor` diagnoses API, cache, and connectivity (tell the user to set up)..

## Examples

```bash
# Find CLI entrypoints / commands
vexor search "typer app commands" --top 5
```

```bash
# Search docs by headings/sections
vexor search "user authentication flow" --path docs --mode outline --ext .md --format porcelain
```

```bash
# Locate config loading/validation logic
vexor search "config loader" --path . --mode code --ext .py
```
