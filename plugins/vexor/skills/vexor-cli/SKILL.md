---
name: vexor-cli
description: Semantic file discovery via `vexor` (use when you need intent-based file lookup in a repo).
license: MIT
---

# Vexor CLI Skill

## Purpose

Vexor is a **semantic grep** for codebases. Use it to find files by *what they do*, not by exact string matches.

## When to Use

Invoke Vexor when you need **intent-based file discovery**:

- "Where is config loaded/validated?"
- "Which file defines the provider backends?"
- "Find the CLI command that prints this output"
- "Locate the code that handles user authentication"

**Do NOT use** for exact string matching—use `grep`/`ripgrep` instead.

## Quick Reference

```bash
# Search (auto-indexes if needed)
vexor search "<QUERY>" [--path <ROOT>] [--mode <MODE>] [--ext .py,.md] [--top 10]

# Pre-index (optional, for warmup/CI)
vexor index [--path <ROOT>] [--mode <MODE>] [--ext .py,.md]

# Check installation
vexor --help
```

If `vexor` is missing: `pip install --user vexor`

## Mode Selection Guide

Choose the **least expensive mode** that works:

| Mode | Use When | Cost |
|------|----------|------|
| `auto` | Default choice (smart routing) | varies |
| `name` | Only need filename matches | lowest |
| `head` | Need quick content hint | low |
| `code` | Need Python/JS/TS symbols (functions/classes) | medium |
| `outline` | Need Markdown structure | medium |
| `brief` | Need PRD/requirements keywords | medium |
| `full` | Need deep content search | highest |

**Auto mode routing:** Python/JS/TS → `code`, Markdown → `outline`, small files → `full`, large files → `head`

## Cache Key Rules

The same flags produce the same cached index. Different flags = different cache.

**Cache key components:**
- `--path` (required, default: current directory)
- `--mode` (default: `auto`)
- `--include-hidden` / `-i` (default: off)
- `--no-recursive` / `-n` (default: recursive)
- `--no-respect-gitignore` (default: respects gitignore)
- `--ext` (default: all extensions)

**Tip:** Keep flags consistent between `index` and `search` to reuse cache.

## Output Formats

| Format | Use Case |
|--------|----------|
| `--format rich` | Default table (human-readable) |
| `--format porcelain` | Tab-separated (scriptable) |
| `--format porcelain-z` | NUL-delimited (safe for filenames with spaces) |

Porcelain fields: `rank`, `similarity`, `path`, `chunk_index`, `start_line`, `end_line`, `preview`

Line fields are `-` when unavailable.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Stale results | Re-run `vexor index` with same flags |
| Need ignored files | Add `--no-respect-gitignore` |
| Low recall | Try `--mode full` (more expensive) |
| API errors | Run `vexor config --show` to check settings and let the user configure |
| Missing command | `pip install --user vexor` |

Run `vexor doctor` to diagnose common issues:
- Command availability on PATH
- Config file status
- Cache directory writability
- API key configuration
- API connectivity (use `--skip-api-test` to skip network check)

## Examples

```bash
# Find config-related files in Python project
vexor search "config loader" --path . --mode code --ext .py

# Search documentation
vexor search "API authentication" --path ./docs --mode outline --ext .md

# Fast filename-only search
vexor search "test utils" --path . --mode name --top 10

# Include hidden and ignored files
vexor search "env secrets" --path . --include-hidden --no-respect-gitignore
```
