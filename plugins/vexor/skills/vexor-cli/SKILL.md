---
name: vexor-cli
description: Semantic file discovery via `vexor` (use when you need intent-based file lookup in a repo).
license: MIT
---

# Vexor CLI

## Purpose (for the agent)

Use Vexor as a fast "semantic grep": build (or refresh) an index for a directory, then query it to find the
most relevant files/sections for the current task.

## When to use

Use this skill when you need intent-based file discovery (not exact string match), e.g.:

- "Where is config loaded / validated?"
- "Which file defines the provider backends?"
- "Find the CLI command that prints this output"
- "Locate the code that chunks/embeds Python by function"

## Checklist (do this in order)

1. Ensure the CLI exists:
   - Prefer `vexor --help`.
   - If `vexor` is missing: `pip install --user vexor`.
2. Search the target root (auto-indexes when needed by default):
   - `vexor search "<QUERY>" --path <ROOT> [--mode <MODE>] [--ext .py,.md] [--top 10]`
3. Optional: pre-index (warm cache / CI) or refresh explicitly:
   - `vexor index --path <ROOT> [--mode <MODE>] [--ext .py,.md] [--include-hidden] [--no-recursive]`
   - Disable auto-index if you want the "search requires index" behavior:
     - `vexor config --set-auto-index false`

## Cache key (reuse the same cached index)

Vexor caches indexes by a key derived from these flags. Keep them consistent to reuse the same cache
(otherwise Vexor will build a separate index for the new flag combination):

- `--path`, `--mode` (defaults to `auto` when omitted)
- `--include-hidden`, `--no-recursive`
- `--no-respect-gitignore` (omit to respect `.gitignore`, including nested `.gitignore`)
- `--ext` (repeatable; each value may be a comma/space-separated list like `--ext .py,.md` or `--ext '.py .md'`)

## Mode guidance (pick the least expensive that works)

- Default: `auto` (Python → `code`, Markdown → `outline`, small → `full`, large → `head`)
- If you only need filenames: `name`
- If you need a quick content hint: `head`
- If you need structure in Markdown docs: `outline`
- If you need Python symbols (functions/classes/methods): `code`
- If recall is still low: `full` (more expensive; many chunks per file)

## Output parsing

- Default output is a rich table. It includes a `Lines` column when available.
- `--format porcelain` / `porcelain-z` emits:
  `rank similarity path chunk_index start_line end_line preview`
  (line fields may be `-` if unavailable).

## Troubleshooting

- If results look stale: rerun `vexor index` for the same cache key (or leave auto-index enabled).
- If you need ignored files: add `--no-respect-gitignore` to both commands.
- If API/config issues: `vexor config --show` and let the user fix them.
