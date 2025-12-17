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
   - Prefer `vexor doctor` or `python -m vexor --help`.
   - If `vexor` is missing: `python -m pip install vexor` (or `pip install vexor` / `pipx install vexor`).
2. Index the target root (required once per cache key):
   - `vexor index --path <ROOT> [--mode <MODE>] [--ext .py ...] [--include-hidden] [--no-recursive]`
3. Search using the same cache key flags:
   - `vexor search "<QUERY>" --path <ROOT> [--mode <MODE>] [--ext .py ...] [--top 10]`

## Cache key (avoid "No cached index found…")

These flags must match between `index` and `search`:

- `--path`, `--mode` (defaults to `auto` when omitted)
- `--include-hidden`, `--no-recursive`
- `--respect-gitignore/--no-respect-gitignore` (default: respect, including nested `.gitignore`)
- `--ext` (repeatable; treat as a set)

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

- If results look stale: rerun `vexor index` for the same cache key.
- If you need ignored files: add `--no-respect-gitignore` to both commands.
- If API/config issues: `vexor config --show` and let the user fix them.
