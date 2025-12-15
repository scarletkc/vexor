---
name: vexor-cli
description: Use Vexor's CLI (`vexor index`, `vexor search`, `vexor config`) to do semantic file discovery in a codebase. Trigger this skill when you need to find files by intent/meaning (not exact text matches), when you forgot a filename/path, when you need to pick the right indexing mode (name/head/brief/full), or when you need to inspect/refresh/clear the cached indexes under `~/.vexor` and configure Gemini/OpenAI embedding providers.
---

# Vexor CLI

## Overview

Use Vexor to build a cached semantic index for a directory, then search it quickly from the terminal.

## Default workflow (index → search)

1. Verify the CLI is available:
   - Run `vexor doctor` (installed entrypoint), or `python -m vexor --help` (repo/dev environments).
2. Build or refresh an index (required before searching):
   - Run `vexor index --path <ROOT> --mode <MODE> [--include-hidden] [--no-recursive] [--ext EXT ...]`.
3. Search the same index key:
   - Run `vexor search "<QUERY>" --path <ROOT> --mode <MODE> [--top K] [--include-hidden] [--no-recursive] [--ext EXT ...] [--format porcelain|porcelain-z]`.

Always pass `--mode` for both `index` and `search` (it is required; there is no default).
Omit `--format` to use the default `rich` table output.

## Pick an indexing mode

Use the mode to control what gets embedded:

- `name`: Embed file names only (fastest, zero content reads).
- `head`: Embed file name + a short head snippet (about the first 1000 characters) for supported text/code/PDF/DOCX/PPTX files; fall back to `name` when unsupported.
- `brief`: Embed a keyword summary extracted from the document head (English + Chinese tokenization); best for specs/PRDs.
- `full`: Chunk up to ~200k characters of supported files into sliding windows and embed each chunk; expect repeated file paths with different previews.

Prefer `head` for general codebase discovery, `name` for quick filename recall, `brief` for requirements docs, and `full` for long documents when `head` is too shallow.

## Options that must match (cache key)

Treat these options as part of the cache identity; a mismatch often looks like “No cached index found…”:

- `--path` / `-p` (root directory; default is current working directory)
- `--mode` / `-m`
- `--include-hidden` / `-i` (hidden files are included only when both `index` and `search` use it)
- `--no-recursive` / `-n` (recursive by default; recursive and non-recursive indexes are stored separately)
- `--ext` / `-e` (repeatable; normalize to `.py`, `.md`, etc.; indexing and searching must use the same set)

## Query and scope tips

- Prefer short, intent-rich queries (e.g., “config loader entrypoint”, “oauth token refresh”, “retry backoff”).
- Use `--ext` early to cut noise and cost (e.g., `--ext .py --ext .md`).
- Increase recall by switching from `name` → `head` → `full` rather than writing longer queries.
- Tune result count with `--top` / `-k` (must be `> 0`).

## Configure providers, keys, and defaults

Store configuration in `~/.vexor/config.json`:

- Set an API key: `vexor config --set-api-key "<TOKEN>"` (or set `VEXOR_API_KEY`).
- Pick a provider: `vexor config --set-provider gemini|openai`.
- Pick a model: `vexor config --set-model <MODEL_NAME>`.
- Set batch size: `vexor config --set-batch-size 0` (0 means a single request).
- Route through a custom endpoint/proxy: `vexor config --set-base-url <URL>` (reset with `--clear-base-url`).
- Inspect current settings: `vexor config --show`.

Use provider-specific environment variables when convenient:

- Gemini: `GOOGLE_GENAI_API_KEY` (legacy)
- OpenAI: `OPENAI_API_KEY`

## Inspect and clear cached indexes

- Show metadata for a specific index key: `vexor index --path <ROOT> --mode <MODE> --show [flags...]`.
- Clear one index key: `vexor index --path <ROOT> --mode <MODE> --clear [flags...]`.
- List all cached indexes: `vexor config --show-index-all`.
- Clear all cached indexes: `vexor config --clear-index-all`.

Indexes are stored in a shared SQLite database at `~/.vexor/index.db`.

## Troubleshooting

- Handle “No cached index found…” by rerunning `vexor index` with the same `--path/--mode/--include-hidden/--no-recursive/--ext` combination you plan to use for searching.
- Handle “Cached index … appears outdated” by rerunning `vexor index` (it re-embeds only changed files when possible).
- Handle “API key is missing…” by setting `vexor config --set-api-key "<TOKEN>"` or the appropriate environment variable.
- Expect `vexor update` to require network access (it fetches the latest version info).
