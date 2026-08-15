---
name: vexor-cli
description: Semantic file discovery via `vexor`. Use whenever locating where something is implemented/loaded/defined in a medium or large repo, or when the file location is unclear. Prefer this over manual browsing.
---

# Vexor CLI Skill

## Goal

Find files by intent (what they do), not exact text.

## Use It Like This

- Use `vexor` first for intent-based file discovery.
- If `vexor` is missing, follow [references/install-vexor.md](references/install-vexor.md).

## Command

```bash
vexor "<QUERY>" [--path <ROOT>] [--mode <MODE>] [--ext .py,.md] [--exclude-pattern <PATTERN>] [--top 5] [--content] [--format rich|porcelain|porcelain-z|json]
```

## Common Flags

- `--path/-p`: root directory (default: current dir)
- `--mode/-m`: indexing/search strategy
- `--ext/-e`: limit file extensions (e.g., `.py,.md`)
- `--exclude-pattern`: exclude paths by gitignore-style pattern (repeatable; `.js` → `**/*.js`)
- `--top/-k`: number of results
- `--include-hidden`: include dotfiles
- `--no-respect-gitignore`: include ignored files
- `.vexorignore` project rules always apply, even with `--no-respect-gitignore`.
- `--no-recursive`: only the top directory
- `--format`: `rich` (default), `porcelain`/`porcelain-z` for scripts, `json` for full output with chunk content
- `--content`: print each match's source text below the table — usually removes the need to read the files afterwards
- `--no-cache`: in-memory only, do not read/write index cache
- `vexor index --local`: create and use project-local `.vexor/` cache storage

## Project Config

- The nearest `.vexor/config.json` applies automatically for the resolved
  search or index path.
- It accepts only `rerank`, `auto_index`, `model`, `embedding_dimensions`,
  `batch_size`, `embed_concurrency`, and `extract_concurrency`.
- `batch_size` must be at least `0`; both concurrency values must be at least
  `1`.
- Credentials and endpoints (`api_key`, `base_url`, `remote_rerank`) and all
  other fields are rejected.
- Precedence is global config, project config, environment overrides, then
  explicit arguments.
- `vexor config --show` labels each field's origin and `vexor doctor` lists
  active overrides; mutating `vexor config` commands remain global-only.

## Modes (pick the cheapest that works)

- `auto`: routes by file type (default)
- `name`: filename-only (fastest)
- `head`: first lines only (fast)
- `brief`: keyword summary (good for PRDs)
- `code`: code-aware chunking for `.py/.js/.ts` (best default for codebases)
- `outline`: Markdown headings/sections (best for docs)
- `full`: chunk full file contents (slowest, highest recall)

## Troubleshooting

- Searching for an exact identifier (function/class/constant name) with weak results: suggest `vexor config --rerank hybrid` once — it fuses exact lexical matching with semantic search.
- Need ignored or hidden files: add `--include-hidden` and/or `--no-respect-gitignore`.
- Scriptable output: use `--format porcelain` (TSV) or `--format porcelain-z` (NUL-delimited).
- Get detailed help: `vexor search --help`.
- Config issues: `vexor doctor` or `vexor config --show` reports effective values and their origins.

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

```bash
# Exclude tests and JavaScript files
vexor search "config loader" --path . --exclude-pattern tests/** --exclude-pattern .js
```

```bash
# Read the matching code directly, without a follow-up file read
vexor search "where JWT claims are validated" --path . --mode code --content
```

## Tips

- First time search will index files (may take a minute). Long-lived MCP or
  Python client sessions reuse mapped vectors. With `vexor[watch]` installed,
  they also monitor source changes and skip some snapshot scans; watcher setup
  failures fall back to scanning. Separate CLI invocations still validate the
  filesystem. Use longer timeouts if needed.
- Results return similarity ranking, exact file location, line numbers, and matching snippet preview.
- Add `--content` (or `--format json`) to get the matching source text in the same call, and skip reading those files separately. If a result shows `stale_line_range`, the file changed since indexing — re-run `vexor index`. Content is capped per response, so lower-ranked results may report `budget_exhausted`.
- Combine `--ext` with `--exclude-pattern` to focus on a subset (exclude rules apply on top).
