# CLI Reference

## Command Reference

| Command | Description |
|---------|-------------|
| `vexor init` | Run the interactive setup wizard |
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
| `vexor mcp [--path PATH]` | Run the MCP stdio server for AI agents |
| `vexor doctor` | Run diagnostic checks (command, config, cache, API key, API connectivity) |
| `vexor update [--upgrade] [--pre]` | Check for new version (optionally upgrade; `--pre` includes pre-releases) |
| `vexor feedback` | Open GitHub issue form (or use `gh`) |
| `vexor alias` | Print a shell alias for `vx` and optionally apply it |

## Common Flags

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
| `--no-cache` | In-memory only; do not read/write index cache |

Porcelain output fields: `rank`, `similarity`, `path`, `chunk_index`,
`start_line`, `end_line`, `preview` (line fields are `-` when unavailable).

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

Index cache keys derive from: `--path`, `--mode`, `--include-hidden`,
`--no-recursive`, `--no-respect-gitignore`, `--ext`, `--exclude-pattern`.

Keep flags consistent to reuse cache; changing flags creates a separate index.

```bash
vexor config --show-index-all    # list all cached indexes
vexor config --clear-index-all   # clear all cached indexes
vexor index --path . --clear     # clear index for specific path
```

Re-running `vexor index` only re-embeds changed files; >50% changes trigger
full rebuild.
