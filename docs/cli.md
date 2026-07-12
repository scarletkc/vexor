# CLI Reference

## Command Reference

| Command | Description |
|---------|-------------|
| `vexor init` | Run the interactive setup wizard |
| `vexor QUERY` | Shortcut for `vexor search QUERY` |
| `vexor search QUERY --path PATH` | Semantic search (auto-indexes if needed) |
| `vexor index --path PATH` | Build/refresh index manually |
| `vexor config --show` | Display effective configuration and each field's origin |
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
| `--no-respect-gitignore` | Include files ignored by Git (does not disable `.vexorignore`) |
| `--format porcelain` | Script-friendly TSV output |
| `--format porcelain-z` | NUL-delimited output |
| `--no-cache` | In-memory only; do not read/write index cache |
| `--local` | With `index`, create and use `<path>/.vexor/index.db` |

Reranking is a config setting rather than a search flag — see
[Configuration → Rerank](configuration.md#rerank) for the available
strategies (`off`, `bm25`, `flashrank`, `remote`, `hybrid`).

## Project Configuration

Search and index commands walk upward from their resolved `--path` and apply
`config.json` from the nearest `.vexor/` marker. Project config v1 accepts only
`rerank`, `auto_index`, `model`, `embedding_dimensions`, `batch_size`,
`embed_concurrency`, and `extract_concurrency`. Credentials and endpoints
(`api_key`, `base_url`, `remote_rerank`) and every other field are rejected.

Precedence is global config, then project config, then environment overrides,
then explicit arguments. `vexor config --show` and `vexor doctor` use the
current working directory and show each effective field's origin. Mutating
`vexor config` options always write `~/.vexor/config.json`; edit the project
file directly for project-specific values. See
[Configuration → Project configuration](configuration.md#project-configuration)
for the full contract.

Porcelain output fields: `rank`, `similarity`, `path`, `chunk_index`,
`start_line`, `end_line`, `preview` (line fields are `-` when unavailable).

## Ignore Files

Use `.vexorignore` for project-specific indexing exclusions. It supports full
gitignore syntax, including negation (`!pattern`), directory-only patterns, and
anchored patterns. Files can appear in any directory; rules apply to that
directory and its descendants, and Vexor follows the ancestor chain from the
repository root when scanning a subdirectory. Outside a Git repository, rules
are anchored at the scanned directory.

`.vexorignore` is always honored, including with `--no-respect-gitignore`. When
both ignore files exist in a directory, `.gitignore` is read first and
`.vexorignore` second, so `.vexorignore` can add exclusions or re-include a path
with a negated pattern. Explicit `--exclude-pattern` rules still apply
separately. Changes to `.vexorignore` are detected automatically by the index
staleness check, so the next search or index refresh updates the indexed file
set.

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

By default, indexes are stored in `~/.vexor/index.db`. Project-local caching is
opt-in: create a `.vexor/` directory in a project root, or run
`vexor index --local`. Either way, Vexor writes a `.gitignore` inside
`.vexor/` on first use so generated indexes and caches cannot be committed by
accident while `config.json` remains eligible for version control. For each
index or search operation, Vexor walks upward from the resolved target path and
uses the nearest `.vexor/` directory it finds. This means nested projects use
their nearest marker. Searching a parent directory above a project root does
not discover markers in its children and therefore uses the global database.

Cache location precedence is: an explicit API/cache override, then the nearest
project `.vexor/`, then `~/.vexor/`. The project directory may also contain the
safe `config.json` overlay described above. Global configuration, update-check
data, FlashRank assets, and local embedding models remain under `~/.vexor/`.

```bash
vexor config --show-index-all    # list all cached indexes
vexor config --clear-index-all   # clear all cached indexes
vexor index --path . --clear     # clear index for specific path
vexor index --path . --local     # create/use ./.vexor/index.db
```

Re-running `vexor index` only re-embeds changed files; >50% changes trigger
full rebuild.
