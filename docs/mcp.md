# MCP Server

`vexor mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io)
server over stdio, exposing Vexor's semantic search to any MCP-capable agent
(Claude Code, Codex, Cursor, Windsurf, Zed, and others). It requires no extra
dependencies and reuses your existing Vexor configuration (`~/.vexor/config.json`)
and index cache.

```bash
vexor mcp [--path PATH]
```

`--path` sets the default directory used when a tool call omits `path`
(defaults to the server's working directory; MCP clients usually launch
servers from the project root, so the default is normally correct).

Run `vexor init` (or configure a provider/API key) before wiring Vexor into an
agent — the MCP server is non-interactive and reports missing configuration as
tool errors instead of launching the setup wizard.

## Client setup

### Claude Code

```bash
claude mcp add vexor -- vexor mcp
```

### Codex

```bash
codex mcp add vexor -- vexor mcp
```

### Generic JSON config (Cursor, Windsurf, and others)

```json
{
  "mcpServers": {
    "vexor": {
      "command": "vexor",
      "args": ["mcp"]
    }
  }
}
```

If `vexor` is not on the client's PATH, use an absolute command path, or
`python` with `"args": ["-m", "vexor", "mcp"]`.

### Running via uvx (no install)

MCP clients can also launch Vexor without installing it, using
`"command": "uvx", "args": ["vexor", "mcp"]`. Notes for this mode:

- Configuration, index caches, and downloaded models live under `~/.vexor`,
  so they persist across uvx's ephemeral environments. CLI commands from
  error messages work with a prefix: `uvx vexor init`, `uvx vexor config --show`.
- uvx caches the resolved environment and does **not** auto-upgrade; use
  `uvx vexor@latest mcp` to always run the newest release, or refresh
  explicitly with `uvx --refresh vexor`. `vexor update --upgrade` applies
  to PATH installs, not uvx environments.
- Local (offline) models need the `local` extra, which the default
  environment does not include — launch with
  `"command": "uvx", "args": ["--from", "vexor[local]", "vexor", "mcp"]`
  and set up the model once via
  `uvx --from "vexor[local]" vexor local --setup`.

### Provider and API key via client config

Instead of running `vexor init`, Vexor can be configured entirely from the
MCP client config through environment variables, with credentials and
configuration kept on separate channels:

- `VEXOR_API_KEY` (secret) — the embedding provider API key; takes precedence
  over a key stored in `~/.vexor/config.json` (the default provider is
  `openai`, so the key alone is enough for the common case).
- `VEXOR_REMOTE_RERANK_API_KEY` (secret) — the remote reranker API key;
  takes precedence over a stored reranker key and is only needed when
  `rerank` is set to `remote`.
- `VEXOR_CONFIG_JSON` (non-secret) — any other Vexor config as a JSON
  object, merged over `~/.vexor/config.json`. Accepts the same fields as
  the config file: `provider`, `model`, `base_url`, `rerank`,
  `embedding_dimensions`, `auto_index`, and so on. Credential fields
  (`api_key`, `remote_rerank.api_key`) are rejected with a clear error so
  secrets stay on the dedicated variables above.

```json
{
  "mcpServers": {
    "vexor": {
      "command": "vexor",
      "args": ["mcp"],
      "env": {
        "VEXOR_API_KEY": "sk-...",
        "VEXOR_CONFIG_JSON": "{\"provider\": \"gemini\", \"rerank\": \"bm25\"}"
      }
    }
  }
}
```

```toml
# Codex (~/.codex/config.toml)
[mcp_servers.vexor]
command = "vexor"
args = ["mcp"]
env = { "VEXOR_API_KEY" = "sk-...", "VEXOR_CONFIG_JSON" = '{"provider": "gemini", "rerank": "bm25"}' }
```

An invalid `VEXOR_CONFIG_JSON` fails loudly with a clear error instead of
being silently ignored.

## Tools

### `vexor_search`

Find files or code from a natural-language description and return ranked
paths, scores, line ranges, and previews. Missing or stale indexes are built
automatically when `auto_index` is enabled; otherwise call `vexor_index`
first.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `query` | string | required | Natural-language description of the file/code you want |
| `path` | string | server default path | Directory to search; absolute, or relative to the server's default path |
| `top` | integer | 5 | Number of results (1–50; out-of-range values are rejected) |
| `mode` | string | `auto` | Index mode (`auto`/`name`/`head`/`brief`/`full`/`code`/`outline`) |
| `include_hidden` | boolean | false | Include dot-prefixed files and directories such as `.github` |
| `respect_gitignore` | boolean | true | Honor `.gitignore` rules; set false to scan ignored files |
| `recursive` | boolean | true | Recurse into subdirectories; set false for top level only |
| `extensions` | string[] | – | Only include these extensions, e.g. `[".py", ".md"]` |
| `exclude_patterns` | string[] | – | Gitignore-style patterns to exclude |

Returns JSON text:

```json
{
  "query": "config loader",
  "path": "C:/projects/demo",
  "backend": "text-embedding-3-small",
  "reranker": null,
  "stale": false,
  "index_empty": false,
  "results": [
    {
      "rank": 1,
      "score": 0.9123,
      "path": "./src/config.py",
      "absolute_path": "C:/projects/demo/src/config.py",
      "start_line": 1,
      "end_line": 20,
      "preview": "config loader entrypoint"
    }
  ]
}
```

### `vexor_index`

Build or refresh the index for a directory. Use it to warm the cache or when
`auto_index` is disabled. It accepts the same arguments as `vexor_search`
except `query` and `top`, and returns
`{path, mode, status, files_indexed}` where `status` is `stored`,
`up_to_date`, or `empty`.

## Behavior notes

- Transport is newline-delimited JSON-RPC 2.0 over stdio; supported protocol
  versions: `2025-06-18`, `2025-03-26`, `2024-11-05`.
- Both tools declare an `outputSchema` and return `structuredContent`
  alongside the JSON text payload (MCP 2025-06-18; older clients ignore the
  extra fields).
- Relative `path` arguments resolve against the server's default path
  (`--path`, or the server's working directory).
- Index cache keys follow the same rules as the CLI (see
  [Cache Behavior](cli.md#cache-behavior)): tool calls with the same
  path/mode/filters share indexes with CLI usage.
- Execution failures (missing directory, provider errors, missing API key)
  are returned as tool results with `isError: true` so the agent can
  self-correct; malformed arguments (wrong types, out-of-range `top`,
  unknown mode) are rejected as JSON-RPC `-32602` errors.
- The server writes exactly one startup line to stderr and never writes
  non-protocol output to stdout.
- On startup a background thread checks PyPI for a newer release (cached
  for 24 hours in `~/.vexor/update_check.json`, 5-second timeout, silent on
  failure) and prints a one-line notice to stderr when an update exists.
  Disable it with `vexor config --set-update-check false`, or set
  `VEXOR_NO_UPDATE_CHECK=1` as a hard off switch. The same daily notice
  (and the same 24h cache) also applies to interactive CLI usage, where it
  only appears when stderr is a terminal.
