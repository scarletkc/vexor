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

## Tools

### `vexor_search`

Semantic file search. Auto-indexes the directory on first use (when
`auto_index` is enabled, the default) and reuses the index afterwards.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `query` | string | required | Natural-language description of the file/code you want |
| `path` | string | server default path | Directory to search |
| `top` | integer | 5 | Number of results (1–50) |
| `mode` | string | `auto` | Index mode (`auto`/`name`/`head`/`brief`/`full`/`code`/`outline`) |
| `include_hidden` | boolean | false | Include hidden files |
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

Build or refresh the index for a directory (optional warm-up; searching
auto-indexes when needed). Accepts the same arguments as `vexor_search`
except `query` and `top`. Returns `{path, mode, status, files_indexed}` where
`status` is `stored`, `up_to_date`, or `empty`.

## Behavior notes

- Transport is newline-delimited JSON-RPC 2.0 over stdio; supported protocol
  versions: `2025-06-18`, `2025-03-26`, `2024-11-05`.
- Index cache keys follow the same rules as the CLI (see "Cache Behavior" in
  the README): tool calls with the same path/mode/filters share indexes with
  CLI usage.
- Execution failures (missing directory, provider errors, missing API key)
  are returned as tool results with `isError: true` so the agent can
  self-correct; malformed arguments are rejected as JSON-RPC errors.
- The server writes exactly one startup line to stderr and never writes
  non-protocol output to stdout.
