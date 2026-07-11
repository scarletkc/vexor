<div align="center">

<img src="https://raw.githubusercontent.com/scarletkc/vexor/refs/heads/main/assets/vexor.svg" alt="Vexor" width="35%" height="auto">

# Vexor

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/vexor.svg)](https://pypi.org/project/vexor/)
[![CI](https://img.shields.io/github/actions/workflow/status/scarletkc/vexor/publish.yml?branch=main)](https://github.com/scarletkc/vexor/actions/workflows/publish.yml)
[![Codecov](https://img.shields.io/codecov/c/github/scarletkc/vexor/main)](https://codecov.io/github/scarletkc/vexor)
[![License](https://img.shields.io/github/license/scarletkc/vexor.svg)](https://github.com/scarletkc/vexor/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/scarletkc/vexor)

</div>

---

**Vexor** is a semantic search engine that builds reusable indexes over files and code.
It supports configurable embedding and reranking providers, and exposes the same core through a Python API, a CLI tool, and an optional desktop frontend.

<video src="https://github.com/user-attachments/assets/4d53eefd-ab35-4232-98a7-f8dc005983a9" controls="controls" style="max-width: 600px;">
      Vexor Demo Video
    </video>

## Featured In

Vexor has been recognized and featured by the community:

- **[Ruan Yifeng's Weekly (Issue #379)](https://github.com/ruanyf/weekly/blob/master/docs/issue-379.md#ai-%E7%9B%B8%E5%85%B3)** - A leading tech newsletter in the Chinese developer community.
- **[Awesome Claude Skills](https://github.com/VoltAgent/awesome-claude-skills?tab=readme-ov-file#development-and-testing)** - Curated list of best-in-class skills for AI agents.

## Why Vexor?

When you remember what a file *does* but forget its name or location, Vexor finds it instantly—no grep patterns or directory traversal needed.

Designed for both humans and AI coding assistants, enabling semantic file discovery in autonomous agent workflows.

## Install

Download standalone binary from [releases](https://github.com/scarletkc/vexor/releases) (no Python required), or:
```bash
pip install vexor  # also works with pipx, uv
```

## Quick Start

### 0. Guided Setup (Recommended)
```bash
vexor init
```
The wizard also runs automatically on first use when no config exists.

### 1. Search
```bash
vexor "api client config"  # defaults to search current directory
# or explicit path:
vexor search "api client config" --path ~/projects/demo --top 5
# in-memory search only:
vexor search "api client config" --no-cache 
```

Vexor auto-indexes on first search. Example output:
```
Vexor semantic file search results
──────────────────────────────────
#   Similarity   File path                       Lines   Preview
1   0.923        ./src/config_loader.py          -       config loader entrypoint
2   0.871        ./src/utils/config_parse.py     -       parse config helpers
3   0.809        ./tests/test_config_loader.py   -       tests for config loader
```

### 2. Explicit Index (Optional)
```bash
vexor index  # indexes current directory
# or explicit path:
vexor index --path ~/projects/demo --mode code
```
Useful for CI warmup or when `auto_index` is disabled.

## Desktop App (Experimental)

> The desktop app is experimental and not actively maintained.
> It may be unstable. For production use, prefer the CLI.

![GUI](https://raw.githubusercontent.com/scarletkc/vexor/refs/heads/main/assets/gui_demo.png)

Download the desktop app from [releases](https://github.com/scarletkc/vexor/releases).

## Python API

Vexor can also be imported and used directly from Python:

```python
from vexor import index, search

index(path=".", mode="head")
response = search("config loader", path=".", mode="name")

for hit in response.results:
    print(hit.path, hit.score)
```

By default it reads `~/.vexor/config.json`. For runtime config overrides, cache
controls, and per-call options, see [`docs/api/python.md`](https://github.com/scarletkc/vexor/tree/main/docs/api/python.md).

## AI Agent Skill

This repo includes a skill for AI agents to use Vexor effectively:

```bash
vexor install --skills claude  # Claude Code
vexor install --skills codex   # Codex
```

Skill source: [`plugins/vexor/skills/vexor-cli`](https://github.com/scarletkc/vexor/raw/refs/heads/main/plugins/vexor/skills/vexor-cli/SKILL.md)

## MCP Server

<!-- mcp-name: io.github.scarletkc/vexor -->

[![vexor MCP server](https://glama.ai/mcp/servers/scarletkc/vexor/badges/score.svg)](https://glama.ai/mcp/servers/scarletkc/vexor)

> [!NOTE]
> The Agent Skill and the MCP server provide the same core capability — pick **one** per agent.
> The skill teaches shell-capable agents (Claude Code, Codex) to drive the full CLI and assumes `vexor` is installed on PATH; the MCP server exposes search as native tools, works in any MCP client (Cursor, Windsurf, Zed, ...), and can bootstrap without prior setup via `uvx` and environment variables.

Vexor ships a built-in [MCP](https://modelcontextprotocol.io) stdio server, so any MCP-capable agent can use semantic file search as a native tool:

```bash
claude mcp add vexor -- vexor mcp   # Claude Code
codex mcp add vexor -- vexor mcp    # Codex
```

Or configure manually in any MCP client, optionally supplying the API key
and any config overrides via `env` (no `vexor init` needed):

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

The server exposes two tools: `vexor_search` (semantic file search) and `vexor_index` (explicit index warm-up). No extra dependencies are required. Vexor is listed on the [official MCP registry](https://registry.modelcontextprotocol.io) as `io.github.scarletkc/vexor`. See [`docs/mcp.md`](https://github.com/scarletkc/vexor/tree/main/docs/mcp.md) for tool schemas, environment variables, and client setup details.

## Configuration

```bash
vexor init                             # guided setup (recommended)
vexor config --set-api-key "YOUR_KEY"  # or env: VEXOR_API_KEY / OPENAI_API_KEY / ...
vexor config --set-provider openai     # default; also gemini/voyageai/custom/local
vexor config --rerank bm25             # recommended: improves search accuracy
vexor config --show                    # view current settings
```

Config lives in `~/.vexor/config.json`. Any field can also be injected via the `VEXOR_CONFIG_JSON` environment variable (useful for MCP client configs and CI), and fully offline use is supported through local embedding models.

See [`docs/configuration.md`](https://github.com/scarletkc/vexor/blob/main/docs/configuration.md) for the complete reference: all config commands, API keys and environment variables, rerank strategies (BM25 / FlashRank / remote), remote vs local providers, embedding dimensions, and offline local model setup.

## CLI Reference

Everyday usage fits in `vexor "query"`, `vexor search`, and `vexor index` (see Quick Start). The full command table, common flags, index modes (`--mode auto/name/head/brief/full/code/outline`), cache behavior, and porcelain output format are documented in [`docs/cli.md`](https://github.com/scarletkc/vexor/blob/main/docs/cli.md).

## Documentation

- [Configuration](https://github.com/scarletkc/vexor/blob/main/docs/configuration.md) — providers, API keys, rerank, embedding dimensions, local models
- [CLI reference](https://github.com/scarletkc/vexor/blob/main/docs/cli.md) — commands, flags, index modes, cache behavior
- [MCP server](https://github.com/scarletkc/vexor/blob/main/docs/mcp.md) — client setup, environment variables, tool schemas
- [Python API](https://github.com/scarletkc/vexor/blob/main/docs/api/python.md) — programmatic usage

## Contributing

Contributions, issues, and PRs welcome! Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org) (e.g. `feat(mcp): add stdio server`). Star if you find it helpful.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=scarletkc/vexor&type=date&legend=top-left)](https://www.star-history.com/#scarletkc/vexor&type=date&legend=top-left)

## License

[MIT](http://github.com/scarletkc/vexor/blob/main/LICENSE)
