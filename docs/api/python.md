# Python API

The Python API mirrors the CLI behavior and exposes the same search/index flow.
Configuration can come from disk, in-memory overrides, or per-call parameters.

## Quick start

```python
from vexor import index, search

index(path=".", mode="head")
response = search("config loader", path=".", mode="name")

for hit in response.results:
    print(hit.path, hit.score)
```

## Client sessions (library usage)

For scoped configuration and per-call data directories, use `VexorClient`:

```python
from vexor import VexorClient

client = VexorClient(data_dir="/tmp/vexor", use_config=False)
client.set_config_json({"provider": "openai", "api_key": "YOUR_KEY"})

response = client.search("config loader", path=".", mode="name")
```

Scoped overrides without global mutation:

```python
from vexor import config_context

with config_context({"provider": "openai", "api_key": "YOUR_KEY"}) as client:
    response = client.search("config loader", path=".", mode="name")
```

## Configuration sources and precedence

Configuration is resolved in this order (highest to lowest priority):

1. Explicit function arguments (e.g., `provider`, `model`, `api_key`)
2. Per-call override via `config=...`
3. In-memory runtime config set by `set_config_json(...)`
4. `~/.vexor/config.json` (unless `use_config=False`)
5. Built-in defaults

For pure library usage that should ignore on-disk config, pass `use_config=False`
and set explicit arguments or `config=...`.

## API reference

### search(...)

Core parameters:

- `query`: text to search for (required)
- `path`: root directory to search
- `mode`: indexing strategy (`auto`, `name`, `head`, `brief`, `code`, `outline`, `full`)
- `top`: number of results
- `include_hidden`, `respect_gitignore`, `recursive`
- `extensions`, `exclude_patterns`
- `data_dir`, `config_dir`, `cache_dir` (per-call overrides for config/cache locations)

Config-related parameters:

- `provider`, `model`, `batch_size`, `embed_concurrency`, `extract_concurrency`, `extract_backend`
- `base_url`, `api_key`, `local_cuda`
- `auto_index`, `use_config`
- `config`: dict or JSON string (per-call override)

Cache control:

- `temporary_index`: build index in memory (no index cache read/write)
- `no_cache`: disable all disk caches (index + embedding + query)

Returns a `SearchResponse` with:

- `results`: list of `SearchResult` items (`path`, `score`, `preview`, `chunk_index`, `start_line`, `end_line`)
- `backend`: backend description string
- `is_stale`, `index_empty`, `reranker`

### index(...)

Build or refresh the index for a directory. Accepts the same indexing and config
parameters as `search`, plus `config` for per-call override.

Returns `IndexResult` with `status`, `cache_path`, `files_indexed`.

### index_in_memory(...)

Build an index without writing to disk. Returns `InMemoryIndex`, which can be
searched via `InMemoryIndex.search(...)`. Defaults to `no_cache=True`.

### clear_index(...)

Remove cached index entries for a directory. Returns the count removed.

### set_data_dir(path)

Set the base directory for Vexor data:

- `config.json`
- `index.db` and embedding/query caches
- `flashrank/` and `models/` directories

Pass `None` to reset to `~/.vexor`.
For scoped overrides, prefer `VexorClient` or per-call `data_dir`/`config_dir`/`cache_dir`.

### set_config_json(payload, replace=False)

Set an in-memory config override from a dict or JSON string.
This does not write `config.json`. Pass `None` to clear the override.

If `replace=True`, the payload is applied to default values instead of
merging with the current runtime or on-disk config (avoids reading disk).

## Config payload schema

The `config` payload (dict/JSON) supports:

- `provider`: `openai`, `gemini`, `custom`, `local`
- `model`: embedding model name
- `api_key`: API key (string or null)
- `base_url`: API base URL for `openai`/`custom` providers
- `batch_size`: integer
- `embed_concurrency`: integer
- `extract_concurrency`: integer
- `extract_backend`: `auto`, `thread`, or `process`
- `auto_index`: boolean
- `local_cuda`: boolean (local provider only)
- `rerank`: `off`, `bm25`, `flashrank`, `remote`
- `flashrank_model`: string or null
- `remote_rerank`: object with `base_url`, `api_key`, `model`

Unknown keys are ignored. `remote_rerank.base_url` is normalized to end with `/rerank`.

API keys can also come from environment variables:
`VEXOR_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_GENAI_API_KEY`,
and `VEXOR_REMOTE_RERANK_API_KEY`.

Environment variables only supply API keys; other settings must be passed
explicitly via function arguments or `config=...`.

## Cache behavior

- Default: index + embedding/query caches are stored on disk.
- `temporary_index=True`: no index cache read/write; embedding/query caches still used.
- `no_cache=True`: disables all disk caches; forces in-memory indexing and embeddings.
- When `no_cache=False`, a process-local LRU may reuse recent embeddings.

## Examples

Per-call override:

```python
from vexor import search

response = search(
    "config loader",
    path=".",
    mode="name",
    config={"rerank": "remote", "remote_rerank": {"model": "bge-reranker-v2-m3"}},
)
```

Runtime override (in memory only):

```python
from vexor import set_config_json, search

set_config_json({"provider": "openai", "api_key": "YOUR_KEY"})
response = search("config loader", path=".", mode="name")
set_config_json(None)
```

No cache:

```python
from vexor import search

response = search("config loader", path=".", mode="name", no_cache=True)
```

In-memory index:

```python
from vexor import index_in_memory

index = index_in_memory(path=".", mode="name")
response = index.search("config loader", top=5)
```
