# Configuration

Vexor is configured through `vexor config` commands (or the interactive
`vexor init` wizard). Global settings persist in `~/.vexor/config.json`.

## Where data lives

Global configuration, update-check data, FlashRank assets, and local embedding
models stay under `~/.vexor/`. Indexes normally use `~/.vexor/index.db`, but a
project containing a `.vexor/` directory uses `<project>/.vexor/index.db` for
searches and indexing within that project. The same directory may contain a
tracked `config.json` with safe project-level overrides.

Run `vexor index --local` to create the project directory and its ignore file.
This does not create a project config. The generated index and caches remain
ignored while `.vexor/config.json` can be committed.

When an upgrade changes the index cache schema, existing index caches are
invalidated and rebuilt automatically on next use; cached embeddings are
reused, so the migration does not re-embed unchanged content.

## Project configuration

For each path-aware search or index operation, Vexor resolves the target path
and walks upward to the nearest `.vexor/` directory. If that marker contains
`config.json`, Vexor overlays it on the global configuration. Nested projects
therefore use their nearest marker; a search above a project does not discover
markers below it. If the nearest marker has no `config.json`, Vexor uses the
global configuration instead of continuing into an outer project.

Project config v1 accepts exactly these fields:

- `rerank`
- `auto_index`
- `model`
- `embedding_dimensions`
- `batch_size`
- `embed_concurrency`
- `extract_concurrency`

Example `<project>/.vexor/config.json`:

```json
{
  "model": "text-embedding-3-small",
  "embedding_dimensions": 1024,
  "batch_size": 32,
  "embed_concurrency": 4,
  "extract_concurrency": 4,
  "auto_index": true,
  "rerank": "hybrid"
}
```

Repository files are untrusted input, so this is a strict allowlist. Project
config explicitly rejects credentials and endpoints (`api_key`, `base_url`,
and the entire `remote_rerank` object), as well as every other unsupported or
unknown field. Configure those values globally, through environment variables,
or with explicit Python API arguments. This is stricter than
`VEXOR_CONFIG_JSON`, which may set non-secret fields such as `base_url`.

Effective precedence, from lowest to highest, is:

1. Built-in defaults
2. Global `~/.vexor/config.json`
3. The nearest project `.vexor/config.json`
4. Environment overrides, including `VEXOR_CONFIG_JSON`
5. Explicit runtime or API arguments

`vexor config --show` and `vexor doctor` resolve the project from the current
working directory: `--show` labels every effective field with its origin, and
`doctor` reports the project config file plus any project or environment
overrides. All mutating `vexor config` options and `vexor init` still write
only the global config; edit `.vexor/config.json` directly to change a
project override.

## Commands

```bash
vexor config --set-provider openai          # default; also supports gemini/voyageai/custom/local
vexor config --set-model text-embedding-3-small
vexor config --set-provider voyageai        # uses voyage defaults when model/base_url are unset
vexor config --set-batch-size 0             # 0 = single request
vexor config --set-embed-concurrency 4       # parallel embedding requests
vexor config --set-extract-concurrency 4     # parallel file extraction workers
vexor config --set-extract-backend auto      # auto|thread|process (default: auto)
vexor config --set-embedding-dimensions 1024 # optional, model/provider dependent
vexor config --clear-embedding-dimensions    # reset to model default dimension
vexor config --set-auto-index true          # auto-index before search (default)
vexor config --set-update-check false       # disable the daily update notice (default: on)
vexor config --rerank bm25                  # optional BM25 rerank for top-k results
vexor config --rerank flashrank             # FlashRank rerank (requires optional extra)
vexor config --rerank remote                # remote rerank via HTTP endpoint
vexor config --rerank hybrid                # full-corpus BM25 + dense rank fusion
vexor config --set-flashrank-model ms-marco-MultiBERT-L-12  # multilingual model
vexor config --set-flashrank-model          # reset FlashRank model to default
vexor config --clear-flashrank              # remove cached FlashRank models
vexor config --set-remote-rerank-url https://proxy.example.com/v1/rerank
vexor config --set-remote-rerank-model bge-reranker-v2-m3
vexor config --set-remote-rerank-api-key $VEXOR_REMOTE_RERANK_API_KEY  # or env var
vexor config --clear-remote-rerank          # clear remote rerank config
vexor config --set-base-url https://proxy.example.com  # optional proxy
vexor config --clear-base-url               # reset to official endpoint
vexor config --show                         # view effective settings and origins
```

Rerank defaults to `off`. **It is highly recommended to configure the
Reranker in advance to improve search accuracy.** FlashRank requires
`pip install "vexor[flashrank]"` and caches models under `~/.vexor/flashrank`.

## API Keys

```bash
vexor config --set-api-key "YOUR_KEY"
```

Or via environment: `VEXOR_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_GENAI_API_KEY`,
or `VOYAGE_API_KEY`; `VEXOR_API_KEY` takes precedence over a stored key.

Any non-secret config field can also be injected as a JSON object via the
`VEXOR_CONFIG_JSON` environment variable (useful for MCP client configs and
CI), merged over the effective global and project config. Credential fields
inside `VEXOR_CONFIG_JSON` are rejected — use the dedicated variables above.

## Rerank

Rerank reorders the semantic results with a secondary ranker. Candidate
sizing uses `clamp(int(--top * 2), 20, 150)`.

`hybrid` is different from the candidate rerankers: it scores BM25 over every
indexed chunk and fuses that full lexical ranking with the full dense ranking
using reciprocal rank fusion. Hybrid result scores are normalized rank-fusion
scores in `[0, 1]`, not cosine similarities. A score of `1.0` means a chunk was
ranked first by both retrieval paths.

Recommended defaults:

- Keep `off` unless you want extra precision.
- Use `bm25` for a lightweight lexical boost with almost no extra latency.
- Use `hybrid` when exact identifiers and lexical-only matches matter.
- BM25 uses a multilingual tokenizer (Bert pre-tokenizer), so it can handle CJK better.
- Use `flashrank` for stronger reranking (requires `pip install "vexor[flashrank]"` and
  downloads a model to `~/.vexor/flashrank`).
- Use `remote` to call a hosted reranker that accepts `{model, query, documents}` and
  returns ranked indexes.
- For Chinese or multi-language content, set `--set-flashrank-model ms-marco-MultiBERT-L-12`.
- If unset, FlashRank defaults to `ms-marco-TinyBERT-L-2-v2`.

## Providers: Remote vs Local

Vexor supports both remote API providers (`openai`, `gemini`, `voyageai`,
`custom`) and a local provider (`local`):

- Remote providers use `api_key` and optional `base_url`.
- `voyageai` defaults to `https://api.voyageai.com/v1` when `base_url` is not set.
- `custom` is OpenAI-compatible and requires both `model` and `base_url`.
- Local provider ignores `api_key/base_url` and only uses `model` plus `local_cuda` (CPU/GPU switch).

## Embedding Dimensions

Embedding dimensions are optional. If unset, the provider/model default is
used. Custom dimensions are validated for:

- OpenAI `text-embedding-3-*`
- Voyage `voyage-3*` and `voyage-code-3*`

```bash
vexor config --set-embedding-dimensions 1024
vexor config --clear-embedding-dimensions
```

If you change dimensions after an index is built, rebuild the index:

```bash
vexor index --path .
```

## Local Model (Offline)

Install the lightweight local backend:

```bash
pip install "vexor[local]"
```

GPU backend (requires CUDA drivers):

```bash
pip install "vexor[local-cuda]"
```

Download a local embedding model and auto-configure Vexor:

```bash
vexor local --setup --model intfloat/multilingual-e5-small
```

Then use `vexor search` / `vexor index` as usual.

Local models are stored in `~/.vexor/models` (clear with `vexor local --clean-up`).

GPU (optional): install `onnxruntime-gpu` (or `vexor[local-cuda]`) and use
`vexor local --setup --cuda` (or `vexor local --cuda`). Switch back with
`vexor local --cpu`.
