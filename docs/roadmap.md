# Roadmap

Strategic note (2026-07 project review): position Vexor as retrieval
infrastructure for AI coding agents — local-first, no account required,
provider-agnostic — rather than only a human-facing search CLI. Agent
integrations (MCP, skills) are the primary distribution channel; the
differentiators against hosted competitors (mgrep, claude-context) are
"pip install and go", bring-your-own-key or fully offline, and data
never leaving the machine.

## P0 — Agent-first distribution

- Chunk content in search results (shipped): `vexor_search`, the Python API,
  and `vexor search --content` / `--format json` return the matching source
  text, read back from the file by line range at search time. Previously a
  caller got a path plus a 160-character preview and had to re-read the file,
  which spent back most of what retrieval saved. Porcelain output is
  unchanged. Follow-ups this unblocks:
  - Feed chunk content to the rerankers. `_build_rerank_document` still scores
    against filename, path, and the truncated preview, so rerank quality is
    capped by preview length. Changing it shifts existing result ordering, so
    it needs its own benchmark — and the collection API work below already
    requires extracting that seam.
  - The token-cost evaluation below is only worth running after this: with
    path-only results, agent+Vexor could not show much saving over grep
    because the follow-up reads dominated.

- Add a filesystem-independent, general-purpose text collection API for
  callers to submit records from databases directly, with operations such
  as `upsert(id, text, metadata)`, `delete(id)`, and
  `search(query, filters, top_k)`. Metadata should support fields such as
  `user_id`, `chat_id`, timestamp, record type, and source, with strict
  field-based filtering at search time. Reuse the existing embedding,
  cache, BM25, hybrid ranking, and reranker layers without requiring text
  to originate from files. Results should return the record ID, original
  text, metadata, and relevance score instead of file paths and line
  numbers. This lets integrations such as Telegram bots keep MySQL as the
  source of truth and incrementally write chat excerpts to Vexor without
  exporting temporary files or maintaining a parallel file tree.
  - Reuse, unchanged: `bm25.py` already holds only pure functions over
    ids and rows (`score_postings` returns `{id: score}`, `rrf_fuse`
    fuses by row index); the `embedding_cache` table is keyed by
    `(model, text_hash)` and its helpers open `cache_db_path()` on their
    own connection, so a separate store still shares one embedding cache;
    and `VexorSearcher.embed_texts` accepts arbitrary text and already
    L2-normalizes, so a dot product is cosine similarity.
  - Filesystem coupling to break: `index_metadata` keys on a path hash
    and `indexed_file` requires `rel_path`, `abs_path`, `size_bytes`, and
    `mtime`; `SearchResult.path` is a required `Path`; and all three
    rerankers build their documents through `_build_rerank_document`,
    which reads `result.path`.
  - Storage location: give collections their own `collections.db` beside
    `index.db` rather than new tables inside it. `clear_all_cache()`
    unlinks the database file and `vexor config --clear-index-all` calls
    it; a file index is rebuildable from disk, but collection records are
    only rebuildable by making the caller re-upsert everything and pay
    for embeddings a second time. Do not bump `CACHE_VERSION` for this
    work — the new tables are additive under
    `CREATE TABLE IF NOT EXISTS`, and a bump would invalidate every
    existing user's file index. Extract the shared `_connect` and
    `_chunk_values` helpers out of `cache.py` so both stores get the same
    WAL and `busy_timeout` behavior under concurrent writers.
  - Schema: `collection` (name, provider, model, dimension, schema
    version) pins the embedding contract, and changing model or dimension
    afterwards must raise and tell the caller to recreate the collection
    instead of mixing vector widths. `collection_record` holds the
    caller's `record_key` (unique per collection), the original text, a
    `text_hash` computed with `embedding_cache_key` so an unchanged
    upsert skips re-embedding, and the metadata JSON.
    `collection_embedding` stores one normalized vector per record: v1
    does not chunk, one record is one vector, and text that exceeds a
    provider limit surfaces the provider error rather than being silently
    truncated. `collection_bm25_doc` and `collection_bm25_posting`
    deliberately mirror the shape of the existing `bm25_doc` and
    `bm25_posting` tables so `bm25.score_postings` consumes their rows
    unchanged.
  - Metadata and filtering: keep metadata a flat mapping of scalars
    (`str`, `int`, `float`, `bool`, `None`) and reject nested values
    rather than accepting values that can never be filtered on, which
    would surface later as a phantom recall bug. Index it in a
    `collection_meta` EAV table carrying both `value_text` and
    `value_num` so strings compare by equality while numbers,
    timestamps, and booleans also support ranges; a `datetime` writes
    ISO text and epoch seconds together. Support `eq`, `ne`, `in`,
    `nin`, `gt`, `gte`, `lt`, `lte`, and `exists` with keys ANDed
    together, and leave OR out of v1. Filtering must be strict: the
    filter compiles to SQL and resolves to a candidate id set *before*
    scoring, with only those vectors loaded into memory. Post-filtering
    a global top-k returns nothing for a single chat whose records never
    reach the global head, which is exactly the Telegram case. An
    unknown metadata key is an empty result; a malformed operator or a
    non-scalar value is an error.
  - Ranking: once the filter resolves ids, dense scoring is a matrix
    product against the query vector, and `hybrid` tokenizes with
    `bm25.tokenize`, loads postings restricted to those ids, then fuses
    through `score_postings` and `rrf_fuse` unchanged. Compute the BM25
    `doc_count` and `avg_doc_len` over the filtered subset rather than
    the whole collection so idf and scoring agree on what the corpus is.
    Resolve the query embedding through the shared `embedding_cache`; the
    per-index `query_cache` layer is index-scoped and not worth
    duplicating here.
  - Reranker seam: extract the result-agnostic core of the three
    `_apply_*_rerank` functions into a helper that takes
    `(query, documents)` and returns ranked `(index, score)` pairs. The
    file path stays inside `_build_rerank_document` and collections pass
    record text instead. This is a behavior-preserving refactor guarded
    by the existing `tests/unit/test_search_service.py` cases, and it is
    the only change to a code path users already depend on.
  - Surface: `vexor/collection_store.py` for the SQLite layer beside
    `cache.py`, `vexor/services/collection_service.py` for orchestration,
    and a `VexorClient.collection(name)` handle exposing `upsert_many`,
    `delete_many`, `get`, `search`, `count`, and `drop`. `upsert_many` is
    the primary write path — one batched embedding call that skips
    records whose `text_hash` is unchanged, the same trick
    `_split_payloads_by_label` already uses for files — and single-record
    `upsert` delegates to it. Provider and model resolution goes through
    the existing `_resolve_settings` so config precedence matches every
    other entry point. Results are a
    `RecordResult(id, text, metadata, score)`.
  - Delivery: land the reranker refactor first as its own PR, then the
    store, service, and Python API with tests, then
    `vexor collection list|info|search|delete|drop` plus an
    `upsert --json -` NDJSON stdin path for bulk import from a database
    dump. Tests must cover strict pre-filtering (a record ranked first
    globally must not appear once filtered out), upsert idempotency,
    cascade deletes, and model or dimension mismatch errors. Hold MCP
    exposure: the server is path-scoped today, and deciding which
    collection an agent may reach is a separate design question.
- Flip the default ranking to hybrid retrieval (shipped opt-in behind
  `--rerank hybrid` in 0.25.0) once the benchmark confirms it beats
  dense-only across embedding models. Current `scripts/eval_hybrid.py`
  status on this repo: hybrid wins with the small local model but still
  trails a strong remote model (bge-m3) on MRR@10. Tune the fusion
  (RRF k, dense/BM25 weights, doc-length normalization) against a larger
  query set and more corpora first, and call the flip out in release
  notes since result ordering shifts for existing users.
- Publish an evaluation: token cost + answer quality of agent+Vexor vs
  grep-only workflows (30–50 QA tasks), feature the chart in the README.
  Benchmarks are what make these tools travel (see mgrep's launch).
  `scripts/eval_hybrid.py` and `scripts/eval_queries.jsonl` are the seed.

## P1 — Performance & experience

- `vexor watch`: background incremental indexing via a file watcher.
  Also removes the per-search full-directory `stat()` staleness scan,
  which is O(N) filesystem work on every query today.
- Replace SQLite vector blobs with `vectors.npy` + `metadata.json`
  (memmap) to reuse across searches.
- Extend the MCP lazy-start path to other CLI commands; agents may invoke
  the CLI dozens of times per session so startup latency multiplies.
- Dependency slimming: move document extractors (`pypdf`, `python-docx`,
  `python-pptx`) behind a `vexor[docs]` extra. They are already imported
  lazily, and cosine similarity now uses direct NumPy operations.
- Apple Silicon support for local embeddings (issue #7): CoreML/MPS
  execution provider for onnxruntime, or documented guidance.
- API performance improvements.
  - Adaptive embedding concurrency based on 429/timeout signals
    (in-process only; do not persist config changes).
  - Async embedding backends (AsyncOpenAI/Async Gemini) with asyncio
    concurrency to reduce thread overhead and improve connection reuse.
  - Adaptive embedding batch size for remote providers (guarded by safe
    min/max and backoff on 429/413).
  - Batch query search API to embed multiple queries per call and reuse
    loaded index vectors (reduce repeated I/O).

## P2 — Coverage & polish

- Add AST-aware `code` mode chunking for Go and Rust (tree-sitter support).
- Project-level config (`<project>/.vexor/config.json`).
  - v1 (implemented): restricted project overlays apply consistently across
    the CLI, Python API, and MCP, while config inspection and diagnostics
    report effective sources. See
    [Project configuration](configuration.md#project-configuration) for the
    schema and precedence contract.
  - v2 (only if v1 sees real use): per-project scan defaults (`mode`,
    `extensions`, `exclude_patterns`) — these are per-invocation CLI
    arguments today, not config fields, so supporting them means new
    config surface and CLI-default plumbing.
- Additional embedding providers (Azure).
- Evaluate an optional LLM reranker that reads a bounded set of retrieved
  candidates and judges their relevance to the query. Keep dense/BM25
  retrieval as the recall layer rather than treating a general-purpose LLM
  as the search engine; define token, latency, privacy, provider, and offline
  behavior before implementation, and benchmark it against existing BM25,
  FlashRank, and remote rerank paths.
  - Separately evaluate LLM-assisted query expansion, HyDE, and multi-step
    retrieval only if the reranker benchmark shows enough quality gain to
    justify the additional cost and complexity.
- OCR-backed head-mode snippets for images.
  - Preferred approach: integrate `rapidocr-onnxruntime` as the local OCR
    backend (pure Python + ONNX Runtime, good privacy story) with lazy
    initialization and per-file caching.
  - Open concern: current RapidOCR wheels require `numpy<2`. Until the
    upstream stack supports NumPy 2.x, keep OCR optional instead of
    enforcing the dependency.
- Evaluate migrating the similarity store to FAISS or another vector
  database for faster search and scalable metadata filtering.
- Official Vexor API relay service to offload local credentials and speed
  up indexing.
- VS Code extension integration (should reuse the MCP server rather than
  a bespoke protocol).

## GUI policy

- The desktop app was retired in 0.26 (release assets had stalled at
  0.19.0 with effectively zero downloads). The code is preserved on the
  `archive/gui` branch; the last shipped builds remain downloadable from
  old releases. A future graphical entry point should be the VS Code
  extension (see P2).

## Growth / distribution (non-code)

- Package for homebrew, scoop, and winget (standalone binaries already
  exist in releases).
- Distribution pushes are gated on the P0 evaluation chart: the shareable
  artifact is the benchmark, not an announcement post (what traveled in
  mgrep's launch was the chart). Once the chart is in the README, follow
  up with Ruan Yifeng's Weekly as the warm channel (Vexor appeared in
  issue #379); Show HN is optional. A launch post without new evidence
  is not worth writing — the always-on discovery channel is the MCP
  registry and skill directories, which are already live.
- README: add a comparison table vs mgrep / claude-context highlighting
  local-first, no account, provider-agnostic, reranking options.

## Engineering TODO

- Add a dev-only consistency test that validates the MCP tool
  `inputSchema` against the server-side argument validation (feed
  known-good/bad payloads through both), so the advertised schema and the
  strict validation cannot drift apart.
- Add a porcelain output contract test to CI so CLI flag or column
  changes cannot silently break scripts and agents that parse
  `--format porcelain` output.
- Make user-facing error handling more systematic.
  - Most messages are centralized in `text.py`, but several runtime
    validation paths still build detailed errors inline. Consider adding
    structured error helpers so CLI, API, and tests can rely on consistent
    wording and recovery guidance.
- Revisit provider adapter boundaries for OpenAI-compatible services.
  - Reusing the OpenAI-compatible backend for Voyage AI and custom
    providers is pragmatic. If more provider-specific request parameters
    appear, introduce dedicated adapter classes instead of adding more
    conditional logic inside the shared OpenAI backend.
