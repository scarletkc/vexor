# Roadmap

Strategic note (2026-07 project review): position Vexor as retrieval
infrastructure for AI coding agents — local-first, no account required,
provider-agnostic — rather than only a human-facing search CLI. Agent
integrations (MCP, skills) are the primary distribution channel; the
differentiators against hosted competitors (mgrep, claude-context) are
"pip install and go", bring-your-own-key or fully offline, and data
never leaving the machine.

## P0 — Agent-first distribution

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
  - v1 (shipped): the nearest project marker may override `rerank`,
    `auto_index`, `model`, `embedding_dimensions`, `batch_size`,
    `embed_concurrency`, and `extract_concurrency`. The strict allowlist
    rejects credentials, endpoints, and every other field. Precedence is
    global config < project config < environment overrides < explicit
    arguments, and `vexor config --show` plus `vexor doctor` report each
    effective field's origin. Mutating config commands remain global-only.
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
