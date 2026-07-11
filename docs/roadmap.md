# Roadmap

Strategic note (2026-07 project review): position Vexor as retrieval
infrastructure for AI coding agents — local-first, no account required,
provider-agnostic — rather than only a human-facing search CLI. Agent
integrations (MCP, skills) are the primary distribution channel; the
differentiators against hosted competitors (mgrep, claude-context) are
"pip install and go", bring-your-own-key or fully offline, and data
never leaving the machine.

## P0 — Agent-first distribution

- Hybrid retrieval as a first-class path: fuse BM25 and dense scores
  (e.g. reciprocal rank fusion) during search instead of offering BM25
  only as an opt-in reranker. Dependencies (`rank-bm25`, `tokenizers`)
  are already present. Pure semantic search is weak on exact identifiers;
  hybrid is the ecosystem default now.
- Publish an evaluation: token cost + answer quality of agent+Vexor vs
  grep-only workflows (30–50 QA tasks), feature the chart in the README.
  Benchmarks are what make these tools travel (see mgrep's launch).

## P1 — Performance & experience

- `vexor watch`: background incremental indexing via a file watcher.
  Also removes the per-search full-directory `stat()` staleness scan,
  which is O(N) filesystem work on every query today.
- Replace SQLite vector blobs with `vectors.npy` + `metadata.json`
  (memmap) to reuse across searches.
- Lazy imports to cut CLI cold start (~0.7s measured) toward ~0.3s;
  agents may invoke the CLI dozens of times per session so startup
  latency multiplies.
- Dependency slimming: move document extractors (`pypdf`, `python-docx`,
  `python-pptx`) behind a `vexor[docs]` extra; replace the `scikit-learn`
  dependency with direct NumPy ops if cosine similarity is the only use.
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
- Support `.vexorignore` for per-project ignore rules.
- Project-level local cache (per-folder cache root override).
- Additional embedding providers (Azure).
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

- The desktop app is in maintenance mode: security/dependency bumps only,
  no new features. Rationale (2026-07 review): the growth channel is
  agents and terminal users; the GUI has no macOS build, no tests, and a
  fragile argv/porcelain contract with the CLI.
- Add a porcelain output contract test to CI so CLI flag or column
  changes cannot silently break the GUI.
- A future graphical entry point should be the VS Code extension.

## Growth / distribution (non-code)

- Package for homebrew, scoop, and winget (standalone binaries already
  exist in releases).
- Write a launch post when MCP + hybrid search ship (Show HN, Chinese
  dev community follow-up — Vexor appeared in Ruan Yifeng's Weekly #379).
- README: add a comparison table vs mgrep / claude-context highlighting
  local-first, no account, provider-agnostic, reranking options.

## Engineering TODO

- Align release version semantics across Python, plugin, and GUI packages.
  - Python/package releases can currently move ahead while the desktop GUI
    remains on its own version. This is workable, but release notes and
    asset naming should make the split explicit, or the GUI should get an
    independent documented release track.
- Split provider-specific config validation out of `config.py` if provider
  support keeps growing.
  - `config.py` now owns default model resolution, provider environment
    variables, base URLs, embedding dimension validation, and config
    persistence. Keep it stable for now, but consider moving provider
    capability metadata into a dedicated module before adding more
    provider-specific rules.
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
