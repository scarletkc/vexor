# Roadmap
- OCR-backed head-mode snippets for images.
  - Preferred approach: integrate `rapidocr-onnxruntime` as the local OCR backend (pure Python + ONNX Runtime, good privacy story) with lazy initialization and per-file caching.
  - Open concern: current RapidOCR wheels require `numpy<2`, which conflicts with allowing newer NumPy versions. Until the upstream stack supports NumPy 2.x, we plan to keep OCR optional instead of enforcing the dependency.
  - OCR (image head/full fragment) is still under development. We prefer to integrate with the local `rapidocr-onnxruntime` to avoid uploading the code to the cloud, but this dependency currently requires `numpy<2`. Until RapidOCR/ONNX Runtime officially supports NumPy 2.x, the default distribution will not force OCR to be enabled, to avoid blocking users from upgrading NumPy.
- Additional embedding providers (Azure). 
- Add AST-aware `code` mode chunking for Go and Rust (tree-sitter support).
- TODO: Search performance improvements.
  - Replace SQLite vector blobs with `vectors.npy` + `metadata.json` (memmap) to reuse across searches.
- TODO: API performance improvements.
  - Adaptive embedding concurrency based on 429/timeout signals (in-process only; do not persist config changes).
  - Async embedding backends (AsyncOpenAI/Async Gemini) with asyncio concurrency to reduce thread overhead and improve connection reuse.
  - Adaptive embedding batch size for remote providers (guarded by safe min/max and backoff on 429/413).
  - Batch query search API to embed multiple queries per call and reuse loaded index vectors (reduce repeated I/O).
- Evaluate migrating the similarity store to FAISS or another vector database for faster search and scalable metadata filtering.
- Official Vexor API relay service to offload local credentials and speed up indexing.
- VS Code extension integration.
- Project-level local cache (per-folder cache root override).
- Support `.vexorignore` for per-project ignore rules.

## Engineering TODO
- Align release version semantics across Python, plugin, and GUI packages.
  - Python/package releases can currently move ahead while the desktop GUI remains on its own version. This is workable, but release notes and asset naming should make the split explicit, or the GUI should get an independent documented release track.
- Split provider-specific config validation out of `config.py` if provider support keeps growing.
  - `config.py` now owns default model resolution, provider environment variables, base URLs, embedding dimension validation, and config persistence. Keep it stable for now, but consider moving provider capability metadata into a dedicated module before adding more provider-specific rules.
- Make user-facing error handling more systematic.
  - Most messages are centralized in `text.py`, but several runtime validation paths still build detailed errors inline. Consider adding structured error helpers so CLI, API, and tests can rely on consistent wording and recovery guidance.
- Revisit provider adapter boundaries for OpenAI-compatible services.
  - Reusing the OpenAI-compatible backend for Voyage AI and custom providers is pragmatic. If more provider-specific request parameters appear, introduce dedicated adapter classes instead of adding more conditional logic inside the shared OpenAI backend.
