# Roadmap
- OCR-backed head-mode snippets for images.
  - Preferred approach: integrate `rapidocr-onnxruntime` as the local OCR backend (pure Python + ONNX Runtime, good privacy story) with lazy initialization and per-file caching.
  - Open concern: current RapidOCR wheels require `numpy<2`, which conflicts with allowing newer NumPy versions. Until the upstream stack supports NumPy 2.x, we plan to keep OCR optional instead of enforcing the dependency.
  - OCR (image head/full fragment) is still under development. We prefer to integrate with the local `rapidocr-onnxruntime` to avoid uploading the code to the cloud, but this dependency currently requires `numpy<2`. Until RapidOCR/ONNX Runtime officially supports NumPy 2.x, the default distribution will not force OCR to be enabled, to avoid blocking users from upgrading NumPy.
- Additional embedding providers (Azure). 
- Add AST-aware `code` mode chunking for Go and Rust (tree-sitter support).
- Evaluate migrating the similarity store to FAISS or another vector database for faster search and scalable metadata filtering.
- Official Vexor API relay service to offload local credentials and speed up indexing.
