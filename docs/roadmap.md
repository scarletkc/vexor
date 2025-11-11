# Roadmap
- Full-document chunked indexing for long files.
- Index and search only files with the specified file extension.
- Keyword-focused indexing mode for requirements/PRDs.
- OCR-backed head-mode snippets for images; investigate lighter-weight Python OCR engines versus hosted OCR APIs while balancing install size, cost, and privacy expectations.
- Additional embedding providers (Azure, local backends).
- Evaluate migrating the similarity store to FAISS or another vector database for faster search and scalable metadata filtering.
- Official Vexor API relay service to offload local credentials and speed up indexing.
