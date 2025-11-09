# Development
Run development tasks with:
```bash
pip install -e .[dev]
python -m vexor
pytest
```
Tests rely on fake embedding backends, so no network access is required.

Cache files and configuration live in `~/.vexor`. Adjust `_label_for_path` or `VexorSearcher._prepare_text` if you need to encode additional context (e.g., relative paths).