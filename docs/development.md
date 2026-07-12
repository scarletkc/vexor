# Development
Run development tasks with:
```bash
pip install -e .[dev]
python -m vexor
pytest
```
Tests rely on fake embedding backends, so no network access is required.

Cache files and configuration live in `~/.vexor` (a project with a `.vexor/` directory keeps its index there instead). The text embedded for each chunk is built by the mode strategies in `vexor/modes.py`; adjust the strategy `label` construction there if you need to encode additional context.