# Development
Run development tasks inside the project virtualenv, creating one only when the
repository has none:
```bash
python -m venv .venv          # only if .venv is missing
python -m pip install -e .[dev]
python -m vexor
python -m pytest
```
Tests rely on fake embedding backends, so no network access is required.

Global cache files and configuration live in `~/.vexor`. A project with a
`.vexor/` directory keeps its index there and may add the restricted
`config.json` overlay documented in `docs/configuration.md`. The text embedded
for each chunk is built by the mode strategies in `vexor/modes.py`; adjust the
strategy `label` construction there if you need to encode additional context.
