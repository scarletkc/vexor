# Repository Guidelines

## Project Structure & Module Organization
Core runtime lives in `vexor/`. CLI entrypoints (`cli.py`, `__main__.py`) sit alongside the public Python API (`api.py`), search/result models (`search.py`), and shared helpers (`cache.py`, `config.py`, `modes.py`, `output.py`, `text.py`, `utils.py`). Service orchestration now spans `vexor/services/` for indexing, search, config, cache inspection, content extraction, keyword extraction, JS parsing, init/setup flows, system diagnostics, and skill installation. Provider adapters live under `vexor/providers/` for Gemini, OpenAI-compatible backends, and local embeddings; `voyageai` and `custom` still flow through the OpenAI-compatible path. Docs live in `docs/`, including the Python API guide at `docs/api/python.md`. Optional desktop app sources live in `gui/` (Vue + Electron), bundled agent assets live in `plugins/vexor/` and `plugins/vexor/skills/vexor-cli/`, and release helpers live in `scripts/` plus `vexor.spec`. Generated artifacts may appear in `build/` and `dist/`. Tests mirror the product surface under `tests/unit/` and `tests/integration/`.

## Build, Test, and Development Commands
- `python3 -m pip install -e .[dev]` installs Vexor plus pytest, coverage, build, twine, and PyInstaller helpers.
- `python3 -m vexor --help` is the basic CLI smoke test once runtime deps are installed; realistic checks include `python3 -m vexor search "api client config" --path . --mode auto --no-cache` and `python3 -m vexor index --path . --mode code`.
- `python3 -m pytest` is the main offline test command; narrow scope with `python3 -m pytest tests/unit -k cache` or `python3 -m pytest tests/integration/test_cli.py -k doctor`.
- `python3 -m pytest --cov=vexor --cov-report=term-missing` should pass before merging to keep coverage steady.
- `python3 -m build` creates the wheel and sdist in `dist/`; release helpers also live in `scripts/bump_version.py`.
- `cd gui && npm install && npm run dev` starts the desktop app in dev mode; `npm run build`, `npm run package`, and `npm run make` cover frontend packaging work.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and roughly 100-character lines. Use `snake_case` for modules/functions, `PascalCase` for classes, and lowercase imperative Typer command names. Type-annotate new Python code, keep CLI validation errors consistent with `typer.BadParameter` where appropriate, and route user-facing copy through `text.py` so CLI messaging stays centralized. Prefer structured rendering paths over ad hoc print strings, especially for Rich tables and porcelain output. Tests should live in `test_<subject>.py`, use fixtures/stubs instead of real APIs, and assert behavior or structured output rather than brittle console formatting.

## Testing Guidelines
Unit coverage now spans cache behavior and upgrades, API/runtime config, content extraction, index/search orchestration, init flows, skill install helpers, plugin manifests, and system/update helpers. Integration suites drive the Typer CLI for search/index/config flows plus porcelain output, local setup, doctor/update, alias creation, star/feedback fallbacks, and end-to-end indexing behavior. Keep provider and network interactions mocked so tests stay offline. Pair each change with happy-path and failure coverage, especially for invalid modes, empty queries, malformed config, stale or missing indexes, optional extras such as `flashrank`, and platform-specific shell behavior.

## Commit & Pull Request Guidelines
History favors concise, imperative commit subjects (for example, `Add vexor logo image`). PRs should explain motivation, list the commands/tests exercised, link issues, and include screenshots when CLI/GUI output changes. Call out config-schema, cache-schema, Python API, bundled skill, or provider/reranker changes so reviewers can check compatibility. Update `README.md`, docs, plugin metadata, and this guide whenever behavior or contributor workflow changes.

## Security, Configuration & Maintenance
Never commit API keys, private proxy endpoints, or local credentials; use `vexor config --set-api-key`, provider env vars, or ignored `.env` files. Persistent data lives under `~/.vexor`: config/cache data in the main directory, FlashRank assets under `~/.vexor/flashrank`, and local embedding models under `~/.vexor/models`. Sanitize filesystem paths, extension filters, and exclude patterns before using them. Treat extracted document text, embedding payloads, and remote rerank/provider responses as untrusted input before writing to disk or surfacing them elsewhere. Keep docs, bundled skills, and release metadata in sync when commands, packaging, or setup flows change.
