# Repository Guidelines

## Project Structure & Module Organization
Runtime code sits inside `vexor/`: Typer entrypoints (`cli.py`, `__main__.py`) delegate to service layers (`services/index_service.py`, `services/search_service.py`, `services/config_service.py`, `services/skill_service.py`) and shared helpers (`cache.py`, `config.py`, `text.py`, `utils.py`, `modes.py`). Provider adapters for Gemini/OpenAI/Local live under `providers/`. Documentation (`docs/development.md`, `docs/roadmap.md`, `docs/workflow-diagram.md`) and assets (`assets/`) explain workflows and future work, while build outputs only appear in `dist/`. Tests mirror the structure—`tests/unit/` covers helpers/services and `tests/integration/` drives CLI runs such as `test_cli.py` and `test_end_to_end.py`.

## Build, Test, and Development Commands
- `pip install -e .[dev]` installs Vexor plus pytest, coverage, and packaging helpers.
- `python -m vexor --help` smoke-tests the CLI; use `vexor index --path . --mode head` and `vexor search "config loader" --path . --mode name` for realistic runs.
- `pytest` stays offline via fake providers; scope down with `pytest tests/unit -k cache` and add `--maxfail=1 -ra` for quick failures.
- `pytest --cov=vexor --cov-report=term-missing` should pass before merging to hold the Codecov badge steady.
- `python -m build` (wheel+sdist) produces release artifacts mirrored by CI.

## Coding Style & Naming Conventions
Follow PEP 8 (4-space indent, ~100-char lines). Use `snake_case` for modules/functions, `PascalCase` for classes, and lowercase imperative Typer command names. Type-annotate new code, surface CLI validation errors via `typer.BadParameter`, and route user-facing copy through `text.py` so Rich styling remains consistent. Tests live in `test_<subject>.py`, prefer fixtures/stubs to real APIs, and assert structured output instead of console strings.

## Testing Guidelines
Unit suites validate deterministic logic (cache pruning, mode validation, extension handling), while integration suites spawn the CLI to verify Rich tables, config flows, and search/index orchestration. Keep fake provider fixtures current so runs remain offline. Pair every change with happy-path and failure coverage (invalid mode, missing config, empty query) and add regressions whenever cache or provider layers change.

## Commit & Pull Request Guidelines
History favors concise, imperative commit subjects (“Add vexor logo image”). PRs should explain motivation, list exercised commands or screenshots, link issues, and confirm `pytest --cov` output. Call out config-path, cache-schema, or provider changes so reviewers can check backward compatibility, and update README/docs when behavior shifts.

## Security, Configuration & Maintenance
Never commit API keys or provider endpoints; rely on `vexor config --set-api-key`, provider env vars, or ignored `.env` files. Cache/config data lives in `~/.vexor`—reset with `vexor config --clear-index-all` when troubleshooting stale indexes. Sanitize filesystem paths, treat embedding outputs as untrusted input before writing to disk, and update README/docs/this guide whenever workflows change.
