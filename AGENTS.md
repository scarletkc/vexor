# Repository Guidelines

## Project Structure & Module Organization
Vexor’s runtime code lives under `vexor/`, with Typer entrypoints in `cli.py`/`__main__.py`, caching primitives in `cache.py`, provider adapters in `providers/`, search/index services in `services/`, and shared helpers in `utils.py` plus `text.py`. Tests mirror this layout—`tests/unit/` covers pure functions and services, while `tests/integration/` drives CLI, config, and end-to-end flows. Visual assets stay in `assets/`, narrative docs in `docs/`, and `dist/` only appears when you build release artifacts.

## Build, Test, and Development Commands
- `pip install -e .[dev]` installs the package plus pytest, coverage, and packaging utilities.
- `python -m vexor --help` exercises the local CLI, while `vexor search ...` or `vexor index ...` run once the entry point is on PATH.
- `pytest` runs the entire suite using fake embedding backends (no network); use `pytest tests/unit -k cache` for targeted checks and `pytest --cov=vexor --cov-report=term-missing` before shipping.
- `python -m build` produces sdist and wheel files in `dist/`, matching the publish workflow.

## Coding Style & Naming Conventions
Follow PEP 8 spacing (4-space indents, ~100-character lines) and keep modules/functions in `snake_case`, classes in `PascalCase`, and Typer commands in imperative form (`search`, `index`). Type-hint new code, raise `typer.BadParameter` for CLI validation, and keep user-facing strings in `text.py` for consistent Rich styling. Tests adopt `test_<subject>.py` names and should rely on fixtures or stubs instead of real API calls.

## Testing Guidelines
Pytest is the only framework: unit suites guard services/utilities, and integration suites exercise Typer flows under `tests/integration/test_cli.py` and `test_end_to_end.py`. Keep the fake providers current so runs stay offline. When adding features, include regression plus happy-path coverage and ensure the Codecov badge trend does not fall—open PRs should never reduce overall coverage.

## Commit & Pull Request Guidelines
Commits in history use concise, imperative subjects (“Add vexor logo image”), so follow that style and keep summaries under ~60 characters. Each PR should describe motivation, list key commands or screenshots for CLI output, link issues, and confirm `pytest --cov` was run. Flag configuration or cache-path changes so reviewers can double-check backward compatibility.
Keep the README, documentation, and this file up to date.

## Security & Configuration Tips
Never hard-code API keys or base URLs; rely on `vexor config --set-api-key`, provider-specific env vars, or `.env` files ignored by Git. Cache/config data live in `~/.vexor`; clean with `vexor config --clear-index-all` when debugging. Treat embedding providers as untrusted inputs—sanitize filesystem paths before writing and prefer defensive checks around remote responses.