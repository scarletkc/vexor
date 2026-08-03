# Repository Instructions

## Non-negotiables

- Run Python commands inside the repository's existing virtual environment. Create `.venv` only
  when none exists; never install into the system interpreter.
- Fix root causes. Do not swallow errors or add silent fallbacks. A degraded path is acceptable
  only when it is deliberate and user-visible.
- Ask when requirements are ambiguous or inconsistent instead of inventing behavior. Do not ship
  temporary, placeholder, or demo-only implementations unless explicitly requested.
- Never commit API keys, private endpoints, local credentials, or local `.env` files. Use Vexor's
  config commands, provider environment variables, or ignored `.env` files.

## Project map

- `vexor/` contains the runtime, `vexor/services/` the orchestration layer, and
  `vexor/providers/` the provider adapters and capability metadata.
- Tests mirror the product surface under `tests/unit/` and `tests/integration/`. Documentation
  lives in `docs/`; `docs/roadmap.md` is the source of truth for priorities.
- `plugins/vexor/skills/vexor-cli/SKILL.md` is the bundled standalone agent guide. Check it when
  commands or behavior change.

## Implementation

- Follow PEP 8 with 4-space indentation and roughly 100-character lines. Type-annotate new code
  and use the project's existing naming conventions.
- Route user-facing CLI copy through `vexor/text.py`. Reuse the services layer, structured output
  paths, and shared helpers before adding new logic; extract the smallest useful helper when
  behavior is duplicated.
- Use `typer.BadParameter` for CLI validation where appropriate. Keep human and porcelain output
  separate, and treat machine-readable output as a compatibility contract.
- Treat filesystem paths, extracted content, embedding payloads, and provider or reranker
  responses as untrusted input.

## Verification

- Use `docs/development.md` for setup. `python -m pytest` is the main offline test command; run
  focused tests while developing and the full suite before merging.
- Cover relevant success and failure paths. Use fixtures or stubs and keep provider and network
  interactions mocked in the offline suite. Assert behavior or structured output rather than
  brittle Rich formatting.
- Before merging, run `python -m pytest --cov=vexor --cov-report=term-missing` and inspect the
  report for obvious regressions.
- Changes to providers, indexing, search, or MCP require a real end-to-end check using configured
  credentials or the local provider. Record the result in the PR.
- Run `python -m vexor --help` for CLI changes and `python -m build` for packaging or release
  changes. Use `scripts/bump_version.py` for version bumps.

## Delivery

- Work on a branch and land changes through a PR, not directly on `main`.
- Commit subjects and PR titles follow Conventional Commits. The validation workflow in
  `.github/workflows/conventional-commits.yml` is authoritative for accepted types.
- PRs explain motivation and verification, link relevant issues, and include terminal output when
  CLI output changes. Call out compatibility-sensitive config, cache, Python API, provider,
  reranker, or bundled-skill changes.
- Update the documentation affected by a behavior or workflow change. The bundled skill is the
  easiest copy site to forget.
