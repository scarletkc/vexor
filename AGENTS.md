# Repository Instructions

## Non-negotiables

- Run Python commands inside the repository's existing virtual environment. Create `.venv` only
  when none exists; never install into the system interpreter.
- Fix root causes. Do not swallow errors or add silent fallbacks. A degraded path is acceptable
  only when it is deliberate and user-visible. When the root cause stays unclear, say so and name
  the missing information; extending `vexor doctor` or verbose output beats a speculative fix.
- Ask when requirements are ambiguous or inconsistent instead of inventing behavior. Do not ship
  temporary, placeholder, or demo-only implementations unless explicitly requested.
- Never commit API keys, private endpoints, or local credentials. Store them via Vexor's config
  commands, provider environment variables, or a git-ignored `.env`.

## Project map

- `vexor/` contains the runtime, `vexor/services/` the orchestration layer (including the MCP
  stdio server in `mcp_service.py`, exposed via `vexor mcp`), and `vexor/providers/` the provider
  adapters and capability metadata. The `voyageai` and `custom` providers route through the
  OpenAI-compatible adapter instead of dedicated modules.
- Tests mirror the product surface under `tests/unit/` and `tests/integration/`. Documentation
  lives in `docs/`; `docs/roadmap.md` is the source of truth for priorities, and the README stays
  lean and links into `docs/`.
- `plugins/vexor/skills/vexor-cli/SKILL.md` is the bundled standalone agent guide. Check it when
  commands or behavior change.

## Implementation

- Follow PEP 8 with 4-space indentation and roughly 100-character lines. Type-annotate new code
  and keep Typer command names lowercase and imperative.
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
- `python -m ruff check .` must pass; it gates the release job. Suppress a rule only with a
  comment saying why, and never by widening the ignore list to hide a real finding.
- Cover relevant success and failure paths, especially optional extras such as `flashrank` and
  platform-specific shell behavior. Use fixtures or stubs and keep provider and network
  interactions mocked in the offline suite. Assert behavior or structured output rather than
  brittle Rich formatting.
- Before merging, run `python -m pytest --cov=vexor --cov-report=term-missing`. There is no hard
  gate, but keep core logic coverage at or above 90%.
- Changes to providers, indexing, search, or MCP require a real end-to-end check using configured
  credentials or the local provider. Record the result in the PR.
- Exercise the affected command for CLI changes (`python -m vexor --help` at minimum). Use
  `python -m build` for packaging changes, `vexor.spec` for standalone binaries, and
  `scripts/bump_version.py` for version bumps.

## Delivery

- Work on a branch named `type/short-slug` and land changes through a PR, not directly on `main`.
- Commit subjects and PR titles follow Conventional Commits, use the imperative mood, stay under
  ~72 characters, and mark breaking changes with `!`. The validation workflow in
  `.github/workflows/conventional-commits.yml` is authoritative for accepted types.
- PRs explain motivation and verification, link relevant issues, and include terminal output when
  CLI output changes. Call out compatibility-sensitive config, cache, Python API, provider,
  reranker, or bundled-skill changes.
- A release may carry a hand-written section in `docs/release-notes/<version>.md`, published above
  the generated changelog. It is optional, but a heading-only file fails the publish job; see
  `docs/development.md`.
- Update the documentation affected by a behavior or workflow change. The bundled skill is the
  easiest copy site to forget.
