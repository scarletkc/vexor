# Vexor Skills

## Overview

Vexor is a vector-powered CLI for semantic file search. It supports configurable remote embedding models (Gemini, OpenAI) and ranks results by cosine similarity. The CLI is built with Python 3.9+, Typer for command handling, and Rich for terminal output.

## Project Structure

```
vexor/
├── __init__.py         # Package version
├── __main__.py         # Python -m entrypoint
├── cli.py              # Typer CLI commands (index, search, config, doctor, update)
├── cache.py            # Cache management for embeddings
├── config.py           # Configuration loading/saving (~/.vexor/config.json)
├── modes.py            # Index mode strategies (name, head, brief, full)
├── search.py           # Cosine similarity search logic
├── text.py             # User-facing messages and Rich styles
├── utils.py            # Path resolution, extension normalization
├── providers/          # Embedding provider adapters
│   ├── gemini.py       # Google Gemini API
│   └── openai.py       # OpenAI API
└── services/           # Business logic layers
    ├── cache_service.py
    ├── config_service.py
    ├── content_extract_service.py
    ├── index_service.py
    ├── keyword_service.py
    ├── search_service.py
    └── system_service.py
tests/
├── unit/               # Unit tests for helpers/services
└── integration/        # CLI integration tests
docs/
├── development.md
├── roadmap.md
└── workflow-diagram.md
```

## Development Commands

```bash
# Install in development mode with test dependencies
pip install -e .[dev]

# Run the CLI
python -m vexor --help
vexor --version

# Index a directory (creates embeddings cache)
vexor index --path . --mode name
vexor index --path . --mode head --include-hidden

# Search with semantic query
vexor search "config loader" --path . --mode name --top 5

# Configure API keys and providers
vexor config --set-api-key "YOUR_KEY"
vexor config --set-provider gemini
vexor config --set-model gemini-embedding-001
vexor config --show

# Run tests
pytest                                    # All tests
pytest tests/unit                        # Unit tests only
pytest tests/unit -k cache               # Tests matching pattern
pytest --cov=vexor --cov-report=term-missing  # With coverage

# Build distribution
python -m build
```

## Index Modes

- **name**: Embed only the file name (fastest, zero content reads)
- **head**: Extract first snippet of text/code/PDF/DOCX/PPTX files
- **brief**: Extract high-frequency keywords from file content (English/Chinese)
- **full**: Chunk entire contents into windows for long documents

## Configuration

Configuration is stored in `~/.vexor/config.json`. Embedding indexes are cached in `~/.vexor/cache/`.

Environment variables:
- `VEXOR_API_KEY`: Primary API key
- `GOOGLE_GENAI_API_KEY`: Gemini-specific key
- `OPENAI_API_KEY`: OpenAI-specific key

Supported providers: `gemini` (default), `openai`

## Code Conventions

- **Style**: PEP 8, 4-space indent, ~100-char lines
- **Naming**: `snake_case` for modules/functions, `PascalCase` for classes
- **Type hints**: Required for new code
- **CLI errors**: Use `typer.BadParameter` for validation errors
- **User messages**: Route through `text.py` (Messages class) for consistency
- **Rich styling**: Use `Styles` class from `text.py`

## Testing Guidelines

- Tests use fake provider fixtures (no network access required)
- Unit tests: Validate deterministic logic (cache, mode validation, extensions)
- Integration tests: Spawn CLI to verify Rich output, config flows
- Always include happy-path and failure coverage
- Test file naming: `test_<subject>.py`

## Key APIs

### CLI Commands
- `vexor index`: Create/refresh cached embeddings
- `vexor search`: Query cached index with semantic search
- `vexor config`: Manage API keys, models, providers
- `vexor doctor`: Check if vexor is on PATH
- `vexor update`: Check for newer releases

### Service Layer
- `IndexService.build_index()`: Create embeddings for directory
- `SearchService.perform_search()`: Execute semantic search
- `ConfigService.apply_config_updates()`: Update configuration

### Mode Strategies
- `NameStrategy`: File name only
- `HeadStrategy`: Name + content snippet
- `BriefStrategy`: Name + keyword summary
- `FullStrategy`: Chunked content windows

## Common Patterns

### Adding a new CLI option
1. Add option to command in `cli.py` with appropriate `typer.Option`
2. Add help text to `Messages` class in `text.py`
3. Pass through service layer
4. Add unit test for validation and integration test for CLI behavior

### Adding a new provider
1. Create adapter in `providers/` implementing embedding interface
2. Add to `SUPPORTED_PROVIDERS` in `config.py`
3. Update provider selection logic in services
4. Add fake provider fixture for tests

### Adding a new index mode
1. Create strategy class implementing `IndexModeStrategy` protocol in `modes.py`
2. Register in `_STRATEGIES` dict
3. Add unit tests for payload generation
