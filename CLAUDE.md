# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RepoRoulette is a Python library for randomly sampling GitHub repositories using multiple methods: ID-based sampling, temporal sampling, BigQuery integration, and GitHub Archive processing. The library provides different strategies for collecting representative samples of repositories for research and analysis.

## Development Commands

This project uses **uv** for dependency management and **uv_build** for the build backend - 2025's cutting-edge Python tooling. Make sure you have uv installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Installation and Setup
```bash
# Install all dependencies (including dev, docs, and bigquery extras)
uv sync --all-extras --group dev

# Install specific extras only
uv sync --extra bigquery          # For BigQuery sampler
uv sync --extra docs              # For documentation
uv sync --group dev               # For development tools

# Install and run without syncing
uv run --extra bigquery python -c "import reporoulette"
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_id_sampler.py
uv run pytest tests/test_gharchive_sampler.py
uv run pytest tests/test_bq_sampler.py
uv run pytest tests/test_temporal_sampler.py

# Skip live-credential tests (BigQuery live tests are marked "integration")
uv run pytest -m "not integration"

# Run tests with coverage
uv run pytest --cov=reporoulette --cov-report=xml
```

### Linting and Formatting
```bash
# Run Ruff linting (used in CI)
uv run ruff check --select=E9,F63,F7,F82 .  # Critical errors only
uv run ruff check .                         # Full linting

# Format code
uv run ruff format .

# Check formatting without changes
uv run ruff format --check .

# Auto-fix linting issues
uv run ruff check --fix .
```

### Type Checking and Docstring Linting
```bash
# Run pyright type checking
uv run pyright

# Run pydoclint for docstring validation (excludes tests)
uv run pydoclint reporoulette/

# Run both type checking and docstring linting
uv run pyright && uv run pydoclint reporoulette/
```

### Building and Distribution
```bash
# Build the package with uv_build backend (creates wheel and source distribution)
uv build

# Build specific formats
uv build --wheel      # Build wheel only
uv build --sdist      # Build source distribution only

# Install from local build
uv pip install dist/reporoulette-*.whl
```

### Build System
This project uses the **uv_build** backend for ultra-fast builds (10-35x faster than setuptools). The flat package layout is configured in `pyproject.toml`:

```toml
[build-system]
requires = ["uv_build>=0.9.11,<0.10.0"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-root = ""
module-name = "reporoulette"
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks (run once)
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files

# Update pre-commit hooks
uv run pre-commit autoupdate
```

## Architecture

### Core Structure
- **Base Classes**: `reporoulette/samplers/base.py` defines `BaseSampler` abstract class with common functionality
- **Sampler Implementations**: Four main sampling strategies in `reporoulette/samplers/`
- **Main Module**: `reporoulette/__init__.py` provides unified interface and convenience functions

### Sampling Methods

1. **IDSampler** (`id_sampler.py`): Probes random repository IDs within GitHub's sequential ID space
2. **TemporalSampler** (`temporal_sampler.py`): Samples repositories by random time periods using GitHub API
3. **BigQuerySampler** (`bigquery_sampler.py`): Uses Google BigQuery's GitHub dataset for advanced queries
4. **GHArchiveSampler** (`gh_sampler.py`): Processes GitHub Archive files for event-based sampling

### Key Design Patterns
- All samplers inherit from `BaseSampler` and implement the `sample()` method
- Consistent configuration through constructors (token, seed, logging)
- Built-in filtering capabilities (`_filter_repos()` method)
- Comprehensive logging and success rate tracking
- Rate limiting and error handling for API interactions

### Dependencies
- **Core**: `requests` for HTTP operations
- **Optional**: `google-cloud-bigquery`, `google-auth` for BigQuery functionality
- **Development**: `pytest`, `pytest-cov`, `ruff`, `pre-commit` for testing and code quality
- **Documentation**: `sphinx`, `furo`, `myst-parser` for building docs

### Configuration
- GitHub tokens via `GITHUB_TOKEN` environment variable
- Google Cloud credentials via `GOOGLE_APPLICATION_CREDENTIALS`
- Logging configuration in `pyproject.toml` under `[tool.reporoulette.logging]`

## Working with the Codebase

### Adding New Samplers
1. Inherit from `BaseSampler` in `reporoulette/samplers/base.py`
2. Implement the abstract `sample()` method
3. Add appropriate error handling and logging
4. Update `__init__.py` files to export the new sampler
5. Add comprehensive tests in `tests/`

### Testing Guidelines
- Each sampler has dedicated test files in `tests/`
- Tests run offline with mocked HTTP/BigQuery clients; only tests marked
  `@pytest.mark.integration` need live credentials (stored in CI secrets)
- Mock external API calls when appropriate
- Test both success and failure scenarios

### Common Patterns
- Use `self.logger` for consistent logging across samplers
- Implement seed support for reproducible sampling
- Handle rate limiting gracefully with configurable safety margins
- Return standardized repository data structures
- Track success rates and attempt counts

### Known Limitations / Follow-ups
- `IDSampler.max_id` defaults to a dated constant (`DEFAULT_MAX_ID`,
  measured 2026-07); call `update_max_id()` (one search API call) or pass
  `max_id` explicitly for full coverage of the newest repositories.
- GitHub's Events API payload change of 2025-10-07 removed repository
  CreateEvents from the public feed: GHArchiveSampler's default population
  is empty for days after that date.
- BigQuery cost control: `scripts/validate_randomization.py --p7` dry-runs
  every query shape for free and reports the bytes each would scan; use it
  before any paid BigQuery run (`sample_active` scans ~100 GB per run).
