# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RepoRoulette is a Python library for randomly sampling GitHub repositories using multiple methods: ID-based sampling, temporal sampling, BigQuery integration, and GitHub Archive processing. The library provides different strategies for collecting representative samples of repositories for research and analysis.

## Development Commands

This project uses **uv** for dependency management and project workflows. Make sure you have uv installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

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
uv run pytest reporoulette/tests/test_id_sampler.py
uv run pytest reporoulette/tests/test_gharchive_sampler.py
uv run pytest reporoulette/tests/test_bq_sampler.py
uv run pytest reporoulette/tests/test_temporal_sampler.py

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

### Building and Distribution
```bash
# Build the package (creates wheel and source distribution)
uv build

# Build specific formats
uv build --wheel      # Build wheel only
uv build --sdist      # Build source distribution only

# Install from local build
uv pip install dist/reporoulette-*.whl
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
5. Add comprehensive tests in `reporoulette/tests/`

### Testing Guidelines
- Each sampler has dedicated test files in `reporoulette/tests/`
- Tests require API credentials (stored in CI secrets)
- Mock external API calls when appropriate
- Test both success and failure scenarios

### Common Patterns
- Use `self.logger` for consistent logging across samplers
- Implement seed support for reproducible sampling
- Handle rate limiting gracefully with configurable safety margins
- Return standardized repository data structures
- Track success rates and attempt counts
