# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RepoRoulette is a Python library for randomly sampling GitHub repositories using multiple methods: ID-based sampling, temporal sampling, BigQuery integration, and GitHub Archive processing. The library provides different strategies for collecting representative samples of repositories for research and analysis.

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest reporoulette/tests/test_id_sampler.py
pytest reporoulette/tests/test_gharchive_sampler.py
pytest reporoulette/tests/test_bq_sampler.py
pytest reporoulette/tests/test_temporal_sampler.py

# Run tests with coverage
pytest --cov=reporoulette
```

### Linting
```bash
# Run flake8 linting (used in CI)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Run with development dependencies (if available)
black reporoulette/  # Code formatting
isort reporoulette/  # Import sorting
```

### Installation
```bash
# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e .[bigquery]  # For BigQuery sampler
pip install -e .[dev]       # For development tools
pip install -e .[docs]      # For documentation
```

### Building and Distribution
```bash
# Build the package
python -m build

# Install dependencies for testing
pip install flake8 pytest google-cloud-bigquery db-dtypes
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
- **Development**: `pytest`, `black`, `isort`, `flake8` for testing and code quality

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