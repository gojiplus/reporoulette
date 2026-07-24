import logging
from datetime import datetime
from typing import Any


def execute_query(
    client: Any, query: str, logger: logging.Logger
) -> list[dict[str, Any]]:
    """Execute a BigQuery query and return results as a list of dictionaries.

    Args:
        client: BigQuery client instance
        query: SQL query string to execute
        logger: Logger instance for error reporting

    Returns:
        Query results as list of dictionaries, empty list on error
    """
    try:
        query_job = client.query(query)
        results = query_job.result()
        # Convert each row to a dict (depending on your client, adjust as needed)
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return []


def _owner_login(repo: dict[str, Any]) -> str | None:
    """Return the owner login for either API-shaped or BigQuery-shaped repo dicts.

    Args:
        repo: Repository dictionary

    Returns:
        Owner login string, or None if absent
    """
    owner: Any = repo.get("owner")
    if isinstance(owner, dict):
        owner = owner.get("login")
    return owner if isinstance(owner, str) else None


def filter_repos(
    repos: list[dict[str, Any]],
    logger: logging.Logger | None = None,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Filter repositories based on provided criteria.

    Supported filters: min_stars (stargazers_count >= value), min_forks
    (forks_count >= value), languages (case-insensitive membership of the
    repo's language), language (single-language shorthand), owner (exact
    owner login). If no repository in the input carries the field a filter
    needs (e.g. BigQuery results have no star or language data), that
    filter is skipped with a warning instead of dropping every repository.
    Unknown filter names are ignored with a warning.

    Args:
        repos: List of repository dictionaries to filter
        logger: Logger for warnings; defaults to the module logger
        **filters: Key-value pairs for filtering criteria

    Returns:
        Filtered list of repositories
    """
    log = logger or logging.getLogger(__name__)
    filtered = repos

    threshold_fields = {"min_stars": "stargazers_count", "min_forks": "forks_count"}
    for key, field in threshold_fields.items():
        if key not in filters:
            continue
        if not any(field in repo for repo in filtered):
            log.warning(f"Filter '{key}' skipped: no repository has a '{field}' field")
            continue
        threshold = filters[key]
        filtered = [r for r in filtered if r.get(field, 0) >= threshold]

    wanted_languages: list[Any] = list(filters.get("languages") or [])
    if "language" in filters:
        wanted_languages.append(filters["language"])
    if wanted_languages:
        if not any("language" in repo for repo in filtered):
            log.warning(
                "Filter 'languages' skipped: no repository has a 'language' field"
            )
        else:
            wanted = {str(lang).lower() for lang in wanted_languages}
            filtered = [
                r
                for r in filtered
                if r.get("language") and str(r["language"]).lower() in wanted
            ]

    if "owner" in filters:
        filtered = [r for r in filtered if _owner_login(r) == filters["owner"]]

    known = {"min_stars", "min_forks", "languages", "language", "owner"}
    for key in filters:
        if key not in known:
            log.warning(f"Unknown filter '{key}' ignored")

    return filtered


def format_timestamp_query(timestamp: str | datetime) -> str:
    """Format a timestamp (string or datetime) for use in a SQL query.

    Args:
        timestamp: Timestamp as string or datetime object

    Returns:
        Formatted timestamp string for SQL queries

    Raises:
        ValueError: If timestamp is not a string or datetime object
    """
    if isinstance(timestamp, str):
        return f"'{timestamp}'"
    elif isinstance(timestamp, datetime):
        return f"'{timestamp.strftime('%Y-%m-%d')}'"
    else:
        raise ValueError("Timestamp must be a string or datetime object")
