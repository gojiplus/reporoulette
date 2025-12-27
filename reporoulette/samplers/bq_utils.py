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


def filter_repos(repos: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    """Filter repositories based on provided criteria.

    Args:
        repos: List of repository dictionaries to filter
        **filters: Key-value pairs for filtering criteria

    Returns:
        Filtered list of repositories

    Note:
        This is a simple implementation that filters by exact matches.
    """
    filtered = repos
    for key, value in filters.items():
        filtered = [repo for repo in filtered if repo.get(key) == value]
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
