"""Small helpers for running BigQuery queries and shaping their results."""

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
        logger.error("Error executing query: %s", e)
        return []


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
    # Redundant given the annotation, and kept anyway: callers are not all
    # type-checked, and a ValueError names the problem where an AttributeError
    # on .strftime would not.
    if isinstance(timestamp, datetime):  # pyright: ignore[reportUnnecessaryIsInstance]
        return f"'{timestamp.strftime('%Y-%m-%d')}'"
    raise ValueError("Timestamp must be a string or datetime object")
