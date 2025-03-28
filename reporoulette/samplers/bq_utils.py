import logging
from typing import List, Dict, Any, Union
from datetime import datetime

def execute_query(client, query: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Execute a BigQuery query and return results as a list of dictionaries.
    """
    try:
        query_job = client.query(query)
        results = query_job.result()
        # Convert each row to a dict (depending on your client, adjust as needed)
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return []

def filter_repos(repos: List[Dict[str, Any]], **filters) -> List[Dict[str, Any]]:
    """
    Filter repositories based on provided criteria.
    
    This is a simple implementation. Extend as needed.
    """
    filtered = repos
    for key, value in filters.items():
        filtered = [repo for repo in filtered if repo.get(key) == value]
    return filtered

def format_timestamp_query(timestamp: Union[str, datetime]) -> str:
    """
    Format a timestamp (string or datetime) for use in a SQL query.
    """
    if isinstance(timestamp, str):
        return f"'{timestamp}'"
    elif isinstance(timestamp, datetime):
        return f"'{timestamp.strftime('%Y-%m-%d')}'"
    else:
        raise ValueError("Timestamp must be a string or datetime object")
