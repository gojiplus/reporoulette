"""
RepoRoulette: Randomly Sample GitHub Repositories

A Python library for randomly sampling GitHub repositories using multiple methods:
- ID-based sampling: Probes random repository IDs
- Temporal sampling: Weighted sampling based on repository activity by time period
- BigQuery sampling: Advanced querying using Google BigQuery's GitHub dataset
- GitHub Archive sampling: Event-based sampling from GitHub Archive files

Example:
    >>> from reporoulette import sample
    >>> results = sample(method='temporal', n_samples=10)
    >>> print(f"Found {len(results['samples'])} repositories")
"""
# reporoulette/__init__.py
import logging
import os
from typing import Optional, Dict, Any

from .samplers.id_sampler import IDSampler
from .samplers.temporal_sampler import TemporalSampler
from .samplers.bigquery_sampler import BigQuerySampler
from .samplers.gh_sampler import GHArchiveSampler

__version__ = '0.1.1'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def sample(
    method: str = 'temporal',
    n_samples: int = 50,
    token: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Sample repositories using the specified method.

    Args:
        method: Sampling method ('id', 'temporal', or 'bigquery')
        n_samples: Number of repositories to sample
        token: GitHub Personal Access Token (not used for BigQuery)
        **kwargs: Additional parameters specific to each sampler

    Returns:
        Dictionary with sampling results and stats
    """
    # Use environment token if none provided
    if token is None:
        token = os.environ.get('GITHUB_TOKEN')

    # Create the appropriate sampler
    if method.lower() == 'id':
        sampler = IDSampler(token=token)
    elif method.lower() == 'temporal':
        sampler = TemporalSampler(token=token)
    elif method.lower() == 'archive':
        sampler = GHArchiveSampler()
    elif method.lower() == 'bigquery':
        credentials_path = kwargs.pop('credentials_path',
                                      os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
        project_id = kwargs.pop('project_id', None)

        sampler = BigQuerySampler(
            credentials_path=credentials_path,
            project_id=project_id
        )
    else:
        logging.error(f"Unknown sampling method: {method}")
        return {"error": f"Unknown sampling method: {method}"}

    # Sample repositories
    results = sampler.sample(n_samples=n_samples, **kwargs)

    # Return results and stats
    return {
        'method': method,
        'params': kwargs,
        'attempts': sampler.attempts,
        'success_rate': sampler.success_rate,
        'samples': results
    }


# Export samplers
__all__ = ['IDSampler', 'TemporalSampler', 'BigQuerySampler', 'GHArchiveSampler', 'sample']
