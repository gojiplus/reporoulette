# reporoulette/samplers/bigquery_sampler.py
import random
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

from .base import BaseSampler

class BigQuerySampler(BaseSampler):
    """
    Sample repositories using Google BigQuery's GitHub dataset.
    
    This sampler leverages the public GitHub dataset in Google BigQuery to
    efficiently sample repositories with complex criteria and at scale.
    """
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Initialize the BigQuery sampler.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file
            project_id: Google Cloud project ID
        """
        super().__init__(token=None)  # GitHub token not used for BigQuery
        
        if not BIGQUERY_AVAILABLE:
            raise ImportError(
                "BigQuery dependencies not installed. Install reporoulette with "
                "the [bigquery] extra: pip install reporoulette[bigquery]"
            )
            
        self.credentials_path = credentials_path
        self.project_id = project_id
        self._init_client()
        
    def _init_client(self):
        """Initialize the BigQuery client."""
        if self.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            self.client = bigquery.Client(
                credentials=credentials,
                project=self.project_id
            )
        else:
            # Use default credentials
            self.client = bigquery.Client(project=self.project_id)
            
    def sample(
        self, 
        n_samples: int = 100,
        created_after: Optional[Union[str, datetime]] = None,
        created_before: Optional[Union[str, datetime]] = None,
        sample_by: str = "created_at",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using BigQuery.
        
        Args:
            n_samples: Number of repositories to sample
            created_after: Only include repos created after this date
            created_before: Only include repos created before this date
            sample_by: Field to use for sampling ('created_at', 'updated_at', or 'pushed_at')
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        # Format dates for the query
        if created_after:
            if isinstance(created_after, str):
                created_after = f"'{created_after}'"
            else:
                created_after = f"'{created_after.strftime('%Y-%m-%d')}'"
        else:
            # Default to 1 year ago
            one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            created_after = f"'{one_year_ago}'"
            
        if created_before:
            if isinstance(created_before, str):
                created_before = f"'{created_before}'"
            else:
                created_before = f"'{created_before.strftime('%Y-%m-%d')}'"
        else:
            created_before = "CURRENT_TIMESTAMP()"
            
        # Build filter conditions
        conditions = [
            f"r.created_at BETWEEN TIMESTAMP({created_after}) AND TIMESTAMP({created_before})"
        ]
        
        if 'min_stars' in kwargs:
            conditions.append(f"r.stargazers_count >= {kwargs['min_stars']}")
            
        if 'min_forks' in kwargs:
            conditions.append(f"r.forks_count >= {kwargs['min_forks']}")
            
        if 'languages' in kwargs and kwargs['languages']:
            lang_list = ", ".join([f"'{lang}'" for lang in kwargs['languages']])
            conditions.append(f"r.language IN ({lang_list})")
            
        if 'has_license' in kwargs and kwargs['has_license']:
            conditions.append("r.license IS NOT NULL")
            
        # Combine conditions
        where_clause = " AND ".join(conditions)
        
        # Build the query
        query = f"""
        SELECT
            r.id,
            r.name,
            r.full_name,
            r.owner_login as owner,
            r.html_url,
            r.description,
            CAST(r.created_at AS STRING) as created_at,
            CAST(r.updated_at AS STRING) as updated_at,
            CAST(r.pushed_at AS STRING) as pushed_at,
            r.stargazers_count,
            r.forks_count,
            r.language,
            r.visibility
        FROM 
            `bigquery-public-data.github_repos.sample_repos` r
        WHERE {where_clause}
        ORDER BY RAND()
        LIMIT {n_samples}
        """
        
        self.logger.info(f"Executing BigQuery: {query}")
        
        try:
            # Execute the query
            query_job = self.client.query(query)
            rows = query_job.result()
            
            # Process results
            valid_repos = []
            for row in rows:
                repo_data = dict(row.items())
                valid_repos.append(repo_data)
                
            self.attempts = 1  # Only one query attempt
            self.success_count = 1 if valid_repos else 0
            self.results = valid_repos
            
            self.logger.info(f"Found {len(valid_repos)} repositories with BigQuery")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error executing BigQuery: {str(e)}")
            self.attempts = 1
            self.success_count = 0
            return []