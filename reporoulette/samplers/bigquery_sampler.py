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
    Uses a cost-effective implementation that samples from specific dates and hours.
    """
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the BigQuery sampler.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file
            project_id: Google Cloud project ID
            seed: Random seed for reproducibility
        """
        super().__init__(token=None)  # GitHub token not used for BigQuery
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            self._seed = seed
            self.logger.info(f"Random seed set to: {seed}")
        
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
    
    def _execute_query(self, query: str) -> List[Dict]:
        """Execute a BigQuery query and return results as a list of dictionaries."""
        try:
            self.logger.info(f"Executing BigQuery: {query}")
            query_job = self.client.query(query)
            rows = query_job.result()
            
            results = []
            for row in rows:
                results.append(dict(row.items()))
            
            return results
        except Exception as e:
            self.logger.error(f"Error executing BigQuery: {str(e)}")
            return []
    
    def sample_by_datetime(
        self,
        n_samples: int = 100,
        hours_to_sample: int = 10,
        repos_per_hour: int = 10,
        years_back: int = 10,
        languages: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using the cost-effective datetime approach.
        
        Args:
            n_samples: Target number of repositories to sample
            hours_to_sample: Number of random hours to sample
            repos_per_hour: Repositories to sample per hour
            years_back: How many years to look back
            languages: List of languages to filter by (default: all languages)
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        # Calculate parameters to ensure we get enough samples
        hours_needed = max(1, (n_samples + repos_per_hour - 1) // repos_per_hour)
        hours_to_sample = max(hours_to_sample, hours_needed)
        
        # Create the language filter
        language_filter = ""
        if languages:
            lang_list = ", ".join([f"'{lang}'" for lang in languages])
            language_filter = f"AND rl.language IN ({lang_list})"
        
        # Build the query
        query = f"""
        -- Define parameters
        DECLARE hours_to_sample INT64 DEFAULT {hours_to_sample};
        DECLARE repos_per_hour INT64 DEFAULT {repos_per_hour};
        DECLARE years_back INT64 DEFAULT {years_back};

        -- Create a table of random dates and hours to sample from
        CREATE TEMP TABLE random_dates AS (
          SELECT 
            FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL CAST(FLOOR(RAND({self._seed if hasattr(self, '_seed') else ''}) * (365 * years_back)) AS INT64) DAY)) AS day,
            CAST(FLOOR(RAND({self._seed if hasattr(self, '_seed') else ''}) * 24) AS INT64) AS hour
          FROM 
            UNNEST(GENERATE_ARRAY(1, hours_to_sample))
        );

        -- Create a table to collect sampled repositories
        CREATE TEMP TABLE sampled_repos (
          repo_name STRING,
          repo_url STRING,
          created_at TIMESTAMP,
          sample_day STRING,
          sample_hour INT64,
          event_type STRING
        );

        -- For each random date, sample repositories efficiently
        FOR date_record IN (SELECT * FROM random_dates) DO
          -- Construct the specific table name for this date
          DECLARE table_name STRING;
          SET table_name = CONCAT('`githubarchive.day.', date_record.day, date_record.hour, '`');
          
          -- Execute a dynamic SQL statement to sample from just this one table
          EXECUTE IMMEDIATE format('''
            INSERT INTO sampled_repos
            SELECT
              repo.name AS repo_name,
              repo.url AS repo_url,
              created_at,
              '%s' AS sample_day,
              %d AS sample_hour,
              type AS event_type
            FROM
              %s TABLESAMPLE SYSTEM (5 PERCENT)
            WHERE
              (type = 'CreateEvent' OR type = 'PushEvent')
            ORDER BY RAND({self._seed if hasattr(self, '_seed') else ''})
            LIMIT %d
          ''', date_record.day, date_record.hour, table_name, repos_per_hour);
        END FOR;

        -- Get language information for repositories
        CREATE TEMP TABLE repo_languages AS (
          -- Sample a recent day's data for language information
          WITH language_sources AS (
            -- First try to get language from PullRequestEvent payloads
            SELECT
              repo.name AS repo_name,
              JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') AS language
            FROM
              `githubarchive.day.20230101` TABLESAMPLE SYSTEM (10 PERCENT)
            WHERE
              type = 'PullRequestEvent'
              AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.base.repo.language') IS NOT NULL
            
            UNION ALL
            
            -- Also try CreateEvent which sometimes has language info
            SELECT
              repo.name AS repo_name,
              JSON_EXTRACT_SCALAR(payload, '$.repository.language') AS language
            FROM
              `githubarchive.day.20230101` TABLESAMPLE SYSTEM (10 PERCENT)
            WHERE
              type = 'CreateEvent'
              AND JSON_EXTRACT_SCALAR(payload, '$.repository.language') IS NOT NULL
          )
          
          SELECT
            repo_name,
            language
          FROM
            language_sources
          GROUP BY
            repo_name, language
        );

        -- Final result - join sampled repos with language info and filter for languages if specified
        SELECT
          r.repo_name AS name,
          r.repo_url AS html_url,
          r.created_at,
          CONCAT(r.sample_day, '-', r.sample_hour) AS sampled_from,
          rl.language,
          'Unknown' AS owner,
          NULL AS description,
          NULL AS stargazers_count,
          NULL AS forks_count,
          'public' AS visibility
        FROM
          sampled_repos r
        JOIN  -- Using INNER JOIN to only keep repos with language info
          repo_languages rl
        ON
          r.repo_name = rl.repo_name
        WHERE 1=1
          {language_filter}
        ORDER BY
          r.repo_name
        LIMIT {n_samples};
        """
        
        valid_repos = self._execute_query(query)
        
        self.attempts = 1  # Only one query attempt
        self.success_count = 1 if valid_repos else 0
        self.results = valid_repos
        
        self.logger.info(f"Found {len(valid_repos)} repositories with BigQuery")
        return self.results
            
    def sample(
        self, 
        n_samples: int = 100,
        created_after: Optional[Union[str, datetime]] = None,
        created_before: Optional[Union[str, datetime]] = None,
        sample_by: str = "created_at",
        use_datetime_sampling: bool = True,
        hours_to_sample: int = 10,
        repos_per_hour: int = 10,
        years_back: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using BigQuery.
        
        Args:
            n_samples: Number of repositories to sample
            created_after: Only include repos created after this date
            created_before: Only include repos created before this date
            sample_by: Field to use for sampling ('created_at', 'updated_at', or 'pushed_at')
            use_datetime_sampling: Whether to use the cost-effective datetime sampling method
            hours_to_sample: Number of random hours to sample (if use_datetime_sampling=True)
            repos_per_hour: Repositories to sample per hour (if use_datetime_sampling=True)
            years_back: How many years to look back (if use_datetime_sampling=True)
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        if use_datetime_sampling:
            return self.sample_by_datetime(
                n_samples=n_samples,
                hours_to_sample=hours_to_sample,
                repos_per_hour=repos_per_hour,
                years_back=years_back,
                **kwargs
            )
        
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
        # Use a consistent seed for the random function in BigQuery if a seed was provided
        rand_function = "RAND()" if not hasattr(self, '_seed') else f"RAND({self._seed})"
        
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
        ORDER BY {rand_function}
        LIMIT {n_samples}
        """
        
        valid_repos = self._execute_query(query)
        
        self.attempts = 1  # Only one query attempt
        self.success_count = 1 if valid_repos else 0
        self.results = valid_repos
        
        self.logger.info(f"Found {len(valid_repos)} repositories with BigQuery")
        return self.results
