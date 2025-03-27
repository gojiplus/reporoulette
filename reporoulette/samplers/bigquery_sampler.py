import random
import os
import time
import logging
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
        seed: Optional[int] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the BigQuery sampler.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file
            project_id: Google Cloud project ID
            seed: Random seed for reproducibility
            log_level: Logging level (default: INFO)
        """
        super().__init__(token=None)  # GitHub token not used for BigQuery
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Add a handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            self._seed = seed
            self.logger.info(f"Random seed set to: {seed}")
        else:
            # Generate a random seed for BigQuery if not provided
            self._seed = random.randint(1, 1000000)
            self.logger.info(f"Generated random seed: {self._seed}")
        
        if not BIGQUERY_AVAILABLE:
            error_msg = (
                "BigQuery dependencies not installed. Install with "
                "pip install google-cloud-bigquery google-auth"
            )
            self.logger.error(error_msg)
            raise ImportError(error_msg)
            
        self.credentials_path = credentials_path
        self.project_id = project_id
        
        # Initialize BigQuery client
        self.logger.info("Initializing BigQuery client")
        self._init_client()
        
    def _init_client(self):
        """Initialize the BigQuery client."""
        try:
            if self.credentials_path:
                self.logger.info(f"Using service account credentials from: {self.credentials_path}")
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self.client = bigquery.Client(
                    credentials=credentials,
                    project=self.project_id
                )
            else:
                # Use default credentials
                self.logger.info("Using default credentials from environment")
                self.client = bigquery.Client(project=self.project_id)
                
            # Log project info
            self.logger.info(f"BigQuery client initialized for project: {self.client.project}")
        except Exception as e:
            self.logger.error(f"Failed to initialize BigQuery client: {str(e)}")
            raise
    
    def _execute_query(self, query: str) -> List[Dict]:
        """
        Execute a BigQuery query and return results as a list of dictionaries.
        
        Args:
            query: BigQuery SQL query to execute
            
        Returns:
            List of dictionaries containing query results
        """
        start_time = time.time()
        
        try:
            # Log query (truncate if too long)
            max_log_length = 1000
            log_query = query if len(query) <= max_log_length else query[:max_log_length] + "..."
            self.logger.info(f"Executing BigQuery query: {log_query}")
            
            # Execute query
            query_job = self.client.query(query)
            self.logger.info(f"Query job ID: {query_job.job_id}")
            
            # Start getting results
            self.logger.info("Waiting for query results...")
            rows = query_job.result()
            
            # Process results
            results = []
            for row in rows:
                results.append(dict(row.items()))
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Query completed in {elapsed_time:.2f} seconds with {len(results)} results")
            
            # Log query statistics if available
            if query_job.total_bytes_processed:
                self.logger.info(
                    f"Processed {query_job.total_bytes_processed / 1024 / 1024:.2f} MB, "
                    f"billed for {query_job.total_bytes_billed / 1024 / 1024:.2f} MB"
                )
            
            return results
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Query failed after {elapsed_time:.2f} seconds: {str(e)}")
            
            # Attempt to provide more detailed error information
            if hasattr(e, 'errors') and e.errors:
                for error in e.errors:
                    self.logger.error(f"Error details: {error}")
            
            # Return empty list on error
            return []
    
    def sample_by_datetime(
        self,
        n_samples: int = 100,
        hours_to_sample: int = 10,
        repos_per_hour: int = 10,
        years_back: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using the cost-effective datetime approach.
        
        This method samples GitHub repositories by randomly selecting hours
        from the GitHub archive and collecting repository information.
        
        Args:
            n_samples: Target number of repositories to sample
            hours_to_sample: Number of random hours to sample
            repos_per_hour: Repositories to sample per hour
            years_back: How many years to look back
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        self.logger.info(
            f"Sampling repositories: n_samples={n_samples}, hours_to_sample={hours_to_sample}, "
            f"repos_per_hour={repos_per_hour}, years_back={years_back}"
        )
        
        # Calculate parameters to ensure we get enough samples
        hours_needed = max(1, (n_samples + repos_per_hour - 1) // repos_per_hour)
        hours_to_sample = max(hours_to_sample, hours_needed)
        
        self.logger.info(f"Adjusted hours_to_sample to {hours_to_sample} to ensure enough samples")
        
        # Query to sample repositories from random time periods
        query = f"""
        -- Define parameters
        DECLARE hours_to_sample INT64 DEFAULT {hours_to_sample};
        DECLARE repos_per_hour INT64 DEFAULT {repos_per_hour};
        DECLARE years_back INT64 DEFAULT {years_back};

        -- Create a table of random dates and hours to sample from
        CREATE TEMP TABLE random_dates AS (
          SELECT 
            FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL CAST(FLOOR(RAND({self._seed}) * (365 * years_back)) AS INT64) DAY)) AS day,
            CAST(FLOOR(RAND({self._seed}) * 24) AS INT64) AS hour
          FROM 
            UNNEST(GENERATE_ARRAY(1, hours_to_sample))
        );

        -- Sample repositories from each random date-hour
        WITH sampled_repos AS (
          SELECT
            date_record.day AS sample_day,
            date_record.hour AS sample_hour,
            repo.name AS repo_name,
            repo.url AS html_url,
            actor.login AS owner,
            created_at,
            type AS event_type,
            ROW_NUMBER() OVER (PARTITION BY date_record.day, date_record.hour ORDER BY RAND({self._seed})) AS rn
          FROM random_dates date_record
          CROSS JOIN (
            -- Dynamically access table for each date-hour
            SELECT CONCAT('`githubarchive.day.', day, '`') AS table_name
            FROM random_dates
          ) table_names,
          -- Use a dynamic table reference
          UNNEST([STRUCT(
            (SELECT AS STRUCT
              ARRAY(
                EXECUTE IMMEDIATE FORMAT(
                  "SELECT repo, actor, created_at, type FROM %s TABLESAMPLE SYSTEM (1 PERCENT) WHERE type IN ('PushEvent', 'CreateEvent', 'PullRequestEvent') LIMIT %d",
                  table_names.table_name, repos_per_hour * 10
                )
              ) AS events
            )
          )]),
          UNNEST(events) AS event
        )
        
        -- Extract final set of repositories
        SELECT
          repo_name AS full_name,
          SPLIT(repo_name, '/')[OFFSET(1)] AS name,
          SPLIT(repo_name, '/')[OFFSET(0)] AS owner,
          html_url,
          created_at,
          CONCAT(sample_day, '-', sample_hour) AS sampled_from,
          event_type
        FROM
          sampled_repos
        WHERE
          rn <= repos_per_hour
        ORDER BY
          RAND({self._seed})
        LIMIT {n_samples};
        """
        
        # Execute the main query
        valid_repos = self._execute_query(query)
        
        self.attempts = 1  # Only one query attempt
        self.success_count = 1 if valid_repos else 0
        self.results = valid_repos
        
        # Log summary of results
        self.logger.info(f"Found {len(valid_repos)} repositories with BigQuery datetime sampling")
        
        return self.results
    
    def sample_standard(
        self,
        n_samples: int = 100,
        created_after: Optional[Union[str, datetime]] = None,
        created_before: Optional[Union[str, datetime]] = None,
        languages: Optional[List[str]] = None,
        min_stars: int = 0,
        min_forks: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using the standard BigQuery approach.
        
        Args:
            n_samples: Number of repositories to sample
            created_after: Only include repos created after this date
            created_before: Only include repos created before this date
            languages: List of languages to filter by
            min_stars: Minimum number of stars
            min_forks: Minimum number of forks
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        self.logger.info(
            f"Standard sampling: n_samples={n_samples}, min_stars={min_stars}, min_forks={min_forks}"
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
            
        self.logger.info(f"Date range: {created_after} to {created_before}")
        
        # Build filter conditions
        conditions = [
            f"r.created_at BETWEEN TIMESTAMP({created_after}) AND TIMESTAMP({created_before})"
        ]
        
        if min_stars > 0:
            conditions.append(f"r.stargazers_count >= {min_stars}")
            
        if min_forks > 0:
            conditions.append(f"r.forks_count >= {min_forks}")
            
        if languages:
            lang_list = ", ".join([f"'{lang}'" for lang in languages])
            self.logger.info(f"Filtering for languages: {lang_list}")
            conditions.append(f"r.language IN ({lang_list})")
            
        if 'has_license' in kwargs and kwargs['has_license']:
            conditions.append("r.license IS NOT NULL")
            
        # Combine conditions
        where_clause = " AND ".join(conditions)
        
        # Build the query with explicit column selection
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
            r.language,  -- Get language directly from repos table
            r.visibility
        FROM 
            `bigquery-public-data.github_repos.sample_repos` r
        WHERE {where_clause}
        ORDER BY RAND({self._seed})
        LIMIT {n_samples}
        """
        
        # Execute query
        valid_repos = self._execute_query(query)
        
        self.attempts = 1  # Only one query attempt
        self.success_count = 1 if valid_repos else 0
        self.results = valid_repos
        
        # Log summary of results
        self.logger.info(f"Found {len(valid_repos)} repositories with standard BigQuery sampling")
        if valid_repos:
            languages_found = set(repo.get('language') for repo in valid_repos if repo.get('language'))
            self.logger.info(f"Languages found: {', '.join(sorted(languages_found))}")
        
        return self.results
            
    def sample(
        self, 
        n_samples: int = 100,
        use_datetime_sampling: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using BigQuery.
        
        Args:
            n_samples: Number of repositories to sample
            use_datetime_sampling: Whether to use the cost-effective datetime sampling method
            **kwargs: Additional parameters for the sampling method
            
        Returns:
            List of repository data
        """
        self.logger.info(f"Starting repository sampling with n_samples={n_samples}")
        start_time = time.time()
        
        # Choose sampling method
        if use_datetime_sampling:
            self.logger.info("Using datetime sampling method")
            results = self.sample_by_datetime(n_samples=n_samples, **kwargs)
        else:
            self.logger.info("Using standard sampling method")
            results = self.sample_standard(n_samples=n_samples, **kwargs)
            
        # Log completion
        elapsed_time = time.time() - start_time
        self.logger.info(f"Sampling completed in {elapsed_time:.2f} seconds, found {len(results)} repositories")
        
        return results
        
    def get_languages(self, repos: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve language information for a list of repositories.
        
        This is a separate method since language information isn't reliably 
        available in the standard BigQuery sampling approach.
        
        Args:
            repos: List of repository dictionaries with at least 'full_name' field
            
        Returns:
            Dictionary mapping repository names to their language information
        """
        self.logger.info(f"Fetching language information for {len(repos)} repositories")
        
        # Extract repo names for the query
        repo_names = [repo['full_name'] for repo in repos if 'full_name' in repo]
        if not repo_names:
            self.logger.warning("No valid repository names found")
            return {}
            
        # GitHub API only accepts 100 repos at a time in a query
        chunk_size = 100
        repo_chunks = [repo_names[i:i + chunk_size] for i in range(0, len(repo_names), chunk_size)]
        
        language_info = {}
        for chunk in repo_chunks:
            # Create the repo list for the query
            repo_list = ", ".join([f"'{repo}'" for repo in chunk])
            
            # Query to get language information - using the UNNEST operation
            # This is the correct way to query the languages table
            query = f"""
            SELECT
                repo_name,
                lang.name AS language,
                lang.bytes AS bytes
            FROM
                `bigquery-public-data.github_repos.languages`,
                UNNEST(language) AS lang
            WHERE
                repo_name IN ({repo_list})
            ORDER BY
                repo_name, bytes DESC
            """
            
            # Execute query
            results = self._execute_query(query)
            
            # Process results
            for result in results:
                repo_name = result.get('repo_name')
                if repo_name:
                    if repo_name not in language_info:
                        language_info[repo_name] = []
                    language_info[repo_name].append({
                        'language': result.get('language'),
                        'bytes': result.get('bytes')
                    })
        
        # Log summary
        repos_with_language = len(language_info)
        self.logger.info(f"Found language information for {repos_with_language}/{len(repos)} repositories")
        
        return language_info

# Command-line interface
if __name__ == "__main__":
    import argparse
    import json
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Sample random GitHub repositories using BigQuery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--count", type=int, default=10, help="Number of repositories to find")
    parser.add_argument("--min-stars", type=int, default=0, help="Minimum number of stars")
    parser.add_argument("--language", type=str, default=None, help="Programming language filter")
    parser.add_argument("--output", type=str, default="repos.jsonl", help="Output file")
    parser.add_argument("--years-back", type=int, default=10, help="How many years back to sample from")
    parser.add_argument("--credentials", type=str, default=None, help="Path to Google Cloud credentials")
    parser.add_argument("--project", type=str, default=None, help="Google Cloud project ID")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--method", choices=["datetime", "standard"], default="datetime", 
                        help="Sampling method to use")
    
    args = parser.parse_args()
    
    # Get credentials path from environment if not provided
    credentials_path = args.credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    
    try:
        # Create sampler
        sampler = BigQuerySampler(
            credentials_path=credentials_path,
            project_id=project_id,
            seed=args.seed
        )
        
        # Set up language filter
        languages = [args.language] if args.language else None
        
        # Sample repositories
        repos = sampler.sample(
            n_samples=args.count,
            use_datetime_sampling=(args.method == "datetime"),
            min_stars=args.min_stars,
            languages=languages,
            years_back=args.years_back
        )
        
        # Write results to file
        with open(args.output, 'w') as f:
            for repo in repos:
                f.write(json.dumps(repo) + '\n')
                
        print(f"Found {len(repos)} repositories. Results saved to {args.output}")
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)
