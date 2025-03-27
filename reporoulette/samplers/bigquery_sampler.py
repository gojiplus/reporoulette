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
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
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
        self.logger.info(f"Initializing BigQuery client (project_id: {project_id or 'default'})")
        self._init_client()
        
        # Initialize tracking variables
        self.attempts = 0
        self.success_count = 0
        self.results = []
        
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
            # Increment attempt counter
            self.attempts += 1
            
            # Log query (truncate if too long)
            max_log_length = 1000
            log_query = query if len(query) <= max_log_length else query[:max_log_length] + "..."
            self.logger.info(f"Executing BigQuery query (attempt {self.attempts}): {log_query}")
            
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
            
            # Increment success counter
            self.success_count += 1
            
            # Log query statistics if available
            if query_job.total_bytes_processed:
                bytes_processed = query_job.total_bytes_processed / 1024 / 1024
                bytes_billed = query_job.total_bytes_billed / 1024 / 1024
                estimated_cost = bytes_billed / 1024 / 1024 * 5  # $5 per TB is standard rate
                
                self.logger.info(
                    f"Query stats: Processed {bytes_processed:.2f} MB, "
                    f"billed for {bytes_billed:.2f} MB "
                    f"(est. cost: ${estimated_cost:.6f})"
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
        event_types: List[str] = ["PushEvent", "CreateEvent", "PullRequestEvent"],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using the cost-effective datetime approach with hour tables.
        
        This method samples GitHub repositories by randomly selecting specific hours
        from the GitHub archive and collecting repository information.
        
        Args:
            n_samples: Target number of repositories to sample
            hours_to_sample: Number of random hours to sample
            repos_per_hour: Repositories to sample per hour
            years_back: How many years to look back
            event_types: Types of GitHub events to consider
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        self.logger.info(
            f"Starting datetime sampling: n_samples={n_samples}, hours_to_sample={hours_to_sample}, "
            f"repos_per_hour={repos_per_hour}, years_back={years_back}"
        )
        
        # Log filter criteria if any
        if kwargs:
            self.logger.info(f"Filter criteria: {kwargs}")
        
        # Calculate parameters to ensure we get enough samples
        hours_needed = max(1, (n_samples + repos_per_hour - 1) // repos_per_hour)
        hours_to_sample = max(hours_to_sample, hours_needed)
        
        self.logger.debug(f"Adjusted hours_to_sample to {hours_to_sample} to ensure enough samples")
        
        # Format event types for the SQL query
        event_types_str = ", ".join([f"'{event_type}'" for event_type in event_types])
        self.logger.debug(f"Using event types: {event_types_str}")
        
        # Query to sample repositories from random time periods using hour tables
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
    
        -- Sample repositories from each random hour
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
            -- Dynamically access hour table for each date-hour
            SELECT CONCAT('`githubarchive.hour.', day, FORMAT('%02d', hour), '`') AS table_name
            FROM random_dates
          ) table_names,
          -- Use a dynamic table reference
          UNNEST([STRUCT(
            (SELECT AS STRUCT
              ARRAY(
                EXECUTE IMMEDIATE FORMAT(
                  "SELECT repo, actor, created_at, type FROM %s WHERE type IN ({event_types_str}) LIMIT %d",
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
        
        # Store results
        self.results = valid_repos
        
        # Apply any filters
        filtered_count_before = len(valid_repos)
        if kwargs:
            self.results = self._filter_repos(valid_repos, **kwargs)
            filtered_count_after = len(self.results)
            if filtered_count_before != filtered_count_after:
                self.logger.info(
                    f"Applied filters: {filtered_count_before - filtered_count_after} repositories filtered out, "
                    f"{filtered_count_after} repositories remaining"
                )
        
        # Log summary of results
        self.logger.info(
            f"Datetime sampling completed: found {len(valid_repos)} repositories "
            f"(success rate: {(self.success_count / self.attempts) * 100:.1f}%)"
        )
        
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
            f"Starting standard sampling: n_samples={n_samples}, min_stars={min_stars}, min_forks={min_forks}"
        )
        
        # Log filter criteria if any
        if kwargs:
            self.logger.info(f"Filter criteria: {kwargs}")
        
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
        
        # Store results
        self.results = valid_repos
        
        # Apply any filters
        filtered_count_before = len(valid_repos)
        if kwargs:
            self.results = self._filter_repos(valid_repos, **kwargs)
            filtered_count_after = len(self.results)
            if filtered_count_before != filtered_count_after:
                self.logger.info(
                    f"Applied filters: {filtered_count_before - filtered_count_after} repositories filtered out, "
                    f"{filtered_count_after} repositories remaining"
                )
        
        # Log summary of results
        self.logger.info(f"Standard sampling completed: found {len(valid_repos)} repositories")
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
        
        # Reset tracking variables
        self.attempts = 0
        self.success_count = 0
        
        # Choose sampling method
        if use_datetime_sampling:
            self.logger.info("Using datetime sampling method")
            results = self.sample_by_datetime(n_samples=n_samples, **kwargs)
        else:
            self.logger.info("Using standard sampling method")
            results = self.sample_standard(n_samples=n_samples, **kwargs)
            
        # Log completion
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Sampling completed in {elapsed_time:.2f} seconds: "
            f"found {len(results)} repositories, "
            f"{self.attempts} attempts, {self.success_count} successful queries"
        )
        
        return results
    
    def _filter_repos(self, repos: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Filter repositories based on criteria.
        
        Args:
            repos: List of repository data
            **kwargs: Filter criteria as key-value pairs
            
        Returns:
            Filtered list of repositories
        """
        if not kwargs:
            return repos
            
        self.logger.debug(f"Filtering {len(repos)} repositories with criteria: {kwargs}")
        filtered = repos
        
        # Owner filter
        if 'owner' in kwargs:
            owner = kwargs['owner']
            filtered = [r for r in filtered if r.get('owner') == owner]
            self.logger.debug(f"Filtered by owner '{owner}': {len(filtered)} repositories remaining")
            
        # Language filter
        if 'language' in kwargs:
            language = kwargs['language']
            filtered = [r for r in filtered if r.get('language') == language]
            self.logger.debug(f"Filtered by language '{language}': {len(filtered)} repositories remaining")
            
        # Min stars filter
        if 'min_stars' in kwargs:
            min_stars = int(kwargs['min_stars'])
            filtered = [r for r in filtered if r.get('stargazers_count', 0) >= min_stars]
            self.logger.debug(f"Filtered by min_stars {min_stars}: {len(filtered)} repositories remaining")
            
        # Min forks filter
        if 'min_forks' in kwargs:
            min_forks = int(kwargs['min_forks'])
            filtered = [r for r in filtered if r.get('forks_count', 0) >= min_forks]
            self.logger.debug(f"Filtered by min_forks {min_forks}: {len(filtered)} repositories remaining")
            
        # Created after filter
        if 'created_after' in kwargs:
            created_after = kwargs['created_after']
            if isinstance(created_after, str):
                created_after = datetime.fromisoformat(created_after.replace('Z', '+00:00'))
            filtered = [r for r in filtered if r.get('created_at') and datetime.fromisoformat(r['created_at'].replace('Z', '+00:00')) >= created_after]
            self.logger.debug(f"Filtered by created_after {created_after}: {len(filtered)} repositories remaining")
            
        # Created before filter
        if 'created_before' in kwargs:
            created_before = kwargs['created_before']
            if isinstance(created_before, str):
                created_before = datetime.fromisoformat(created_before.replace('Z', '+00:00'))
            filtered = [r for r in filtered if r.get('created_at') and datetime.fromisoformat(r['created_at'].replace('Z', '+00:00')) <= created_before]
            self.logger.debug(f"Filtered by created_before {created_before}: {len(filtered)} repositories remaining")
        
        return filtered
        
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
        start_time = time.time()
        
        # Extract repo names for the query
        repo_names = [repo['full_name'] for repo in repos if 'full_name' in repo]
        if not repo_names:
            self.logger.warning("No valid repository names found")
            return {}
            
        # GitHub API only accepts 100 repos at a time in a query
        chunk_size = 100
        repo_chunks = [repo_names[i:i + chunk_size] for i in range(0, len(repo_names), chunk_size)]
        
        self.logger.info(f"Processing {len(repo_chunks)} chunks of repository names (max {chunk_size} per chunk)")
        
        language_info = {}
        for i, chunk in enumerate(repo_chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(repo_chunks)} with {len(chunk)} repositories")
            
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
            chunk_start_time = time.time()
            results = self._execute_query(query)
            chunk_elapsed = time.time() - chunk_start_time
            
            self.logger.info(f"Chunk {i+1} query completed in {chunk_elapsed:.2f} seconds with {len(results)} language records")
            
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
        total_languages = sum(len(langs) for langs in language_info.values())
        elapsed_time = time.time() - start_time
        
        self.logger.info(
            f"Language query completed in {elapsed_time:.2f} seconds: "
            f"found information for {repos_with_language}/{len(repos)} repositories "
            f"({total_languages} language entries total)"
        )
        
        # Log most common languages
        if language_info:
            all_languages = []
            for repo_langs in language_info.values():
                for lang in repo_langs:
                    if 'language' in lang:
                        all_languages.append(lang['language'])
            
            language_counts = {}
            for lang in all_languages:
                language_counts[lang] = language_counts.get(lang, 0) + 1
                
            # Get top 10 languages
            top_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_langs_str = ", ".join([f"{lang}: {count}" for lang, count in top_languages])
            self.logger.info(f"Top languages: {top_langs_str}")
        
        return language_info