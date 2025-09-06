import random
import logging
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

from .base import BaseSampler
from .bq_utils import execute_query, filter_repos, format_timestamp_query

class BigQuerySampler(BaseSampler):
    """
    Sample repositories using Google BigQuery's GitHub dataset.
    
    This sampler leverages the public GitHub dataset in Google BigQuery to
    efficiently sample repositories with complex criteria and at scale.
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
        """
        super().__init__(token=None)  # GitHub token not used for BigQuery

        # Configure logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)
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

        self.logger.info(f"Initializing BigQuery client (project_id: {project_id or 'default'})")
        self._init_client()

        # Initialize tracking variables
        self.attempts = 0
        self.success_count = 0
        self.results = []

    def _init_client(self) -> None:
        """Initialize the BigQuery client."""
        try:
            if self.credentials_path:
                self.logger.info(f"Using service account credentials from: {self.credentials_path}")
                credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
                self.client = bigquery.Client(credentials=credentials, project=self.project_id)
            else:
                self.logger.info("Using default credentials from environment")
                self.client = bigquery.Client(project=self.project_id)
            self.logger.info(f"BigQuery client initialized for project: {self.client.project}")
        except Exception as e:
            self.logger.error(f"Failed to initialize BigQuery client: {str(e)}")
            raise

    def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a BigQuery query using the utility function.
        """
        self.attempts += 1
        results = execute_query(self.client, query, self.logger)
        if results:
            self.success_count += 1
        return results

    def _build_count_query(self, days_to_sample: int, years_back: int) -> str:
        """
        Build the SQL query that creates a temporary table of random days
        and counts unique repositories for each day from GitHub Archive tables.
        """
        return f"""
        -- Define parameters
        DECLARE days_to_sample INT64 DEFAULT {days_to_sample};
        DECLARE years_back INT64 DEFAULT {years_back};

        -- Create a table of random dates to sample from
        CREATE TEMP TABLE random_dates AS (
          SELECT 
            FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL CAST(FLOOR(RAND() * (365 * years_back)) AS INT64) DAY)) AS day
          FROM UNNEST(GENERATE_ARRAY(1, days_to_sample))
        );

        -- Count unique repositories per day
        SELECT
          rd.day AS sample_day,
          COUNT(DISTINCT event.repo_name) AS repo_count
        FROM random_dates rd
        CROSS JOIN (
          SELECT repo.name AS repo_name, created_at
          FROM (
            EXECUTE IMMEDIATE FORMAT(
              "SELECT repo.name, created_at
               FROM `githubarchive.day.%s`
               WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL %d YEAR)
               LIMIT 100000",
              rd.day,
              years_back
            )
          )
        ) event
        GROUP BY rd.day
        ORDER BY repo_count DESC
        """

    def _build_day_query(self, day_data: Dict[str, Any], i: int, years_back: int) -> str:
        """
        Build an individual SQL query to sample repositories from a specific day.
        """
        day = day_data.get('sample_day')
        repo_count = day_data.get('repo_count', 0)
        samples_to_take = day_data.get('samples_to_take', 1)
        
        return f"""
        -- Day {i+1}: {day} with {repo_count} repositories
        SELECT DISTINCT
            event.repo_name AS full_name,
            SPLIT(event.repo_name, '/')[SAFE_OFFSET(1)] AS name,
            SPLIT(event.repo_name, '/')[SAFE_OFFSET(0)] AS owner,
            event.repo_url AS html_url,
            event.created_at,
            '{day}' AS sampled_from,
            event.event_type,
            {repo_count} AS day_repo_count,
            {samples_to_take} AS samples_allocated
        FROM (
            EXECUTE IMMEDIATE FORMAT(
                "SELECT
                    repo.name AS repo_name,
                    repo.url AS repo_url,
                    created_at,
                    type AS event_type
                 FROM `githubarchive.day.%s`
                 WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL %d YEAR)
                 LIMIT 100000",
                '{day}',
                {years_back}
            )
        ) AS event
        ORDER BY RAND({self._seed} + {i})
        LIMIT {samples_to_take}
        """

    def _combine_day_queries(self, day_queries: List[str], n_samples: int) -> str:
        """
        Combine individual day queries into one final query and deduplicate the results.
        """
        combined_query = "\nUNION ALL\n".join(day_queries)
        return f"""
        -- Final combined query with deduplication
        SELECT DISTINCT
            full_name,
            name,
            owner,
            html_url,
            created_at,
            sampled_from,
            event_type,
            day_repo_count,
            samples_allocated
        FROM (
            {combined_query}
        )
        ORDER BY RAND({self._seed})
        LIMIT {n_samples}
        """

    def sample_by_day(
        self,
        n_samples: int = 100,
        days_to_sample: int = 10,
        repos_per_day: int = 50,
        years_back: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using a day-based approach with GitHub Archive tables.
        """
        self.logger.info(
            f"Sampling {n_samples} repositories across {days_to_sample} days"
        )
        if kwargs:
            self.logger.info(f"Filter criteria: {kwargs}")

        # Adjust days to sample if needed
        days_needed = max(1, (n_samples + repos_per_day - 1) // repos_per_day)
        days_to_sample = max(days_to_sample, days_needed)
        self.logger.debug(f"Adjusted days_to_sample: {days_to_sample}")

        count_query = self._build_count_query(days_to_sample, years_back)
        day_counts = self._execute_query(count_query)
        if not day_counts:
            self.logger.warning("No repositories found for the selected days")
            return []

        total_repos = sum(day.get('repo_count', 0) for day in day_counts)
        for day in day_counts:
            day_repo_count = day.get('repo_count', 0)
            day_weight = day_repo_count / total_repos if total_repos > 0 else 0
            day['samples_to_take'] = min(
                max(1, int(n_samples * day_weight)),
                min(repos_per_day, day_repo_count)
            )

        self.logger.info(f"Found {len(day_counts)} days with {total_repos} total repositories")

        day_queries = []
        for i, day in enumerate(day_counts):
            if day.get('samples_to_take', 0) <= 0:
                continue
            day_query = self._build_day_query(day, i, years_back)
            day_queries.append(day_query)

        final_query = self._combine_day_queries(day_queries, n_samples)
        valid_repos = self._execute_query(final_query)
        self.results = valid_repos

        filtered_count_before = len(valid_repos)
        if kwargs:
            self.results = filter_repos(valid_repos, **kwargs)
            filtered_count_after = len(self.results)
            if filtered_count_before != filtered_count_after:
                self.logger.info(
                    f"Applied filters: {filtered_count_after}/{filtered_count_before} repositories retained"
                )

        self.logger.info(f"Completed day-based sampling: found {len(self.results)} repositories")

        if valid_repos:
            day_counts_map = {}
            for repo in valid_repos:
                day_sampled = repo.get('sampled_from', 'unknown')
                day_repo_count = repo.get('day_repo_count', 0)
                allocated = repo.get('samples_allocated', 0)
                if day_sampled not in day_counts_map:
                    day_counts_map[day_sampled] = {'count': 0, 'repos': day_repo_count, 'allocated': allocated}
                day_counts_map[day_sampled]['count'] += 1
            self.logger.info(f"Sampled from {len(day_counts_map)} different days")
            for day_str, data in sorted(day_counts_map.items()):
                self.logger.debug(
                    f"Day {day_str}: {data['count']}/{data['allocated']} samples from {data['repos']} repos"
                )

        return self.results

    def sample_active(
        self,
        n_samples: int = 100,
        created_after: Optional[Union[str, datetime]] = None,
        created_before: Optional[Union[str, datetime]] = None,
        languages: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories with recent commit activity.
        """
        self.logger.info(f"Sampling {n_samples} active repositories based on commit history")
        if kwargs:
            self.logger.info(f"Filter criteria: {kwargs}")

        if created_after:
            created_after = format_timestamp_query(created_after)
        else:
            one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            created_after = f"'{one_year_ago}'"

        if created_before:
            created_before = format_timestamp_query(created_before)
        else:
            created_before = "CURRENT_TIMESTAMP()"

        self.logger.info(f"Time period: {created_after} to {created_before}")

        lang_list = None
        if languages:
            lang_list = ", ".join([f"'{lang}'" for lang in languages])
            self.logger.info(f"Filtering for languages: {lang_list}")

        query = f"""
        WITH repo_set AS (
            SELECT DISTINCT
                repo AS full_name,
                SPLIT(repo, '/')[OFFSET(1)] AS name,
                SPLIT(repo, '/')[OFFSET(0)] AS owner
            FROM 
                `bigquery-public-data.github_repos.commits` c,
                UNNEST(c.repo_name) AS repo
            WHERE 
                TIMESTAMP_SECONDS(c.committer.time_sec) BETWEEN TIMESTAMP({created_after}) AND TIMESTAMP({created_before})
                {("AND SPLIT(repo, '/')[OFFSET(0)] IN (" + lang_list + ")") if languages else ''}
        )
        SELECT
            full_name,
            name,
            owner
        FROM 
            repo_set
        ORDER BY RAND({self._seed})
        LIMIT {n_samples}
        """
        valid_repos = self._execute_query(query)
        self.results = valid_repos

        filtered_count_before = len(valid_repos)
        if kwargs:
            self.results = filter_repos(valid_repos, **kwargs)
            filtered_count_after = len(self.results)
            if filtered_count_before != filtered_count_after:
                self.logger.info(
                    f"Applied filters: {filtered_count_after}/{filtered_count_before} repositories retained"
                )

        self.logger.info(f"Completed active repository sampling: found {len(self.results)} repositories")
        return self.results

    def sample(
        self, 
        n_samples: int = 100,
        population: str = "all",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using BigQuery.
        
        Args:
            n_samples: Number of repositories to sample
            population: Type of repository population to sample from ('all' or 'active')
            **kwargs: Additional filtering criteria
            
        Returns:
            List of repository dictionaries
        """
        self.logger.info(f"Starting repository sampling: n_samples={n_samples}, population={population}")
        start_time = time.time()

        self.attempts = 0
        self.success_count = 0

        if population == "active":
            self.logger.info("Targeting active repositories with recent commits")
            results = self.sample_active(n_samples=n_samples, **kwargs)
        else:  # all
            self.logger.info("Sampling from all repositories across time periods")
            results = self.sample_by_day(n_samples=n_samples, **kwargs)

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Sampling completed in {elapsed_time:.2f}s: {len(results)} repositories found"
        )

        return results

    def get_languages(self, repos: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve language information for a list of repositories.
        """
        self.logger.info(f"Fetching language information for {len(repos)} repositories")
        start_time = time.time()
        
        repo_names = [repo['full_name'] for repo in repos if 'full_name' in repo]
        if not repo_names:
            self.logger.warning("No valid repository names found")
            return {}
        
        # Process all repositories in a single query
        repo_list = ", ".join([f"'{repo}'" for repo in repo_names])
        
        query = f"""
        SELECT
            repo_name,
            ARRAY_AGG(
                STRUCT(
                    lang.name AS language, 
                    lang.bytes AS bytes
                ) 
                ORDER BY lang.bytes DESC
            ) AS languages
        FROM
            `bigquery-public-data.github_repos.languages`,
            UNNEST(language) AS lang
        WHERE
            repo_name IN ({repo_list})
        GROUP BY
            repo_name
        """
        
        query_start_time = time.time()
        results = self._execute_query(query)
        query_elapsed = time.time() - query_start_time
        self.logger.info(f"Query completed in {query_elapsed:.2f}s: found language data for {len(results)} repositories")
        
        # Process results
        language_info = {}
        for result in results:
            repo_name = result.get('repo_name')
            if repo_name and 'languages' in result:
                language_info[repo_name] = result['languages']
        
        # Calculate stats
        repos_with_language = len(language_info)
        total_languages = sum(len(langs) for langs in language_info.values())
        elapsed_time = time.time() - start_time
        
        self.logger.info(
            f"Language query completed in {elapsed_time:.2f}s: found data for {repos_with_language}/{len(repos)} repos"
        )
        
        # Generate language statistics if data was found
        if language_info:
            all_languages = []
            for repo_langs in language_info.values():
                for lang_entry in repo_langs:
                    if 'language' in lang_entry:
                        all_languages.append(lang_entry['language'])
            
            language_counts = {}
            for lang in all_languages:
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            top_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_langs_str = ", ".join([f"{lang}: {count}" for lang, count in top_languages])
            self.logger.info(f"Top languages: {top_langs_str}")
        
        return language_info