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

    def _init_client(self):
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

    def _execute_query(self, query: str) -> List[Dict]:
        """
        Execute a BigQuery query using the utility function.
        """
        self.attempts += 1
        results = execute_query(self.client, query, self.logger)
        if results:
            self.success_count += 1
        return results

    def _adjust_hours_to_sample(self, n_samples: int, repos_per_hour: int, hours_to_sample: int) -> int:
        """
        Adjust the number of hours to sample based on target samples and max repos per hour.
        """
        hours_needed = max(1, (n_samples + repos_per_hour - 1) // repos_per_hour)
        return max(hours_to_sample, hours_needed)

    def _build_count_query(self, hours_to_sample: int, years_back: int) -> str:
        """
        Build the SQL query that creates a temporary table of random (day, hour)
        pairs and counts unique repositories for each (day, hour) from daily GitHub Archive tables.
        """
        return f"""
        -- Define parameters
        DECLARE hours_to_sample INT64 DEFAULT {hours_to_sample};
        DECLARE years_back INT64 DEFAULT {years_back};

        -- Create a table of random dates and hours to sample from
        CREATE TEMP TABLE random_dates AS (
          SELECT 
            FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL CAST(FLOOR(RAND({self._seed}) * (365 * years_back)) AS INT64) DAY)) AS day,
            CAST(FLOOR(RAND({self._seed}) * 24) AS INT64) AS hour
          FROM UNNEST(GENERATE_ARRAY(1, hours_to_sample))
        );

        -- Count unique repositories per (day, hour)
        SELECT
          rd.day AS sample_day,
          rd.hour AS sample_hour,
          CONCAT(rd.day, '-', FORMAT('%02d', rd.hour)) AS hour_key,
          COUNT(DISTINCT event.repo_name) AS repo_count
        FROM random_dates rd
        CROSS JOIN (
          SELECT DISTINCT day
          FROM random_dates
        ) dt
        CROSS JOIN (
          SELECT repo.name AS repo_name, created_at
          FROM (
            EXECUTE IMMEDIATE FORMAT(
              "SELECT repo.name, created_at
               FROM `githubarchive.day.%s`
               WHERE EXTRACT(HOUR FROM created_at) = %d
               AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL %d YEAR)
               LIMIT 100000",
              dt.day,
              rd.hour,
              years_back
            )
          )
        ) event
        WHERE rd.day = dt.day
        GROUP BY rd.day, rd.hour
        ORDER BY repo_count DESC
        """

    def _build_bucket_query(self, hour: Dict[str, Any], i: int, years_back: int) -> str:
        """
        Build an individual SQL query to sample repositories from a specific (day, hour) bucket.
        """
        day = hour.get('sample_day')
        hour_num = hour.get('sample_hour')
        hour_key = hour.get('hour_key')
        repo_count = hour.get('repo_count', 0)
        samples_to_take = hour.get('samples_to_take', 1)
        return f"""
        -- Hour bucket {i+1}: {hour_key} with {repo_count} repositories
        SELECT DISTINCT
            event.repo_name AS full_name,
            SPLIT(event.repo_name, '/')[SAFE_OFFSET(1)] AS name,
            SPLIT(event.repo_name, '/')[SAFE_OFFSET(0)] AS owner,
            event.repo_url AS html_url,
            event.created_at,
            '{hour_key}' AS sampled_from,
            event.event_type,
            {repo_count} AS hour_repo_count,
            {samples_to_take} AS samples_allocated
        FROM (
            EXECUTE IMMEDIATE FORMAT(
                "SELECT
                    repo.name AS repo_name,
                    repo.url AS repo_url,
                    created_at,
                    type AS event_type
                 FROM `githubarchive.day.%s`
                 WHERE EXTRACT(HOUR FROM created_at) = %d
                 AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL %d YEAR)
                 LIMIT 100000",
                '{day}',
                {hour_num},
                {years_back}
            )
        ) AS event
        ORDER BY RAND({self._seed} + {i})
        LIMIT {samples_to_take}
        """

    def _combine_bucket_queries(self, bucket_queries: List[str], n_samples: int) -> str:
        """
        Combine individual bucket queries into one final query and deduplicate the results.
        """
        combined_query = "\nUNION ALL\n".join(bucket_queries)
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
            hour_repo_count,
            samples_allocated
        FROM (
            {combined_query}
        )
        ORDER BY RAND({self._seed})
        LIMIT {n_samples}
        """

    def sample_by_datetime_weighted(
        self,
        n_samples: int = 100,
        time_buckets_to_sample: int = 30,
        repos_per_bucket: int = 10,
        years_back: int = 10,
        time_bucket_interval: str = "DAY",  # Options: "HOUR", "DAY", "WEEK", "MONTH"
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories with proper weighting based on unique repository counts in time buckets.
        """
        self.logger.info(
            f"Starting time-weighted sampling: n_samples={n_samples}, "
            f"time_buckets_to_sample={time_buckets_to_sample}, "
            f"time_bucket_interval={time_bucket_interval}"
        )

        # Validate time bucket interval
        valid_intervals = ["HOUR", "DAY", "WEEK", "MONTH"]
        if time_bucket_interval not in valid_intervals:
            raise ValueError(f"time_bucket_interval must be one of {valid_intervals}")

        if kwargs:
            self.logger.info(f"Filter criteria: {kwargs}")

        self.logger.info("Querying for repository counts per time bucket...")
        time_bucket_query = f"""
        -- Get repository counts by time bucket
        WITH time_buckets AS (
          SELECT
            TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(c.committer.time_sec), {time_bucket_interval}) AS time_bucket,
            COUNT(DISTINCT repo) AS repo_count
          FROM
            `bigquery-public-data.github_repos.commits` c,
            UNNEST(c.repo_name) AS repo
          WHERE
            c.committer.time_sec >= UNIX_SECONDS(TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {years_back} YEAR))
          GROUP BY
            time_bucket
          ORDER BY
            time_bucket DESC
        )
        SELECT
          FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', time_bucket) AS time_bucket_str,
          time_bucket,
          repo_count,
          PERCENT_RANK() OVER (ORDER BY repo_count) AS percentile
        FROM
          time_buckets
        ORDER BY repo_count DESC
        LIMIT {time_buckets_to_sample}
        """
        time_buckets = self._execute_query(time_bucket_query)
        if not time_buckets:
            self.logger.warning("No time buckets found. Check your query parameters.")
            return []

        total_repos = sum(bucket.get('repo_count', 0) for bucket in time_buckets)
        avg_repos = total_repos / len(time_buckets) if time_buckets else 0
        self.logger.info(
            f"Found {len(time_buckets)} time buckets with {total_repos} total repositories "
            f"(avg {avg_repos:.1f} repos per bucket)"
        )

        # Allocate samples for each bucket
        for bucket in time_buckets:
            bucket_repo_count = bucket.get('repo_count', 0)
            bucket_weight = bucket_repo_count / total_repos if total_repos > 0 else 0
            bucket['samples_to_take'] = min(
                max(1, int(n_samples * bucket_weight)),
                min(repos_per_bucket, bucket_repo_count)
            )

        self.logger.info("Sampling repositories from selected time buckets...")
        bucket_queries = []
        for i, bucket in enumerate(time_buckets):
            time_bucket_str = bucket.get('time_bucket_str')
            repo_count = bucket.get('repo_count', 0)
            samples_to_take = bucket.get('samples_to_take', 1)
            if samples_to_take <= 0:
                continue
            bucket_query = f"""
            -- Time bucket {i+1}: {time_bucket_str} with {repo_count} repositories
            SELECT DISTINCT
                repo AS full_name,
                SPLIT(repo, '/')[OFFSET(1)] AS name,
                SPLIT(repo, '/')[OFFSET(0)] AS owner,
                '{time_bucket_str}' AS sampled_time_bucket,
                {repo_count} AS time_bucket_repo_count
            FROM 
                `bigquery-public-data.github_repos.commits` c,
                UNNEST(c.repo_name) AS repo
            WHERE 
                TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(c.committer.time_sec), {time_bucket_interval}) = TIMESTAMP('{time_bucket_str}')
            ORDER BY
                RAND({self._seed} + {i})
            LIMIT {samples_to_take}
            """
            bucket_queries.append(bucket_query)

        combined_query = "\nUNION ALL\n".join(bucket_queries)
        final_query = f"""
        -- Combined query across all time buckets
        SELECT *
        FROM (
            {combined_query}
        )
        ORDER BY RAND({self._seed})
        LIMIT {n_samples}
        """
        valid_repos = self._execute_query(final_query)
        self.results = valid_repos

        filtered_count_before = len(valid_repos)
        if kwargs:
            self.results = filter_repos(valid_repos, **kwargs)
            filtered_count_after = len(self.results)
            if filtered_count_before != filtered_count_after:
                self.logger.info(
                    f"Applied filters: {filtered_count_before - filtered_count_after} repositories filtered out, "
                    f"{filtered_count_after} repositories remaining"
                )

        self.logger.info(
            f"Time-weighted sampling completed: found {len(valid_repos)} repositories "
            f"(success rate: {(self.success_count / self.attempts) * 100:.1f}%)"
        )

        if valid_repos:
            bucket_counts = {}
            for repo in valid_repos:
                bucket = repo.get('sampled_time_bucket', 'unknown')
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            self.logger.info(f"Time bucket distribution in results: {bucket_counts}")

        return self.results

    def sample_by_datetime(
        self,
        n_samples: int = 100,
        hours_to_sample: int = 10,
        repos_per_hour: int = 50,
        years_back: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using a datetime-based approach with daily GitHub Archive tables.
        """
        self.logger.info(
            f"Starting datetime sampling: n_samples={n_samples}, hours_to_sample={hours_to_sample}, "
            f"repos_per_hour={repos_per_hour}, years_back={years_back}"
        )
        if kwargs:
            self.logger.info(f"Filter criteria: {kwargs}")

        hours_to_sample = self._adjust_hours_to_sample(n_samples, repos_per_hour, hours_to_sample)
        self.logger.debug(f"Adjusted hours_to_sample: {hours_to_sample}")

        count_query = self._build_count_query(hours_to_sample, years_back)
        hour_counts = self._execute_query(count_query)
        if not hour_counts:
            self.logger.warning("No repository counts found for the selected hours")
            return []

        total_repos = sum(hour.get('repo_count', 0) for hour in hour_counts)
        for hour in hour_counts:
            hour_repo_count = hour.get('repo_count', 0)
            hour_weight = hour_repo_count / total_repos if total_repos > 0 else 0
            hour['samples_to_take'] = min(
                max(1, int(n_samples * hour_weight)),
                min(repos_per_hour, hour_repo_count)
            )

        self.logger.info(f"Found {len(hour_counts)} hours with {total_repos} total repositories")

        bucket_queries = []
        for i, hour in enumerate(hour_counts):
            if hour.get('samples_to_take', 0) <= 0:
                continue
            bucket_query = self._build_bucket_query(hour, i, years_back)
            bucket_queries.append(bucket_query)

        final_query = self._combine_bucket_queries(bucket_queries, n_samples)
        valid_repos = self._execute_query(final_query)
        self.results = valid_repos

        filtered_count_before = len(valid_repos)
        if kwargs:
            self.results = filter_repos(valid_repos, **kwargs)
            filtered_count_after = len(self.results)
            if filtered_count_before != filtered_count_after:
                self.logger.info(
                    f"Applied filters: {filtered_count_before - filtered_count_after} repositories filtered out, "
                    f"{filtered_count_after} repositories remaining"
                )

        self.logger.info(
            f"Datetime sampling completed: found {len(valid_repos)} repositories "
            f"(success rate: {(self.success_count / self.attempts) * 100:.1f}%)"
        )

        if valid_repos:
            hour_counts_map = {}
            for repo in valid_repos:
                hour_sampled = repo.get('sampled_from', 'unknown')
                hour_repo_count = repo.get('hour_repo_count', 0)
                allocated = repo.get('samples_allocated', 0)
                if hour_sampled not in hour_counts_map:
                    hour_counts_map[hour_sampled] = {'count': 0, 'repos': hour_repo_count, 'allocated': allocated}
                hour_counts_map[hour_sampled]['count'] += 1
            self.logger.info(f"Sampled from {len(hour_counts_map)} different hours")
            for hour_str, data in sorted(hour_counts_map.items()):
                self.logger.debug(
                    f"Hour {hour_str}: {data['count']}/{data['allocated']} samples from {data['repos']} repos"
                )

        return self.results

    def sample_standard(
        self,
        n_samples: int = 100,
        created_after: Optional[Union[str, datetime]] = None,
        created_before: Optional[Union[str, datetime]] = None,
        languages: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using the standard BigQuery approach.
        """
        self.logger.info(f"Starting standard sampling: n_samples={n_samples}")
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

        self.logger.info(f"Date range: {created_after} to {created_before}")

        lang_list = None
        if languages:
            lang_list = ", ".join([f"'{lang}'" for lang in languages])
            self.logger.info(f"Filtering for owners: {lang_list}")

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
                {('AND SPLIT(repo, \'/\')[OFFSET(0)] IN (' + lang_list + ')') if languages else ''}
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
                    f"Applied filters: {filtered_count_before - filtered_count_after} repositories filtered out, "
                    f"{filtered_count_after} repositories remaining"
                )

        self.logger.info(f"Standard sampling completed: found {len(valid_repos)} repositories")
        return self.results

    def sample(
        self, 
        n_samples: int = 100,
        sampling_method: str = "time_weighted",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories using BigQuery.
        """
        self.logger.info(f"Starting repository sampling with n_samples={n_samples}, method={sampling_method}")
        start_time = time.time()

        self.attempts = 0
        self.success_count = 0

        if sampling_method == "time_weighted":
            self.logger.info("Using time-weighted sampling method")
            results = self.sample_by_datetime_weighted(n_samples=n_samples, **kwargs)
        elif sampling_method == "datetime":
            self.logger.info("Using datetime sampling method")
            results = self.sample_by_datetime(n_samples=n_samples, **kwargs)
        else:  # standard
            self.logger.info("Using standard sampling method")
            results = self.sample_standard(n_samples=n_samples, **kwargs)

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Sampling completed in {elapsed_time:.2f} seconds: "
            f"found {len(results)} repositories, "
            f"{self.attempts} attempts, {self.success_count} successful queries"
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

        chunk_size = 100
        repo_chunks = [repo_names[i:i + chunk_size] for i in range(0, len(repo_names), chunk_size)]
        self.logger.info(f"Processing {len(repo_chunks)} chunks (max {chunk_size} per chunk)")

        language_info = {}
        for i, chunk in enumerate(repo_chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(repo_chunks)} with {len(chunk)} repositories")
            repo_list = ", ".join([f"'{repo}'" for repo in chunk])
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
            chunk_start_time = time.time()
            results = self._execute_query(query)
            chunk_elapsed = time.time() - chunk_start_time
            self.logger.info(f"Chunk {i+1} query completed in {chunk_elapsed:.2f} seconds with {len(results)} language records")

            for result in results:
                repo_name = result.get('repo_name')
                if repo_name:
                    if repo_name not in language_info:
                        language_info[repo_name] = []
                    language_info[repo_name].append({
                        'language': result.get('language'),
                        'bytes': result.get('bytes')
                    })

        repos_with_language = len(language_info)
        total_languages = sum(len(langs) for langs in language_info.values())
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Language query completed in {elapsed_time:.2f} seconds: "
            f"found information for {repos_with_language}/{len(repos)} repositories "
            f"({total_languages} language entries total)"
        )

        if language_info:
            all_languages = []
            for repo_langs in language_info.values():
                for lang in repo_langs:
                    if 'language' in lang:
                        all_languages.append(lang['language'])
            language_counts = {}
            for lang in all_languages:
                language_counts[lang] = language_counts.get(lang, 0) + 1
            top_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_langs_str = ", ".join([f"{lang}: {count}" for lang, count in top_languages])
            self.logger.info(f"Top languages: {top_langs_str}")

        return language_info
