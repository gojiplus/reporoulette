"""Sampling from BigQuery's public GitHub datasets."""

import logging
import random
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from .base import BaseSampler
from .bq_utils import execute_query, format_timestamp_query

if TYPE_CHECKING:
    from google.cloud import (
        bigquery,
    )
    from google.oauth2 import service_account

# Runtime imports with fallback
bigquery = None
service_account = None
_bigquery_available = False
try:
    from google.cloud import bigquery
    from google.oauth2 import service_account

    _bigquery_available = True
except ImportError:
    pass

BIGQUERY_AVAILABLE = _bigquery_available


class BigQuerySampler(BaseSampler):
    """Sample repositories using Google BigQuery's GitHub dataset.

    This sampler leverages the public GitHub dataset in Google BigQuery to
    efficiently sample repositories with complex criteria and at scale.
    The population is repositories generating GH Archive events on the
    sampled days, so the sample is biased toward active repositories.
    """

    def __init__(
        self,
        credentials_path: str | None = None,
        project_id: str | None = None,
        seed: int | None = None,
        log_level: int = logging.INFO,
    ):
        """Initialize the BigQuery sampler with credentials and configuration."""
        super().__init__(token=None)  # GitHub token not used for BigQuery

        # Configure logger
        self.logger: logging.Logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            self._seed = seed
            self.logger.info("Random seed set to: %s", seed)
        else:
            self._seed = random.randint(1, 1000000)
            self.logger.info("Generated random seed: %s", self._seed)

        if not BIGQUERY_AVAILABLE:
            error_msg = (
                "BigQuery dependencies not installed. Install with "
                "pip install google-cloud-bigquery google-auth"
            )
            self.logger.error(error_msg)
            raise ImportError(error_msg)

        self.credentials_path: str | None = credentials_path
        self.project_id: str | None = project_id
        self.client: Any = None  # BigQuery client will be initialized in _init_client

        self.logger.info(
            "Initializing BigQuery client (project_id: %s)", project_id or "default"
        )
        self._init_client()

        # Initialize tracking variables
        self.attempts: int = 0
        self.success_count: int = 0
        self.results: list[dict[str, Any]] = []

    def _init_client(self) -> None:
        """Initialize the BigQuery client."""
        if not BIGQUERY_AVAILABLE:
            raise ImportError("BigQuery dependencies are not available")

        try:
            if self.credentials_path:
                self.logger.info(
                    "Using service account credentials from: %s", self.credentials_path
                )
                if service_account is None:
                    raise ImportError("google.oauth2.service_account is not available")
                # google-auth ships no type stubs for this constructor
                credentials = service_account.Credentials.from_service_account_file(  # pyright: ignore[reportUnknownMemberType]
                    self.credentials_path
                )
                if bigquery is None:
                    raise ImportError("google.cloud.bigquery is not available")
                self.client = bigquery.Client(
                    credentials=credentials, project=self.project_id
                )
            else:
                self.logger.info("Using default credentials from environment")
                if bigquery is None:
                    raise ImportError("google.cloud.bigquery is not available")
                self.client = bigquery.Client(project=self.project_id)
            self.logger.info(
                "BigQuery client initialized for project: %s", self.client.project
            )
        except Exception as e:
            self.logger.error("Failed to initialize BigQuery client: %s", e)
            raise

    def _execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a BigQuery query and track success metrics."""
        self.attempts += 1
        results = execute_query(self.client, query, self.logger)
        if results:
            self.success_count += 1
        return results

    def _random_day_suffixes(self, days_to_sample: int, years_back: int) -> list[str]:
        """Pick seeded random day-table suffixes client-side.

        Suffixes are YYYYMMDD without the leading '2', generated here so
        the wildcard query can prune to just those tables instead of
        scanning every githubarchive.day table.

        Args:
            days_to_sample: Number of random days to draw (collisions are
                deduplicated, so fewer may be returned)
            years_back: How many years back to draw from

        Returns:
            Sorted list of _TABLE_SUFFIX literals
        """
        rng = random.Random(self._seed)
        today = datetime.now(UTC).date()
        suffixes = {
            (today - timedelta(days=rng.randint(1, 365 * years_back))).strftime(
                "%Y%m%d"
            )[1:]
            for _ in range(days_to_sample)
        }
        return sorted(suffixes)

    def _build_count_query(self, days_to_sample: int, years_back: int) -> str:
        """Build SQL query to count repositories on seeded random days.

        The day list is generated client-side and inlined as _TABLE_SUFFIX
        literals: BigQuery only prunes wildcard tables on constant filters,
        so computing the days in SQL forced a scan of every day table
        (~0.2 TB); with literals the query scans only the selected days.
        """
        suffix_list = ", ".join(
            f"'{s}'" for s in self._random_day_suffixes(days_to_sample, years_back)
        )
        # S608: wildcard-table pruning requires inlined constant suffixes
        # (see docstring); every value is generated client-side from the seed.
        return f"""
        SELECT
          CONCAT('2', _TABLE_SUFFIX) AS sample_day,
          COUNT(DISTINCT repo.name) AS repo_count
        FROM `githubarchive.day.2*`
        WHERE _TABLE_SUFFIX IN ({suffix_list})
        GROUP BY sample_day
        HAVING COUNT(DISTINCT repo.name) > 0
        ORDER BY repo_count DESC
        """  # noqa: S608

    def _build_day_query(self, day_data: dict[str, Any], i: int) -> str:
        """Build SQL query to sample repositories from a specific day."""
        day = day_data.get("sample_day")
        repo_count = day_data.get("repo_count", 0)
        samples_to_take = day_data.get("samples_to_take", 1)

        # S608: same constant-pruning constraint; day/count/seed are
        # internally generated, never user strings.
        return f"""
        SELECT
            repo.name AS full_name,
            SPLIT(repo.name, '/')[SAFE_OFFSET(1)] AS name,
            SPLIT(repo.name, '/')[SAFE_OFFSET(0)] AS owner,
            CONCAT('https://github.com/', repo.name) AS html_url,
            MIN(created_at) AS created_at,
            '{day}' AS sampled_from,
            ANY_VALUE(type) AS event_type,
            {repo_count} AS day_repo_count,
            {samples_to_take} AS samples_allocated
        FROM `githubarchive.day.{day}`
        WHERE repo.name IS NOT NULL
          AND repo.name LIKE '%/%'
        GROUP BY repo.name
        ORDER BY FARM_FINGERPRINT(CONCAT(repo.name, '-{self._seed}-{i}'))
        LIMIT {samples_to_take}
        """  # noqa: S608

    def _combine_day_queries(self, day_queries: list[str], n_samples: int) -> str:
        """Combine day queries into final query and deduplicate results."""
        # UNION ALL operands that carry ORDER BY/LIMIT must be parenthesized
        combined_query = "\nUNION ALL\n".join(f"({q})" for q in day_queries)
        return f"""
        SELECT
            full_name,
            ANY_VALUE(name) AS name,
            ANY_VALUE(owner) AS owner,
            ANY_VALUE(html_url) AS html_url,
            MIN(created_at) AS created_at,
            ANY_VALUE(sampled_from) AS sampled_from,
            ANY_VALUE(event_type) AS event_type,
            ANY_VALUE(day_repo_count) AS day_repo_count,
            ANY_VALUE(samples_allocated) AS samples_allocated
        FROM (
            {combined_query}
        )
        GROUP BY full_name
        ORDER BY FARM_FINGERPRINT(CONCAT(full_name, '-{self._seed}'))
        LIMIT {n_samples}
        """  # noqa: S608

    def sample_by_day(
        self,
        n_samples: int = 100,
        days_to_sample: int = 10,
        repos_per_day: int = 50,
        years_back: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Sample repositories using a day-based approach with GitHub Archive tables."""
        self.logger.info(
            "Sampling %s repositories across %s days", n_samples, days_to_sample
        )
        if kwargs:
            self.logger.info("Filter criteria: %s", kwargs)

        # Adjust days to sample if needed
        days_needed = max(1, (n_samples + repos_per_day - 1) // repos_per_day)
        days_to_sample = max(days_to_sample, days_needed)
        self.logger.debug("Adjusted days_to_sample: %s", days_to_sample)

        count_query = self._build_count_query(days_to_sample, years_back)
        day_counts = self._execute_query(count_query)
        if not day_counts:
            self.logger.warning("No repositories found for the selected days")
            return []

        total_repos = sum(day.get("repo_count", 0) for day in day_counts)
        for day in day_counts:
            day_repo_count = day.get("repo_count", 0)
            day_weight = day_repo_count / total_repos if total_repos > 0 else 0
            weighted_samples = max(1, int(n_samples * day_weight))
            max_allowed = min(repos_per_day, day_repo_count)
            day["samples_to_take"] = min(weighted_samples, max_allowed)

        self.logger.info(
            "Found %s days with %s total repositories", len(day_counts), total_repos
        )

        day_queries: list[str] = []
        for i, day in enumerate(day_counts):
            if day.get("samples_to_take", 0) <= 0:
                continue
            day_query = self._build_day_query(day, i)
            day_queries.append(day_query)

        final_query = self._combine_day_queries(day_queries, n_samples)
        valid_repos = self._execute_query(final_query)
        self.results: list[dict[str, Any]] = valid_repos

        filtered_count_before = len(valid_repos)
        if kwargs:
            self.results = self._filter_repos(valid_repos, **kwargs)
            filtered_count_after = len(self.results)
            if filtered_count_before != filtered_count_after:
                self.logger.info(
                    "Applied filters: %s/%s repositories retained",
                    filtered_count_after,
                    filtered_count_before,
                )

        self.logger.info(
            "Completed day-based sampling: found %s repositories", len(self.results)
        )

        if valid_repos:
            day_counts_map: dict[str, dict[str, int]] = {}
            for repo in valid_repos:
                day_sampled = repo.get("sampled_from", "unknown")
                day_repo_count = repo.get("day_repo_count", 0)
                allocated = repo.get("samples_allocated", 0)
                if day_sampled not in day_counts_map:
                    day_counts_map[day_sampled] = {
                        "count": 0,
                        "repos": day_repo_count,
                        "allocated": allocated,
                    }
                day_counts_map[day_sampled]["count"] += 1
            self.logger.info("Sampled from %s different days", len(day_counts_map))
            for day_str, data in sorted(day_counts_map.items()):
                self.logger.debug(
                    "Day %s: %s/%s samples from %s repos",
                    day_str,
                    data["count"],
                    data["allocated"],
                    data["repos"],
                )

        return self.results

    def sample_active(
        self,
        n_samples: int = 100,
        created_after: str | datetime | None = None,
        created_before: str | datetime | None = None,
        languages: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Sample repositories with recent commit activity.

        Args:
            n_samples: Number of repositories to sample
            created_after: Filter commits after this timestamp
            created_before: Filter commits before this timestamp
            languages: List of programming languages to filter by
                (uses github_repos.languages)
            **kwargs: Additional filter criteria

        Returns:
            List of repository dictionaries
        """
        self.logger.info(
            "Sampling %s active repositories based on commit history", n_samples
        )
        if kwargs:
            self.logger.info("Filter criteria: %s", kwargs)

        if created_after:
            created_after_str = format_timestamp_query(created_after)
        else:
            one_year_ago = (datetime.now(UTC) - timedelta(days=365)).strftime(
                "%Y-%m-%d"
            )
            created_after_str = f"'{one_year_ago}'"

        if created_before:
            created_before_str = format_timestamp_query(created_before)
        else:
            created_before_str = "CURRENT_TIMESTAMP()"

        self.logger.info("Time period: %s to %s", created_after_str, created_before_str)

        query = self._build_active_query(
            created_after_str, created_before_str, languages, n_samples
        )

        valid_repos = self._execute_query(query)
        self.results = valid_repos

        filtered_count_before = len(valid_repos)
        if kwargs:
            self.results = self._filter_repos(valid_repos, **kwargs)
            filtered_count_after = len(self.results)
            if filtered_count_before != filtered_count_after:
                self.logger.info(
                    "Applied filters: %s/%s repositories retained",
                    filtered_count_after,
                    filtered_count_before,
                )

        self.logger.info(
            "Completed active repository sampling: found %s repositories",
            len(self.results),
        )
        return self.results

    def _build_active_query(
        self,
        created_after_str: str,
        created_before_str: str,
        languages: list[str] | None,
        n_samples: int,
    ) -> str:
        """Build the SQL query for sample_active.

        Args:
            created_after_str: SQL-formatted lower timestamp bound
            created_before_str: SQL-formatted upper timestamp bound
            languages: Optional list of languages to filter by
            n_samples: Number of repositories to sample

        Returns:
            SQL query string
        """
        if languages:
            lang_list = ", ".join([f"'{lang}'" for lang in languages])
            self.logger.info("Filtering for languages: %s", lang_list)
            # Use JOIN with languages table for proper language filtering
            return f"""
            WITH repo_set AS (
                SELECT DISTINCT
                    repo AS full_name,
                    SPLIT(repo, '/')[OFFSET(1)] AS name,
                    SPLIT(repo, '/')[OFFSET(0)] AS owner
                FROM
                    `bigquery-public-data.github_repos.commits` c,
                    UNNEST(c.repo_name) AS repo
                WHERE
                    TIMESTAMP_SECONDS(c.committer.time_sec)
                        BETWEEN TIMESTAMP({created_after_str})
                        AND TIMESTAMP({created_before_str})
            )
            SELECT
                rs.full_name,
                ANY_VALUE(rs.name) AS name,
                ANY_VALUE(rs.owner) AS owner,
                CONCAT('https://github.com/', rs.full_name) AS html_url
            FROM repo_set rs
            JOIN `bigquery-public-data.github_repos.languages` l
                ON rs.full_name = l.repo_name,
                UNNEST(l.language) AS lang
            WHERE lang.name IN ({lang_list})
            GROUP BY rs.full_name
            ORDER BY FARM_FINGERPRINT(CONCAT(rs.full_name, '-{self._seed}'))
            LIMIT {n_samples}
            """  # noqa: S608
        return f"""
        WITH repo_set AS (
            SELECT DISTINCT
                repo AS full_name,
                SPLIT(repo, '/')[OFFSET(1)] AS name,
                SPLIT(repo, '/')[OFFSET(0)] AS owner
            FROM
                `bigquery-public-data.github_repos.commits` c,
                UNNEST(c.repo_name) AS repo
            WHERE
                TIMESTAMP_SECONDS(c.committer.time_sec)
                    BETWEEN TIMESTAMP({created_after_str})
                    AND TIMESTAMP({created_before_str})
        )
        SELECT
            full_name,
            name,
            owner,
            CONCAT('https://github.com/', full_name) AS html_url
        FROM repo_set
        ORDER BY FARM_FINGERPRINT(CONCAT(full_name, '-{self._seed}'))
        LIMIT {n_samples}
        """  # noqa: S608

    def sample(
        self, n_samples: int = 100, population: str = "all", **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Sample repositories using BigQuery.

        Args:
            n_samples: Number of repositories to sample
            population: Type of repository population to sample from ('all' or 'active')
            **kwargs: Additional filtering criteria

        Returns:
            List of repository dictionaries
        """
        self.logger.info(
            "Starting repository sampling: n_samples=%s, population=%s",
            n_samples,
            population,
        )
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
            "Sampling completed in %.2fs: %s repositories found",
            elapsed_time,
            len(results),
        )

        return results

    def _build_languages_query(self, repo_names: list[str]) -> str:
        """Build the SQL query fetching language breakdowns for repositories.

        Args:
            repo_names: Repository full names to look up

        Returns:
            SQL query string
        """
        repo_list = ", ".join([f"'{repo}'" for repo in repo_names])
        return f"""
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
        """  # noqa: S608

    def get_languages(
        self, repos: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieve language information for a list of repositories."""
        self.logger.info(
            "Fetching language information for %s repositories", len(repos)
        )
        start_time = time.time()

        repo_names = [repo["full_name"] for repo in repos if "full_name" in repo]
        if not repo_names:
            self.logger.warning("No valid repository names found")
            return {}

        # Process all repositories in a single query
        query = self._build_languages_query(repo_names)

        query_start_time = time.time()
        results = self._execute_query(query)
        query_elapsed = time.time() - query_start_time
        self.logger.info(
            "Query completed in %.2fs: found language data for %s repositories",
            query_elapsed,
            len(results),
        )

        # Process results
        language_info: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            repo_name = result.get("repo_name")
            if repo_name and "languages" in result:
                language_info[repo_name] = result["languages"]

        # Calculate stats
        repos_with_language = len(language_info)
        elapsed_time = time.time() - start_time

        self.logger.info(
            "Language query completed in %.2fs: found data for %s/%s repos",
            elapsed_time,
            repos_with_language,
            len(repos),
        )

        # Generate language statistics if data was found
        if language_info:
            all_languages: list[str] = [
                lang_entry["language"]
                for repo_langs in language_info.values()
                for lang_entry in repo_langs
                if "language" in lang_entry
            ]

            language_counts: dict[str, int] = {}
            for lang in all_languages:
                language_counts[lang] = language_counts.get(lang, 0) + 1

            top_languages = sorted(
                language_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
            top_langs_str = ", ".join(
                [f"{lang}: {count}" for lang, count in top_languages]
            )
            self.logger.info("Top languages: %s", top_langs_str)

        return language_info
