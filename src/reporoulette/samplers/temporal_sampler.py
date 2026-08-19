"""Sampling repositories created in randomly chosen time windows."""

import logging
import random
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from ..logging_config import get_logger
from .base import HTTP_OK, BaseSampler


class TemporalSampler(BaseSampler):
    """Sample repositories by randomly picking days and fetching repos pushed then.

    This sampler selects random days within a specified date range,
    weights them by repository count, and retrieves repositories with
    proportional sampling.
    The population is repositories *pushed* on the sampled days, so the sample
    is biased toward actively maintained repositories, and the Search API's
    1,000-results-per-query cap limits coverage on high-activity days.
    """

    def __init__(
        self,
        token: str | None = None,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        rate_limit_safety: int = 5,
        seed: int | None = None,
        years_back: int = 10,
        log_level: int = logging.INFO,
    ):
        """Initialize the temporal sampler.

        Args:
            token: GitHub Personal Access Token
            start_date: Start of date range to sample from
            end_date: End of date range to sample from
            rate_limit_safety: Stop when this many search API requests remain.
                The search bucket allows only 30 requests/minute, so this must
                stay below 30 (a safety of 100 would block every request).
            seed: Random seed for reproducibility
            years_back: How many years back to sample from (if start_date not specified)
            log_level: Logging level (default: INFO)
        """
        super().__init__(token)

        # Configure logger
        self.logger: logging.Logger = get_logger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            self._seed = seed
            self.logger.info("Random seed set to: %s", seed)
        else:
            self._seed = None

        # GitHub search qualifiers are UTC; naive inputs are assumed UTC so
        # user-supplied dates and the aware default stay comparable.
        def as_utc(value: datetime) -> datetime:
            return value if value.tzinfo else value.replace(tzinfo=UTC)

        # Default to current time for end_date if not specified
        if end_date is None:
            self.end_date: datetime = datetime.now(UTC)
        elif isinstance(end_date, str):
            self.end_date = as_utc(datetime.fromisoformat(end_date))
        else:
            self.end_date = as_utc(end_date)

        # Use years_back parameter instead of fixed 90 days
        if start_date is None:
            self.start_date: datetime = self.end_date - timedelta(days=365 * years_back)
        elif isinstance(start_date, str):
            self.start_date = as_utc(datetime.fromisoformat(start_date))
        else:
            self.start_date = as_utc(start_date)

        # Ensure dates have no time component for consistent day-level sampling
        self.start_date = self.start_date.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.end_date = self.end_date.replace(
            hour=23, minute=59, second=59, microsecond=999999
        )

        self.rate_limit_safety = rate_limit_safety
        # Requests go through the Search API, whose bucket allows only
        # 30 requests/minute (vs 5,000/hour for core)
        self.rate_limit_resource = "search"
        self.api_base_url = "https://api.github.com"

        time_delta = self.end_date - self.start_date

        self.logger.info(
            "Initialized TemporalSampler with date range: %s to %s (%s days)",
            self.start_date.strftime("%Y-%m-%d"),
            self.end_date.strftime("%Y-%m-%d"),
            time_delta.days,
        )

        # Initialize tracking variables
        self.attempts: int = 0
        self.success_count: int = 0
        self.results: list[dict[str, Any]] = []

    def _random_date(self) -> datetime:
        """Generate a random date within the specified range.

        Returns:
            Random datetime object with time set to beginning of day
        """
        time_delta = self.end_date - self.start_date
        random_days = random.randint(0, time_delta.days)
        random_date = self.start_date + timedelta(days=random_days)

        # Set to beginning of day
        return random_date.replace(hour=0, minute=0, second=0, microsecond=0)

    def _format_date_for_query(self, dt: datetime) -> tuple[str, str]:
        """Format a date for GitHub API query.

        Args:
            dt: Date to format

        Returns:
            Tuple of (start, end) strings for the day period
        """
        # Set to beginning of day
        dt_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        # Set to end of day
        dt_next_day = dt_day + timedelta(days=1)

        # Format for GitHub API with Z suffix for UTC
        start_str = dt_day.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        end_str = dt_next_day.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

        return start_str, end_str

    def _build_search_query(
        self,
        start_time_str: str,
        end_time_str: str,
        min_stars: int = 0,
        min_size_kb: int = 0,
        language: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Build a search query string for the GitHub API.

        Args:
            start_time_str: Start time in ISO format
            end_time_str: End time in ISO format
            min_stars: Minimum number of stars
            min_size_kb: Minimum repository size in KB
            language: Programming language to filter by
            **kwargs: Additional filters

        Returns:
            Query string
        """
        # Construct query for repositories updated in this time period
        query_parts = [f"pushed:{start_time_str}..{end_time_str}"]

        # Add language filter if specified. Multiple language: qualifiers are
        # ANDed by the search API (matching nothing), so with more than one
        # language the qualifier is omitted and filtering happens client-side
        # in _filter_repos.
        if language:
            query_parts.append(f"language:{language}")
        elif kwargs.get("languages"):
            languages = kwargs["languages"]
            if len(languages) == 1:
                query_parts.append(f"language:{languages[0]}")
            else:
                self.logger.warning(
                    "Multiple languages %s: querying without a language qualifier "
                    "and filtering client-side",
                    languages,
                )

        # Add star filter if specified
        if min_stars > 0:
            query_parts.append(f"stars:>={min_stars}")

        # Add size filter if specified
        if min_size_kb > 0:
            query_parts.append(f"size:>={min_size_kb}")

        # Join query parts
        return " ".join(query_parts)

    def sample(
        self,
        n_samples: int = 100,  # Number of repositories to collect
        days_to_sample: int = 10,  # Changed from hours_to_sample
        per_page: int = 100,
        min_wait: float = 1.0,
        min_stars: int = 0,
        min_size_kb: int = 0,
        language: str | None = None,
        max_attempts: int = 100,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Sample repositories from random days, weighted by each day's repo count.

        Args:
            n_samples: Target number of repositories to collect. Collection
                proceeds a full search page at a time, so the returned list
                can exceed this target (it is a lower bound, not an exact
                size).
            days_to_sample: Number of random days to initially sample for
                count assessment
            per_page: Number of results per page (max 100)
            min_wait: Minimum wait time between API requests
            min_stars: Minimum number of stars (0 for no filtering)
            min_size_kb: Minimum repository size in KB (0 for no filtering)
            language: Programming language to filter by
            max_attempts: Maximum collection-loop iterations before giving up
            **kwargs: Additional filters to apply

        Returns:
            List of repository data
        """
        self.logger.info(
            "Starting weighted temporal sampling: days_to_sample=%s, "
            "n_samples=%s, per_page=%s, min_stars=%s, min_size_kb=%s, "
            "language=%s",
            days_to_sample,
            n_samples,
            per_page,
            min_stars,
            min_size_kb,
            language or "None",
        )

        if self.token:
            self.logger.info("Using GitHub API token for authentication")
        else:
            self.logger.warning(
                "No GitHub API token provided. Rate limits will be restricted."
            )

        # Initialize variables
        all_repos: list[dict[str, Any]] = []
        # Maps a day to its repo count and formatted label. Annotated because
        # everything downstream -- valid_days, weights, top_days -- derives its
        # type from here, so one bare {} made four collections Unknown.
        period_data: dict[datetime, dict[str, Any]] = {}
        self.attempts: int = 0
        self.success_count: int = 0
        start_time = time.time()

        # Generate random days for initial sampling
        initial_days: list[datetime] = []
        for _ in range(days_to_sample):
            random_dt = self._random_date()
            initial_days.append(random_dt)

        # Sort chronologically for better logging
        initial_days.sort()

        self.logger.info("Generated %s random days to sample", len(initial_days))

        # Step 1: Get the first page of results and total counts for each
        # day in one pass
        for i, day in enumerate(initial_days):
            # Check rate limit periodically
            if i % 5 == 0:
                remaining = self._check_rate_limit("search")
                if remaining is not None and remaining <= self.rate_limit_safety:
                    self.logger.warning(
                        "Approaching GitHub API rate limit (%s remaining). "
                        "Stopping initial sampling after %s/%s days.",
                        remaining,
                        i,
                        days_to_sample,
                    )
                    break

            start_time_str, end_time_str = self._format_date_for_query(day)
            day_str = day.strftime("%Y-%m-%d")

            # Build query
            query = self._build_search_query(
                start_time_str, end_time_str, min_stars, min_size_kb, language, **kwargs
            )

            # Construct the URL for first page
            url = (
                f"{self.api_base_url}/search/repositories"
                f"?q={query}&sort=updated&order=desc&per_page={per_page}&page=1"
            )

            self.logger.info("Sampling day %s/%s: %s", i + 1, days_to_sample, day_str)

            try:
                self.attempts += 1
                response = self._make_github_request(url, min_wait=min_wait, timeout=10)

                if response is None:
                    self.logger.warning(
                        "Request failed or rate limited for day %s", day_str
                    )
                    continue
                if response.status_code == HTTP_OK:
                    results = response.json()
                    count = results["total_count"]

                    if count > 0:
                        self.success_count += 1
                        self.logger.info("Found %s repositories on %s", count, day_str)

                        # Store period data for weighting only - do NOT add repos yet
                        # to avoid page 1 bias (most recently updated repos)
                        period_data[day] = {
                            "count": count,
                            "day_str": day_str,
                        }
                    else:
                        self.logger.info("No repositories found on %s", day_str)

                else:
                    self.logger.warning(
                        "API error: Status code %s, Response: %s...",
                        response.status_code,
                        response.text[:200],
                    )

            except Exception as e:
                self.logger.error("Error sampling day %s: %s", day_str, e)
                time.sleep(min_wait * 2)  # Longer delay on error

        # Step 2: Create weighted distribution based on repository counts
        # Filter out days with zero repositories
        valid_days = {
            p: data["count"] for p, data in period_data.items() if data["count"] > 0
        }

        if not valid_days:
            self.logger.warning(
                "No repositories found in any sampled days. Returning empty list."
            )
            return []

        # Step 3: Create probability distribution for weighted sampling
        days = list(valid_days.keys())
        weights = [valid_days[day] for day in days]
        total_weight = sum(weights)

        # Normalize weights to get probabilities
        probs = [weight / total_weight for weight in weights]

        self.logger.info(
            "Created weighted distribution across %s days (total weight: %s)",
            len(days),
            total_weight,
        )

        # Log the top days with highest weights
        top_days = sorted(valid_days.items(), key=lambda x: x[1], reverse=True)[:5]
        self.logger.info("Top 5 days by repository count:")
        for day, count in top_days:
            day_str = period_data[day]["day_str"]
            self.logger.info("  %s: %s repositories", day_str, count)

        # Step 4: Sample repositories from days based on weighted distribution
        # iterations counts every pass (including skipped days) so the loop is
        # bounded and the rate-limit check cannot be starved by `continue`;
        # self.attempts keeps counting actual API requests for success_rate.
        iterations = 0
        while len(all_repos) < n_samples and iterations < max_attempts:
            iterations += 1

            # Check if we're approaching rate limit
            if iterations % 5 == 0:
                remaining = self._check_rate_limit("search")
                if remaining is not None and remaining <= self.rate_limit_safety:
                    self.logger.warning(
                        "Approaching GitHub API rate limit (%s remaining). "
                        "Stopping after collecting %s/%s repositories.",
                        remaining,
                        len(all_repos),
                        n_samples,
                    )
                    break

            # Select a day using weighted random choice
            day = random.choices(days, weights=probs, k=1)[0]
            day_info = period_data[day]
            day_str = day_info["day_str"]
            count = day_info["count"]

            # Skip if we've already collected enough from this day
            # (To avoid repeatedly sampling the same popular day)
            if (
                sum(1 for repo in all_repos if repo.get("sampled_from") == day_str)
                >= count / 2
            ):
                continue

            start_time_str, end_time_str = self._format_date_for_query(day)
            self.attempts += 1

            # Log the day we're querying
            self.logger.info(
                "Sampling weighted day: %s (weight: %s) - collected %s/%s "
                "repositories so far",
                day_str,
                count,
                len(all_repos),
                n_samples,
            )

            # Build query
            query = self._build_search_query(
                start_time_str,
                end_time_str,
                min_stars,
                min_size_kb,
                language,
                **kwargs,
            )

            # For days with many repos, select a random page within the first N pages
            max_page = min(10, (count // per_page) + 1)
            page = random.randint(1, max_page)

            # Construct the URL
            url = (
                f"{self.api_base_url}/search/repositories?q={query}&sort=updated&"
                f"order=desc&per_page={per_page}&page={page}"
            )

            try:
                query_start_time = time.time()
                response = self._make_github_request(url, min_wait=min_wait, timeout=10)
                query_elapsed = time.time() - query_start_time

                if response is None:
                    self.logger.warning(
                        "Request failed or rate limited for day %s", day_str
                    )
                    continue
                if response.status_code == HTTP_OK:
                    results = response.json()

                    if results["total_count"] > 0:
                        repos = results["items"]
                        self.success_count += 1

                        self.logger.info(
                            "Found %s repositories (fetched %s from page %s in %.2fs)",
                            results["total_count"],
                            len(repos),
                            page,
                            query_elapsed,
                        )

                        # Process repos to match our standard format
                        period_repos: list[dict[str, Any]] = []
                        for repo in repos:
                            # Skip repos we already have
                            if any(
                                r["full_name"] == repo["full_name"] for r in all_repos
                            ):
                                continue

                            repo_data = {
                                "id": repo["id"],
                                "name": repo["name"],
                                "full_name": repo["full_name"],
                                "owner": repo["owner"]["login"],
                                "html_url": repo["html_url"],
                                "description": repo.get("description"),
                                "created_at": repo["created_at"],
                                "updated_at": repo["updated_at"],
                                "pushed_at": repo.get("pushed_at"),
                                "stargazers_count": repo.get("stargazers_count", 0),
                                "forks_count": repo.get("forks_count", 0),
                                "language": repo.get("language"),
                                "visibility": repo.get("visibility", "public"),
                                "size": repo.get("size", 0),
                                "sampled_from": day_str,
                            }

                            period_repos.append(repo_data)

                        # Add new repos from this period
                        all_repos.extend(period_repos)
                        added_count = len(period_repos)
                        self.logger.info(
                            "Added %s new repositories from this day", added_count
                        )

                        # If we've added enough repos, we can stop
                        if len(all_repos) >= n_samples:
                            self.logger.info(
                                "Reached target of %s repositories. Stopping sampling.",
                                n_samples,
                            )
                            break
                    else:
                        self.logger.info("No repositories found on %s", day_str)

                else:
                    self.logger.warning(
                        "API error: Status code %s, Response: %s...",
                        response.status_code,
                        response.text[:200],
                    )

            except Exception as e:
                self.logger.error("Error sampling day %s: %s", day_str, e)
                time.sleep(min_wait * 2)

        if len(all_repos) < n_samples and iterations >= max_attempts:
            self.logger.warning(
                "Stopped after max_attempts=%s iterations with %s/%s "
                "repositories collected",
                max_attempts,
                len(all_repos),
                n_samples,
            )

        # Report summary
        elapsed_time = time.time() - start_time
        success_rate = (
            (self.success_count / self.attempts) * 100 if self.attempts > 0 else 0
        )

        self.logger.info(
            "Sampling completed in %.2fs: %s attempts, %s successful "
            "(%.1f%%), collected %s repositories",
            elapsed_time,
            self.attempts,
            self.success_count,
            success_rate,
            len(all_repos),
        )

        # Apply any additional filters
        filtered_count_before = len(all_repos)
        self.results: list[dict[str, Any]] = self._filter_repos(all_repos, **kwargs)
        filtered_count_after = len(self.results)

        if filtered_count_before != filtered_count_after:
            self.logger.info(
                "Applied filters: %s repositories filtered out, %s remaining",
                filtered_count_before - filtered_count_after,
                filtered_count_after,
            )

        return self.results
