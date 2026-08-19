"""Random sampling by probing GitHub's sequential repository ID space."""

import logging
import random
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from ..logging_config import get_logger
from .base import HTTP_OK, BaseSampler


class IDSampler(BaseSampler):
    """Sample repositories using random ID probing.

    This sampler generates random repository IDs within a specified range
    and attempts to retrieve repositories with those IDs from GitHub.
    The resulting sample is approximately uniform over all existing public
    repositories with ID <= max_id.
    """

    # Approximate newest repository ID, measured 2026-07 via live validation
    # (newest observed ID: ~1.31B). IDs grow by roughly 200-300M per year;
    # call update_max_id() for the current ceiling.
    DEFAULT_MAX_ID = 1_300_000_000

    def __init__(
        self,
        token: str | None = None,
        min_id: int = 1,
        max_id: int = DEFAULT_MAX_ID,
        rate_limit_safety: int = 100,
        seed: int | None = None,
        log_level: int = logging.INFO,
    ):
        """Initialize the ID sampler.

        Args:
            token: GitHub Personal Access Token
            min_id: Minimum repository ID to sample from
            max_id: Maximum repository ID to sample from. The default is a
                dated constant (see DEFAULT_MAX_ID); repositories with
                higher IDs are unreachable, so call update_max_id() or pass
                the current ceiling explicitly for full coverage.
            rate_limit_safety: Stop when this many API requests remain
            seed: Random seed for reproducibility
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(token)

        # Configure logger
        self.logger: logging.Logger = get_logger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        # Set random seed if provided
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            self.logger.info("Random seed set to: %s", seed)

        self.min_id: int = min_id
        self.max_id: int = max_id
        self.rate_limit_safety: int = rate_limit_safety

        self.logger.info(
            "Initialized IDSampler with min_id=%s, max_id=%s", min_id, self.max_id
        )

    def update_max_id(self) -> int:
        """Refresh max_id from the newest repository visible on GitHub.

        Makes one search API call for repositories created in the last day
        and sets max_id to the highest ID observed (approximately the
        current ID ceiling). Falls back to the existing max_id with a
        warning if the call fails.

        Returns:
            The (possibly updated) max_id
        """
        since = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            response = requests.get(
                f"{self.api_base_url}/search/repositories",
                params={"q": f"created:>={since}", "per_page": 100},
                headers=self._get_headers(),
                timeout=30,
            )
            response.raise_for_status()
            items = response.json()["items"]
            if items:
                self.max_id = max(item["id"] for item in items)
                self.logger.info("Updated max_id to %s", self.max_id)
        except Exception as e:
            self.logger.warning(
                "Could not update max_id (keeping %s): %s", self.max_id, e
            )
        return self.max_id

    def _passes_filters(self, repo_data: dict[str, Any], **kwargs: Any) -> bool:
        """Check if a repository passes all filter criteria.

        Args:
            repo_data: Repository data dictionary
            **kwargs: Filter criteria

        Returns:
            True if the repository passes all filters
        """
        if (
            "min_stars" in kwargs
            and repo_data.get("stargazers_count", 0) < kwargs["min_stars"]
        ):
            return False

        if (
            "min_forks" in kwargs
            and repo_data.get("forks_count", 0) < kwargs["min_forks"]
        ):
            return False

        if kwargs.get("languages"):
            repo_lang = repo_data.get("language")
            if repo_lang is None or repo_lang not in kwargs["languages"]:
                return False

        return True

    def sample(
        self,
        n_samples: int = 10,
        min_wait: float = 0.1,
        max_attempts: int = 1000,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Sample repositories by trying random IDs.

        Args:
            n_samples: Number of valid repositories to collect
            min_wait: Minimum wait time between API requests
            max_attempts: Maximum number of IDs to try
            **kwargs: Additional filters to apply (filtering happens during collection)

        Returns:
            List of repository data
        """
        self.logger.info(
            "Starting sampling: target=%s, max_attempts=%s", n_samples, max_attempts
        )

        if self.token:
            self.logger.info("Using GitHub API token for authentication")
        else:
            self.logger.warning(
                "No GitHub API token provided. Rate limits will be restricted."
            )

        filtered_repos: list[dict[str, Any]] = []
        self.attempts: int = 0
        self.success_count: int = 0

        # Log request rate/interval
        self.logger.info("Minimum wait between requests: %s seconds", min_wait)

        # Log filter criteria if any
        if kwargs:
            self.logger.info("Filter criteria: %s", kwargs)

        start_time = time.time()

        while len(filtered_repos) < n_samples and self.attempts < max_attempts:
            # Periodically log progress
            if self.attempts > 0 and self.attempts % 10 == 0:
                elapsed = time.time() - start_time
                rate = self.attempts / elapsed if elapsed > 0 else 0
                success_rate = (
                    (self.success_count / self.attempts) * 100
                    if self.attempts > 0
                    else 0
                )
                self.logger.info(
                    "Progress: %s/%s repos, %s attempts (%.1f%% success, %.2f req/s)",
                    len(filtered_repos),
                    n_samples,
                    self.attempts,
                    success_rate,
                    rate,
                )

            # Check rate limit every 10 attempts or if we're getting close
            should_check_limit = self.attempts % 10 == 0 or (
                self.attempts > 0 and (self.attempts % max(max_attempts // 20, 1)) == 0
            )

            if should_check_limit:
                remaining = self._check_rate_limit()
                if remaining is not None and remaining <= self.rate_limit_safety:
                    self.logger.warning(
                        "Approaching GitHub API rate limit (%s remaining). "
                        "Stopping with %s samples.",
                        remaining,
                        len(filtered_repos),
                    )
                    break

            # Generate random repository ID
            repo_id = random.randint(self.min_id, self.max_id)
            self.logger.debug("Trying repository ID: %s", repo_id)

            # Try to fetch the repository by ID
            url = f"{self.api_base_url}/repositories/{repo_id}"
            try:
                response = self._make_github_request(url, min_wait=min_wait, timeout=10)
                self.attempts += 1

                # Check if request succeeded
                if response is None:
                    self.logger.debug(
                        "Request failed or rate limited for ID %s", repo_id
                    )
                    continue

                # Check if repository exists
                if response.status_code == HTTP_OK:
                    repo_json = response.json()
                    self.success_count += 1

                    # Log repository details at debug level
                    self.logger.debug(
                        "Repository details: name=%s, owner=%s, stars=%s, language=%s",
                        repo_json["name"],
                        repo_json["owner"]["login"],
                        repo_json.get("stargazers_count", 0),
                        repo_json.get("language"),
                    )

                    repo_data: dict[str, Any] = {
                        "id": repo_id,
                        "name": repo_json["name"],
                        "full_name": repo_json["full_name"],
                        "owner": repo_json["owner"]["login"],
                        "html_url": repo_json["html_url"],
                        "description": repo_json.get("description"),
                        "created_at": repo_json["created_at"],
                        "updated_at": repo_json["updated_at"],
                        "pushed_at": repo_json.get("pushed_at"),
                        "stargazers_count": repo_json.get("stargazers_count", 0),
                        "forks_count": repo_json.get("forks_count", 0),
                        "language": repo_json.get("language"),
                        "visibility": repo_json.get("visibility", "unknown"),
                    }

                    # Apply filters during collection
                    if self._passes_filters(repo_data, **kwargs):
                        filtered_repos.append(repo_data)
                        self.logger.info(
                            "Found valid repository (%s/%s): %s (id: %s)",
                            len(filtered_repos),
                            n_samples,
                            repo_json["full_name"],
                            repo_id,
                        )
                    else:
                        self.logger.debug(
                            "Repository %s filtered out", repo_json["full_name"]
                        )
                else:
                    self.logger.debug(
                        "Invalid repository ID: %s (Status code: %s, Response: %s...)",
                        repo_id,
                        response.status_code,
                        response.text[:100],
                    )

            except Exception as e:
                self.logger.error("Error sampling repository ID %s: %s", repo_id, e)
                time.sleep(min_wait * 5)

        # Calculate final stats
        elapsed = time.time() - start_time
        success_rate = (
            (self.success_count / self.attempts) * 100 if self.attempts > 0 else 0
        )
        rate = self.attempts / elapsed if elapsed > 0 else 0

        self.logger.info(
            "Sampling completed in %.2fs: %s attempts, %s repositories "
            "(%.1f%% success, %.2f req/s)",
            elapsed,
            self.attempts,
            len(filtered_repos),
            success_rate,
            rate,
        )

        self.results: list[dict[str, Any]] = filtered_repos
        return self.results
