# reporoulette/samplers/id_sampler.py
import logging
import random
import time
from typing import Any

from ..logging_config import get_logger
from .base import HTTP_OK, BaseSampler


class IDSampler(BaseSampler):
    """Sample repositories using random ID probing.

    This sampler generates random repository IDs within a specified range
    and attempts to retrieve repositories with those IDs from GitHub.
    """

    def __init__(
        self,
        token: str | None = None,
        min_id: int = 1,
        max_id: int = 850000000,
        rate_limit_safety: int = 100,
        seed: int | None = None,
        log_level: int = logging.INFO,
    ):
        """Initialize the ID sampler.

        Args:
            token: GitHub Personal Access Token
            min_id: Minimum repository ID to sample from
            max_id: Maximum repository ID to sample from (default: 850M based on
                    validation testing that found repositories at ID 800M+)
            rate_limit_safety: Stop when this many API requests remain
            seed: Random seed for reproducibility
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(token)

        # Configure logger
        self.logger: logging.Logger = get_logger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            self.logger.info(f"Random seed set to: {seed}")

        self.min_id: int = min_id
        self.max_id: int = max_id
        self.rate_limit_safety: int = rate_limit_safety

        self.logger.info(
            f"Initialized IDSampler with min_id={min_id}, max_id={self.max_id}"
        )

    def _passes_filters(self, repo_data: dict[str, Any], **kwargs: Any) -> bool:
        """Check if a repository passes all filter criteria.

        Args:
            repo_data: Repository data dictionary
            **kwargs: Filter criteria

        Returns:
            True if the repository passes all filters
        """
        if "min_stars" in kwargs:
            if repo_data.get("stargazers_count", 0) < kwargs["min_stars"]:
                return False

        if "min_forks" in kwargs:
            if repo_data.get("forks_count", 0) < kwargs["min_forks"]:
                return False

        if "languages" in kwargs and kwargs["languages"]:
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
            f"Starting sampling: target={n_samples}, max_attempts={max_attempts}"
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
        self.logger.info(f"Minimum wait between requests: {min_wait} seconds")

        # Log filter criteria if any
        if kwargs:
            self.logger.info(f"Filter criteria: {kwargs}")

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
                    f"Progress: {len(filtered_repos)}/{n_samples} repos found, "
                    f"{self.attempts} attempts ({success_rate:.1f}% success rate), "
                    f"{rate:.2f} requests/sec"
                )

            # Check rate limit every 10 attempts or if we're getting close
            should_check_limit = self.attempts % 10 == 0 or (
                self.attempts > 0 and (self.attempts % max(max_attempts // 20, 1)) == 0
            )

            if should_check_limit:
                remaining = self._check_rate_limit()
                if remaining <= self.rate_limit_safety:
                    self.logger.warning(
                        f"Approaching GitHub API rate limit ({remaining} remaining). "
                        f"Stopping with {len(filtered_repos)} samples."
                    )
                    break

            # Generate random repository ID
            repo_id = random.randint(self.min_id, self.max_id)
            self.logger.debug(f"Trying repository ID: {repo_id}")

            # Try to fetch the repository by ID
            url = f"{self.api_base_url}/repositories/{repo_id}"
            try:
                response = self._make_github_request(url, min_wait=min_wait, timeout=10)
                self.attempts += 1

                # Check if request succeeded
                if response is None:
                    self.logger.debug(
                        f"Request failed or rate limited for ID {repo_id}"
                    )
                    continue

                # Check if repository exists
                if response.status_code == HTTP_OK:
                    repo_json = response.json()
                    self.success_count += 1

                    # Log repository details at debug level
                    self.logger.debug(
                        f"Repository details: name={repo_json['name']}, "
                        f"owner={repo_json['owner']['login']}, "
                        f"stars={repo_json.get('stargazers_count', 0)}, "
                        f"language={repo_json.get('language')}"
                    )

                    repo_data = {
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
                            f"Found valid repository ({len(filtered_repos)}/{n_samples}): "
                            f"{repo_json['full_name']} (id: {repo_id})"
                        )
                    else:
                        self.logger.debug(
                            f"Repository {repo_json['full_name']} filtered out"
                        )
                else:
                    self.logger.debug(
                        f"Invalid repository ID: {repo_id} "
                        f"(Status code: {response.status_code}, Response: {response.text[:100]}...)"
                    )

            except Exception as e:
                self.logger.error(f"Error sampling repository ID {repo_id}: {str(e)}")
                time.sleep(min_wait * 5)

        # Calculate final stats
        elapsed = time.time() - start_time
        success_rate = (
            (self.success_count / self.attempts) * 100 if self.attempts > 0 else 0
        )
        rate = self.attempts / elapsed if elapsed > 0 else 0

        self.logger.info(
            f"Sampling completed in {elapsed:.2f} seconds: "
            f"{self.attempts} attempts, found {len(filtered_repos)} repositories "
            f"({success_rate:.1f}% success rate, {rate:.2f} requests/sec)"
        )

        self.results: list[dict[str, Any]] = filtered_repos
        return self.results
