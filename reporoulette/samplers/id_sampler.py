# reporoulette/samplers/id_sampler.py
import logging
import random
import time
from typing import Any

from ..logging_config import get_logger
from .base import HTTP_NOT_FOUND, HTTP_OK, BaseSampler


class IDSampler(BaseSampler):
    """Sample repositories using random ID probing.

    This sampler generates random repository IDs within a specified range
    and attempts to retrieve repositories with those IDs from GitHub.
    """

    def __init__(
        self,
        token: str | None = None,
        min_id: int = 1,
        max_id: int = 850000000,  # Updated from 500M based on validation testing
        rate_limit_safety: int = 100,
        seed: int | None = None,  # Add seed parameter
        log_level: int = logging.INFO,
        auto_discover_max: bool = False,
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
            auto_discover_max: Automatically discover current max repository ID
                              (recommended for future-proof sampling)
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

        # Auto-discover max ID if requested
        if auto_discover_max:
            discovered_max = self._discover_max_repository_id()
            if discovered_max > 0:
                self.max_id = discovered_max
                self.logger.info(f"Auto-discovered max repository ID: {self.max_id}")

        self.logger.info(
            f"Initialized IDSampler with min_id={min_id}, max_id={self.max_id}"
        )

    def _discover_max_repository_id(self, initial_guess: int = 800000000) -> int:
        """Dynamically discover the current maximum repository ID using binary search.

        Args:
            initial_guess: Starting point for the search

        Returns:
            Estimated maximum repository ID, or 0 if discovery fails
        """
        self.logger.info("Starting dynamic repository ID range discovery...")

        # Binary search for the maximum valid ID
        low = 1
        high = initial_guess
        max_found = 0
        attempts = 0
        max_attempts = 30  # Limit search attempts

        while low <= high and attempts < max_attempts:
            mid = (low + high) // 2
            attempts += 1

            try:
                url = f"{self.api_base_url}/repositories/{mid}"
                response = self._make_github_request(url, min_wait=0.5, timeout=5)

                if response is None:
                    break  # Request failed or rate limited
                elif response.status_code == HTTP_OK:
                    # Repository exists, search higher
                    max_found = max(max_found, mid)
                    low = mid + 1
                    self.logger.debug(
                        f"Found repository at ID {mid}, searching higher..."
                    )
                elif response.status_code == HTTP_NOT_FOUND:
                    # Repository doesn't exist, search lower
                    high = mid - 1
                    self.logger.debug(f"No repository at ID {mid}, searching lower...")
                else:
                    self.logger.warning(
                        f"Unexpected status code {response.status_code} for ID {mid}"
                    )

            except Exception as e:
                self.logger.error(f"Error discovering max ID: {str(e)}")
                break

        # Refine the estimate with a linear search near the boundary
        if max_found > 0:
            self.logger.info(f"Refining max ID estimate starting from {max_found}")

            # Try a few IDs above the max found
            for offset in range(1, 11):
                test_id = max_found + offset * 100000
                try:
                    url = f"{self.api_base_url}/repositories/{test_id}"
                    response = self._make_github_request(url, min_wait=0.5, timeout=5)

                    if response is None:
                        break  # Request failed or rate limited
                    elif response.status_code == HTTP_OK:
                        max_found = test_id
                        self.logger.debug(f"Found higher repository at ID {test_id}")
                    elif response.status_code == HTTP_NOT_FOUND:
                        break

                except Exception:
                    break

        if max_found > 0:
            # Add some buffer for newly created repos
            max_found = int(max_found * 1.05)
            self.logger.info(
                f"Discovered maximum repository ID: {max_found} (with 5% buffer)"
            )
        else:
            self.logger.warning("Failed to discover maximum repository ID")

        return max_found

    def sample(
        self,
        n_samples: int = 10,
        min_wait: float = 0.1,  # Add min_wait parameter
        max_attempts: int = 1000,  # Add max_attempts parameter
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Sample repositories by trying random IDs.

        Args:
            n_samples: Number of valid repositories to collect
            min_wait: Minimum wait time between API requests
            max_attempts: Maximum number of IDs to try
            **kwargs: Additional filters to apply

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

        valid_repos: list[dict[str, Any]] = []
        self.attempts: int = 0
        self.success_count: int = 0

        # Log request rate/interval
        self.logger.info(f"Minimum wait between requests: {min_wait} seconds")

        # Log filter criteria if any
        if kwargs:
            self.logger.info(f"Filter criteria: {kwargs}")

        start_time = time.time()

        while len(valid_repos) < n_samples and self.attempts < max_attempts:
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
                    f"Progress: {len(valid_repos)}/{n_samples} repos found, "
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
                        f"Stopping with {len(valid_repos)} samples."
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
                    repo_data = response.json()
                    self.success_count += 1

                    # Log repository details at debug level
                    self.logger.debug(
                        f"Repository details: name={repo_data['name']}, "
                        f"owner={repo_data['owner']['login']}, "
                        f"stars={repo_data.get('stargazers_count', 0)}, "
                        f"language={repo_data.get('language')}"
                    )

                    valid_repos.append(
                        {
                            "id": repo_id,
                            "name": repo_data["name"],
                            "full_name": repo_data["full_name"],
                            "owner": repo_data["owner"]["login"],
                            "html_url": repo_data["html_url"],
                            "description": repo_data.get("description"),
                            "created_at": repo_data["created_at"],
                            "updated_at": repo_data["updated_at"],
                            "pushed_at": repo_data.get("pushed_at"),
                            "stargazers_count": repo_data.get("stargazers_count", 0),
                            "forks_count": repo_data.get("forks_count", 0),
                            "language": repo_data.get("language"),
                            "visibility": repo_data.get("visibility", "unknown"),
                        }
                    )
                    self.logger.info(
                        f"Found valid repository ({len(valid_repos)}/{n_samples}): "
                        f"{repo_data['full_name']} (id: {repo_id})"
                    )
                else:
                    self.logger.debug(
                        f"Invalid repository ID: {repo_id} "
                        f"(Status code: {response.status_code}, Response: {response.text[:100]}...)"
                    )

            except Exception as e:
                self.logger.error(f"Error sampling repository ID {repo_id}: {str(e)}")
                time.sleep(min_wait * 5)  # Longer delay on error

        # Calculate final stats
        elapsed = time.time() - start_time
        success_rate = (
            (self.success_count / self.attempts) * 100 if self.attempts > 0 else 0
        )
        rate = self.attempts / elapsed if elapsed > 0 else 0

        self.logger.info(
            f"Sampling completed in {elapsed:.2f} seconds: "
            f"{self.attempts} attempts, found {len(valid_repos)} repositories "
            f"({success_rate:.1f}% success rate, {rate:.2f} requests/sec)"
        )

        # Apply any filters
        filtered_count_before = len(valid_repos)
        self.results: list[dict[str, Any]] = self._filter_repos(valid_repos, **kwargs)
        filtered_count_after = len(self.results)

        if filtered_count_before != filtered_count_after:
            self.logger.info(
                f"Applied filters: {filtered_count_before - filtered_count_after} repositories filtered out, "
                f"{filtered_count_after} repositories remaining"
            )

        return self.results
