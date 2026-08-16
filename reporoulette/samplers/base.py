# reporoulette/samplers/base.py
import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import requests

# HTTP status code constants
HTTP_OK = 200
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404

# Default timeout and rate limit constants
DEFAULT_WAIT_TIME = 60
DEFAULT_RATE_LIMIT_SAFETY = 100
DEFAULT_MIN_REQUEST_INTERVAL = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 10


class BaseSampler(ABC):
    """Base class for all repository samplers."""

    def __init__(
        self,
        token: str | None = None,
        rate_limit_safety: int = DEFAULT_RATE_LIMIT_SAFETY,
    ) -> None:
        self.token: str | None = token
        self.rate_limit_safety: int = rate_limit_safety
        # Which rate-limit bucket this sampler's requests draw from;
        # samplers using the Search API must override with "search"
        self.rate_limit_resource: str = "core"
        self.results: list[dict[str, Any]] = []
        self.attempts: int = 0
        self.success_count: int = 0
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.api_base_url: str = "https://api.github.com"

        # Rate limiting configuration
        self.min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL
        self.max_retries_on_rate_limit: int = DEFAULT_MAX_RETRIES
        self.default_timeout: int = DEFAULT_TIMEOUT

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of sampling attempts.

        Returns:
            Percentage of successful attempts
        """
        if self.attempts == 0:
            return 0.0
        return (self.success_count / self.attempts) * 100

    @abstractmethod
    def sample(self, n_samples: int, **kwargs: Any) -> list[dict[str, Any]]:
        """Sample repositories according to the specific strategy.

        Args:
            n_samples: Number of repositories to sample
            **kwargs: Additional parameters specific to each sampler

        Returns:
            List of repository data dictionaries
        """
        pass

    @staticmethod
    def _owner_login(repo: dict[str, Any]) -> str | None:
        """Return the owner login for API-shaped or flat repo dictionaries.

        Args:
            repo: Repository dictionary

        Returns:
            Owner login string, or None if absent
        """
        owner: Any = repo.get("owner")
        if isinstance(owner, dict):
            owner = owner.get("login")
        return owner if isinstance(owner, str) else None

    @staticmethod
    def _parse_iso(value: str | datetime) -> datetime:
        """Parse an ISO timestamp (Z-suffixed or not) to an aware datetime.

        Naive inputs are assumed UTC so mixed inputs stay comparable.

        Args:
            value: Timestamp string or datetime

        Returns:
            Timezone-aware datetime
        """
        if not isinstance(value, datetime):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value

    def _filter_repos(
        self, repos: list[dict[str, Any]], **filters: Any
    ) -> list[dict[str, Any]]:
        """Filter repositories based on criteria.

        Shared by all samplers. Supported filters: min_stars, min_forks,
        min_size_kb (>= thresholds), languages / language (case-insensitive
        membership), owner (exact login), created_after / created_before
        (ISO strings or datetimes), and max_repos (seeded random subsample).
        If no repository in the input carries the field a filter needs
        (e.g. BigQuery and GH Archive results have no star or language
        data), that filter is skipped with a warning rather than dropping
        every repository. Unknown filter names are ignored with a warning.

        Args:
            repos: List of repository data to filter
            **filters: Criteria to filter by (e.g., min_stars, languages)

        Returns:
            Filtered list of repositories
        """
        if not filters:
            return repos

        filtered = list(repos)

        threshold_fields = {
            "min_stars": "stargazers_count",
            "min_forks": "forks_count",
            "min_size_kb": "size",
        }
        for key, field in threshold_fields.items():
            if key not in filters:
                continue
            if not any(field in r for r in filtered):
                self.logger.warning(
                    f"Filter '{key}' skipped: no repository has a '{field}' field"
                )
                continue
            threshold = filters[key]
            filtered = [r for r in filtered if r.get(field, 0) >= threshold]

        wanted_languages: list[Any] = list(filters.get("languages") or [])
        if filters.get("language"):
            wanted_languages.append(filters["language"])
        if wanted_languages:
            if not any("language" in r for r in filtered):
                self.logger.warning(
                    "Filter 'languages' skipped: no repository has a 'language' field"
                )
            else:
                wanted = {str(lang).lower() for lang in wanted_languages}
                filtered = [
                    r
                    for r in filtered
                    if r.get("language") and str(r["language"]).lower() in wanted
                ]

        if "owner" in filters:
            filtered = [r for r in filtered if self._owner_login(r) == filters["owner"]]

        if "created_after" in filters:
            cutoff = self._parse_iso(filters["created_after"])
            filtered = [
                r
                for r in filtered
                if r.get("created_at") and self._parse_iso(r["created_at"]) >= cutoff
            ]

        if "created_before" in filters:
            cutoff = self._parse_iso(filters["created_before"])
            filtered = [
                r
                for r in filtered
                if r.get("created_at") and self._parse_iso(r["created_at"]) <= cutoff
            ]

        if "max_repos" in filters and len(filtered) > filters["max_repos"]:
            seed = getattr(self, "_seed", None)
            if seed is not None:
                random.seed(seed)
            random.shuffle(filtered)
            filtered = filtered[: filters["max_repos"]]

        known = {
            "min_stars",
            "min_forks",
            "min_size_kb",
            "languages",
            "language",
            "owner",
            "created_after",
            "created_before",
            "max_repos",
        }
        for key in filters:
            if key not in known:
                self.logger.warning(f"Unknown filter '{key}' ignored")

        return filtered

    def _get_headers(self) -> dict[str, str]:
        """Get headers for GitHub API requests.

        Returns:
            Dictionary with authorization headers
        """
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    def _check_rate_limit(self, resource: str = "core") -> int | None:
        """Check GitHub API rate limit and return remaining requests.

        Args:
            resource: Which rate limit resource to check ("core" or "search")

        Returns:
            Number of remaining API requests, or None if the check itself
            failed (so a flaky /rate_limit endpoint doesn't veto real
            requests the way a literal 0 would)
        """
        headers = self._get_headers()

        try:
            self.logger.debug(f"Checking GitHub API rate limit for {resource}")
            response = requests.get(f"{self.api_base_url}/rate_limit", headers=headers)
            if response.status_code == HTTP_OK:
                data = response.json()
                remaining = data["resources"][resource]["remaining"]
                reset_time = data["resources"][resource]["reset"]
                self.logger.debug(
                    f"Rate limit status ({resource}): {remaining} requests remaining, reset at timestamp {reset_time}"
                )
                return remaining
            else:
                self.logger.warning(
                    f"Failed to check rate limit. Status code: {response.status_code}"
                )
                return None
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {str(e)}")
            return None

    def _make_github_request(
        self,
        url: str,
        min_wait: float | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> requests.Response | None:
        """Make a rate-limited request to GitHub API.

        Args:
            url: URL to request
            min_wait: Minimum wait time between requests (uses default if None)
            timeout: Request timeout in seconds (uses default if None)
            max_retries: Maximum retries for rate limit exceeded (uses default if None)

        Returns:
            Response object or None if failed
        """
        # Use default values if not specified
        if min_wait is None:
            min_wait = self.min_request_interval
        if timeout is None:
            timeout = self.default_timeout
        if max_retries is None:
            max_retries = self.max_retries_on_rate_limit

        headers = self._get_headers()

        return self._attempt_request(url, headers, timeout, min_wait, max_retries)

    def _attempt_request(
        self,
        url: str,
        headers: dict[str, str],
        timeout: int,
        min_wait: float,
        max_retries: int,
    ) -> requests.Response | None:
        """Helper method to reduce complexity of _make_github_request."""
        for attempt in range(max_retries + 1):
            # Check the bucket this sampler's requests actually draw from
            remaining = self._check_rate_limit(self.rate_limit_resource)
            if remaining is not None and remaining <= self.rate_limit_safety:
                self.logger.warning(
                    f"Approaching GitHub API rate limit ({remaining} remaining). "
                    f"Request aborted for safety."
                )
                return None

            try:
                response = requests.get(url, headers=headers, timeout=timeout)

                # Handle rate limit responses
                if (
                    response.status_code == HTTP_FORBIDDEN
                    and "rate limit exceeded" in response.text.lower()
                ):
                    if attempt < max_retries:
                        msg = (
                            f"Rate limit exceeded on attempt {attempt + 1}, retrying..."
                        )
                        self.logger.warning(msg)
                        self._handle_rate_limit_exceeded(response)
                        continue
                    else:
                        self.logger.error(
                            f"Rate limit exceeded after {max_retries} retries"
                        )
                        return None

                # Add delay for rate limiting
                time.sleep(min_wait)
                return response

            except Exception as e:
                if attempt < max_retries:
                    msg = f"Error making GitHub request (attempt {attempt + 1}): {str(e)}, retrying..."
                    self.logger.warning(msg)
                    time.sleep(min_wait * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    msg = f"Error making GitHub request to {url} after {max_retries} retries: {str(e)}"
                    self.logger.error(msg)
                    return None

        return None

    def _handle_rate_limit_exceeded(self, response: requests.Response) -> None:
        """Handle rate limit exceeded responses.

        Args:
            response: Response object with rate limit information
        """
        wait_time = DEFAULT_WAIT_TIME  # Default fallback
        try:
            # Try to get the reset time from headers
            if "x-ratelimit-reset" in response.headers:
                reset_time = int(response.headers["x-ratelimit-reset"])
                current_time = time.time()
                wait_time = max(reset_time - current_time + 5, 10)
                self.logger.warning(
                    f"Rate limit exceeded. Reset at {time.ctime(reset_time)}. "
                    f"Waiting {wait_time:.1f} seconds..."
                )
            else:
                # Check if we can get reset time from rate_limit endpoint
                try:
                    headers = self._get_headers()
                    rate_limit_response = requests.get(
                        f"{self.api_base_url}/rate_limit", headers=headers
                    )
                    if rate_limit_response.status_code == HTTP_OK:
                        data = rate_limit_response.json()
                        reset_time = data["resources"]["core"]["reset"]
                        current_time = time.time()
                        wait_time = max(reset_time - current_time + 10, 10)
                        self.logger.warning(
                            f"Rate limit exceeded. Reset at {time.ctime(reset_time)}. "
                            f"Waiting {wait_time:.1f} seconds..."
                        )
                except Exception:
                    pass  # Fall back to default wait time
        except Exception as e:
            self.logger.error(f"Error parsing rate limit headers: {str(e)}")

        if wait_time == DEFAULT_WAIT_TIME:
            self.logger.warning(
                f"Rate limit exceeded. Waiting {wait_time:.1f} seconds (default)..."
            )
        time.sleep(wait_time)

    def configure_rate_limiting(
        self,
        min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL,
        max_retries_on_rate_limit: int = DEFAULT_MAX_RETRIES,
        default_timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Configure rate limiting parameters for GitHub API requests.

        Args:
            min_request_interval: Minimum seconds between requests
            max_retries_on_rate_limit: Maximum retries when rate limited
            default_timeout: Default request timeout in seconds
        """
        self.min_request_interval = min_request_interval
        self.max_retries_on_rate_limit = max_retries_on_rate_limit
        self.default_timeout = default_timeout

        self.logger.info(
            f"Rate limiting configured: interval={min_request_interval}s, "
            f"max_retries={max_retries_on_rate_limit}, timeout={default_timeout}s"
        )

    def get_rate_limit_config(self) -> dict[str, Any]:
        """Get current rate limiting configuration.

        Returns:
            Dictionary with current rate limiting settings
        """
        return {
            "min_request_interval": self.min_request_interval,
            "max_retries_on_rate_limit": self.max_retries_on_rate_limit,
            "default_timeout": self.default_timeout,
            "rate_limit_safety": self.rate_limit_safety,
        }
