"""Event-based sampling from GH Archive's hourly event dumps."""

import gzip
import json
import logging
import random
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from ..logging_config import get_logger
from .base import HTTP_OK, BaseSampler


class GHArchiveSampler(BaseSampler):
    """Sample repositories by downloading and processing GH Archive files.

    This sampler randomly selects days from GitHub's event history, downloads
    the corresponding archive files, and extracts repository information.
    With the default CreateEvent filter the population is repositories
    *created* on the sampled days; with other event types it is an
    activity-biased event population.
    """

    def __init__(
        self,
        token: str | None = None,
        seed: int | None = None,
        log_level: int = logging.INFO,
    ):
        """Initialize the GH Archive sampler.

        Args:
            token: GitHub Personal Access Token (not strictly needed for GH Archive)
            seed: Random seed for reproducibility
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(token)

        # Configure logger
        self.logger: logging.Logger = get_logger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        # Initialize state
        self._seed = seed
        self.attempts: int = 0
        self.success_count = 0
        self.results = []

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            self.logger.info("Random seed set to: %s", seed)

        self.logger.info("Initialized GHArchiveSampler")

    def sample(self, n_samples: int = 100, **kwargs: Any) -> list[dict[str, Any]]:
        """Sample repositories using the GH Archive approach.

        This is the implementation of the abstract method from BaseSampler,
        which delegates to the gh_sampler method with the provided parameters.

        Args:
            n_samples: Number of repositories to sample
            **kwargs: Additional parameters to pass to gh_sampler

        Returns:
            List of repository data
        """
        self.logger.info("Sample method called with n_samples=%s", n_samples)

        # Call the main implementation method
        return self.gh_sampler(n_samples=n_samples, **kwargs)

    def gh_sampler(
        self,
        n_samples: int = 100,
        days_to_sample: int = 5,
        repos_per_day: int = 20,
        years_back: int = 10,
        event_types: list[str] | None = None,
        hours_per_day: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Sample repositories from GH Archive's hourly files.

        The population sampled depends on hours_per_day. With the default
        (None), all 24 hourly files of each sampled day are processed, so
        each day's population is exactly the repositories with matching
        events that day. With hours_per_day=H, only H randomly chosen hours
        are downloaded (H/24 of the bandwidth), and the population becomes
        repositories with matching events in the sampled hours - repos
        active in low-traffic hours are then over-represented relative to
        the full-day population, the same bias structure the per-day cap
        already introduces across days.

        Args:
            n_samples: Target number of repositories to sample
            days_to_sample: Number of random days to sample
            repos_per_day: Maximum repositories to sample per day
            years_back: How many years to look back
            event_types: Types of GitHub events to consider. Note: GitHub's
                Events API payload change of 2025-10-07 removed repository
                CreateEvents from the public feed, so with the default
                event_types no repository created after that date can be
                sampled; days after it have an empty population.
            hours_per_day: Number of random hours (of 24) to download per
                day; None processes the full day
            **kwargs: Additional filters to apply

        Returns:
            List of repository data
        """
        self.logger.info(
            "Sampling via archives: n_samples=%s, days_to_sample=%s, "
            "repos_per_day=%s, years_back=%s, event_types=%s, hours_per_day=%s",
            n_samples,
            days_to_sample,
            repos_per_day,
            years_back,
            event_types,
            hours_per_day,
        )

        # Set default event types if None provided
        if event_types is None:
            event_types = ["CreateEvent"]

        # Log filter criteria if any
        if kwargs:
            self.logger.info("Filter criteria: %s", kwargs)

        start_time = time.time()

        # Calculate parameters to ensure we get enough samples
        days_needed = max(1, (n_samples + repos_per_day - 1) // repos_per_day)
        days_to_sample = max(days_to_sample, days_needed)
        self.logger.debug(
            "Adjusted days_to_sample to %s to ensure enough data", days_to_sample
        )

        # Generate random days
        random_days: list[datetime] = []
        now = datetime.now(UTC)

        for _ in range(days_to_sample):
            # Random days back (within years_back)
            days_back = random.randint(1, years_back * 365)

            target_date = now - timedelta(days=days_back)
            # Set to beginning of day
            target_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)

            random_days.append(target_date)

        # Process each random day
        # Keyed by full_name to avoid duplicates. Annotated because the return
        # value and every log line downstream inherit its type.
        all_repos: dict[str, dict[str, Any]] = {}
        processed_days = 0
        errors = 0
        self.attempts = 0
        self.success_count = 0

        for i, target_date in enumerate(random_days):
            self.attempts += 1
            # Format the date string for logging
            day_str = target_date.strftime("%Y-%m-%d")
            self.logger.info("Processing day %s/%s: %s", i + 1, days_to_sample, day_str)

            day_repos: dict[str, dict[str, Any]] = {}
            day_events_processed = 0
            day_events_skipped = 0
            day_events_error = 0
            day_process_start = time.time()

            # GH Archive publishes hourly files ({date}-{hour}.json.gz);
            # there is no consolidated daily file.
            archive_date = target_date.strftime("%Y-%m-%d")
            hours = list(range(24))
            if hours_per_day is not None:
                hours = sorted(random.sample(hours, min(hours_per_day, 24)))
            hours_processed = 0
            for hour in hours:
                archive_url = (
                    f"https://data.gharchive.org/{archive_date}-{hour}.json.gz"
                )
                try:
                    response = requests.get(archive_url, stream=True, timeout=30)
                    if response.status_code != HTTP_OK:
                        self.logger.warning(
                            "Failed to download archive %s: HTTP %s",
                            archive_url,
                            response.status_code,
                        )
                        continue

                    with gzip.GzipFile(fileobj=response.raw) as f:
                        for line in f:
                            try:
                                event = json.loads(line.decode("utf-8"))
                                day_events_processed += 1

                                # Only process specified event types
                                event_type = event.get("type")
                                if event_type not in event_types:
                                    day_events_skipped += 1
                                    continue

                                # Add an additional check to only record CreateEvent
                                # repositories if CreateEvent is the ONLY type in
                                # event_types
                                if event_types == ["CreateEvent"] and not (
                                    # Only actual repository-creation events
                                    event_type == "CreateEvent"
                                    and event.get("payload", {}).get("ref_type")
                                    == "repository"
                                ):
                                    day_events_skipped += 1
                                    continue

                                # Extract repo information
                                repo = event.get("repo", {})
                                repo_name = repo.get("name")

                                # Skip if no valid repo name
                                if not repo_name or "/" not in repo_name:
                                    day_events_skipped += 1
                                    continue

                                # Skip if we already have this repo
                                if repo_name in day_repos:
                                    day_events_skipped += 1
                                    continue

                                # Get additional repo information
                                owner, name = repo_name.split("/", 1)

                                repo_data: dict[str, Any] = {
                                    "full_name": repo_name,
                                    "name": name,
                                    "owner": owner,
                                    "html_url": f"https://github.com/{repo_name}",
                                    "created_at": event.get("created_at"),
                                    "sampled_from": day_str,
                                    "event_type": event.get("type"),
                                }

                                # Store repo in our day collection
                                day_repos[repo_name] = repo_data

                            except json.JSONDecodeError:
                                day_events_error += 1
                                continue
                            except Exception as e:
                                day_events_error += 1
                                self.logger.warning("Error processing event: %s", e)
                                continue
                    hours_processed += 1

                except Exception as e:
                    self.logger.warning(
                        "Error processing archive %s: %s", archive_url, e
                    )

            if hours_processed == 0:
                self.logger.warning("No archive hours processed for %s", day_str)
                errors += 1
                continue

            if (
                not day_repos
                and day_events_processed > 0
                and event_types == ["CreateEvent"]
            ):
                self.logger.warning(
                    "No repository-creation events in %s: GitHub removed "
                    "repository CreateEvents from the public events feed on "
                    "2025-10-07, so days after that date have an empty "
                    "population under the default event_types",
                    day_str,
                )

            # Randomly sample repositories for this day
            day_repos_list = list(day_repos.values())
            random.shuffle(day_repos_list)
            sampled_day_repos = day_repos_list[:repos_per_day]

            # Add sampled repos to overall collection
            for repo_data in sampled_day_repos:
                all_repos[repo_data["full_name"]] = repo_data

            day_process_time = time.time() - day_process_start
            processed_days += 1
            self.success_count += 1

            self.logger.info(
                "Found %s repositories from %s (%s/24 hours, %s events "
                "in %.2fs, skipped %s, errors %s)",
                len(sampled_day_repos),
                day_str,
                hours_processed,
                day_events_processed,
                day_process_time,
                day_events_skipped,
                day_events_error,
            )

            # Log progress towards overall target
            self.logger.info(
                "Progress: %s/%s repositories collected", len(all_repos), n_samples
            )

            # Break if we have enough repositories
            if len(all_repos) >= n_samples:
                self.logger.info(
                    "Reached target sample size of %s repositories", n_samples
                )
                break

        # Convert to list
        result_repos = list(all_repos.values())

        # Calculate processing statistics
        elapsed = time.time() - start_time
        days_per_second = processed_days / elapsed if elapsed > 0 else 0
        repos_per_second = len(result_repos) / elapsed if elapsed > 0 else 0

        # Log summary
        self.logger.info(
            "Completed archive sampling in %.2fs: %s repositories from "
            "%s days (errors: %s, rate: %.2f days/s, %.2f repos/s)",
            elapsed,
            len(result_repos),
            processed_days,
            errors,
            days_per_second,
            repos_per_second,
        )

        # Randomize the final list to avoid time-based patterns
        if self._seed is not None:
            random.seed(self._seed)
        random.shuffle(result_repos)

        # Limit to requested sample size
        result = result_repos[:n_samples]

        # Apply any filters
        filtered_count_before = len(result)
        if kwargs:
            result = self._filter_repos(result, **kwargs)
            filtered_count_after = len(result)
            if filtered_count_before != filtered_count_after:
                self.logger.info(
                    "Applied filters: %s repositories filtered out, %s remaining",
                    filtered_count_before - filtered_count_after,
                    filtered_count_after,
                )

        self.results: list[dict[str, Any]] = result

        return result
