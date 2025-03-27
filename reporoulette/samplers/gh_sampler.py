# reporoulette/samplers/gh_archive_sampler.py
import logging
import sys
import random
import time
from typing import List, Dict, Any, Optional

from .base import BaseSampler

class GHArchiveSampler(BaseSampler):
    """
    Sample repositories by downloading and processing GH Archive files.
    
    This sampler randomly selects hours from GitHub's event history, downloads
    the corresponding archive files, and extracts repository information.
    """
    def __init__(self, token: Optional[str] = None, seed: Optional[int] = None, log_level: int = logging.INFO):
        """
        Initialize the GH Archive sampler.
        
        Args:
            token: GitHub Personal Access Token (not strictly needed for GH Archive)
            seed: Random seed for reproducibility
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(token)
        
        # Configure logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Create console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Initialize state
        self._seed = seed
        self.attempts = 0
        self.success_count = 0
        self.results = []
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            self.logger.info(f"Random seed set to: {seed}")
            
        self.logger.info(f"Initialized GHArchiveSampler")
    
    def gh_sampler(
        self,
        n_samples: int = 100,
        hours_to_sample: int = 10,
        repos_per_hour: int = 10,
        years_back: int = 10,
        event_types: List[str] = ["PushEvent", "CreateEvent", "PullRequestEvent"],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories by downloading and processing GH Archive files.
        
        This method samples GitHub repositories by randomly selecting hours,
        downloading the corresponding archive files, and extracting repository information.
        
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
        import requests
        import gzip
        import json
        from datetime import datetime, timedelta
        
        self.logger.info(
            f"Sampling via archives: n_samples={n_samples}, hours_to_sample={hours_to_sample}, "
            f"repos_per_hour={repos_per_hour}, years_back={years_back}, "
            f"event_types={event_types}"
        )
        
        # Log filter criteria if any
        if kwargs:
            self.logger.info(f"Filter criteria: {kwargs}")
        
        start_time = time.time()
        
        # Calculate parameters to ensure we get enough samples
        hours_needed = max(1, (n_samples + repos_per_hour - 1) // repos_per_hour)
        hours_to_sample = max(hours_to_sample, hours_needed)
        self.logger.debug(f"Adjusted hours_to_sample to {hours_to_sample} to ensure enough data")
        
        # Generate random hours
        random_hours = []
        now = datetime.now()
        
        for _ in range(hours_to_sample):
            # Random days back (within years_back)
            days_back = random.randint(1, years_back * 365)
            # Random hour of the day
            hour = random.randint(0, 23)
            
            target_date = now - timedelta(days=days_back)
            target_date = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            random_hours.append(target_date)
            
        self.logger.debug(f"Generated {len(random_hours)} random hours to sample")
        
        # Process each random hour
        all_repos = []
        processed_hours = 0
        errors = 0
        
        for i, target_date in enumerate(random_hours):
            # Format the date for the archive URL: YYYY-MM-DD-H
            archive_date = target_date.strftime('%Y-%m-%d-%-H')
            archive_url = f"https://data.gharchive.org/{archive_date}.json.gz"
            
            self.logger.info(f"Processing archive {i+1}/{hours_to_sample}: {archive_date}, URL: {archive_url}")
            
            try:
                # Download the archive
                download_start = time.time()
                self.logger.debug(f"Downloading archive: {archive_url}")
                response = requests.get(archive_url, stream=True, timeout=30)
                response.raise_for_status()  # Raise an exception for HTTP errors
                download_time = time.time() - download_start
                self.logger.debug(f"Download completed in {download_time:.2f} seconds")
                
                # Process the archive
                hour_repos = {}  # Use dict to avoid duplicates within the same hour
                process_start = time.time()
                events_processed = 0
                events_skipped = 0
                events_error = 0
                
                # Decompress and process line by line
                self.logger.debug(f"Processing archive content")
                with gzip.GzipFile(fileobj=response.raw) as f:
                    for line in f:
                        try:
                            event = json.loads(line.decode('utf-8'))
                            events_processed += 1
                            
                            # Only process specified event types
                            if event.get('type') not in event_types:
                                events_skipped += 1
                                continue
                            
                            # Extract repo information
                            repo = event.get('repo', {})
                            repo_name = repo.get('name')
                            
                            # Skip if no valid repo name
                            if not repo_name or '/' not in repo_name:
                                events_skipped += 1
                                continue
                            
                            # Skip if we already have this repo from this hour
                            if repo_name in hour_repos:
                                events_skipped += 1
                                continue
                            
                            # Get additional repo information
                            owner, name = repo_name.split('/', 1)
                            
                            repo_data = {
                                'full_name': repo_name,
                                'name': name,
                                'owner': owner,
                                'html_url': repo.get('url') or f"https://github.com/{repo_name}",
                                'created_at': event.get('created_at'),
                                'sampled_from': archive_date,
                                'event_type': event.get('type')
                            }
                            
                            # Store repo in our hour collection
                            hour_repos[repo_name] = repo_data
                            
                            # Log periodically about repos found
                            if len(hour_repos) % 5 == 0:
                                self.logger.debug(f"Found {len(hour_repos)} repositories so far in this archive")
                            
                            # Break if we have enough repos for this hour
                            if len(hour_repos) >= repos_per_hour:
                                self.logger.debug(f"Reached target of {repos_per_hour} repositories for this hour")
                                break
                                
                        except json.JSONDecodeError:
                            events_error += 1
                            continue  # Skip invalid JSON lines
                        except Exception as e:
                            events_error += 1
                            self.logger.warning(f"Error processing event: {e}")
                            continue
                
                process_time = time.time() - process_start
                
                # Add repos from this hour to our overall collection
                all_repos.extend(list(hour_repos.values()))
                processed_hours += 1
                
                self.logger.info(
                    f"Found {len(hour_repos)} repositories from {archive_date} "
                    f"(processed {events_processed} events in {process_time:.2f} seconds, "
                    f"skipped {events_skipped}, errors {events_error})"
                )
                
                # Log progress towards overall target
                self.logger.info(f"Progress: {len(all_repos)}/{n_samples} repositories collected")
                
                # Break if we have enough repositories
                if len(all_repos) >= n_samples:
                    all_repos = all_repos[:n_samples]  # Trim to exact count
                    self.logger.info(f"Reached target sample size of {n_samples} repositories")
                    break
                    
            except requests.RequestException as e:
                self.logger.error(f"Failed to download archive {archive_url}: {e}")
                errors += 1
                continue
            except Exception as e:
                self.logger.error(f"Error processing archive {archive_url}: {e}")
                errors += 1
                continue
        
        # Calculate processing statistics
        elapsed = time.time() - start_time
        hours_per_second = processed_hours / elapsed if elapsed > 0 else 0
        repos_per_second = len(all_repos) / elapsed if elapsed > 0 else 0
        
        # Log summary
        self.logger.info(
            f"Completed archive sampling in {elapsed:.2f} seconds: "
            f"found {len(all_repos)} repositories from {processed_hours} hours "
            f"(errors: {errors}, rate: {hours_per_second:.2f} hours/sec, "
            f"{repos_per_second:.2f} repos/sec)"
        )
        
        # Randomize the final list to avoid time-based patterns
        if self._seed is not None:
            random.seed(self._seed)
        random.shuffle(all_repos)
        
        # Limit to requested sample size
        result = all_repos[:n_samples]
        
        # Apply any filters
        filtered_count_before = len(result)
        if kwargs:
            result = self._filter_repos(result, **kwargs)
            filtered_count_after = len(result)
            if filtered_count_before != filtered_count_after:
                self.logger.info(
                    f"Applied filters: {filtered_count_before - filtered_count_after} repositories filtered out, "
                    f"{filtered_count_after} repositories remaining"
                )
        
        self.attempts = hours_to_sample
        self.success_count = processed_hours
        self.results = result
        
        return result
        
    def _filter_repos(self, repos: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Filter repositories based on criteria.
        
        Args:
            repos: List of repository data
            **kwargs: Filter criteria as key-value pairs
            
        Returns:
            Filtered list of repositories
        """
        # Implementation of filter logic
        filtered = repos
        
        # Example: filter by owner
        if 'owner' in kwargs:
            owner = kwargs['owner']
            filtered = [r for r in filtered if r['owner'] == owner]
            self.logger.debug(f"Filtered by owner '{owner}': {len(filtered)} repos remaining")
            
        # Example: filter by min stars (if available)
        if 'min_stars' in kwargs and any('stargazers_count' in r for r in repos):
            min_stars = int(kwargs['min_stars'])
            filtered = [r for r in filtered if r.get('stargazers_count', 0) >= min_stars]
            self.logger.debug(f"Filtered by min_stars {min_stars}: {len(filtered)} repos remaining")
            
        return filtered