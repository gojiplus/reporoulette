# reporoulette/samplers/gh_archive_sampler.py
import logging
import sys
import random
import time
import requests
import gzip
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from .base import BaseSampler

class GHArchiveSampler(BaseSampler):
    """
    Sample repositories by downloading and processing GH Archive files.
    
    This sampler randomly selects days from GitHub's event history, downloads
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
    
    def sample(self, n_samples: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """
        Sample repositories using the GH Archive approach.
        
        This is the implementation of the abstract method from BaseSampler,
        which delegates to the gh_sampler method with the provided parameters.
        
        Args:
            n_samples: Number of repositories to sample
            **kwargs: Additional parameters to pass to gh_sampler
            
        Returns:
            List of repository data
        """
        self.logger.info(f"Sample method called with n_samples={n_samples}")
        
        # Call the main implementation method
        return self.gh_sampler(n_samples=n_samples, **kwargs)
    
    def gh_sampler(
        self,
        n_samples: int = 100,
        days_to_sample: int = 5,
        repos_per_day: int = 20,
        years_back: int = 10,
        event_types: List[str] = ["CreateEvent"],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories by downloading and processing full day's GH Archive files.
        
        Args:
            n_samples: Target number of repositories to sample
            days_to_sample: Number of random days to sample
            repos_per_day: Maximum repositories to sample per day
            years_back: How many years to look back
            event_types: Types of GitHub events to consider
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        self.logger.info(
            f"Sampling via archives: n_samples={n_samples}, days_to_sample={days_to_sample}, "
            f"repos_per_day={repos_per_day}, years_back={years_back}, "
            f"event_types={event_types}"
        )
        
        # Log filter criteria if any
        if kwargs:
            self.logger.info(f"Filter criteria: {kwargs}")
        
        start_time = time.time()
        
        # Calculate parameters to ensure we get enough samples
        days_needed = max(1, (n_samples + repos_per_day - 1) // repos_per_day)
        days_to_sample = max(days_to_sample, days_needed)
        self.logger.debug(f"Adjusted days_to_sample to {days_to_sample} to ensure enough data")
        
        # Generate random days
        random_days = []
        now = datetime.now()
        
        for _ in range(days_to_sample):
            # Random days back (within years_back)
            days_back = random.randint(1, years_back * 365)
            
            target_date = now - timedelta(days=days_back)
            # Set to beginning of day
            target_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            random_days.append(target_date)
        
        # Process each random day
        all_repos = {}  # Use dict to avoid duplicates
        processed_days = 0
        errors = 0
        
        for i, target_date in enumerate(random_days):
            # Format the date string for logging
            day_str = target_date.strftime('%Y-%m-%d')
            self.logger.info(f"Processing day {i+1}/{days_to_sample}: {day_str}")
            
            day_repos = {}
            day_events_processed = 0
            day_events_skipped = 0
            day_events_error = 0
            day_process_start = time.time()
            
            # Download full day's archive
            try:
                # Download the entire day's archive (consolidated file)
                archive_date = target_date.strftime('%Y-%m-%d')
                archive_url = f"https://data.gharchive.org/{archive_date}.json.gz"
                
                response = requests.get(archive_url, stream=True, timeout=30)
                if response.status_code != 200:
                    self.logger.warning(f"Failed to download archive {archive_url}: HTTP {response.status_code}")
                    errors += 1
                    continue
                
                # Process the archive
                with gzip.GzipFile(fileobj=response.raw) as f:
                    for line in f:
                        try:
                            event = json.loads(line.decode('utf-8'))
                            day_events_processed += 1
                            
                            # Only process specified event types
                            event_type = event.get('type')
                            if event_type not in event_types:
                                day_events_skipped += 1
                                continue
                            
                            # Add an additional check to only record CreateEvent repositories
                            # if CreateEvent is the ONLY type in event_types
                            if len(event_types) == 1 and event_types[0] == 'CreateEvent':
                                # For CreateEvent, we only want actual repository creation
                                # This checks for repository create events specifically
                                if not (event_type == 'CreateEvent' and 
                                        event.get('payload', {}).get('ref_type') == 'repository'):
                                    day_events_skipped += 1
                                    continue
                            
                            # Extract repo information
                            repo = event.get('repo', {})
                            repo_name = repo.get('name')
                            
                            # Skip if no valid repo name
                            if not repo_name or '/' not in repo_name:
                                day_events_skipped += 1
                                continue
                            
                            # Skip if we already have this repo
                            if repo_name in day_repos:
                                day_events_skipped += 1
                                continue
                            
                            # Get additional repo information
                            owner, name = repo_name.split('/', 1)
                            
                            repo_data = {
                                'full_name': repo_name,
                                'name': name,
                                'owner': owner,
                                'html_url': repo.get('url') or f"https://github.com/{repo_name}",
                                'created_at': event.get('created_at'),
                                'sampled_from': day_str,
                                'event_type': event.get('type')
                            }
                            
                            # Store repo in our day collection
                            day_repos[repo_name] = repo_data
                            
                        except json.JSONDecodeError:
                            day_events_error += 1
                            continue
                        except Exception as e:
                            day_events_error += 1
                            self.logger.warning(f"Error processing event: {e}")
                            continue
                
                # Randomly sample repositories for this day
                day_repos_list = list(day_repos.values())
                random.shuffle(day_repos_list)
                sampled_day_repos = day_repos_list[:repos_per_day]
                
                # Add sampled repos to overall collection
                for repo_data in sampled_day_repos:
                    all_repos[repo_data['full_name']] = repo_data
                
                day_process_time = time.time() - day_process_start
                processed_days += 1
                
                self.logger.info(
                    f"Found {len(sampled_day_repos)} repositories from {day_str} "
                    f"(processed {day_events_processed} events in {day_process_time:.2f} seconds, "
                    f"skipped {day_events_skipped}, errors {day_events_error})"
                )
                
                # Log progress towards overall target
                self.logger.info(f"Progress: {len(all_repos)}/{n_samples} repositories collected")
                
                # Break if we have enough repositories
                if len(all_repos) >= n_samples:
                    self.logger.info(f"Reached target sample size of {n_samples} repositories")
                    break
            
            except Exception as e:
                self.logger.warning(f"Error processing day {day_str}: {e}")
                errors += 1
        
        # Convert to list
        result_repos = list(all_repos.values())
        
        # Calculate processing statistics
        elapsed = time.time() - start_time
        days_per_second = processed_days / elapsed if elapsed > 0 else 0
        repos_per_second = len(result_repos) / elapsed if elapsed > 0 else 0
        
        # Log summary
        self.logger.info(
            f"Completed archive sampling in {elapsed:.2f} seconds: "
            f"found {len(result_repos)} repositories from {processed_days} days "
            f"(errors: {errors}, rate: {days_per_second:.2f} days/sec, "
            f"{repos_per_second:.2f} repos/sec)"
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
                    f"Applied filters: {filtered_count_before - filtered_count_after} repositories filtered out, "
                    f"{filtered_count_after} repositories remaining"
                )
        
        self.attempts = days_to_sample
        self.success_count = processed_days
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
