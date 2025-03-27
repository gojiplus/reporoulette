import random
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union

import requests

from .base import BaseSampler

class TemporalSampler(BaseSampler):
    """
    Sample repositories by randomly selecting time points and fetching repos updated in those periods.
    
    This sampler selects random date/hour combinations within a specified range and
    retrieves repositories that were updated during those time periods.
    """
    def __init__(
        self,
        token: Optional[str] = None,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        rate_limit_safety: int = 100,
        seed: Optional[int] = None,
        years_back: int = 10,
        log_level: int = logging.INFO
    ):
        """
        Initialize the temporal sampler.
        
        Args:
            token: GitHub Personal Access Token
            start_date: Start of date range to sample from
            end_date: End of date range to sample from
            rate_limit_safety: Stop when this many API requests remain
            seed: Random seed for reproducibility
            years_back: How many years back to sample from (if start_date not specified)
            log_level: Logging level (default: INFO)
        """
        super().__init__(token)
        
        # Configure logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Add a handler if none exists
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
            self._seed = None
        
        # Default to current time for end_date if not specified
        if end_date is None:
            self.end_date = datetime.now()
        elif isinstance(end_date, str):
            self.end_date = datetime.fromisoformat(end_date)
        else:
            self.end_date = end_date
            
        # Use years_back parameter instead of fixed 90 days
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=365 * years_back)
        elif isinstance(start_date, str):
            self.start_date = datetime.fromisoformat(start_date)
        else:
            self.start_date = start_date
            
        self.rate_limit_safety = rate_limit_safety
        self.api_base_url = "https://api.github.com"
        
        time_delta = self.end_date - self.start_date
        
        self.logger.info(
            f"Initialized TemporalSampler with date range: "
            f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')} "
            f"({time_delta.days} days)"
        )
        
        # Initialize tracking variables
        self.attempts = 0
        self.success_count = 0
        self.results = []
        
    def _random_datetime(self) -> datetime:
        """
        Generate a random datetime within the specified range.
        
        Returns:
            Random datetime object rounded to the hour
        """
        time_delta = self.end_date - self.start_date
        random_seconds = random.randint(0, int(time_delta.total_seconds()))
        random_dt = self.start_date + timedelta(seconds=random_seconds)
        
        # Round to the hour for consistency
        return random_dt.replace(minute=0, second=0, microsecond=0)
    
    def _format_date_for_query(self, dt: datetime) -> Tuple[str, str]:
        """
        Format a datetime for GitHub API query.
        
        Args:
            dt: Datetime to format
            
        Returns:
            Tuple of (start, end) strings for the hour period
        """
        # Round to the hour
        dt_hour = dt.replace(minute=0, second=0, microsecond=0)
        dt_next_hour = dt_hour + timedelta(hours=1)
        
        # Format for GitHub API with Z suffix for UTC
        start_str = dt_hour.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
        end_str = dt_next_hour.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
        
        return start_str, end_str
    
    def _check_rate_limit(self) -> int:
        """
        Check GitHub API rate limit and return remaining requests.
        
        Returns:
            Number of remaining API requests
        """
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'
            
        try:
            self.logger.debug("Checking GitHub API rate limit")
            response = requests.get(f"{self.api_base_url}/rate_limit", headers=headers)
            if response.status_code == 200:
                data = response.json()
                remaining = data['resources']['core']['remaining']
                reset_time = data['resources']['core']['reset']
                reset_datetime = datetime.fromtimestamp(reset_time)
                self.logger.debug(
                    f"Rate limit status: {remaining} requests remaining, "
                    f"reset at {reset_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                return remaining
            else:
                self.logger.warning(
                    f"Failed to check rate limit: {response.status_code}, "
                    f"Response: {response.text[:100]}"
                )
                return 0
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {str(e)}")
            return 0
    
    def _calculate_rate_limit_wait_time(self) -> float:
        """
        Calculate wait time until rate limit reset.
        
        Returns:
            Seconds to wait until reset (plus 10 second buffer)
        """
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'
            
        try:
            self.logger.debug("Calculating rate limit wait time")
            response = requests.get(f"{self.api_base_url}/rate_limit", headers=headers)
            if response.status_code == 200:
                data = response.json()
                reset_time = data['resources']['core']['reset']
                now = time.time()
                wait_time = max(0, reset_time - now) + 10  # Add 10s buffer
                reset_datetime = datetime.fromtimestamp(reset_time)
                self.logger.info(
                    f"Rate limit will reset at {reset_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"(in {wait_time:.1f} seconds)"
                )
                return wait_time
            self.logger.warning(f"Failed to get rate limit reset time: {response.status_code}")
            return 60  # Default to 60s if can't determine
        except Exception as e:
            self.logger.error(f"Error calculating rate limit wait time: {str(e)}")
            return 60  # Default to 60s if can't determine
    
    def sample(
        self, 
        hours_to_sample: int = 10,
        per_page: int = 100,
        min_wait: float = 1.0,
        min_stars: int = 0,
        min_size_kb: int = 0,
        language: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories by randomly selecting time periods.
        
        Args:
            hours_to_sample: Number of random hours to sample
            per_page: Number of results per page (max 100)
            min_wait: Minimum wait time between API requests
            min_stars: Minimum number of stars (0 for no filtering)
            min_size_kb: Minimum repository size in KB (0 for no filtering)
            language: Programming language to filter by
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        self.logger.info(
            f"Starting temporal sampling: hours_to_sample={hours_to_sample}, "
            f"per_page={per_page}, min_stars={min_stars}, min_size_kb={min_size_kb}, "
            f"language={language or 'None'}"
        )
        
        # Log filter criteria if any
        if kwargs:
            self.logger.info(f"Additional filter criteria: {kwargs}")
        
        headers = {}
        if self.token:
            self.logger.info("Using GitHub API token for authentication")
            headers['Authorization'] = f'token {self.token}'
        else:
            self.logger.warning("No GitHub API token provided. Rate limits will be restricted.")
            
        all_repos = []
        attempted_periods = set()
        self.attempts = 0
        self.success_count = 0
        
        # Capture start time for reporting
        start_time = time.time()
        
        # Generate all random datetimes upfront
        random_hours = []
        for _ in range(hours_to_sample):
            random_dt = self._random_datetime()
            random_hours.append(random_dt)
        
        # Sort chronologically for better logging
        random_hours.sort()
        
        self.logger.info(f"Generated {len(random_hours)} random time periods to sample")
        
        # Process each random hour
        for i, random_time in enumerate(random_hours):
            # Check rate limit periodically
            if i % 5 == 0:
                remaining = self._check_rate_limit()
                if remaining <= self.rate_limit_safety:
                    self.logger.warning(
                        f"Approaching GitHub API rate limit ({remaining} remaining). "
                        f"Stopping after {i}/{hours_to_sample} time periods."
                    )
                    break
                    
            start_time_str, end_time_str = self._format_date_for_query(random_time)
            
            # Skip if we've already tried this period
            period_key = f"{start_time_str}-{end_time_str}"
            if period_key in attempted_periods:
                self.logger.debug(f"Skipping already attempted period: {period_key}")
                continue
                
            attempted_periods.add(period_key)
            self.attempts += 1
            
            # Log the period we're querying
            hour_str = random_time.strftime("%Y-%m-%d %H:00")
            self.logger.info(
                f"Sampling period {i+1}/{hours_to_sample}: {hour_str} "
                f"(collected {len(all_repos)} repositories so far)"
            )
            
            # Construct query for repositories updated in this time period
            query_parts = [f"pushed:{start_time_str}..{end_time_str}"]
            
            # Add language filter if specified
            if language:
                query_parts.append(f"language:{language}")
                self.logger.debug(f"Added language filter: {language}")
            elif 'languages' in kwargs and kwargs['languages']:
                query_parts.append(f"language:{kwargs['languages'][0]}")
                self.logger.debug(f"Added language filter: {kwargs['languages'][0]}")
                
            # Add star filter if specified
            if min_stars > 0:
                query_parts.append(f"stars:>={min_stars}")
                self.logger.debug(f"Added minimum stars filter: {min_stars}")
                
            # Add size filter if specified
            if min_size_kb > 0:
                query_parts.append(f"size:>={min_size_kb}")
                self.logger.debug(f"Added minimum size filter: {min_size_kb} KB")
                
            # Join query parts
            query = " ".join(query_parts)
            
            # Construct the URL
            url = f"{self.api_base_url}/search/repositories?q={query}&sort=updated&order=desc&per_page={per_page}"
            self.logger.debug(f"Query URL: {url}")
            
            try:
                query_start_time = time.time()
                response = requests.get(url, headers=headers)
                query_elapsed = time.time() - query_start_time
                
                if response.status_code == 200:
                    results = response.json()
                    
                    if results['total_count'] > 0:
                        repos = results['items']
                        self.success_count += 1
                        
                        self.logger.info(
                            f"Found {results['total_count']} repositories "
                            f"(fetched {len(repos)} in {query_elapsed:.2f} seconds)"
                        )
                        
                        # Process repos to match our standard format
                        period_repos = []
                        for repo in repos:
                            # Skip repos we already have
                            if any(r['full_name'] == repo['full_name'] for r in all_repos):
                                self.logger.debug(f"Skipping duplicate repository: {repo['full_name']}")
                                continue
                                
                            repo_data = {
                                'id': repo['id'],
                                'name': repo['name'],
                                'full_name': repo['full_name'],
                                'owner': repo['owner']['login'],
                                'html_url': repo['html_url'],
                                'description': repo.get('description'),
                                'created_at': repo['created_at'],
                                'updated_at': repo['updated_at'],
                                'pushed_at': repo.get('pushed_at'),
                                'stargazers_count': repo.get('stargazers_count', 0),
                                'forks_count': repo.get('forks_count', 0),
                                'language': repo.get('language'),
                                'visibility': repo.get('visibility', 'public'),
                                'size': repo.get('size', 0),  # Size in KB
                                'sampled_from': hour_str  # Add the time period this repo was sampled from
                            }
                            
                            period_repos.append(repo_data)
                            
                        # Add all new repos from this period
                        all_repos.extend(period_repos)
                        self.logger.info(f"Added {len(period_repos)} new repositories from this period")
                    else:
                        self.logger.info(f"No repositories found in period {hour_str}")
                
                elif response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                    # Handle rate limiting - wait until reset
                    wait_time = self._calculate_rate_limit_wait_time()
                    self.logger.warning(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    # Don't count this as an attempt
                    self.attempts -= 1
                    continue
                else:
                    self.logger.warning(
                        f"API error: Status code {response.status_code}, "
                        f"Response: {response.text[:200]}..."
                    )
                    
                # Mandatory wait between requests to avoid rate limiting
                # Use a fixed wait time with small jitter
                wait_time = min_wait + random.uniform(0, 0.5)
                self.logger.debug(f"Waiting {wait_time:.1f} seconds before next request...")
                time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Error sampling time period {hour_str}: {str(e)}")
                time.sleep(min_wait * 2)  # Longer delay on error
        
        # Report summary
        elapsed_time = time.time() - start_time
        success_rate = (self.success_count / self.attempts) * 100 if self.attempts > 0 else 0
        
        self.logger.info(
            f"Sampling completed in {elapsed_time:.2f} seconds: "
            f"{self.attempts} attempts, {self.success_count} successful ({success_rate:.1f}%), "
            f"collected {len(all_repos)} repositories across {self.success_count} time periods"
        )
        
        # Apply any additional filters
        filtered_count_before = len(all_repos)
        self.results = self._filter_repos(all_repos, **kwargs)
        filtered_count_after = len(self.results)
        
        if filtered_count_before != filtered_count_after:
            self.logger.info(
                f"Applied filters: {filtered_count_before - filtered_count_after} repositories filtered out, "
                f"{filtered_count_after} repositories remaining"
            )
        
        return self.results
        
    def _filter_repos(self, repos: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Apply additional filters to the list of repositories.
        
        Args:
            repos: List of repository dictionaries
            **kwargs: Filter criteria
            
        Returns:
            Filtered list of repositories
        """
        if not kwargs:
            return repos
            
        self.logger.debug(f"Filtering {len(repos)} repositories with criteria: {kwargs}")
        filtered_repos = repos.copy()
        
        # Filter by languages if specified
        if 'languages' in kwargs and kwargs['languages']:
            languages = [lang.lower() for lang in kwargs['languages']]
            before_count = len(filtered_repos)
            filtered_repos = [
                repo for repo in filtered_repos 
                if repo.get('language') and repo.get('language').lower() in languages
            ]
            self.logger.debug(
                f"Filtered by languages {languages}: "
                f"{before_count - len(filtered_repos)} repos removed, {len(filtered_repos)} remaining"
            )
            
        # Filter by language (single language) if specified
        elif 'language' in kwargs and kwargs['language']:
            language = kwargs['language'].lower()
            before_count = len(filtered_repos)
            filtered_repos = [
                repo for repo in filtered_repos 
                if repo.get('language') and repo.get('language').lower() == language
            ]
            self.logger.debug(
                f"Filtered by language '{language}': "
                f"{before_count - len(filtered_repos)} repos removed, {len(filtered_repos)} remaining"
            )
            
        # Filter by min_stars if specified
        if 'min_stars' in kwargs:
            min_stars = kwargs['min_stars']
            before_count = len(filtered_repos)
            filtered_repos = [
                repo for repo in filtered_repos
                if repo.get('stargazers_count', 0) >= min_stars
            ]
            self.logger.debug(
                f"Filtered by min_stars {min_stars}: "
                f"{before_count - len(filtered_repos)} repos removed, {len(filtered_repos)} remaining"
            )
            
        # Filter by min_size_kb if specified
        if 'min_size_kb' in kwargs:
            min_size = kwargs['min_size_kb']
            before_count = len(filtered_repos)
            filtered_repos = [
                repo for repo in filtered_repos
                if repo.get('size', 0) >= min_size
            ]
            self.logger.debug(
                f"Filtered by min_size_kb {min_size}: "
                f"{before_count - len(filtered_repos)} repos removed, {len(filtered_repos)} remaining"
            )
            
        # Filter by owner if specified
        if 'owner' in kwargs:
            owner = kwargs['owner']
            before_count = len(filtered_repos)
            filtered_repos = [
                repo for repo in filtered_repos
                if repo.get('owner') == owner
            ]
            self.logger.debug(
                f"Filtered by owner '{owner}': "
                f"{before_count - len(filtered_repos)} repos removed, {len(filtered_repos)} remaining"
            )
            
        # Filter by created_after if specified
        if 'created_after' in kwargs:
            created_after = kwargs['created_after']
            if isinstance(created_after, str):
                created_after = datetime.fromisoformat(created_after.replace('Z', '+00:00'))
            before_count = len(filtered_repos)
            filtered_repos = [
                repo for repo in filtered_repos 
                if repo.get('created_at') and datetime.fromisoformat(repo['created_at'].replace('Z', '+00:00')) >= created_after
            ]
            self.logger.debug(
                f"Filtered by created_after {created_after}: "
                f"{before_count - len(filtered_repos)} repos removed, {len(filtered_repos)} remaining"
            )
            
        # Filter by created_before if specified
        if 'created_before' in kwargs:
            created_before = kwargs['created_before']
            if isinstance(created_before, str):
                created_before = datetime.fromisoformat(created_before.replace('Z', '+00:00'))
            before_count = len(filtered_repos)
            filtered_repos = [
                repo for repo in filtered_repos 
                if repo.get('created_at') and datetime.fromisoformat(repo['created_at'].replace('Z', '+00:00')) <= created_before
            ]
            self.logger.debug(
                f"Filtered by created_before {created_before}: "
                f"{before_count - len(filtered_repos)} repos removed, {len(filtered_repos)} remaining"
            )
            
        # Filter by max_repos if specified (limit total number of repos)
        if 'max_repos' in kwargs:
            max_repos = kwargs['max_repos']
            if len(filtered_repos) > max_repos:
                # Shuffle first if seed is set to maintain reproducibility
                if self._seed is not None:
                    random.seed(self._seed)
                    random.shuffle(filtered_repos)
                filtered_repos = filtered_repos[:max_repos]
                self.logger.debug(f"Limited result to {max_repos} repositories")
        
        return filtered_repos