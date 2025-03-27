# reporoulette/samplers/temporal_sampler.py
import random
import time
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
        rate_limit_safety: int = 100
    ):
        """
        Initialize the temporal sampler.
        
        Args:
            token: GitHub Personal Access Token
            start_date: Start of date range to sample from
            end_date: End of date range to sample from
            rate_limit_safety: Stop when this many API requests remain
        """
        super().__init__(token)
        
        # Default to last 90 days if no range specified
        if end_date is None:
            self.end_date = datetime.now()
        elif isinstance(end_date, str):
            self.end_date = datetime.fromisoformat(end_date)
        else:
            self.end_date = end_date
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=90)
        elif isinstance(start_date, str):
            self.start_date = datetime.fromisoformat(start_date)
        else:
            self.start_date = start_date
            
        self.rate_limit_safety = rate_limit_safety
        
    def _random_datetime(self) -> datetime:
        """
        Generate a random datetime within the specified range.
        
        Returns:
            Random datetime
        """
        time_delta = self.end_date - self.start_date
        random_seconds = random.randint(0, int(time_delta.total_seconds()))
        return self.start_date + timedelta(seconds=random_seconds)
    
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
        
        # Format for GitHub API
        start_str = dt_hour.strftime('%Y-%m-%dT%H:%M:%S')
        end_str = dt_next_hour.strftime('%Y-%m-%dT%H:%M:%S')
        
        return start_str, end_str
    
    def _check_rate_limit(self, headers: Dict[str, str]) -> int:
        """Check GitHub API rate limit and return remaining requests."""
        try:
            response = requests.get("https://api.github.com/rate_limit", headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data['resources']['core']['remaining']
            return 0
        except Exception:
            return 0
    
    def sample(
        self, 
        n_samples: int = 10, 
        per_page: int = 100,
        min_wait: float = 1.0,  # Increased default wait time
        max_attempts: int = 50,  # Add max attempts limit
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories by randomly selecting time periods.
        
        Args:
            n_samples: Target number of repositories to collect (may return fewer)
            per_page: Number of results per page (max 100)
            min_wait: Minimum wait time between API requests
            max_attempts: Maximum number of time periods to try
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'
            
        valid_repos = []
        attempted_periods = set()
        self.attempts = 0
        self.success_count = 0
        
        # Set a reasonable limit on attempts
        while self.attempts < max_attempts:
            # Check rate limit
            remaining = self._check_rate_limit(headers)
            if remaining <= self.rate_limit_safety:
                self.logger.warning(
                    f"Approaching GitHub API rate limit. Stopping with {len(valid_repos)} samples."
                )
                break
                
            # Generate random time point
            random_time = self._random_datetime()
            start_time, end_time = self._format_date_for_query(random_time)
            
            # Skip if we've already tried this period
            period_key = f"{start_time}-{end_time}"
            if period_key in attempted_periods:
                continue
                
            attempted_periods.add(period_key)
            
            # Construct query for repositories updated in this time period
            query = f"updated:{start_time}..{end_time}"
            
            # Add language filter if specified
            if 'languages' in kwargs and kwargs['languages']:
                # Take the first language for the query, we'll filter the rest later
                query += f" language:{kwargs['languages'][0]}"
                
            # Construct the URL - always get first page only
            url = f"https://api.github.com/search/repositories?q={query}&sort=updated&order=desc&per_page={per_page}"
            
            try:
                self.logger.info(f"Querying time period {start_time} to {end_time}")
                response = requests.get(url, headers=headers)
                self.attempts += 1
                
                if response.status_code == 200:
                    results = response.json()
                    
                    if results['total_count'] > 0:
                        # Get the first page of results
                        repos = results['items']
                        self.success_count += 1
                        
                        # Process repos to match our standard format
                        for repo in repos:
                            valid_repos.append({
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
                            })
                            
                        self.logger.info(f"Found {len(repos)} repositories in period {start_time} to {end_time}")
                        
                        # If we have enough samples, break
                        if len(valid_repos) >= n_samples:
                            break
                    else:
                        self.logger.debug(f"No repositories found in period {start_time} to {end_time}")
                elif response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                    # Handle rate limiting - wait until reset
                    wait_time = self._calculate_rate_limit_wait_time(headers)
                    self.logger.warning(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    # Don't count this as an attempt
                    self.attempts -= 1
                    continue
                else:
                    self.logger.warning(f"API error: {response.status_code}, {response.text}")
                    
                # Mandatory wait between requests to avoid rate limiting
                # Use a fixed wait time with small jitter
                wait_time = min_wait + random.uniform(0, 0.5)
                self.logger.debug(f"Waiting {wait_time:.1f} seconds before next request...")
                time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Error sampling time period {start_time} to {end_time}: {str(e)}")
                time.sleep(min_wait * 2)  # Longer delay on error
        
        self.logger.info(f"Completed with {self.attempts} attempts, found {len(valid_repos)} repositories")
        
        # Apply any filters
        self.results = self._filter_repos(valid_repos, **kwargs)
        return self.results
