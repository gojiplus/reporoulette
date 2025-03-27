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
        seed: Optional[int] = None,  # Reproducibility parameter
        years_back: int = 10  # Default years to look back
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
        """
        super().__init__(token)
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            self.logger.info(f"Random seed set to: {seed}")
        
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
            response = requests.get(f"{self.api_base_url}/rate_limit", headers=headers)
            if response.status_code == 200:
                data = response.json()
                remaining = data['resources']['core']['remaining']
                self.logger.debug(f"Rate limit remaining: {remaining}")
                return remaining
            else:
                self.logger.warning(f"Failed to check rate limit: {response.status_code}")
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
            response = requests.get(f"{self.api_base_url}/rate_limit", headers=headers)
            if response.status_code == 200:
                data = response.json()
                reset_time = data['resources']['core']['reset']
                now = time.time()
                wait_time = max(0, reset_time - now) + 10  # Add 10s buffer
                return wait_time
            return 60  # Default to 60s if can't determine
        except Exception:
            return 60  # Default to 60s if can't determine
    
    def sample(
        self, 
        n_samples: int = 10, 
        per_page: int = 10,
        min_wait: float = 1.0,
        max_attempts: int = 50,
        min_stars: int = 0,
        min_size_kb: int = 0,
        select_random: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories by randomly selecting time periods.
        
        Args:
            n_samples: Target number of repositories to collect
            per_page: Number of results per page (max 100)
            min_wait: Minimum wait time between API requests
            max_attempts: Maximum number of time periods to try
            min_stars: Minimum number of stars (0 for no filtering)
            min_size_kb: Minimum repository size in KB (0 for no filtering)
            select_random: If True, select one random repo from each period instead of all
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
        
        # Capture start time for reporting
        start_time = time.time()
        
        # Continue until we have enough samples or hit max attempts
        while len(valid_repos) < n_samples and self.attempts < max_attempts:
            # Check rate limit
            remaining = self._check_rate_limit()
            if remaining <= self.rate_limit_safety:
                self.logger.warning(
                    f"Approaching GitHub API rate limit. Stopping with {len(valid_repos)} samples."
                )
                break
                
            # Generate random time point
            random_time = self._random_datetime()
            start_time_str, end_time_str = self._format_date_for_query(random_time)
            
            # Skip if we've already tried this period
            period_key = f"{start_time_str}-{end_time_str}"
            if period_key in attempted_periods:
                continue
                
            attempted_periods.add(period_key)
            self.attempts += 1
            
            # Log the period we're querying
            hour_str = random_time.strftime("%Y-%m-%d %H:00")
            self.logger.info(f"Sampling hour {self.attempts}/{max_attempts}: {hour_str}")
            
            # Construct query for repositories updated in this time period
            query_parts = [f"pushed:{start_time_str}..{end_time_str}"]
            
            # Add language filter if specified
            if 'languages' in kwargs and kwargs['languages']:
                query_parts.append(f"language:{kwargs['languages'][0]}")
            elif 'language' in kwargs:
                query_parts.append(f"language:{kwargs['language']}")
                
            # Add star filter if specified
            if min_stars > 0:
                query_parts.append(f"stars:>={min_stars}")
                
            # Add size filter if specified
            if min_size_kb > 0:
                query_parts.append(f"size:>={min_size_kb}")
                
            # Join query parts
            query = " ".join(query_parts)
            
            # Construct the URL
            url = f"{self.api_base_url}/search/repositories?q={query}&sort=updated&order=desc&per_page={per_page}"
            
            try:
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    if results['total_count'] > 0:
                        repos = results['items']
                        self.success_count += 1
                        
                        # Either select one random repo or process all
                        if select_random and repos:
                            repos = [random.choice(repos)]
                        
                        # Process repos to match our standard format
                        for repo in repos:
                            # Skip repos we already have
                            if any(r['full_name'] == repo['full_name'] for r in valid_repos):
                                continue
                                
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
                                'size': repo.get('size', 0)  # Size in KB
                            })
                            
                            # If we have enough samples, break
                            if len(valid_repos) >= n_samples:
                                break
                        
                        self.logger.info(f"Found {len(repos)} repositories, collected {len(valid_repos)}/{n_samples}")
                    else:
                        self.logger.debug(f"No repositories found in period {hour_str}")
                
                elif response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                    # Handle rate limiting - wait until reset
                    wait_time = self._calculate_rate_limit_wait_time()
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
                self.logger.error(f"Error sampling time period {hour_str}: {str(e)}")
                time.sleep(min_wait * 2)  # Longer delay on error
        
        # Report summary
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Completed with {self.attempts} attempts ({self.success_count} successful), "
            f"found {len(valid_repos)}/{n_samples} repositories in {elapsed_time:.2f} seconds"
        )
        
        # Apply any additional filters
        self.results = self._filter_repos(valid_repos, **kwargs)
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
        filtered_repos = repos.copy()
        
        # Filter by languages if specified
        if 'languages' in kwargs and kwargs['languages']:
            languages = [lang.lower() for lang in kwargs['languages']]
            filtered_repos = [
                repo for repo in filtered_repos 
                if repo.get('language') and repo.get('language').lower() in languages
            ]
            
        # Filter by min_stars if not already done in query
        if 'min_stars' in kwargs:
            min_stars = kwargs['min_stars']
            filtered_repos = [
                repo for repo in filtered_repos
                if repo.get('stargazers_count', 0) >= min_stars
            ]
            
        # Filter by min_size if not already done in query
        if 'min_size_kb' in kwargs:
            min_size = kwargs['min_size_kb']
            filtered_repos = [
                repo for repo in filtered_repos
                if repo.get('size', 0) >= min_size
            ]
            
        # Add more filters as needed
            
        return filtered_repos

# Command-line interface
if __name__ == "__main__":
    import argparse
    import json
    import os
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Sample random GitHub repositories by time periods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--count", type=int, default=10, help="Number of repositories to find")
    parser.add_argument("--min-stars", type=int, default=5, help="Minimum number of stars")
    parser.add_argument("--language", type=str, default="python", help="Programming language filter")
    parser.add_argument("--min-size", type=int, default=100, help="Minimum repository size in KB")
    parser.add_argument("--output", type=str, default="repos.jsonl", help="Output file")
    parser.add_argument("--years-back", type=int, default=10, help="How many years back to sample from")
    parser.add_argument("--token", type=str, default=None, help="GitHub API token")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token or os.environ.get("GITHUB_TOKEN")
    
    # Create sampler
    sampler = TemporalSampler(
        token=token,
        years_back=args.years_back,
        seed=args.seed
    )
    
    try:
        # Sample repositories
        repos = sampler.sample(
            n_samples=args.count,
            min_stars=args.min_stars,
            min_size_kb=args.min_size,
            language=args.language
        )
        
        # Write results to file
        with open(args.output, 'w') as f:
            for repo in repos:
                f.write(json.dumps(repo) + '\n')
                
        print(f"Found {len(repos)} repositories. Results saved to {args.output}")
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)
