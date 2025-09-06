import random
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union

import requests

from .base import BaseSampler
from ..logging_config import get_logger

class TemporalSampler(BaseSampler):
    """
    Sample repositories by randomly selecting days and fetching repos updated in those periods.
    
    This sampler selects random days within a specified date range,
    weights them by repository count, and retrieves repositories with proportional sampling.
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
        self.logger = get_logger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
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
            
        # Ensure dates have no time component for consistent day-level sampling
        self.start_date = self.start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        self.end_date = self.end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            
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
        
    def _random_date(self) -> datetime:
        """
        Generate a random date within the specified range.
        
        Returns:
            Random datetime object with time set to beginning of day
        """
        time_delta = self.end_date - self.start_date
        random_days = random.randint(0, time_delta.days)
        random_date = self.start_date + timedelta(days=random_days)
        
        # Set to beginning of day
        return random_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _format_date_for_query(self, dt: datetime) -> Tuple[str, str]:
        """
        Format a date for GitHub API query.
        
        Args:
            dt: Date to format
            
        Returns:
            Tuple of (start, end) strings for the day period
        """
        # Set to beginning of day
        dt_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        # Set to end of day
        dt_next_day = dt_day + timedelta(days=1)
        
        # Format for GitHub API with Z suffix for UTC
        start_str = dt_day.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
        end_str = dt_next_day.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
        
        return start_str, end_str
    
    def _build_search_query(
        self,
        start_time_str: str,
        end_time_str: str,
        min_stars: int = 0,
        min_size_kb: int = 0,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Build a search query string for the GitHub API.
        
        Args:
            start_time_str: Start time in ISO format
            end_time_str: End time in ISO format
            min_stars: Minimum number of stars
            min_size_kb: Minimum repository size in KB
            language: Programming language to filter by
            **kwargs: Additional filters
            
        Returns:
            Query string
        """
        # Construct query for repositories updated in this time period
        query_parts = [f"pushed:{start_time_str}..{end_time_str}"]
        
        # Add language filter if specified
        if language:
            query_parts.append(f"language:{language}")
        elif 'languages' in kwargs and kwargs['languages']:
            query_parts.append(f"language:{kwargs['languages'][0]}")
            
        # Add star filter if specified
        if min_stars > 0:
            query_parts.append(f"stars:>={min_stars}")
            
        # Add size filter if specified
        if min_size_kb > 0:
            query_parts.append(f"size:>={min_size_kb}")
            
        # Join query parts
        return " ".join(query_parts)
    
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
        days_to_sample: int = 10,  # Changed from hours_to_sample
        repos_to_collect: int = 100,
        per_page: int = 100,
        min_wait: float = 1.0,
        min_stars: int = 0,
        min_size_kb: int = 0,
        language: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories by randomly selecting days with weighting based on repo count.
        
        Args:
            days_to_sample: Number of random days to initially sample for count assessment
            repos_to_collect: Target number of repositories to collect
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
            f"Starting weighted temporal sampling: days_to_sample={days_to_sample}, "
            f"repos_to_collect={repos_to_collect}, per_page={per_page}, "
            f"min_stars={min_stars}, min_size_kb={min_size_kb}, language={language or 'None'}"
        )
        
        headers = {}
        if self.token:
            self.logger.info("Using GitHub API token for authentication")
            headers['Authorization'] = f'token {self.token}'
        else:
            self.logger.warning("No GitHub API token provided. Rate limits will be restricted.")
        
        # Initialize variables
        all_repos = []
        period_data = {}  # Maps periods to their repo counts and first page data
        self.attempts = 0
        self.success_count = 0
        start_time = time.time()
        
        # Generate random days for initial sampling
        initial_days = []
        for _ in range(days_to_sample):
            random_dt = self._random_date()
            initial_days.append(random_dt)
        
        # Sort chronologically for better logging
        initial_days.sort()
        
        self.logger.info(f"Generated {len(initial_days)} random days to sample")
        
        # Step 1: Get the first page of results and total counts for each day in one pass
        for i, day in enumerate(initial_days):
            # Check rate limit periodically
            if i % 5 == 0:
                remaining = self._check_rate_limit()
                if remaining <= self.rate_limit_safety:
                    self.logger.warning(
                        f"Approaching GitHub API rate limit ({remaining} remaining). "
                        f"Stopping initial sampling after {i}/{days_to_sample} days."
                    )
                    break
            
            start_time_str, end_time_str = self._format_date_for_query(day)
            day_str = day.strftime("%Y-%m-%d")
            
            # Build query
            query = self._build_search_query(
                start_time_str, end_time_str, min_stars, min_size_kb, language, **kwargs
            )
            
            # Construct the URL for first page
            url = f"{self.api_base_url}/search/repositories?q={query}&sort=updated&order=desc&per_page={per_page}&page=1"
            
            self.logger.info(f"Sampling day {i+1}/{days_to_sample}: {day_str}")
            
            try:
                self.attempts += 1
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    results = response.json()
                    count = results['total_count']
                    
                    if count > 0:
                        self.success_count += 1
                        self.logger.info(f"Found {count} repositories on {day_str}")
                        
                        # Store period data including count and first page results
                        period_data[day] = {
                            'count': count,
                            'first_page': results['items'],
                            'day_str': day_str
                        }
                        
                        # Process first page repos and add to collection
                        period_repos = []
                        for repo in results['items']:
                            # Skip repos we already have
                            if any(r['full_name'] == repo['full_name'] for r in all_repos):
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
                                'sampled_from': day_str  # Add the day this repo was sampled from
                            }
                            
                            period_repos.append(repo_data)
                        
                        # Add first page repos to our collection
                        all_repos.extend(period_repos)
                        self.logger.info(f"Added {len(period_repos)} repositories from first page")
                    else:
                        self.logger.info(f"No repositories found on {day_str}")
                
                elif response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
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
                
                # Mandatory wait between requests
                wait_time = min_wait + random.uniform(0, 0.5)
                time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Error sampling day {day_str}: {str(e)}")
                time.sleep(min_wait * 2)  # Longer delay on error
        
        # Step 2: Create weighted distribution based on repository counts
        # Filter out days with zero repositories
        valid_days = {p: data['count'] for p, data in period_data.items() if data['count'] > 0}
        
        if not valid_days:
            self.logger.warning("No repositories found in any sampled days. Returning empty list.")
            return []
        
        # Get enough repositories to meet our target
        if len(all_repos) < repos_to_collect:
            # Step 3: Create probability distribution for weighted sampling
            days = list(valid_days.keys())
            weights = [valid_days[day] for day in days]
            total_weight = sum(weights)
            
            # Normalize weights to get probabilities
            probs = [weight / total_weight for weight in weights]
            
            self.logger.info(
                f"Created weighted distribution across {len(days)} days "
                f"(total weight: {total_weight})"
            )
            
            # Log the top days with highest weights
            top_days = sorted(valid_days.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info("Top 5 days by repository count:")
            for day, count in top_days:
                day_str = period_data[day]['day_str']
                self.logger.info(f"  {day_str}: {count} repositories")
            
            # Step 4: Sample additional repositories from days based on weighted distribution
            while len(all_repos) < repos_to_collect:
                # Check if we're approaching rate limit
                if self.attempts % 5 == 0:
                    remaining = self._check_rate_limit()
                    if remaining <= self.rate_limit_safety:
                        self.logger.warning(
                            f"Approaching GitHub API rate limit ({remaining} remaining). "
                            f"Stopping after collecting {len(all_repos)}/{repos_to_collect} repositories."
                        )
                        break
                
                # Select a day using weighted random choice
                day = random.choices(days, weights=probs, k=1)[0]
                day_info = period_data[day]
                day_str = day_info['day_str']
                count = day_info['count']
                
                # Skip if we've already collected enough from this day
                # (To avoid repeatedly sampling the same popular day)
                if sum(1 for repo in all_repos if repo.get('sampled_from') == day_str) >= count / 2:
                    continue
                
                start_time_str, end_time_str = self._format_date_for_query(day)
                self.attempts += 1
                
                # Log the day we're querying
                self.logger.info(
                    f"Sampling weighted day: {day_str} (weight: {count}) "
                    f"- collected {len(all_repos)}/{repos_to_collect} repositories so far"
                )
                
                # Build query
                query = self._build_search_query(
                    start_time_str, end_time_str, min_stars, min_size_kb, language, **kwargs
                )
                
                # For days with many repos, select a random page within the first N pages
                # Skip page 1 since we already have it
                max_page = min(10, (count // per_page) + 1)
                page = 1 if max_page <= 1 else random.randint(2, max_page)
                
                # Construct the URL for additional page
                url = f"{self.api_base_url}/search/repositories?q={query}&sort=updated&order=desc&per_page={per_page}&page={page}"
                
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
                                f"(fetched {len(repos)} from page {page} in {query_elapsed:.2f} seconds)"
                            )
                            
                            # Process repos to match our standard format
                            period_repos = []
                            for repo in repos:
                                # Skip repos we already have
                                if any(r['full_name'] == repo['full_name'] for r in all_repos):
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
                                    'sampled_from': day_str  # Add the day this repo was sampled from
                                }
                                
                                period_repos.append(repo_data)
                                
                            # Add new repos from this period
                            all_repos.extend(period_repos)
                            added_count = len(period_repos)
                            self.logger.info(f"Added {added_count} new repositories from this day")
                            
                            # If we've added enough repos, we can stop
                            if len(all_repos) >= repos_to_collect:
                                self.logger.info(f"Reached target of {repos_to_collect} repositories. Stopping sampling.")
                                break
                        else:
                            self.logger.info(f"No repositories found on {day_str}")
                    
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
                    wait_time = min_wait + random.uniform(0, 0.5)
                    self.logger.debug(f"Waiting {wait_time:.1f} seconds before next request...")
                    time.sleep(wait_time)
                    
                except Exception as e:
                    self.logger.error(f"Error sampling day {day_str}: {str(e)}")
                    time.sleep(min_wait * 2)  # Longer delay on error
        
        # Report summary
        elapsed_time = time.time() - start_time
        success_rate = (self.success_count / self.attempts) * 100 if self.attempts > 0 else 0
        
        self.logger.info(
            f"Sampling completed in {elapsed_time:.2f} seconds: "
            f"{self.attempts} attempts, {self.success_count} successful ({success_rate:.1f}%), "
            f"collected {len(all_repos)} repositories"
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