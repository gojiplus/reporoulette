# reporoulette/samplers/id_sampler.py
import random
import time
import logging
from typing import List, Dict, Any, Optional

import requests

from .base import BaseSampler
from ..logging_config import get_logger

class IDSampler(BaseSampler):
    """
    Sample repositories using random ID probing.
    
    This sampler generates random repository IDs within a specified range
    and attempts to retrieve repositories with those IDs from GitHub.
    """
    def __init__(
        self, 
        token: Optional[str] = None,
        min_id: int = 1,
        max_id: int = 500000000,
        rate_limit_safety: int = 100,
        seed: Optional[int] = None,  # Add seed parameter
        log_level: int = logging.INFO
    ):
        """
        Initialize the ID sampler.
        
        Args:
            token: GitHub Personal Access Token
            min_id: Minimum repository ID to sample from
            max_id: Maximum repository ID to sample from
            rate_limit_safety: Stop when this many API requests remain
            seed: Random seed for reproducibility
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(token)
        
        # Configure logger
        self.logger = get_logger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            self.logger.info(f"Random seed set to: {seed}")
            
        self.min_id = min_id
        self.max_id = max_id
        self.rate_limit_safety = rate_limit_safety
        
        self.logger.info(f"Initialized IDSampler with min_id={min_id}, max_id={max_id}")
    
    def _check_rate_limit(self, headers: Dict[str, str]) -> int:
        """
        Check GitHub API rate limit and return remaining requests.
        
        Args:
            headers: HTTP headers containing authentication token
            
        Returns:
            int: Number of remaining API requests, or 0 if check fails
        """
        try:
            self.logger.debug("Checking GitHub API rate limit")
            response = requests.get("https://api.github.com/rate_limit", headers=headers)
            if response.status_code == 200:
                data = response.json()
                remaining = data['resources']['core']['remaining']
                reset_time = data['resources']['core']['reset']
                self.logger.debug(f"Rate limit status: {remaining} requests remaining, reset at timestamp {reset_time}")
                return remaining
            self.logger.warning(f"Failed to check rate limit. Status code: {response.status_code}")
            return 0
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {str(e)}")
            return 0
    
    def sample(
        self, 
        n_samples: int = 10, 
        min_wait: float = 0.1,  # Add min_wait parameter
        max_attempts: int = 1000,  # Add max_attempts parameter
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories by trying random IDs.
        
        Args:
            n_samples: Number of valid repositories to collect
            min_wait: Minimum wait time between API requests
            max_attempts: Maximum number of IDs to try
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        self.logger.info(f"Starting sampling: target={n_samples}, max_attempts={max_attempts}")
        
        headers = {}
        if self.token:
            self.logger.info("Using GitHub API token for authentication")
            headers['Authorization'] = f'token {self.token}'
        else:
            self.logger.warning("No GitHub API token provided. Rate limits will be restricted.")
        
        valid_repos = []
        self.attempts = 0
        self.success_count = 0
        
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
                success_rate = (self.success_count / self.attempts) * 100 if self.attempts > 0 else 0
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
                remaining = self._check_rate_limit(headers)
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
            url = f"https://api.github.com/repositories/{repo_id}"
            try:
                response = requests.get(url, headers=headers)
                self.attempts += 1
                
                # Check if repository exists
                if response.status_code == 200:
                    repo_data = response.json()
                    self.success_count += 1
                    
                    # Log repository details at debug level
                    self.logger.debug(
                        f"Repository details: name={repo_data['name']}, "
                        f"owner={repo_data['owner']['login']}, "
                        f"stars={repo_data.get('stargazers_count', 0)}, "
                        f"language={repo_data.get('language')}"
                    )
                    
                    valid_repos.append({
                        'id': repo_id,
                        'name': repo_data['name'],
                        'full_name': repo_data['full_name'],
                        'owner': repo_data['owner']['login'],
                        'html_url': repo_data['html_url'],
                        'description': repo_data.get('description'),
                        'created_at': repo_data['created_at'],
                        'updated_at': repo_data['updated_at'],
                        'pushed_at': repo_data.get('pushed_at'),
                        'stargazers_count': repo_data.get('stargazers_count', 0),
                        'forks_count': repo_data.get('forks_count', 0),
                        'language': repo_data.get('language'),
                        'visibility': repo_data.get('visibility', 'unknown'),
                    })
                    self.logger.info(
                        f"Found valid repository ({len(valid_repos)}/{n_samples}): "
                        f"{repo_data['full_name']} (id: {repo_id})"
                    )
                elif response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                    # Handle rate limiting
                    wait_time = 60  # Simple fallback
                    try:
                        # Try to get the reset time from headers
                        if 'x-ratelimit-reset' in response.headers:
                            reset_time = int(response.headers['x-ratelimit-reset'])
                            current_time = time.time()
                            wait_time = max(reset_time - current_time + 5, 10)
                            self.logger.warning(
                                f"Rate limit exceeded. Reset at {time.ctime(reset_time)}. "
                                f"Waiting {wait_time:.1f} seconds..."
                            )
                    except Exception as e:
                        self.logger.error(f"Error parsing rate limit headers: {str(e)}")
                        
                    self.logger.warning(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    # Don't count this as an attempt
                    self.attempts -= 1
                    continue
                else:
                    self.logger.debug(
                        f"Invalid repository ID: {repo_id} "
                        f"(Status code: {response.status_code}, Response: {response.text[:100]}...)"
                    )
                    
                # Wait between requests with a small random jitter
                wait_time = min_wait + random.uniform(0, min_wait * 0.5)
                time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Error sampling repository ID {repo_id}: {str(e)}")
                time.sleep(min_wait * 5)  # Longer delay on error
        
        # Calculate final stats
        elapsed = time.time() - start_time
        success_rate = (self.success_count / self.attempts) * 100 if self.attempts > 0 else 0
        rate = self.attempts / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"Sampling completed in {elapsed:.2f} seconds: "
            f"{self.attempts} attempts, found {len(valid_repos)} repositories "
            f"({success_rate:.1f}% success rate, {rate:.2f} requests/sec)"
        )
        
        # Apply any filters
        filtered_count_before = len(valid_repos)
        self.results = self._filter_repos(valid_repos, **kwargs)
        filtered_count_after = len(self.results)
        
        if filtered_count_before != filtered_count_after:
            self.logger.info(
                f"Applied filters: {filtered_count_before - filtered_count_after} repositories filtered out, "
                f"{filtered_count_after} repositories remaining"
            )
        
        return self.results