# reporoulette/samplers/id_sampler.py
import random
import time
from typing import List, Dict, Any, Optional

import requests

from .base import BaseSampler

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
        rate_limit_safety: int = 100
    ):
        """
        Initialize the ID sampler.
        
        Args:
            token: GitHub Personal Access Token
            min_id: Minimum repository ID to sample from
            max_id: Maximum repository ID to sample from
            rate_limit_safety: Stop when this many API requests remain
        """
        super().__init__(token)
        self.min_id = min_id
        self.max_id = max_id
        self.rate_limit_safety = rate_limit_safety
    
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
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample repositories by trying random IDs.
        
        Args:
            n_samples: Number of valid repositories to collect
            **kwargs: Additional filters to apply
            
        Returns:
            List of repository data
        """
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'
        
        valid_repos = []
        self.attempts = 0
        self.success_count = 0
        
        while len(valid_repos) < n_samples:
            # Check rate limit
            remaining = self._check_rate_limit(headers)
            if remaining <= self.rate_limit_safety:
                self.logger.warning(
                    f"Approaching GitHub API rate limit. Stopping with {len(valid_repos)} samples."
                )
                break
                
            # Generate random repository ID
            repo_id = random.randint(self.min_id, self.max_id)
            
            # Try to fetch the repository by ID
            url = f"https://api.github.com/repositories/{repo_id}"
            try:
                response = requests.get(url, headers=headers)
                self.attempts += 1
                
                # Check if repository exists
                if response.status_code == 200:
                    repo_data = response.json()
                    self.success_count += 1
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
                    self.logger.info(f"Found valid repository: {repo_data['full_name']}")
                else:
                    self.logger.debug(f"Invalid repository ID: {repo_id} (Status code: {response.status_code})")
                    
                # Small delay to avoid hitting rate limits
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error sampling repository ID {repo_id}: {str(e)}")
                time.sleep(1)  # Longer delay on error
        
        # Apply any filters
        self.results = self._filter_repos(valid_repos, **kwargs)
        return self.results