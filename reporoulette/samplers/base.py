# reporoulette/samplers/base.py
from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any, Optional

class BaseSampler(ABC):
    """
    Base class for all repository samplers.
    """
    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token
        self.results: List[Dict[str, Any]] = []
        self.attempts: int = 0
        self.success_count: int = 0
        self.logger = logging.getLogger(__name__)
    
    @property
    def success_rate(self) -> float:
        """
        Calculate the success rate of sampling attempts.
        
        Returns:
            float: Percentage of successful attempts
        """
        if self.attempts == 0:
            return 0.0
        return (self.success_count / self.attempts) * 100
    
    @abstractmethod
    def sample(self, n_samples: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Sample repositories according to the specific strategy.
        
        Args:
            n_samples: Number of repositories to sample
            **kwargs: Additional parameters specific to each sampler
            
        Returns:
            List of repository data dictionaries
        """
        pass
    
    def _filter_repos(self, repos: List[Dict[str, Any]], **filters: Any) -> List[Dict[str, Any]]:
        """
        Filter repositories based on criteria.
        
        Args:
            repos: List of repository data to filter
            **filters: Criteria to filter by (e.g., min_stars, languages)
            
        Returns:
            Filtered list of repositories
        """
        filtered = repos
        
        if 'min_stars' in filters:
            filtered = [r for r in filtered if r.get('stargazers_count', 0) >= filters['min_stars']]
            
        if 'min_forks' in filters:
            filtered = [r for r in filtered if r.get('forks_count', 0) >= filters['min_forks']]
            
        if 'languages' in filters and filters['languages']:
            filtered = [r for r in filtered if r.get('language') in filters['languages']]
            
        return filtered