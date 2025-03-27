import logging
import sys
from typing import List, Dict, Any

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Create a simple class to house the gh_sampler method
class GitHubSampler:
    def __init__(self, seed=None):
        self.logger = logging.getLogger("GitHubSampler")
        self._seed = seed
        self.attempts = 0
        self.success_count = 0
        self.results = []
    
    # Paste the gh_sampler function here
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
        import random
        from datetime import datetime, timedelta
        
        self.logger.info(
            f"Sampling via archives: n_samples={n_samples}, hours_to_sample={hours_to_sample}, "
            f"repos_per_hour={repos_per_hour}, years_back={years_back}"
        )
        
        # Calculate parameters to ensure we get enough samples
        hours_needed = max(1, (n_samples + repos_per_hour - 1) // repos_per_hour)
        hours_to_sample = max(hours_to_sample, hours_needed)
        
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
        
        # Process each random hour
        all_repos = []
        processed_hours = 0
        errors = 0
        
        for target_date in random_hours:
            # Format the date for the archive URL: YYYY-MM-DD-H
            archive_date = target_date.strftime('%Y-%m-%d-%-H')
            archive_url = f"https://data.gharchive.org/{archive_date}.json.gz"
            
            self.logger.info(f"Processing archive for {archive_date}, URL: {archive_url}")
            
            try:
                # Download the archive
                response = requests.get(archive_url, stream=True, timeout=30)
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Process the archive
                hour_repos = {}  # Use dict to avoid duplicates within the same hour
                
                # Decompress and process line by line
                with gzip.GzipFile(fileobj=response.raw) as f:
                    for line in f:
                        try:
                            event = json.loads(line.decode('utf-8'))
                            
                            # Only process specified event types
                            if event.get('type') not in event_types:
                                continue
                            
                            # Extract repo information
                            repo = event.get('repo', {})
                            repo_name = repo.get('name')
                            
                            # Skip if no valid repo name
                            if not repo_name or '/' not in repo_name:
                                continue
                            
                            # Skip if we already have this repo from this hour
                            if repo_name in hour_repos:
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
                            
                            # Break if we have enough repos for this hour
                            if len(hour_repos) >= repos_per_hour:
                                break
                        except json.JSONDecodeError:
                            continue  # Skip invalid JSON lines
                        except Exception as e:
                            self.logger.warning(f"Error processing event: {e}")
                            continue
                
                # Add repos from this hour to our overall collection
                all_repos.extend(list(hour_repos.values()))
                processed_hours += 1
                
                self.logger.info(f"Found {len(hour_repos)} repositories from {archive_date}")
                
                # Break if we have enough repositories
                if len(all_repos) >= n_samples:
                    all_repos = all_repos[:n_samples]  # Trim to exact count
                    break
                    
            except requests.RequestException as e:
                self.logger.error(f"Failed to download archive {archive_url}: {e}")
                errors += 1
                continue
            except Exception as e:
                self.logger.error(f"Error processing archive {archive_url}: {e}")
                errors += 1
                continue
        
        # Log summary
        self.logger.info(
            f"Completed archive sampling: found {len(all_repos)} repositories "
            f"from {processed_hours} hours (errors: {errors})"
        )
        
        # Randomize the final list to avoid time-based patterns
        if self._seed is not None:
            random.seed(self._seed)
        random.shuffle(all_repos)
        
        # Limit to requested sample size
        result = all_repos[:n_samples]
        
        self.attempts = hours_to_sample
        self.success_count = processed_hours
        self.results = result
        
        return result
