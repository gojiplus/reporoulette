import unittest
from unittest.mock import patch, MagicMock
import io
import gzip
import json
from datetime import datetime
import os
import sys

# Add the parent directory to sys.path to find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the actual class - use the correct module path
from reporoulette.samplers.gh_archive_sampler import GHArchiveSampler

class TestGHArchiveSampler(unittest.TestCase):
    
    def setUp(self):
        # Create a real instance with controlled parameters
        self.sampler = GHArchiveSampler(seed=42)
        
        # Mock the logger to avoid log output during tests
        self.sampler.logger = MagicMock()
    
    @patch('reporoulette.samplers.gh_archive_sampler.requests.get')
    def test_gh_sampler_basic(self, mock_get):
        # Mock the response from requests.get
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        
        # Create sample GitHub events
        events = [
            {
                "type": "PushEvent",
                "repo": {
                    "name": "owner1/repo1",
                    "url": "https://github.com/owner1/repo1"
                },
                "created_at": "2023-01-01T12:00:00Z"
            },
            {
                "type": "CreateEvent",
                "repo": {
                    "name": "owner2/repo2",
                    "url": "https://github.com/owner2/repo2"
                },
                "created_at": "2023-01-01T12:05:00Z"
            },
            {
                "type": "PullRequestEvent",
                "repo": {
                    "name": "owner3/repo3",
                    "url": "https://github.com/owner3/repo3"
                },
                "created_at": "2023-01-01T12:10:00Z"
            },
            # Add an event with a type we're not looking for
            {
                "type": "IssuesEvent",
                "repo": {
                    "name": "owner4/repo4",
                    "url": "https://github.com/owner4/repo4"
                },
                "created_at": "2023-01-01T12:15:00Z"
            }
        ]
        
        # Gzip the events and prepare the mock response
        gz_content = io.BytesIO()
        with gzip.GzipFile(fileobj=gz_content, mode='w') as f:
            for event in events:
                f.write((json.dumps(event) + '\n').encode('utf-8'))
        gz_content.seek(0)
        
        mock_response.raw = gz_content
        mock_get.return_value = mock_response
        
        # Call the method with minimal parameters for testing
        result = self.sampler.gh_sampler(
            n_samples=2,
            hours_to_sample=1,
            repos_per_hour=3,
            years_back=1,
            event_types=["PushEvent", "CreateEvent", "PullRequestEvent"]
        )
        
        # Verify the results
        self.assertEqual(len(result), 2)
        self.assertTrue(all(repo['full_name'] in ['owner1/repo1', 'owner2/repo2', 'owner3/repo3'] 
                           for repo in result))
        self.assertTrue(all(repo['event_type'] in ["PushEvent", "CreateEvent", "PullRequestEvent"] 
                           for repo in result))
        
        # Verify that IssuesEvent type was filtered out
        self.assertFalse(any(repo['full_name'] == 'owner4/repo4' for repo in result))
        
        # Verify that the instance attributes were updated
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 1)
        self.assertEqual(self.sampler.results, result)
    
    @patch('reporoulette.samplers.gh_archive_sampler.requests.get')
    def test_sample_method(self, mock_get):
        # Mock the response from requests.get
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        
        # Create sample GitHub events (same as above)
        events = [
            {
                "type": "PushEvent",
                "repo": {
                    "name": "owner1/repo1",
                    "url": "https://github.com/owner1/repo1"
                },
                "created_at": "2023-01-01T12:00:00Z"
            },
            {
                "type": "CreateEvent",
                "repo": {
                    "name": "owner2/repo2",
                    "url": "https://github.com/owner2/repo2"
                },
                "created_at": "2023-01-01T12:05:00Z"
            }
        ]
        
        # Gzip the events and prepare the mock response
        gz_content = io.BytesIO()
        with gzip.GzipFile(fileobj=gz_content, mode='w') as f:
            for event in events:
                f.write((json.dumps(event) + '\n').encode('utf-8'))
        gz_content.seek(0)
        
        mock_response.raw = gz_content
        mock_get.return_value = mock_response
        
        # Call the abstract sample method which should delegate to gh_sampler
        result = self.sampler.sample(
            n_samples=1,
            hours_to_sample=1,
            repos_per_hour=2
        )
        
        # Verify the results
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]['full_name'] in ['owner1/repo1', 'owner2/repo2'])
        
        # Verify that the instance attributes were updated
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 1)
        
    @patch('reporoulette.samplers.gh_archive_sampler.requests.get')
    def test_gh_sampler_error_handling(self, mock_get):
        # Mock a request exception
        mock_get.side_effect = Exception("Mock network error")
        
        # Call the function
        result = self.sampler.gh_sampler(
            n_samples=2,
            hours_to_sample=1,
            repos_per_hour=2,
            years_back=1
        )
        
        # Verify results are empty
        self.assertEqual(len(result), 0)
        
        # Verify the instance attributes were updated
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 0)
        self.assertEqual(self.sampler.results, [])

if __name__ == '__main__':
    unittest.main()