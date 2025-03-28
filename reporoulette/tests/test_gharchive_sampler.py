# test_gharchive_sampler.py
import unittest
from unittest.mock import patch, MagicMock
import io
import gzip
import json
from datetime import datetime

# Import the actual class - correct import path for CI environment
from reporoulette.samplers.gh_sampler import GHArchiveSampler

class TestGHArchiveSampler(unittest.TestCase):
    
    def setUp(self):
        # Create a real instance with controlled parameters
        self.sampler = GHArchiveSampler(seed=42)
        
        # Mock the logger to avoid log output during tests
        self.sampler.logger = MagicMock()
    
    # Correct mock path for requests in gh_sampler module
    @patch('requests.get')
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
    
    @patch('requests.get')
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
        
    @patch('requests.get')
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


# test_id_sampler.py
import unittest
from unittest.mock import patch, MagicMock

from reporoulette.samplers.id_sampler import IDSampler

class TestIDSampler(unittest.TestCase):
    
    def setUp(self):
        # Create a real instance
        self.sampler = IDSampler(seed=42)
        
        # Mock logger
        self.sampler.logger = MagicMock()
    
    @patch('requests.get')
    def test_id_sampler_basic(self, mock_get):
        # Mock response for successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 12345,
            "name": "test-repo",
            "full_name": "test-owner/test-repo",
            "owner": {"login": "test-owner"},
            "html_url": "https://github.com/test-owner/test-repo",
            "description": "Test repository",
            "created_at": "2023-01-01T12:00:00Z",
            "updated_at": "2023-01-02T12:00:00Z",
            "pushed_at": "2023-01-03T12:00:00Z",
            "stargazers_count": 10,
            "forks_count": 5,
            "language": "Python",
            "visibility": "public"
        }
        mock_get.return_value = mock_response
        
        # Mock the rate limit check to always return a high number
        self.sampler._check_rate_limit = MagicMock(return_value=1000)
        
        # Call the sample method
        result = self.sampler.sample(n_samples=1, max_attempts=1)
        
        # Verify result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'test-repo')
        self.assertEqual(result[0]['owner'], 'test-owner')
        self.assertEqual(result[0]['language'], 'Python')
        
        # Verify attributes
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 1)
    
    @patch('requests.get')
    def test_id_sampler_error_handling(self, mock_get):
        # Mock the rate limit check to always return a high number
        self.sampler._check_rate_limit = MagicMock(return_value=1000)
        
        # Mock a failed request
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        # Call the sample method
        result = self.sampler.sample(n_samples=1, max_attempts=1)
        
        # Verify empty result
        self.assertEqual(len(result), 0)
        
        # Verify attributes
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 0)


if __name__ == '__main__':
    unittest.main()