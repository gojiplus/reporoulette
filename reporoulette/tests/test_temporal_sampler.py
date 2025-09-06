import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from reporoulette.samplers.temporal_sampler import TemporalSampler

class TestTemporalSampler(unittest.TestCase):
    
    def setUp(self):
        # Create a real instance with date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        self.sampler = TemporalSampler(
            seed=42, 
            start_date=start_date,
            end_date=end_date
        )
        
        # Mock logger
        self.sampler.logger = MagicMock()
    
    @patch('requests.get')
    def test_temporal_sampler_basic(self, mock_get):
        # Mock response for successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_count": 1,
            "items": [{
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
            }]
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
    def test_temporal_sampler_empty_results(self, mock_get):
        # Mock the rate limit check to always return a high number
        self.sampler._check_rate_limit = MagicMock(return_value=1000)
        
        # Mock a request with no results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_count": 0,
            "items": []
        }
        mock_get.return_value = mock_response
        
        # Call the sample method
        result = self.sampler.sample(n_samples=1, max_attempts=1)
        
        # Verify empty result
        self.assertEqual(len(result), 0)
        
        # Verify attributes
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 0)
    
    @patch('requests.get')
    def test_temporal_sampler_with_filters(self, mock_get):
        # Mock response for successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_count": 1,
            "items": [{
                "id": 12345,
                "name": "test-repo",
                "full_name": "test-owner/test-repo",
                "owner": {"login": "test-owner"},
                "html_url": "https://github.com/test-owner/test-repo",
                "description": "Test repository",
                "created_at": "2023-01-01T12:00:00Z",
                "updated_at": "2023-01-02T12:00:00Z",
                "pushed_at": "2023-01-03T12:00:00Z",
                "stargazers_count": 20,
                "forks_count": 5,
                "language": "Python",
                "visibility": "public"
            }]
        }
        mock_get.return_value = mock_response
        
        # Mock the rate limit check to always return a high number
        self.sampler._check_rate_limit = MagicMock(return_value=1000)
        
        # Call the sample method with filters
        result = self.sampler.sample(
            n_samples=1, 
            max_attempts=1,
            min_stars=10,
            languages=["Python"]
        )
        
        # Verify result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['stargazers_count'], 20)
        self.assertEqual(result[0]['language'], 'Python')
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        self.sampler.attempts = 10
        self.sampler.success_count = 3
        self.assertEqual(self.sampler.success_rate, 30.0)
        
        # Test zero attempts
        self.sampler.attempts = 0
        self.sampler.success_count = 0
        self.assertEqual(self.sampler.success_rate, 0.0)

if __name__ == '__main__':
    unittest.main()