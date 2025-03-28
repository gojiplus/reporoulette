import unittest
from unittest.mock import patch, MagicMock

from reporoulette.samplers.id_sampler import IDSampler

class TestIDSampler(unittest.TestCase):
    
    def setUp(self):
        # Create a real instance
        self.sampler = IDSampler(seed=42)
        
        # Mock logger
        self.sampler.logger = MagicMock()
    
    @patch('requests.get')  # Patch the requests.get directly
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
    
    @patch('requests.get')  # Patch the requests.get directly
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