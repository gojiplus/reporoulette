import unittest
from unittest.mock import patch, MagicMock
import io
import gzip
import json
from datetime import datetime

class TestGHSampler(unittest.TestCase):
    
    def setUp(self):
        # Create a mock instance with basic required attributes
        self.instance = MagicMock()
        self.instance.logger = MagicMock()
        self.instance._seed = 42  # Fixed seed for reproducibility
        
        # Attach the method we want to test to our mock instance
        from types import MethodType
        import sys
        # Assuming the code is in a module named 'repo_sampler'
        # If it's in a different module, replace accordingly
        module_name = 'your_module_name'
        if module_name not in sys.modules:
            import types
            sys.modules[module_name] = types.ModuleType(module_name)
        
        # Get the function and bind it to our mock instance
        import inspect
        source = inspect.getsource(gh_sampler)
        namespace = {}
        exec(source, namespace)
        self.instance.gh_sampler = MethodType(namespace['gh_sampler'], self.instance)
    
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
        
        # Call the function with minimal parameters for testing
        result = self.instance.gh_sampler(
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
        self.assertEqual(self.instance.attempts, 1)
        self.assertEqual(self.instance.success_count, 1)
        self.assertEqual(self.instance.results, result)
        
    @patch('requests.get')
    def test_gh_sampler_error_handling(self, mock_get):
        # Mock a request exception
        mock_get.side_effect = Exception("Mock network error")
        
        # Call the function
        result = self.instance.gh_sampler(
            n_samples=2,
            hours_to_sample=1,
            repos_per_hour=2,
            years_back=1
        )
        
        # Verify results are empty
        self.assertEqual(len(result), 0)
        
        # Verify the instance attributes were updated
        self.assertEqual(self.instance.attempts, 1)
        self.assertEqual(self.instance.success_count, 0)
        self.assertEqual(self.instance.results, [])

if __name__ == '__main__':
    unittest.main()