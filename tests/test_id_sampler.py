import logging
import unittest
from unittest.mock import MagicMock, patch

from reporoulette.samplers.id_sampler import IDSampler


class TestIDSampler(unittest.TestCase):
    def setUp(self):
        # Create a real instance
        self.sampler = IDSampler(seed=42)

        # Mock logger
        self.sampler.logger = MagicMock()

    @patch("requests.get")  # Patch the requests.get directly
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
            "visibility": "public",
        }
        mock_get.return_value = mock_response

        # Mock the rate limit check to always return a high number
        self.sampler._check_rate_limit = MagicMock(return_value=1000)

        # Call the sample method
        result = self.sampler.sample(n_samples=1, max_attempts=1)

        # Verify result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "test-repo")
        self.assertEqual(result[0]["owner"], "test-owner")
        self.assertEqual(result[0]["language"], "Python")

        # Verify attributes
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 1)

    @patch("requests.get")  # Patch the requests.get directly
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

    @patch("requests.get")
    def test_id_sampler_with_validation(self, mock_get):
        """Test IDSampler with validation metrics."""
        # Create sample repository data
        mock_repos = [
            {
                "id": i,
                "name": f"repo{i}",
                "full_name": f"owner{i}/repo{i}",
                "owner": {"login": f"owner{i}"},
                "html_url": f"https://github.com/owner{i}/repo{i}",
                "created_at": f"2023-0{(i % 9) + 1}-01T12:00:00Z",
                "stargazers_count": i * 10,
                "language": ["Python", "JavaScript", "Java"][i % 3],
                "visibility": "public",
            }
            for i in range(1, 6)
        ]

        # Mock responses
        responses = []
        for repo in mock_repos:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = repo
            responses.append(mock_response)

        mock_get.side_effect = responses
        self.sampler._check_rate_limit = MagicMock(return_value=1000)

        # Sample repositories
        results = self.sampler.sample(n_samples=5, max_attempts=5)

        # Basic validation of results
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)

        # Check that each result has required fields
        for repo in results:
            self.assertIn("id", repo)
            self.assertIn("name", repo)
            self.assertIn("full_name", repo)

    def test_default_range_covers_known_ids(self):
        """Test that new default max_id covers known high repository IDs."""
        # Create sampler with default parameters
        sampler = IDSampler(log_level=logging.WARNING)

        # Verify new default covers the repository ID we found in validation
        known_high_id = 800000000  # Found during validation testing
        self.assertGreaterEqual(
            sampler.max_id,
            known_high_id,
            f"Default max_id {sampler.max_id} should cover known repository ID {known_high_id}",
        )

        # Verify the update actually happened
        old_default = 500000000
        self.assertGreater(
            sampler.max_id,
            old_default,
            f"Default max_id {sampler.max_id} should be greater than old default {old_default}",
        )

    def test_filter_during_collection(self):
        """Test that _passes_filters correctly filters repositories."""
        # Test the filter logic directly
        sampler = IDSampler(seed=42)

        # Repo with enough stars should pass
        high_star_repo = {
            "stargazers_count": 150,
            "forks_count": 10,
            "language": "Python",
        }
        self.assertTrue(sampler._passes_filters(high_star_repo, min_stars=100))

        # Repo with too few stars should not pass
        low_star_repo = {
            "stargazers_count": 50,
            "forks_count": 10,
            "language": "Python",
        }
        self.assertFalse(sampler._passes_filters(low_star_repo, min_stars=100))

        # Test language filter
        python_repo = {"stargazers_count": 10, "language": "Python"}
        self.assertTrue(
            sampler._passes_filters(python_repo, languages=["Python", "Java"])
        )

        js_repo = {"stargazers_count": 10, "language": "JavaScript"}
        self.assertFalse(sampler._passes_filters(js_repo, languages=["Python", "Java"]))

        # Test no filter passes everything
        any_repo = {"stargazers_count": 1}
        self.assertTrue(sampler._passes_filters(any_repo))


if __name__ == "__main__":
    unittest.main()
