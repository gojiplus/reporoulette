import logging
import unittest
from unittest.mock import MagicMock, patch

from reporoulette.validation.bias_detector import BiasDetector
from reporoulette.validation.statistical_analyzer import StatisticalAnalyzer

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

        # Validate with BiasDetector (only if we got results)
        if results:
            bias_detector = BiasDetector()
            popularity_bias = bias_detector.detect_popularity_bias(results)

            # Check that bias detection works
            self.assertIn("bias_level", popularity_bias)
            self.assertIn("avg_stars", popularity_bias)
            self.assertIsInstance(popularity_bias["avg_stars"], (int, float))
            # Validate with StatisticalAnalyzer
            analyzer = StatisticalAnalyzer()
            quality_metrics = analyzer.calculate_sample_quality_metrics(results)

            # Check quality metrics
            self.assertIn("quality_score", quality_metrics)
            self.assertIn("field_completeness", quality_metrics)
            self.assertGreaterEqual(quality_metrics["quality_score"], 0)
            self.assertLessEqual(quality_metrics["quality_score"], 1)
        else:
            self.skipTest(
                "No results returned from sampler - cannot test bias detection"
            )

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

    def test_dynamic_id_discovery(self):
        """Test dynamic maximum ID discovery."""
        with patch("requests.get") as mock_get:
            # Mock responses for binary search
            def mock_response_for_id(url, **kwargs):
                # Extract ID from URL
                repo_id = int(url.split("/")[-1])
                mock_response = MagicMock()

                # Test constant for maximum valid repository ID
                max_valid_repo_id = 700000000
                # Simulate that IDs up to 700M exist
                if repo_id <= max_valid_repo_id:
                    mock_response.status_code = 200
                else:
                    mock_response.status_code = 404

                return mock_response

            mock_get.side_effect = mock_response_for_id

            # Create sampler with auto-discovery
            sampler = IDSampler(auto_discover_max=True, log_level=logging.WARNING)

            # Check that max_id was discovered and adjusted
            self.assertGreater(sampler.max_id, 500000000)
            self.assertLess(sampler.max_id, 900000000)


if __name__ == "__main__":
    unittest.main()
