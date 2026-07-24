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
        """Test that the default max_id covers known high repository IDs."""
        sampler = IDSampler(log_level=logging.WARNING)

        # Live validation (2026-07) observed repository IDs around 1.31B;
        # the previous 850M default excluded ~35% of the ID space.
        known_high_id = 1_290_000_000
        self.assertGreaterEqual(
            sampler.max_id,
            known_high_id,
            f"Default max_id {sampler.max_id} should cover known repository ID {known_high_id}",
        )

    @patch("reporoulette.samplers.id_sampler.requests.get")
    def test_update_max_id(self, mock_get):
        """update_max_id sets max_id to the highest observed repo ID."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "items": [{"id": 1_400_000_001}, {"id": 1_399_999_000}]
        }
        mock_get.return_value = response

        sampler = IDSampler(log_level=logging.WARNING)
        result = sampler.update_max_id()

        self.assertEqual(result, 1_400_000_001)
        self.assertEqual(sampler.max_id, 1_400_000_001)

    @patch("reporoulette.samplers.id_sampler.requests.get")
    def test_update_max_id_keeps_current_on_failure(self, mock_get):
        mock_get.side_effect = Exception("network down")

        sampler = IDSampler(log_level=logging.WARNING)
        before = sampler.max_id
        result = sampler.update_max_id()

        self.assertEqual(result, before)
        self.assertEqual(sampler.max_id, before)

    @patch("time.sleep")
    @patch("requests.get")
    def test_probing_uniformity_smoke(self, mock_get, _mock_sleep):
        """Statistical smoke test: probed IDs are uniform over the ID space.

        Mock GitHub so exactly 10% of IDs exist (id % 10 == 0), draw 1000
        probes with a fixed seed, and check the measured success rate plus a
        chi-square uniformity test over 10 deciles. Fixed seed makes this
        fully deterministic.
        """
        sampler = IDSampler(
            seed=7, min_id=1, max_id=1_000_000, log_level=logging.WARNING
        )
        sampler.logger = MagicMock()
        sampler._check_rate_limit = MagicMock(return_value=100000)

        probed = []

        def fake_get(url, *args, **kwargs):
            repo_id = int(url.rstrip("/").rsplit("/", 1)[1])
            probed.append(repo_id)
            response = MagicMock()
            if repo_id % 10 == 0:
                response.status_code = 200
                response.json.return_value = {
                    "id": repo_id,
                    "name": f"repo{repo_id}",
                    "full_name": f"owner{repo_id}/repo{repo_id}",
                    "owner": {"login": f"owner{repo_id}"},
                    "html_url": f"https://github.com/owner{repo_id}/repo{repo_id}",
                    "created_at": "2023-01-01T12:00:00Z",
                    "updated_at": "2023-01-02T12:00:00Z",
                }
            else:
                response.status_code = 404
                response.text = "Not Found"
            return response

        mock_get.side_effect = fake_get

        sampler.sample(n_samples=1000, max_attempts=1000, min_wait=0)

        self.assertEqual(sampler.attempts, 1000)
        self.assertGreaterEqual(sampler.success_rate, 7.0)
        self.assertLessEqual(sampler.success_rate, 13.0)

        # Chi-square goodness-of-fit over 10 equal-width deciles;
        # critical value for df=9 at alpha=0.05 is 16.92 (no scipy needed).
        buckets = [0] * 10
        span = (sampler.max_id - sampler.min_id + 1) / 10
        for repo_id in probed:
            buckets[min(int((repo_id - sampler.min_id) / span), 9)] += 1
        expected = len(probed) / 10
        chi_sq = sum((obs - expected) ** 2 / expected for obs in buckets)
        self.assertLess(
            chi_sq,
            16.92,
            f"Probed IDs deviate from uniform: chi-square={chi_sq:.2f}, "
            f"buckets={buckets}",
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
