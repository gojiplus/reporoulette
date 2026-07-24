import re
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from reporoulette.samplers.temporal_sampler import TemporalSampler


def search_response(total_count, items):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"total_count": total_count, "items": items}
    return response


def make_repo(name, owner="test-owner", language="Python"):
    return {
        "id": hash(name) % 10**8,
        "name": name,
        "full_name": f"{owner}/{name}",
        "owner": {"login": owner},
        "html_url": f"https://github.com/{owner}/{name}",
        "description": "Test repository",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-02T12:00:00Z",
        "pushed_at": "2023-01-03T12:00:00Z",
        "stargazers_count": 10,
        "forks_count": 5,
        "language": language,
        "visibility": "public",
    }


class TestTemporalSampler(unittest.TestCase):
    def setUp(self):
        # Create a real instance with date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        self.sampler = TemporalSampler(
            seed=42, start_date=start_date, end_date=end_date
        )

        # Mock logger
        self.sampler.logger = MagicMock()

    @patch("requests.get")
    def test_temporal_sampler_basic(self, mock_get):
        # Mock response for successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_count": 1,
            "items": [
                {
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
            ],
        }
        mock_get.return_value = mock_response

        # Mock the rate limit check to always return a high number
        self.sampler._check_rate_limit = MagicMock(return_value=1000)

        # Call the sample method
        result = self.sampler.sample(n_samples=1, days_to_sample=1)

        # Verify result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "test-repo")
        self.assertEqual(result[0]["owner"], "test-owner")
        self.assertEqual(result[0]["language"], "Python")

        # Verify attributes - with the page 1 bias fix, we need 2 requests:
        # 1 for initial day assessment, 1 for weighted sampling
        self.assertEqual(self.sampler.attempts, 2)
        self.assertEqual(self.sampler.success_count, 2)

    @patch("requests.get")
    def test_temporal_sampler_empty_results(self, mock_get):
        # Mock the rate limit check to always return a high number
        self.sampler._check_rate_limit = MagicMock(return_value=1000)

        # Mock a request with no results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"total_count": 0, "items": []}
        mock_get.return_value = mock_response

        # Call the sample method
        result = self.sampler.sample(n_samples=1, days_to_sample=1)

        # Verify empty result
        self.assertEqual(len(result), 0)

        # Verify attributes
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 0)

    @patch("requests.get")
    def test_temporal_sampler_with_filters(self, mock_get):
        # Mock response for successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_count": 1,
            "items": [
                {
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
                    "visibility": "public",
                }
            ],
        }
        mock_get.return_value = mock_response

        # Mock the rate limit check to always return a high number
        self.sampler._check_rate_limit = MagicMock(return_value=1000)

        # Call the sample method with filters
        result = self.sampler.sample(
            n_samples=1, days_to_sample=1, min_stars=10, languages=["Python"]
        )

        # Verify result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["stargazers_count"], 20)
        self.assertEqual(result[0]["language"], "Python")

    @patch("time.sleep")
    @patch("requests.get")
    def test_terminates_when_all_days_capped(self, mock_get, _mock_sleep):
        # Regression: the collection loop had no iteration bound, and the
        # per-day cap `continue` skipped the attempt counter that gated the
        # rate-limit escape, so this scenario used to loop forever.
        mock_get.return_value = search_response(2, [make_repo("only-repo")])
        self.sampler._check_rate_limit = MagicMock(return_value=1000)

        result = self.sampler.sample(
            n_samples=10, days_to_sample=1, min_wait=0, max_attempts=20
        )

        self.assertEqual(len(result), 1)
        self.sampler.logger.warning.assert_any_call(
            "Stopped after max_attempts=20 iterations with 1/10 repositories collected"
        )

    def test_default_rate_limit_safety_below_search_ceiling(self):
        # Regression: the search bucket allows only 30 requests/minute; a
        # default safety of 100 made `remaining <= safety` always true, so
        # the sampler could never issue a single live search request.
        sampler = TemporalSampler(seed=1)
        self.assertLess(sampler.rate_limit_safety, 30)

    def test_query_single_language_uses_qualifier(self):
        query = self.sampler._build_search_query(
            "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", languages=["Python"]
        )
        self.assertIn("language:Python", query)

    def test_query_multiple_languages_omits_qualifier(self):
        # Regression: multiple languages used to silently query only the
        # first one; multiple language: qualifiers AND together and match
        # nothing, so the qualifier is dropped and filtering is client-side.
        query = self.sampler._build_search_query(
            "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", languages=["Python", "Go"]
        )
        self.assertNotIn("language:", query)

    def test_query_star_and_size_qualifiers(self):
        query = self.sampler._build_search_query(
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            min_stars=10,
            min_size_kb=50,
        )
        self.assertIn("stars:>=10", query)
        self.assertIn("size:>=50", query)

    @patch("time.sleep")
    @patch("requests.get")
    def test_seed_reproducibility(self, mock_get, _mock_sleep):
        # Deterministic mock keyed off the requested page: same seed must
        # produce the same day/page draws and therefore identical samples.
        def fake_get(url, *args, **kwargs):
            match = re.search(r"page=(\d+)", url)
            page = int(match.group(1)) if match else 1
            items = [make_repo(f"repo-p{page}-{i}") for i in range(3)]
            return search_response(500, items)

        mock_get.side_effect = fake_get

        def run_once():
            # Samplers seed the global random module, so each run constructs
            # its sampler immediately before sampling.
            sampler = TemporalSampler(
                seed=42, start_date="2024-01-01", end_date="2024-06-30"
            )
            sampler.logger = MagicMock()
            sampler._check_rate_limit = MagicMock(return_value=1000)
            return sampler.sample(
                n_samples=6, days_to_sample=2, min_wait=0, max_attempts=10
            )

        first = [repo["full_name"] for repo in run_once()]
        second = [repo["full_name"] for repo in run_once()]

        self.assertGreater(len(first), 0)
        self.assertEqual(first, second)

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        self.sampler.attempts = 10
        self.sampler.success_count = 3
        self.assertEqual(self.sampler.success_rate, 30.0)

        # Test zero attempts
        self.sampler.attempts = 0
        self.sampler.success_count = 0
        self.assertEqual(self.sampler.success_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
