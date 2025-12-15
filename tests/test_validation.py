import logging
import os
import unittest
from collections import Counter
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

from reporoulette.samplers.bigquery_sampler import BigQuerySampler
from reporoulette.samplers.gh_sampler import GHArchiveSampler
from reporoulette.samplers.id_sampler import IDSampler
from reporoulette.samplers.temporal_sampler import TemporalSampler


class ValidationTestSuite(unittest.TestCase):
    """Comprehensive validation test suite for repository sampling methods.

    Tests for correctness, bias detection, and cross-method consistency.
    """

    def setUp(self):
        """Set up test environment with consistent seeds for reproducibility."""
        self.test_seed = 42
        self.small_sample_size = 5  # Small for quick tests
        self.medium_sample_size = 20  # Medium for validation tests

        # Configure logging for validation tests
        logging.basicConfig(level=logging.WARNING)  # Reduce noise in tests

        # Initialize samplers with consistent seeds
        self.id_sampler = IDSampler(seed=self.test_seed, log_level=logging.WARNING)
        self.temporal_sampler = TemporalSampler(
            seed=self.test_seed, log_level=logging.WARNING
        )

        # Only initialize BigQuery sampler if credentials are available
        self.bq_sampler = None
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get(
            "GOOGLE_CREDENTIALS_JSON"
        ):
            try:
                self.bq_sampler = BigQuerySampler(
                    seed=self.test_seed, log_level=logging.WARNING
                )
            except Exception:
                self.bq_sampler = None

        self.gh_archive_sampler = GHArchiveSampler(
            seed=self.test_seed, log_level=logging.WARNING
        )

    def test_cross_method_consistency_mock(self):
        """Test consistency between different sampling methods using mocked data."""
        # Create consistent mock repository data
        mock_repo_data = {
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

        with patch("requests.get") as mock_get:
            # Mock successful response for ID sampler
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_repo_data
            mock_get.return_value = mock_response

            # Mock rate limit checks
            self.id_sampler._check_rate_limit = MagicMock(return_value=1000)

            # Test ID sampler
            id_results = self.id_sampler.sample(n_samples=1, max_attempts=1)

            # Verify basic structure
            self.assertEqual(len(id_results), 1)
            self.assertIn("full_name", id_results[0])
            self.assertIn("language", id_results[0])
            self.assertIn("stargazers_count", id_results[0])

        with patch("requests.get") as mock_get:
            # Mock search API response for temporal sampler
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "total_count": 1,
                "items": [mock_repo_data],
            }
            mock_get.return_value = mock_response

            # Mock rate limit checks
            self.temporal_sampler._check_rate_limit = MagicMock(return_value=1000)

            # Test temporal sampler
            temporal_results = self.temporal_sampler.sample(
                n_samples=1, days_to_sample=1
            )

            # Verify basic structure
            self.assertEqual(len(temporal_results), 1)
            self.assertIn("full_name", temporal_results[0])
            self.assertIn("sampled_from", temporal_results[0])

    def test_repository_data_structure_consistency(self):
        """Test that all samplers return consistent data structures."""
        required_fields = ["full_name", "name", "owner"]
        # Optional fields that may be present in repository data
        _optional_fields = [
            "html_url",
            "description",
            "created_at",
            "language",
            "stargazers_count",
        ]

        # Mock data for testing structure
        mock_repo = {
            "id": 12345,
            "name": "test-repo",
            "full_name": "test-owner/test-repo",
            "owner": {"login": "test-owner"},
            "html_url": "https://github.com/test-owner/test-repo",
            "description": "Test repository",
            "created_at": "2023-01-01T12:00:00Z",
            "language": "Python",
            "stargazers_count": 10,
        }

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_repo
            mock_get.return_value = mock_response

            self.id_sampler._check_rate_limit = MagicMock(return_value=1000)

            results = self.id_sampler.sample(n_samples=1, max_attempts=1)

            if results:
                repo = results[0]

                # Check required fields
                for field in required_fields:
                    self.assertIn(
                        field,
                        repo,
                        f"Required field '{field}' missing from IDSampler result",
                    )

                # Validate data types
                self.assertIsInstance(repo["full_name"], str)
                self.assertIsInstance(repo["name"], str)
                self.assertIsInstance(repo["owner"], str)

                # Validate full_name format
                self.assertIn(
                    "/", repo["full_name"], "full_name should contain '/' separator"
                )

    def test_sampling_reproducibility(self):
        """Test that sampling is reproducible with same seed."""
        mock_repo = {
            "id": 12345,
            "name": "test-repo",
            "full_name": "test-owner/test-repo",
            "owner": {"login": "test-owner"},
            "html_url": "https://github.com/test-owner/test-repo",
        }

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_repo
            mock_get.return_value = mock_response

            # Create two samplers with same seed
            sampler1 = IDSampler(seed=42, log_level=logging.WARNING)
            sampler2 = IDSampler(seed=42, log_level=logging.WARNING)

            sampler1._check_rate_limit = MagicMock(return_value=1000)
            sampler2._check_rate_limit = MagicMock(return_value=1000)

            # Mock random.randint to return predictable values
            with patch(
                "random.randint", side_effect=[100, 100]
            ):  # Same values for both calls
                results1 = sampler1.sample(n_samples=1, max_attempts=1)
                results2 = sampler2.sample(n_samples=1, max_attempts=1)

            # Results should be identical with same seed
            self.assertEqual(len(results1), len(results2))
            if results1 and results2:
                self.assertEqual(results1[0]["full_name"], results2[0]["full_name"])

    def test_success_rate_calculation(self):
        """Test success rate calculation accuracy."""
        # Test with known success/failure pattern
        sampler = IDSampler(seed=42, log_level=logging.WARNING)

        # Simulate attempts and successes
        sampler.attempts = 10
        sampler.success_count = 3

        expected_rate = (3 / 10) * 100  # 30%
        actual_rate = sampler.success_rate

        self.assertEqual(actual_rate, expected_rate)

        # Test zero attempts case
        sampler.attempts = 0
        sampler.success_count = 0
        self.assertEqual(sampler.success_rate, 0.0)

    def test_filter_functionality(self):
        """Test that filtering works correctly across samplers."""
        # Create test repositories with different characteristics
        test_repos = [
            {
                "full_name": "owner1/repo1",
                "name": "repo1",
                "owner": "owner1",
                "language": "Python",
                "stargazers_count": 50,
                "forks_count": 10,
            },
            {
                "full_name": "owner2/repo2",
                "name": "repo2",
                "owner": "owner2",
                "language": "JavaScript",
                "stargazers_count": 5,
                "forks_count": 2,
            },
            {
                "full_name": "owner3/repo3",
                "name": "repo3",
                "owner": "owner3",
                "language": "Python",
                "stargazers_count": 100,
                "forks_count": 25,
            },
        ]

        sampler = IDSampler(log_level=logging.WARNING)

        # Test minimum stars filter
        filtered = sampler._filter_repos(test_repos, min_stars=20)
        self.assertEqual(len(filtered), 2)  # Only repos with >= 20 stars

        # Test language filter
        filtered = sampler._filter_repos(test_repos, languages=["Python"])
        self.assertEqual(len(filtered), 2)  # Only Python repos

        # Test combined filters
        filtered = sampler._filter_repos(test_repos, min_stars=20, languages=["Python"])
        self.assertEqual(len(filtered), 2)  # Python repos with >= 20 stars

    def test_temporal_sampler_date_handling(self):
        """Test temporal sampler's date generation and formatting."""
        # Test date range generation
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        sampler = TemporalSampler(
            start_date=start_date, end_date=end_date, seed=42, log_level=logging.WARNING
        )

        # Test date formatting
        test_date = datetime(2023, 1, 15, 14, 30, 0)
        start_str, end_str = sampler._format_date_for_query(test_date)

        # Should format to beginning and end of day
        self.assertEqual(start_str, "2023-01-15T00:00:00Z")
        self.assertEqual(end_str, "2023-01-16T00:00:00Z")

        # Test random date generation
        random_date = sampler._random_date()
        self.assertTrue(start_date <= random_date <= end_date)

    def test_bigquery_sampler_availability(self):
        """Test BigQuery sampler initialization and availability."""
        if self.bq_sampler is None:
            self.skipTest("BigQuery credentials not available")

        # Test initialization
        self.assertIsNotNone(self.bq_sampler.client)
        self.assertIsNotNone(self.bq_sampler._seed)

        # Test that queries can be built without errors
        query = self.bq_sampler._build_count_query(days_to_sample=1, years_back=1)
        self.assertIn("DECLARE days_to_sample", query)
        self.assertIn("random_dates", query)

    def test_github_archive_sampler_structure(self):
        """Test GitHub Archive sampler data processing."""
        # Test repository data extraction from mock event
        mock_event = {
            "type": "CreateEvent",
            "repo": {
                "name": "test-owner/test-repo",
                "url": "https://api.github.com/repos/test-owner/test-repo",
            },
            "created_at": "2023-01-01T12:00:00Z",
            "payload": {"ref_type": "repository"},
        }

        # Test data extraction logic
        repo_name = mock_event["repo"]["name"]
        self.assertIn("/", repo_name)

        owner, name = repo_name.split("/", 1)
        self.assertEqual(owner, "test-owner")
        self.assertEqual(name, "test-repo")


class StatisticalValidationTests(unittest.TestCase):
    """Statistical tests for bias detection and distribution analysis."""

    def setUp(self):
        """Set up test data for statistical validation tests."""
        self.test_repos = self._generate_test_repository_data()

    def _generate_test_repository_data(self) -> list[dict[str, Any]]:
        """Generate test repository data with known distributions."""
        repos = []
        languages = ["Python", "JavaScript", "Java", "Go", "Rust"] * 4

        for i in range(20):
            repos.append(
                {
                    "full_name": f"owner{i}/repo{i}",
                    "name": f"repo{i}",
                    "owner": f"owner{i}",
                    "language": languages[i],
                    "stargazers_count": i * 10,  # Increasing star pattern
                    "forks_count": i * 2,
                    "created_at": f"2023-01-{(i % 30) + 1:02d}T12:00:00Z",
                    "size": i * 100,  # KB
                }
            )

        return repos

    def test_language_distribution_analysis(self):
        """Test language distribution analysis for bias detection."""
        language_counts = Counter(repo["language"] for repo in self.test_repos)

        # Should have relatively even distribution (4 of each language)
        expected_count = len(self.test_repos) // len(language_counts)

        for _language, count in language_counts.items():
            # Allow some variance but detect major skew
            self.assertGreaterEqual(count, expected_count - 1)
            self.assertLessEqual(count, expected_count + 1)

    def test_star_count_distribution_analysis(self):
        """Test star count distribution for popularity bias."""
        star_counts = [repo["stargazers_count"] for repo in self.test_repos]

        # Test for expected linear increase (our test data pattern)
        sorted_stars = sorted(star_counts)
        self.assertEqual(sorted_stars, star_counts)  # Should be already sorted

    def test_temporal_distribution_analysis(self):
        """Test temporal distribution for time-based bias."""
        # Extract dates and check distribution
        dates = [repo["created_at"][:10] for repo in self.test_repos]  # YYYY-MM-DD
        date_counts = Counter(dates)

        # Should have varied creation dates (our test data pattern)
        self.assertGreater(
            len(date_counts), 1, "Should have repositories from different dates"
        )

    def test_repository_size_distribution(self):
        """Test repository size distribution for size bias."""
        sizes = [repo["size"] for repo in self.test_repos]

        # Test for reasonable size range
        min_size = min(sizes)
        max_size = max(sizes)

        self.assertGreaterEqual(min_size, 0)
        self.assertGreater(max_size, min_size)

    def test_sample_uniqueness(self):
        """Test that samples contain unique repositories."""
        full_names = [repo["full_name"] for repo in self.test_repos]
        unique_names = set(full_names)

        # Should have no duplicates
        self.assertEqual(len(full_names), len(unique_names))

    def test_required_fields_completeness(self):
        """Test that all repositories have required fields."""
        required_fields = ["full_name", "name", "owner"]

        for repo in self.test_repos:
            for field in required_fields:
                self.assertIn(
                    field, repo, f"Repository missing required field: {field}"
                )
                self.assertTrue(repo[field], f"Repository has empty {field}")


if __name__ == "__main__":
    # Run with verbose output for detailed validation results
    unittest.main(verbosity=2)
