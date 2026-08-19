import base64
import logging
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reporoulette.samplers.bigquery_sampler import BigQuerySampler


def make_offline_sampler(seed=42):
    """Construct a BigQuerySampler without credentials or the google-cloud deps."""
    with (
        patch("reporoulette.samplers.bigquery_sampler.BIGQUERY_AVAILABLE", True),
        patch.object(BigQuerySampler, "_init_client"),
    ):
        sampler = BigQuerySampler(seed=seed, log_level=logging.CRITICAL)
    sampler.client = MagicMock()
    return sampler


class FakeQueryJob:
    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class TestBigQuerySamplerQueries(unittest.TestCase):
    """Offline assertions on the generated SQL - no credentials needed."""

    def setUp(self):
        self.sampler = make_offline_sampler(seed=42)

    def all_queries(self, sampler):
        count_q = sampler._build_count_query(days_to_sample=5, years_back=2)
        day_data = {"sample_day": "20240101", "repo_count": 100, "samples_to_take": 10}
        day_q = sampler._build_day_query(day_data, 0)
        combined_q = sampler._combine_day_queries([day_q], n_samples=10)
        return {"count": count_q, "day": day_q, "combined": combined_q}

    def test_no_invalid_rand_anywhere(self):
        # BigQuery's RAND() takes no arguments; RAND(seed) is a runtime SQL
        # error that execute_query silently swallowed into empty results.
        for name, query in self.all_queries(self.sampler).items():
            self.assertNotIn("RAND(", query, f"{name} query still uses RAND(")

    def test_queries_are_seeded(self):
        # The count query is seeded via client-side day selection (covered by
        # the same/different-seed tests); day and combined queries embed the
        # seed in their FARM_FINGERPRINT ordering.
        queries = self.all_queries(self.sampler)
        for name in ("day", "combined"):
            self.assertIn("FARM_FINGERPRINT", queries[name], f"{name} not seeded")
            self.assertIn("42", queries[name], f"{name} does not embed the seed")

    def test_count_query_prunes_wildcard_tables(self):
        # Day literals must be inlined so BigQuery prunes the wildcard scan;
        # computing days in SQL forced a ~0.2 TB scan of every day table.
        query = self.all_queries(self.sampler)["count"]
        self.assertIn("_TABLE_SUFFIX IN (", query)
        self.assertNotIn("GENERATE_ARRAY", query)

    def test_active_and_languages_queries_valid_and_seeded(self):
        for languages in (None, ["Python", "Go"]):
            query = self.sampler._build_active_query(
                "'2024-01-01'", "CURRENT_TIMESTAMP()", languages, 10
            )
            self.assertNotIn("RAND(", query)
            self.assertIn("FARM_FINGERPRINT", query)
            self.assertIn("42", query)
        lang_query = self.sampler._build_languages_query(["owner/repo"])
        self.assertIn("'owner/repo'", lang_query)
        self.assertIn("GROUP BY", lang_query)

    def test_count_query_wildcard_excludes_views(self):
        # Regression: the githubarchive.day dataset contains views
        # (yesterday, today) next to the date tables, and a bare `day.*`
        # wildcard fails with "Views cannot be queried through prefix".
        query = self.all_queries(self.sampler)["count"]
        self.assertIn("`githubarchive.day.2*`", query)
        self.assertNotIn("`githubarchive.day.*`", query)

    def test_day_query_dedups_by_repo_name(self):
        query = self.all_queries(self.sampler)["day"]
        self.assertIn("GROUP BY repo.name", query)

    def test_combined_query_dedups_across_days(self):
        query = self.all_queries(self.sampler)["combined"]
        self.assertIn("GROUP BY full_name", query)

    def test_union_operands_are_parenthesized(self):
        # Regression: day subqueries end in ORDER BY ... LIMIT, and BigQuery
        # rejects bare UNION ALL between such operands ("Expected \")\" but
        # got keyword UNION").
        day_data = {"sample_day": "20240101", "repo_count": 100, "samples_to_take": 5}
        day_queries = [self.sampler._build_day_query(day_data, i) for i in range(2)]
        combined = self.sampler._combine_day_queries(day_queries, n_samples=10)
        self.assertIn(")\nUNION ALL\n(", combined)

    def test_sample_active_queries_seeded_and_valid(self):
        for languages in (None, ["Python", "Go"]):
            sampler = make_offline_sampler(seed=42)
            sampler.client.query.return_value = FakeQueryJob([])
            sampler.sample_active(n_samples=5, languages=languages)
            query = sampler.client.query.call_args[0][0]
            self.assertNotIn("RAND(", query)
            self.assertIn("FARM_FINGERPRINT", query)
            self.assertIn("42", query)

    def test_same_seed_same_sql(self):
        queries_a = self.all_queries(make_offline_sampler(seed=42))
        queries_b = self.all_queries(make_offline_sampler(seed=42))
        self.assertEqual(queries_a, queries_b)

    def test_different_seed_different_sql(self):
        queries_a = self.all_queries(make_offline_sampler(seed=42))
        queries_b = self.all_queries(make_offline_sampler(seed=43))
        self.assertNotEqual(queries_a["count"], queries_b["count"])
        self.assertNotEqual(queries_a["day"], queries_b["day"])


class TestBigQuerySamplerSampleByDay(unittest.TestCase):
    """End-to-end sample_by_day with a mocked client."""

    def run_sample(self, **kwargs):
        sampler = make_offline_sampler(seed=42)
        day_counts = [
            {"sample_day": "20240101", "repo_count": 300},
            {"sample_day": "20240202", "repo_count": 100},
        ]
        repos = [
            {
                "full_name": f"owner{i}/repo{i}",
                "name": f"repo{i}",
                "owner": f"owner{i}",
                "html_url": f"https://github.com/owner{i}/repo{i}",
                "created_at": "2024-01-01T00:00:00Z",
                "sampled_from": "20240101",
                "event_type": "PushEvent",
                "day_repo_count": 300,
                "samples_allocated": 5,
            }
            for i in range(5)
        ]
        sampler.client.query.side_effect = [
            FakeQueryJob(day_counts),
            FakeQueryJob(repos),
        ]
        results = sampler.sample_by_day(
            n_samples=5, days_to_sample=2, repos_per_day=5, **kwargs
        )
        return sampler, results

    def test_returns_mocked_repos(self):
        sampler, results = self.run_sample()
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["full_name"], "owner0/repo0")
        self.assertEqual(sampler.attempts, 2)
        self.assertEqual(sampler.success_count, 2)

    def test_filter_kwargs_do_not_wipe_results(self):
        # Regression: filter_repos used to match kwargs against nonexistent
        # fields ('min_stars') and return [] for any filtered BigQuery sample.
        _, results = self.run_sample(min_stars=10)
        self.assertEqual(len(results), 5)

    def test_owner_filter_applies(self):
        _, results = self.run_sample(owner="owner3")
        self.assertEqual([r["full_name"] for r in results], ["owner3/repo3"])


@pytest.mark.integration
class TestBigQuerySamplerLive(unittest.TestCase):
    """Live BigQuery tests - require credentials, run with -m integration."""

    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls):
        """Set up credentials for all tests - handles multiple credential options."""
        # Option 1: Load credentials directly from file if exists
        credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH")

        # Option 2: Load encoded credentials from environment variable
        encoded_credentials = os.environ.get("ENCODED_GOOGLE_CREDENTIALS")

        # Option 3: Load JSON credentials from environment variable
        json_credentials = os.environ.get("GOOGLE_CREDENTIALS_JSON")

        # Logic to determine which credentials to use
        if credentials_path and Path(credentials_path).exists():
            cls.credentials_path = credentials_path
            cls.credentials_method = "file"
        elif encoded_credentials:
            # Decode base64 encoded credentials and save to temp file
            decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
            temp_path = "temp_credentials.json"
            Path(temp_path).write_text(decoded_credentials)
            cls.credentials_path = temp_path
            cls.credentials_method = "encoded"
        elif json_credentials:
            # Save JSON credentials to temp file
            temp_path = "temp_credentials.json"
            Path(temp_path).write_text(json_credentials)
            cls.credentials_path = temp_path
            cls.credentials_method = "json"
        else:
            # Try to use application default credentials
            cls.credentials_path = None
            cls.credentials_method = "default"

        cls.project_id = os.environ.get("GOOGLE_PROJECT_ID", "in-electoral-rolls")

        try:
            # Initialize sampler with minimal logging to reduce output noise
            cls.sampler = BigQuerySampler(
                credentials_path=cls.credentials_path,
                project_id=cls.project_id,
                seed=12345,
                log_level=logging.INFO,
            )
            cls.credentials_available = True
            cls.logger.info("Using credentials method: %s", cls.credentials_method)
            cls.logger.info("Using project ID: %s", cls.project_id)
        except Exception as e:
            cls.credentials_available = False
            cls.sampler = None
            cls.logger.warning("BigQuery credentials not available: %s", e)
            cls.logger.warning("BigQuery tests will be skipped")

    @classmethod
    def tearDownClass(cls):
        """Clean up any temporary files."""
        if (
            hasattr(cls, "credentials_method")
            and cls.credentials_method in ["encoded", "json"]
            and hasattr(cls, "credentials_path")
            and cls.credentials_path
            and Path(cls.credentials_path).exists()
        ):
            Path(cls.credentials_path).unlink()

    def test_initialization(self):
        """Test that sampler initializes correctly."""
        if not self.credentials_available:
            self.skipTest("BigQuery credentials not available")

        self.assertEqual(self.sampler.project_id, self.project_id)
        self.assertIsNotNone(self.sampler._seed)
        self.assertIsNotNone(self.sampler.client)

    def test_sample_by_day_small(self):
        """Test sample_by_day with minimal parameters to reduce costs."""
        if not self.credentials_available:
            self.skipTest("BigQuery credentials not available")

        # Only sample 2 repos from 1 day going back just 1 year
        repos = self.sampler.sample_by_day(
            n_samples=2, days_to_sample=1, repos_per_day=2, years_back=1
        )

        # Basic validation
        self.assertIsInstance(repos, list)
        # We might get 0-2 repos depending on data
        self.assertLessEqual(len(repos), 2)

        # If we got repos, check their structure
        if repos:
            repo = repos[0]
            self.assertIn("full_name", repo)
            self.assertIn("name", repo)
            self.assertIn("owner", repo)

    def test_sample_active_small(self):
        """Test sample_active with minimal parameters to reduce costs."""
        if not self.credentials_available:
            self.skipTest("BigQuery credentials not available")

        # Only sample 2 repos from the last month
        from datetime import UTC, datetime, timedelta

        one_month_ago = (datetime.now(UTC) - timedelta(days=30)).strftime("%Y-%m-%d")

        repos = self.sampler.sample_active(n_samples=2, created_after=one_month_ago)

        # Basic validation
        self.assertIsInstance(repos, list)
        # We might get 0-2 repos depending on data
        self.assertLessEqual(len(repos), 2)

        # If we got repos, check their structure
        if repos:
            repo = repos[0]
            self.assertIn("full_name", repo)
            self.assertIn("name", repo)
            self.assertIn("owner", repo)

    def test_get_languages_small(self):
        """Test get_languages with some known repos to reduce costs."""
        if not self.credentials_available:
            self.skipTest("BigQuery credentials not available")

        # Use a few popular repos that are sure to have language data
        test_repos = [
            {"full_name": "tensorflow/tensorflow"},
            {"full_name": "pytorch/pytorch"},
        ]

        languages = self.sampler.get_languages(test_repos)

        # Basic validation
        self.assertIsInstance(languages, dict)

        # If authentication fails, we get empty dict - that's acceptable
        # for CI environments with credential issues
        if len(languages) == 0:
            self.skipTest(
                "No language data returned - likely authentication issue in CI"
            )

        # If we get data, validate its structure
        self.assertGreaterEqual(len(languages), 1)

        # Check structure of language data if available
        if languages and "tensorflow/tensorflow" in languages:
            langs = languages["tensorflow/tensorflow"]
            self.assertIsInstance(langs, list)
            self.assertGreaterEqual(len(langs), 1)
            self.assertIn("language", langs[0])
            self.assertIn("bytes", langs[0])


if __name__ == "__main__":
    unittest.main()
