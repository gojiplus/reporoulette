import unittest
import os
import base64
import logging
from reporoulette.samplers.bigquery_sampler import BigQuerySampler


class TestBigQuerySampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up credentials for all tests - handles multiple credential options"""
        # Option 1: Load credentials directly from file if exists
        credentials_path = os.environ.get('GOOGLE_CREDENTIALS_PATH')

        # Option 2: Load encoded credentials from environment variable
        encoded_credentials = os.environ.get('ENCODED_GOOGLE_CREDENTIALS')

        # Option 3: Load JSON credentials from environment variable
        json_credentials = os.environ.get('GOOGLE_CREDENTIALS_JSON')

        # Logic to determine which credentials to use
        if credentials_path and os.path.exists(credentials_path):
            cls.credentials_path = credentials_path
            cls.credentials_method = "file"
        elif encoded_credentials:
            # Decode base64 encoded credentials and save to temp file
            decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8')
            temp_path = "temp_credentials.json"
            with open(temp_path, 'w') as f:
                f.write(decoded_credentials)
            cls.credentials_path = temp_path
            cls.credentials_method = "encoded"
        elif json_credentials:
            # Save JSON credentials to temp file
            temp_path = "temp_credentials.json"
            with open(temp_path, 'w') as f:
                f.write(json_credentials)
            cls.credentials_path = temp_path
            cls.credentials_method = "json"
        else:
            # Try to use application default credentials
            cls.credentials_path = None
            cls.credentials_method = "default"

        cls.project_id = os.environ.get('GOOGLE_PROJECT_ID', 'in-electoral-rolls')

        # Initialize sampler with minimal logging to reduce output noise
        cls.sampler = BigQuerySampler(
            credentials_path=cls.credentials_path,
            project_id=cls.project_id,
            seed=12345,
            log_level=logging.INFO
        )

        print(f"Using credentials method: {cls.credentials_method}")
        print(f"Using project ID: {cls.project_id}")

    @classmethod
    def tearDownClass(cls):
        """Clean up any temporary files"""
        if cls.credentials_method in ["encoded", "json"] and os.path.exists(cls.credentials_path):
            os.remove(cls.credentials_path)

    def test_initialization(self):
        """Test that sampler initializes correctly"""
        self.assertEqual(self.sampler.project_id, self.project_id)
        self.assertIsNotNone(self.sampler._seed)
        self.assertIsNotNone(self.sampler.client)

    def test_sample_by_day_small(self):
        """Test sample_by_day with minimal parameters to reduce costs"""
        # Only sample 2 repos from 1 day going back just 1 year
        repos = self.sampler.sample_by_day(
            n_samples=2,
            days_to_sample=1,
            repos_per_day=2,
            years_back=1
        )

        # Basic validation
        self.assertIsInstance(repos, list)
        # We might get 0-2 repos depending on data
        self.assertLessEqual(len(repos), 2)

        # If we got repos, check their structure
        if repos:
            repo = repos[0]
            self.assertIn('full_name', repo)
            self.assertIn('name', repo)
            self.assertIn('owner', repo)

    def test_sample_active_small(self):
        """Test sample_active with minimal parameters to reduce costs"""
        # Only sample 2 repos from the last month
        from datetime import datetime, timedelta
        one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        repos = self.sampler.sample_active(
            n_samples=2,
            created_after=one_month_ago
        )

        # Basic validation
        self.assertIsInstance(repos, list)
        # We might get 0-2 repos depending on data
        self.assertLessEqual(len(repos), 2)

        # If we got repos, check their structure
        if repos:
            repo = repos[0]
            self.assertIn('full_name', repo)
            self.assertIn('name', repo)
            self.assertIn('owner', repo)

    def test_get_languages_small(self):
        """Test get_languages with some known repos to reduce costs"""
        # Use a few popular repos that are sure to have language data
        test_repos = [
            {'full_name': 'tensorflow/tensorflow'},
            {'full_name': 'pytorch/pytorch'}
        ]

        languages = self.sampler.get_languages(test_repos)

        # Basic validation
        self.assertIsInstance(languages, dict)
        # We should have data for at least one repo
        self.assertGreaterEqual(len(languages), 1)

        # Check structure of language data
        if languages and 'tensorflow/tensorflow' in languages:
            langs = languages['tensorflow/tensorflow']
            self.assertIsInstance(langs, list)
            self.assertGreaterEqual(len(langs), 1)
            self.assertIn('language', langs[0])
            self.assertIn('bytes', langs[0])


if __name__ == '__main__':
    # To run a specific test only (cheaper):
    # unittest.main(argv=['first-arg-is-ignored', 'TestBigQuerySampler.test_initialization'])

    # Or run all tests:
    unittest.main()
