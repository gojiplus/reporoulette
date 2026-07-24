import gzip
import io
import json
import re
import unittest
from unittest.mock import MagicMock, patch

from reporoulette.samplers.gh_sampler import GHArchiveSampler


def make_event(repo_name, event_type="CreateEvent", ref_type="repository"):
    event = {
        "type": event_type,
        "repo": {"name": repo_name, "url": f"https://github.com/{repo_name}"},
        "created_at": "2023-01-01T12:00:00Z",
    }
    if event_type == "CreateEvent":
        event["payload"] = {"ref_type": ref_type}
    return event


def archive_response(events):
    """Build a mock archive download with a fresh gzip stream."""
    gz_content = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_content, mode="w") as f:
        for event in events:
            f.write((json.dumps(event) + "\n").encode("utf-8"))
    gz_content.seek(0)

    response = MagicMock()
    response.status_code = 200
    response.raw = gz_content
    return response


class TestGHArchiveSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = GHArchiveSampler(seed=42)
        self.sampler.logger = MagicMock()

    @patch("reporoulette.samplers.gh_sampler.requests.get")
    def test_gh_sampler_basic(self, mock_get):
        events = [
            make_event("owner1/repo1", "PushEvent"),
            make_event("owner2/repo2", "CreateEvent"),
            make_event("owner3/repo3", "PullRequestEvent"),
            make_event("owner4/repo4", "IssuesEvent"),
        ]
        mock_get.side_effect = lambda *a, **kw: archive_response(events)

        result = self.sampler.gh_sampler(
            n_samples=3,
            days_to_sample=1,
            repos_per_day=3,
            years_back=1,
            event_types=["PushEvent", "CreateEvent", "PullRequestEvent"],
        )

        # IssuesEvent is not in event_types, so only 3 repos qualify
        self.assertEqual(len(result), 3)
        self.assertEqual(
            sorted(r["full_name"] for r in result),
            ["owner1/repo1", "owner2/repo2", "owner3/repo3"],
        )
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 1)
        self.assertEqual(self.sampler.results, result)

    @patch("reporoulette.samplers.gh_sampler.requests.get")
    def test_default_create_event_requires_repository_ref(self, mock_get):
        events = [
            make_event("owner1/created", "CreateEvent", ref_type="repository"),
            make_event("owner2/branch-only", "CreateEvent", ref_type="branch"),
            make_event("owner3/pushed", "PushEvent"),
        ]
        mock_get.side_effect = lambda *a, **kw: archive_response(events)

        result = self.sampler.gh_sampler(
            n_samples=5, days_to_sample=1, repos_per_day=5, years_back=1
        )

        # Default event_types=["CreateEvent"] keeps only repository creations
        self.assertEqual([r["full_name"] for r in result], ["owner1/created"])

    @patch("reporoulette.samplers.gh_sampler.requests.get")
    def test_sample_method_delegates_to_gh_sampler(self, mock_get):
        events = [
            make_event("owner1/repo1"),
            make_event("owner2/repo2"),
        ]
        mock_get.side_effect = lambda *a, **kw: archive_response(events)

        result = self.sampler.sample(
            n_samples=1, days_to_sample=1, repos_per_day=2, years_back=1
        )

        self.assertEqual(len(result), 1)
        self.assertIn(result[0]["full_name"], ["owner1/repo1", "owner2/repo2"])

    @patch("reporoulette.samplers.gh_sampler.requests.get")
    def test_early_break_attempt_accounting(self, mock_get):
        # Regression: attempts used to be set to the planned days_to_sample
        # even when the target was reached after the first day.
        events = [make_event(f"owner{i}/repo{i}") for i in range(5)]
        mock_get.side_effect = lambda *a, **kw: archive_response(events)

        result = self.sampler.gh_sampler(
            n_samples=3, days_to_sample=3, repos_per_day=5, years_back=1
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 1)

    @patch("reporoulette.samplers.gh_sampler.requests.get")
    def test_uses_hourly_archive_urls(self, mock_get):
        # Regression: GH Archive only publishes hourly files
        # ({date}-{hour}.json.gz); the daily URL the sampler previously used
        # does not exist, so every real download 404ed.
        urls = []

        def record(url, *args, **kwargs):
            urls.append(url)
            return archive_response([make_event("owner1/repo1")])

        mock_get.side_effect = record

        self.sampler.gh_sampler(
            n_samples=1, days_to_sample=1, repos_per_day=1, years_back=1
        )

        self.assertEqual(len(urls), 24)
        hours = [int(re.search(r"-(\d+)\.json\.gz$", u).group(1)) for u in urls]
        self.assertEqual(hours, list(range(24)))

    @patch("reporoulette.samplers.gh_sampler.requests.get")
    def test_hours_per_day_limits_downloads(self, mock_get):
        urls = []

        def record(url, *args, **kwargs):
            urls.append(url)
            return archive_response([make_event("owner1/repo1")])

        mock_get.side_effect = record

        self.sampler.gh_sampler(
            n_samples=1,
            days_to_sample=1,
            repos_per_day=1,
            years_back=1,
            hours_per_day=3,
        )

        self.assertEqual(len(urls), 3)
        hours = [int(re.search(r"-(\d+)\.json\.gz$", u).group(1)) for u in urls]
        self.assertEqual(len(set(hours)), 3)
        self.assertTrue(all(0 <= h <= 23 for h in hours))

    @patch("reporoulette.samplers.gh_sampler.requests.get")
    def test_failed_download_counts_attempt_not_success(self, mock_get):
        # First day: all 24 hourly downloads 404; second day succeeds.
        events = [make_event(f"owner{i}/repo{i}") for i in range(3)]
        calls = {"n": 0}

        def flaky(url, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 24:
                not_found = MagicMock()
                not_found.status_code = 404
                return not_found
            return archive_response(events)

        mock_get.side_effect = flaky

        result = self.sampler.gh_sampler(
            n_samples=10, days_to_sample=2, repos_per_day=5, years_back=1
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(self.sampler.attempts, 2)
        self.assertEqual(self.sampler.success_count, 1)

    @patch("reporoulette.samplers.gh_sampler.requests.get")
    def test_gh_sampler_error_handling(self, mock_get):
        mock_get.side_effect = Exception("Mock network error")

        result = self.sampler.gh_sampler(
            n_samples=2,
            days_to_sample=1,
            repos_per_day=2,
            years_back=1,
        )

        self.assertEqual(len(result), 0)
        self.assertEqual(self.sampler.attempts, 1)
        self.assertEqual(self.sampler.success_count, 0)
        self.assertEqual(self.sampler.results, [])


if __name__ == "__main__":
    unittest.main()
