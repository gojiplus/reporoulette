import logging
import unittest

from reporoulette.samplers.id_sampler import IDSampler

FIELD_ALIASES = {
    "stars": "stargazers_count",
    "forks": "forks_count",
    "created": "created_at",
}


def api_repo(name, owner="octocat", **fields):
    repo = {
        "full_name": f"{owner}/{name}",
        "name": name,
        "stargazers_count": 0,
        "forks_count": 0,
        "language": None,
        "owner": {"login": owner},
        "created_at": "2024-01-01T00:00:00Z",
    }
    for key, value in fields.items():
        repo[FIELD_ALIASES.get(key, key)] = value
    return repo


def bq_repo(name, owner="octocat"):
    return {
        "full_name": f"{owner}/{name}",
        "name": name,
        "owner": owner,
        "html_url": f"https://github.com/{owner}/{name}",
        "created_at": "2024-01-01T00:00:00Z",
    }


class TestSharedFilterRepos(unittest.TestCase):
    """The unified BaseSampler._filter_repos, shared by all samplers."""

    def setUp(self):
        self.sampler = IDSampler(log_level=logging.CRITICAL)

    def filter(self, repos, **kwargs):
        return self.sampler._filter_repos(repos, **kwargs)

    def test_min_stars_threshold(self):
        repos = [
            api_repo("low", stars=5),
            api_repo("mid", stars=20),
            api_repo("high", stars=100),
        ]
        result = self.filter(repos, min_stars=20)
        self.assertEqual([r["name"] for r in result], ["mid", "high"])

    def test_min_forks_threshold(self):
        repos = [api_repo("a", forks=1), api_repo("b", forks=10)]
        result = self.filter(repos, min_forks=5)
        self.assertEqual([r["name"] for r in result], ["b"])

    def test_languages_case_insensitive(self):
        repos = [
            api_repo("py", language="Python"),
            api_repo("go", language="Go"),
            api_repo("none", language=None),
        ]
        result = self.filter(repos, languages=["python"])
        self.assertEqual([r["name"] for r in result], ["py"])

    def test_single_language_shorthand(self):
        repos = [api_repo("py", language="Python"), api_repo("go", language="Go")]
        result = self.filter(repos, language="GO")
        self.assertEqual([r["name"] for r in result], ["go"])

    def test_owner_filter_api_and_flat_shapes(self):
        repos = [
            api_repo("a", owner="alice"),
            bq_repo("b", owner="bob"),
        ]
        self.assertEqual([r["name"] for r in self.filter(repos, owner="alice")], ["a"])
        self.assertEqual([r["name"] for r in self.filter(repos, owner="bob")], ["b"])

    def test_missing_field_skips_filter_instead_of_wiping_results(self):
        # BigQuery/GH Archive result dicts carry no star data; before the
        # audit fixes, min_stars matched nothing and returned [].
        repos = [bq_repo("a"), bq_repo("b")]
        with self.assertLogs("IDSampler", level="WARNING") as captured:
            result = self.filter(repos, min_stars=5)
        self.assertEqual(len(result), 2)
        self.assertTrue(any("min_stars" in msg for msg in captured.output))

    def test_missing_language_field_skips_filter(self):
        repos = [bq_repo("a"), bq_repo("b")]
        with self.assertLogs("IDSampler", level="WARNING") as captured:
            result = self.filter(repos, languages=["Python"])
        self.assertEqual(len(result), 2)
        self.assertTrue(any("languages" in msg for msg in captured.output))

    def test_unknown_filter_ignored_with_warning(self):
        repos = [api_repo("a", stars=50)]
        with self.assertLogs("IDSampler", level="WARNING") as captured:
            result = self.filter(repos, min_starz=10)
        self.assertEqual(len(result), 1)
        self.assertTrue(any("min_starz" in msg for msg in captured.output))

    def test_created_after_and_before(self):
        repos = [
            api_repo("old", created="2020-06-01T00:00:00Z"),
            api_repo("new", created="2024-06-01T00:00:00Z"),
        ]
        self.assertEqual(
            [r["name"] for r in self.filter(repos, created_after="2022-01-01")],
            ["new"],
        )
        self.assertEqual(
            [r["name"] for r in self.filter(repos, created_before="2022-01-01")],
            ["old"],
        )

    def test_max_repos_seeded_subsample(self):
        repos = [api_repo(f"r{i}") for i in range(20)]
        sampler_a = IDSampler(seed=42, log_level=logging.CRITICAL)
        sampler_b = IDSampler(seed=42, log_level=logging.CRITICAL)
        picked_a = [r["name"] for r in sampler_a._filter_repos(repos, max_repos=5)]
        picked_b = [r["name"] for r in sampler_b._filter_repos(repos, max_repos=5)]
        self.assertEqual(len(picked_a), 5)
        self.assertEqual(picked_a, picked_b)

    def test_combined_filters(self):
        repos = [
            api_repo("a", stars=100, language="Python"),
            api_repo("b", stars=100, language="Go"),
            api_repo("c", stars=1, language="Python"),
        ]
        result = self.filter(repos, min_stars=50, languages=["Python"])
        self.assertEqual([r["name"] for r in result], ["a"])


if __name__ == "__main__":
    unittest.main()
