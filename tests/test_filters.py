import unittest

from reporoulette.samplers.bq_utils import filter_repos


def api_repo(name, stars=0, forks=0, language=None, owner="octocat"):
    return {
        "full_name": f"{owner}/{name}",
        "name": name,
        "stargazers_count": stars,
        "forks_count": forks,
        "language": language,
        "owner": {"login": owner},
    }


def bq_repo(name, owner="octocat"):
    return {
        "full_name": f"{owner}/{name}",
        "name": name,
        "owner": owner,
        "html_url": f"https://github.com/{owner}/{name}",
        "created_at": "2024-01-01T00:00:00Z",
    }


class TestFilterRepos(unittest.TestCase):
    def test_min_stars_threshold(self):
        repos = [
            api_repo("low", stars=5),
            api_repo("mid", stars=20),
            api_repo("high", stars=100),
        ]
        result = filter_repos(repos, min_stars=20)
        self.assertEqual([r["name"] for r in result], ["mid", "high"])

    def test_min_forks_threshold(self):
        repos = [api_repo("a", forks=1), api_repo("b", forks=10)]
        result = filter_repos(repos, min_forks=5)
        self.assertEqual([r["name"] for r in result], ["b"])

    def test_languages_case_insensitive(self):
        repos = [
            api_repo("py", language="Python"),
            api_repo("go", language="Go"),
            api_repo("none", language=None),
        ]
        result = filter_repos(repos, languages=["python"])
        self.assertEqual([r["name"] for r in result], ["py"])

    def test_single_language_shorthand(self):
        repos = [api_repo("py", language="Python"), api_repo("go", language="Go")]
        result = filter_repos(repos, language="GO")
        self.assertEqual([r["name"] for r in result], ["go"])

    def test_owner_filter_api_and_bq_shapes(self):
        repos = [
            api_repo("a", owner="alice"),
            bq_repo("b", owner="bob"),
        ]
        self.assertEqual([r["name"] for r in filter_repos(repos, owner="alice")], ["a"])
        self.assertEqual([r["name"] for r in filter_repos(repos, owner="bob")], ["b"])

    def test_missing_field_skips_filter_instead_of_wiping_results(self):
        # BigQuery result dicts carry no star data; before the fix,
        # min_stars=5 matched a nonexistent field and returned [].
        repos = [bq_repo("a"), bq_repo("b")]
        with self.assertLogs(level="WARNING") as captured:
            result = filter_repos(repos, min_stars=5)
        self.assertEqual(len(result), 2)
        self.assertTrue(any("min_stars" in msg for msg in captured.output))

    def test_missing_language_field_skips_filter(self):
        repos = [bq_repo("a"), bq_repo("b")]
        with self.assertLogs(level="WARNING") as captured:
            result = filter_repos(repos, languages=["Python"])
        self.assertEqual(len(result), 2)
        self.assertTrue(any("languages" in msg for msg in captured.output))

    def test_unknown_filter_ignored_with_warning(self):
        repos = [api_repo("a", stars=50)]
        with self.assertLogs(level="WARNING") as captured:
            result = filter_repos(repos, min_starz=10)
        self.assertEqual(len(result), 1)
        self.assertTrue(any("min_starz" in msg for msg in captured.output))

    def test_combined_filters(self):
        repos = [
            api_repo("a", stars=100, language="Python"),
            api_repo("b", stars=100, language="Go"),
            api_repo("c", stars=1, language="Python"),
        ]
        result = filter_repos(repos, min_stars=50, languages=["Python"])
        self.assertEqual([r["name"] for r in result], ["a"])


if __name__ == "__main__":
    unittest.main()
