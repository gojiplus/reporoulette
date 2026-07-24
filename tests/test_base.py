import logging
import unittest
from unittest.mock import MagicMock, patch

from reporoulette.samplers.id_sampler import IDSampler
from reporoulette.samplers.temporal_sampler import TemporalSampler


def rate_limited_get(core_remaining, search_remaining):
    """Mock requests.get routing /rate_limit and repo URLs."""

    def fake_get(url, *args, **kwargs):
        response = MagicMock()
        if url.endswith("/rate_limit"):
            response.status_code = 200
            response.json.return_value = {
                "resources": {
                    "core": {"remaining": core_remaining, "reset": 0},
                    "search": {"remaining": search_remaining, "reset": 0},
                }
            }
        else:
            response.status_code = 200
            response.json.return_value = {"ok": True}
        return response

    return fake_get


class TestRateLimitBuckets(unittest.TestCase):
    """The per-request pre-check must consult the bucket the sampler uses."""

    @patch("time.sleep")
    @patch("requests.get")
    def test_temporal_aborts_on_low_search_bucket(self, mock_get, _sleep):
        # Regression: the pre-check used to consult "core", so a temporal
        # sampler could blow through the 30/min search bucket.
        mock_get.side_effect = rate_limited_get(core_remaining=5000, search_remaining=0)
        sampler = TemporalSampler(seed=1, log_level=logging.CRITICAL)
        self.assertEqual(sampler.rate_limit_resource, "search")

        response = sampler._make_github_request("https://api.github.com/x", min_wait=0)
        self.assertIsNone(response)

    @patch("time.sleep")
    @patch("requests.get")
    def test_id_sampler_unaffected_by_search_bucket(self, mock_get, _sleep):
        mock_get.side_effect = rate_limited_get(core_remaining=5000, search_remaining=0)
        sampler = IDSampler(seed=1, log_level=logging.CRITICAL)
        self.assertEqual(sampler.rate_limit_resource, "core")

        response = sampler._make_github_request("https://api.github.com/x", min_wait=0)
        self.assertIsNotNone(response)

    @patch("time.sleep")
    @patch("requests.get")
    def test_failed_rate_limit_check_does_not_block_request(self, mock_get, _sleep):
        # Regression: a transient /rate_limit failure returned 0, which was
        # always <= the safety margin and silently vetoed the real request.
        def fake_get(url, *args, **kwargs):
            response = MagicMock()
            if url.endswith("/rate_limit"):
                response.status_code = 500
            else:
                response.status_code = 200
                response.json.return_value = {"ok": True}
            return response

        mock_get.side_effect = fake_get
        sampler = IDSampler(seed=1, log_level=logging.CRITICAL)
        self.assertIsNone(sampler._check_rate_limit())

        response = sampler._make_github_request("https://api.github.com/x", min_wait=0)
        self.assertIsNotNone(response)


if __name__ == "__main__":
    unittest.main()
