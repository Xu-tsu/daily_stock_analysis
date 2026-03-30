# -*- coding: utf-8 -*-
"""Tests for SearXNG fallback handling."""

import time
import unittest
from unittest.mock import Mock, patch

import requests

import macro_data_collector as mdc


class SearxngFallbackTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cache = dict(mdc._SEARXNG_CACHE)
        self._old_disabled_until = mdc._SEARXNG_DISABLED_UNTIL
        self._old_disabled_reason = mdc._SEARXNG_DISABLED_REASON

        with mdc._SEARXNG_CACHE_LOCK:
            mdc._SEARXNG_CACHE.clear()
        mdc._SEARXNG_DISABLED_UNTIL = 0.0
        mdc._SEARXNG_DISABLED_REASON = ""

    def tearDown(self) -> None:
        with mdc._SEARXNG_CACHE_LOCK:
            mdc._SEARXNG_CACHE.clear()
            mdc._SEARXNG_CACHE.update(self._old_cache)
        mdc._SEARXNG_DISABLED_UNTIL = self._old_disabled_until
        mdc._SEARXNG_DISABLED_REASON = self._old_disabled_reason

    def test_searxng_connection_failure_enters_cooldown(self) -> None:
        with patch.object(
            mdc.requests,
            "get",
            side_effect=requests.ConnectionError("connection refused"),
        ) as mock_get:
            first = mdc._searxng_search("A股 今日 行情 资金", max_results=3)
            second = mdc._searxng_search("中国股市 最新消息", max_results=3)

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(mock_get.call_count, 1)
        self.assertGreater(mdc._get_searxng_cooldown_remaining(time.time()), 0.0)

    def test_successful_response_clears_cooldown_and_caches_results(self) -> None:
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "results": [
                {
                    "title": "test-title",
                    "url": "https://example.com",
                    "content": "test-content",
                    "engine": "stub",
                }
            ]
        }
        mdc._SEARXNG_DISABLED_UNTIL = time.time() - 1
        mdc._SEARXNG_DISABLED_REASON = "old failure"

        with patch.object(mdc.requests, "get", return_value=response) as mock_get:
            first = mdc._searxng_search("A股 今日 行情 资金", max_results=1)
            second = mdc._searxng_search("A股 今日 行情 资金", max_results=1)

        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(mdc._get_searxng_cooldown_remaining(time.time()), 0.0)


if __name__ == "__main__":
    unittest.main()
