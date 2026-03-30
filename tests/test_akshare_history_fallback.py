# -*- coding: utf-8 -*-
import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.base import DataFetchError


class TestAkshareHistoryFallback(unittest.TestCase):
    @patch("data_provider.akshare_fetcher._should_use_sina_history_fallback", return_value=False)
    def test_sina_history_branch_is_disabled_on_windows_guard(self, _mock_guard):
        fetcher = AkshareFetcher(sleep_min=0, sleep_max=0)

        with self.assertRaises(DataFetchError) as ctx:
            fetcher._fetch_stock_data_sina("600000", "2025-01-01", "2025-03-30")

        self.assertIn("disabled on Windows", str(ctx.exception))

    def test_history_fetch_falls_through_to_tencent_after_previous_failures(self):
        fetcher = AkshareFetcher(sleep_min=0, sleep_max=0)
        calls = []
        expected = pd.DataFrame(
            [{"日期": "2025-03-28", "开盘": 10.0, "收盘": 10.2, "最高": 10.3, "最低": 9.9, "成交量": 1000, "成交额": 10000}]
        )

        def fail_em(*_args, **_kwargs):
            calls.append("em")
            raise DataFetchError("em failed")

        def fail_sina(*_args, **_kwargs):
            calls.append("sina")
            raise DataFetchError("sina failed")

        def succeed_tx(*_args, **_kwargs):
            calls.append("tx")
            return expected

        with (
            patch.object(fetcher, "_fetch_stock_data_em", side_effect=fail_em),
            patch.object(fetcher, "_fetch_stock_data_sina", side_effect=fail_sina),
            patch.object(fetcher, "_fetch_stock_data_tx", side_effect=succeed_tx),
        ):
            result = fetcher._fetch_stock_data("600000", "2025-01-01", "2025-03-30")

        self.assertEqual(calls, ["em", "sina", "tx"])
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
