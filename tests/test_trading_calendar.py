# -*- coding: utf-8 -*-
"""Tests for trading-day counting helpers."""

import os
import sys
import unittest
from datetime import date
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.trading_calendar import (
    count_stock_trading_days,
    count_trading_days,
    is_market_open,
)


class TestTradingDayCount(unittest.TestCase):
    @patch("src.core.trading_calendar._XCALS_AVAILABLE", False)
    def test_weekday_fallback_still_excludes_weekends(self) -> None:
        self.assertTrue(is_market_open("cn", date(2026, 3, 30)))
        self.assertFalse(is_market_open("cn", date(2026, 3, 29)))

    @patch(
        "src.core.trading_calendar.is_market_open",
        side_effect=lambda _market, check_date: check_date.weekday() < 5,
    )
    def test_weekends_are_excluded_from_holding_period(self, _mock_market_open) -> None:
        self.assertEqual(count_trading_days("2026-03-27", "2026-03-27"), 0)
        self.assertEqual(count_trading_days("2026-03-27", "2026-03-30"), 1)
        self.assertEqual(count_trading_days("2026-03-27", "2026-03-31"), 2)

    @patch(
        "src.core.trading_calendar.is_market_open",
        side_effect=lambda _market, check_date: check_date.weekday() < 5,
    )
    def test_stock_count_defaults_to_a_share_market(self, _mock_market_open) -> None:
        self.assertEqual(
            count_stock_trading_days("600011", "2026-03-27", "2026-03-30"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
