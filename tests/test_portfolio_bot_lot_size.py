# -*- coding: utf-8 -*-
"""Tests for A-share lot size validation in portfolio bot commands."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from portfolio_bot import handle_portfolio_command


class TestPortfolioBotLotSize(unittest.TestCase):
    def test_buy_rejects_non_round_lot(self) -> None:
        result = handle_portfolio_command("买入 600519 50股 10元")
        self.assertIn("100股整数倍", result)

    def test_sell_rejects_non_round_lot(self) -> None:
        result = handle_portfolio_command("卖出 600519 40股 10元")
        self.assertIn("100股整数倍", result)


if __name__ == "__main__":
    unittest.main()
