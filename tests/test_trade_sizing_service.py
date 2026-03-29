# -*- coding: utf-8 -*-
"""Tests for A-share lot sizing helpers."""

import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.trade_sizing_service import (
    TradeCostProfile,
    estimate_a_share_trade,
    load_cn_trade_cost_profile,
    max_affordable_a_share_shares,
    round_down_to_lot,
)


class TestTradeSizingService(unittest.TestCase):
    def test_round_down_to_lot(self) -> None:
        self.assertEqual(round_down_to_lot(550), 500)
        self.assertEqual(round_down_to_lot(99), 0)

    def test_max_affordable_shares_respects_fee_and_lot(self) -> None:
        profile = TradeCostProfile()
        self.assertEqual(
            max_affordable_a_share_shares(cash_budget=5000, price=10.0, profile=profile),
            400,
        )

    def test_estimate_sell_trade_includes_stamp_tax(self) -> None:
        estimate = estimate_a_share_trade(
            shares=500,
            price=10.0,
            side="sell",
            profile=TradeCostProfile(),
        )

        self.assertEqual(estimate["gross_amount"], 5000.0)
        self.assertEqual(estimate["fee"], 5.0)
        self.assertEqual(estimate["tax"], 5.0)
        self.assertEqual(estimate["net_cash"], 4990.0)

    def test_load_trade_cost_profile_from_portfolio_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "portfolio.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE portfolio_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    quantity REAL,
                    price REAL,
                    fee REAL,
                    tax REAL,
                    side TEXT,
                    market TEXT
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO portfolio_trades(quantity, price, fee, tax, side, market)
                VALUES (?, ?, ?, ?, ?, 'cn')
                """,
                [
                    (100, 10.0, 5.0, 0.0, "buy"),
                    (10000, 10.0, 31.0, 100.0, "sell"),
                ],
            )
            conn.commit()
            conn.close()

            profile = load_cn_trade_cost_profile(db_paths=[db_path])

        self.assertAlmostEqual(profile.min_fee, 5.0, places=6)
        self.assertAlmostEqual(profile.fee_rate, 0.00031, places=6)
        self.assertAlmostEqual(profile.sell_tax_rate, 0.001, places=6)
        self.assertEqual(profile.source, "portfolio_trades")
        self.assertEqual(profile.sample_size, 2)


if __name__ == "__main__":
    unittest.main()
