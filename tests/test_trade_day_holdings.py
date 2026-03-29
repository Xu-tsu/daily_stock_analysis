# -*- coding: utf-8 -*-
"""Regression tests for trading-day based hold_days persistence."""

import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pdf_parser
import trade_journal


class TestTradeDayHoldings(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_db_path = trade_journal.DB_PATH
        trade_journal.DB_PATH = os.path.join(self._tmpdir.name, "scanner_history.db")
        trade_journal.init_trade_tables()

    def tearDown(self) -> None:
        trade_journal.DB_PATH = self._old_db_path
        self._tmpdir.cleanup()

    def _fetch_sell_hold_days(self) -> int:
        conn = sqlite3.connect(trade_journal.DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT hold_days FROM trade_log WHERE trade_type = 'sell' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        return int(row["hold_days"])

    def _fetch_trade_costs(self) -> tuple[float, float]:
        conn = sqlite3.connect(trade_journal.DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT fee, tax FROM trade_log WHERE trade_type = 'sell' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        return float(row["fee"]), float(row["tax"])

    @patch("trade_journal.save_market_context")
    @patch("trade_journal._get_current_technical", return_value={})
    @patch("trade_journal.count_stock_trading_days", return_value=1)
    def test_record_sell_uses_trading_days(
        self,
        _mock_count,
        _mock_technical,
        _mock_save_context,
    ) -> None:
        trade_journal.record_buy("600011", "Huanneng", 100, 10.0)
        trade_journal.record_sell("600011", "Huanneng", 100, 10.2)

        self.assertEqual(self._fetch_sell_hold_days(), 1)

    @patch("pdf_parser.count_stock_trading_days", return_value=1)
    @patch(
        "pdf_parser.parse_pdf",
        return_value={
            "doc_type": "delivery",
            "page_count": 1,
            "trades": [
                {
                    "date": "2026-03-27",
                    "code": "600011",
                    "name": "Huanneng",
                    "direction": "buy",
                    "shares": 100,
                    "price": 10.0,
                    "amount": 1000.0,
                    "fee": 2.0,
                    "transfer_fee": 0.2,
                    "stamp_tax": 0.0,
                },
                {
                    "date": "2026-03-30",
                    "code": "600011",
                    "name": "Huanneng",
                    "direction": "sell",
                    "shares": 100,
                    "price": 10.2,
                    "amount": 1020.0,
                    "fee": 2.1,
                    "transfer_fee": 0.1,
                    "stamp_tax": 1.02,
                },
            ],
            "chunks": [],
        },
    )
    def test_pdf_import_uses_trading_days(
        self,
        _mock_parse_pdf,
        _mock_count,
    ) -> None:
        result = pdf_parser.import_pdf_to_trade_journal("dummy.pdf")

        self.assertEqual(result["imported"], 2)
        self.assertEqual(self._fetch_sell_hold_days(), 1)
        fee, tax = self._fetch_trade_costs()
        self.assertAlmostEqual(fee, 2.2, places=6)
        self.assertAlmostEqual(tax, 1.02, places=6)


if __name__ == "__main__":
    unittest.main()
