# -*- coding: utf-8 -*-
"""Tests for manual trade feedback persistence and prompt formatting."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services import trade_feedback_service


class TestTradeFeedbackService(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = os.path.join(self._tmpdir.name, "scanner_history.db")

    def test_record_trade_feedback_uses_recent_sell_context(self) -> None:
        with patch.dict(os.environ, {"SCANNER_DB_PATH": self.db_path}, clear=False):
            with patch.object(
                trade_feedback_service,
                "get_recent_trade_context",
                return_value=[
                    {
                        "id": 11,
                        "trade_date": "2026-03-30",
                        "trade_type": "sell",
                        "code": "002506",
                        "name": "协鑫集成",
                        "shares": 300,
                        "price": 7.61,
                    }
                ],
            ):
                result = trade_feedback_service.record_trade_feedback(
                    raw_text="7.61清仓后立刻被拉升到了7.77",
                    parsed_feedback={
                        "stock_hint": "",
                        "reference_action": "clear",
                        "price": 7.61,
                        "outcome_price": 7.77,
                        "feedback_tag": "sold_too_early",
                        "summary": "",
                    },
                    source="feishu",
                )

            self.assertTrue(result["saved"])
            self.assertEqual(result["code"], "002506")
            self.assertEqual(result["name"], "协鑫集成")
            self.assertEqual(result["feedback_tag"], "sold_too_early")
            self.assertAlmostEqual(result["outcome_delta_pct"], 2.1, places=1)
            self.assertIn("底仓做T", result["guidance"])

            feedback_rows = trade_feedback_service.get_recent_feedback(limit=5)
            self.assertEqual(len(feedback_rows), 1)
            self.assertEqual(feedback_rows[0]["code"], "002506")
            self.assertIn("卖飞", feedback_rows[0]["summary"])

    def test_format_feedback_for_prompt_includes_learning_guidance(self) -> None:
        with patch.dict(os.environ, {"SCANNER_DB_PATH": self.db_path}, clear=False):
            with patch.object(trade_feedback_service, "get_recent_trade_context", return_value=[]):
                trade_feedback_service.record_trade_feedback(
                    raw_text="5.05低吸后下午拉到了5.38",
                    parsed_feedback={
                        "stock_hint": "协鑫集成",
                        "reference_action": "buy",
                        "price": 5.05,
                        "outcome_price": 5.38,
                        "feedback_tag": "dip_buy_success",
                        "summary": "5.05低吸后下午拉到5.38，说明分歧低吸有效",
                    },
                    source="feishu",
                )

            prompt_text = trade_feedback_service.format_feedback_for_prompt(limit=5)

            self.assertIn("人工反馈", prompt_text)
            self.assertIn("协鑫集成", prompt_text)
            self.assertIn("分歧低吸", prompt_text)
            self.assertIn("回踩承接", prompt_text)


if __name__ == "__main__":
    unittest.main()

