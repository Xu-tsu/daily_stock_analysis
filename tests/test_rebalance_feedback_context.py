# -*- coding: utf-8 -*-
"""Tests for injecting manual trade feedback into rebalance prompts."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rebalance_engine


class TestRebalanceFeedbackContext(unittest.TestCase):
    @patch("rebalance_engine.format_feedback_for_prompt", return_value="- 卖飞样本：主线强势股不宜一刀切清仓")
    def test_build_recent_feedback_prompt_block(self, _mock_feedback) -> None:
        block = rebalance_engine._build_recent_feedback_prompt_block()
        self.assertIn("最近的实盘反馈纠偏", block)
        self.assertIn("卖飞样本", block)
        self.assertIn("纠偏样本", block)

    @patch("rebalance_engine.format_feedback_for_prompt", return_value="")
    def test_build_recent_feedback_prompt_block_returns_empty_when_no_feedback(self, _mock_feedback) -> None:
        self.assertEqual(rebalance_engine._build_recent_feedback_prompt_block(), "")

    def test_apply_candidate_timing_guards_marks_missed_entry(self) -> None:
        candidates = [
            {
                "code": "002361",
                "name": "神剑股份",
                "reason": "主线候选，资金活跃。",
                "buy_price_range": "7.05-7.10",
            }
        ]
        snapshots = [
            {
                "code": "002361",
                "price": 7.77,
                "change_pct": 5.6,
                "main_net": 4200,
                "entry_state": "overextended",
                "missed_entry": True,
                "preferred_buy_range": "7.05-7.16",
                "timing_note": "当前价格已明显脱离原低吸窗口，等待回踩。",
                "relay_role": "板块领涨龙头",
            }
        ]

        guarded = rebalance_engine._apply_candidate_timing_guards(candidates, snapshots)

        self.assertEqual(guarded[0]["buy_price_range"], "7.05-7.16")
        self.assertIn("现阶段不追高", guarded[0]["reason"])
        self.assertEqual(guarded[0]["relay_role"], "板块领涨龙头")


if __name__ == "__main__":
    unittest.main()
