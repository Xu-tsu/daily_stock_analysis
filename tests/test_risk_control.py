# -*- coding: utf-8 -*-
"""Tests for adaptive risk-control guardrails."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from risk_control import check_stop_loss


def _market_context(*, confirmed: bool, bias: str = "positive", macro: str = "low", quant: str = "low", auction: str = "strong"):
    return {
        "bias": bias,
        "market_score": 0.86 if bias == "positive" else -0.72,
        "macro_risk_level": macro,
        "top_sectors": [
            {"name": "Power", "rank": 1, "change_pct": 2.1, "main_net": 6800},
        ],
        "sector_confirmation": {
            "sector": "Power",
            "confirmed": confirmed,
            "strength": "strong" if confirmed else "weak",
            "rank": 1 if confirmed else 6,
            "change_pct": 2.1 if confirmed else -0.3,
            "main_net": 6800 if confirmed else -800,
        },
        "hot_money_probe": {"signal": "active" if confirmed else "quiet"},
        "quant_pressure": {"signal": quant},
        "opening_auction": {"direction": auction},
    }


class TestAdaptiveStopLoss(unittest.TestCase):
    def test_loss_review_line_can_keep_base_when_market_and_sector_confirm(self) -> None:
        holdings = [
            {
                "code": "600011",
                "name": "Huanneng",
                "sector": "Power",
                "cost_price": 10.0,
                "current_price": 9.45,
                "buy_date": "2026-03-29",
            }
        ]

        alerts = check_stop_loss(
            holdings,
            market_context=_market_context(confirmed=True),
        )

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].level, "warning")
        self.assertEqual(alerts[0].action, "review")
        self.assertIn("保留底仓做T", alerts[0].message)

    def test_loss_review_line_exits_when_sector_confirmation_is_missing(self) -> None:
        holdings = [
            {
                "code": "600011",
                "name": "Huanneng",
                "sector": "Power",
                "cost_price": 10.0,
                "current_price": 9.4,
                "buy_date": "2026-03-29",
            }
        ]

        alerts = check_stop_loss(
            holdings,
            market_context=_market_context(confirmed=False, macro="medium", auction="weak"),
        )

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].action, "force_sell")
        self.assertIn("板块", alerts[0].message)

    def test_deep_loss_still_forces_exit_even_in_strong_market(self) -> None:
        holdings = [
            {
                "code": "600011",
                "name": "Huanneng",
                "sector": "Power",
                "cost_price": 10.0,
                "current_price": 9.1,
                "buy_date": "2026-03-29",
            }
        ]

        alerts = check_stop_loss(
            holdings,
            market_context=_market_context(confirmed=True),
        )

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].level, "critical")
        self.assertEqual(alerts[0].action, "force_sell")
        self.assertIn("强制退出线", alerts[0].message)


if __name__ == "__main__":
    unittest.main()
