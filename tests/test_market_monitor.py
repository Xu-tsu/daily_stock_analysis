# -*- coding: utf-8 -*-
"""Unit tests for pre-open auction monitoring helpers."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from market_monitor import (
    build_intraday_checkpoint_summary,
    build_intraday_portfolio_advice,
    build_opening_auction_watchlist,
    compose_market_decision_context,
    format_intraday_checkpoint_alert,
    format_opening_auction_alert,
    summarize_opening_auction_state,
)


class TestOpeningAuctionMonitor(unittest.TestCase):
    def test_build_opening_auction_watchlist_merges_candidates_and_holdings(self) -> None:
        candidates = [
            {"code": "000001", "name": "平安银行"},
            {"code": "300750", "name": "宁德时代"},
        ]
        holdings = [
            {"code": "000001", "name": "平安银行"},
            {"code": "600519", "name": "贵州茅台"},
        ]

        with patch("market_monitor._get_current_holdings", return_value=holdings):
            watchlist = build_opening_auction_watchlist(candidates)

        self.assertEqual([item["code"] for item in watchlist], ["000001", "300750", "600519"])
        self.assertEqual(watchlist[0]["reason"], "candidate")
        self.assertEqual(watchlist[-1]["reason"], "holding")

    def test_summarize_opening_auction_state_detects_strong_and_weak_signals(self) -> None:
        state = {
            "watchlist": [
                {"code": "000001", "name": "平安银行", "reason": "candidate"},
                {"code": "600519", "name": "贵州茅台", "reason": "holding"},
            ],
            "samples": [
                {
                    "timestamp": "2026-03-30 09:15:00",
                    "stocks": [
                        {
                            "code": "000001",
                            "name": "平安银行",
                            "reason": "candidate",
                            "auction_price": 11.01,
                            "change_pct": 0.20,
                            "auction_amount": 500.0,
                            "bid_stack": 12000,
                            "ask_stack": 10000,
                            "imbalance_pct": 9.09,
                            "amount_ratio_pct": 0.015,
                        },
                        {
                            "code": "600519",
                            "name": "贵州茅台",
                            "reason": "holding",
                            "auction_price": 1410.0,
                            "change_pct": -0.20,
                            "auction_amount": 700.0,
                            "bid_stack": 600,
                            "ask_stack": 900,
                            "imbalance_pct": -20.0,
                            "amount_ratio_pct": 0.015,
                        },
                    ],
                },
                {
                    "timestamp": "2026-03-30 09:24:30",
                    "stocks": [
                        {
                            "code": "000001",
                            "name": "平安银行",
                            "reason": "candidate",
                            "auction_price": 11.08,
                            "change_pct": 1.28,
                            "auction_amount": 1850.0,
                            "bid_stack": 26000,
                            "ask_stack": 11000,
                            "imbalance_pct": 40.54,
                            "amount_ratio_pct": 0.052,
                        },
                        {
                            "code": "600519",
                            "name": "贵州茅台",
                            "reason": "holding",
                            "auction_price": 1388.0,
                            "change_pct": -0.94,
                            "auction_amount": 1600.0,
                            "bid_stack": 500,
                            "ask_stack": 2100,
                            "imbalance_pct": -61.54,
                            "amount_ratio_pct": 0.041,
                        },
                    ],
                },
            ],
        }

        summary = summarize_opening_auction_state(state)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["watchlist_size"], 2)
        self.assertEqual(summary["strong"][0]["code"], "000001")
        self.assertEqual(summary["weak"][0]["code"], "600519")

        alert = format_opening_auction_alert(summary)
        self.assertIsNotNone(alert)
        self.assertIn("集合竞价资金监控", alert)
        self.assertIn("000001 平安银行", alert)
        self.assertIn("600519 贵州茅台", alert)

    def test_format_opening_auction_alert_returns_none_without_signal(self) -> None:
        summary = {
            "timestamp": "2026-03-30 09:24:30",
            "sample_count": 2,
            "watchlist_size": 1,
            "strong": [],
            "weak": [],
            "stocks": [],
        }

        self.assertIsNone(format_opening_auction_alert(summary))

    def test_build_intraday_checkpoint_summary_detects_positive_afternoon_bias(self) -> None:
        index_quotes = {
            "sh000001": {"change_pct": 0.92},
            "sz399001": {"change_pct": 1.15},
            "sz399006": {"change_pct": 1.48},
        }
        sectors = [
            {"name": "机器人", "change_pct": 2.8, "main_net": 8200},
            {"name": "算力", "change_pct": 2.1, "main_net": 6100},
            {"name": "CPO", "change_pct": 1.4, "main_net": 3900},
        ]
        anomaly_result = {"anomalies": []}

        summary = build_intraday_checkpoint_summary(
            checkpoint="afternoon_review",
            index_quotes=index_quotes,
            sectors=sectors,
            anomaly_result=anomaly_result,
        )
        summary["portfolio_advice"] = {
            "has_holdings": True,
            "position_advice": "测试仓位建议",
            "actions": [
                {
                    "code": "300750",
                    "name": "宁德时代",
                    "action": "buy",
                    "ratio": "加仓约3000元",
                    "change_pct": -1.2,
                    "pnl_pct": 2.4,
                    "strategy_label": "强势回踩加仓",
                    "reason": "测试策略标签展示。",
                }
            ],
            "rotation_candidates": [],
        }

        self.assertEqual(summary["bias"], "positive")
        self.assertEqual(summary["bias_label"], "午后偏强")

        alert = format_intraday_checkpoint_alert(summary)
        self.assertIn("12:30 午后方向判断", alert)
        self.assertIn("午后偏强", alert)
        self.assertIn("机器人", alert)
        self.assertIn("[强势回踩加仓]", alert)

    def test_build_intraday_portfolio_advice_adds_winner_and_blocks_averaging_down(self) -> None:
        summary = {
            "bias": "positive",
            "bias_label": "午后偏强",
            "avg_change_pct": 0.92,
            "high_risk_count": 0,
        }
        portfolio = {
            "cash": 12000,
            "total_asset": 50000,
            "actual_position_ratio": 0.52,
            "holdings": [
                {
                    "code": "300750",
                    "name": "宁德时代",
                    "sector": "机器人",
                    "shares": 20,
                    "sellable_shares": 20,
                    "cost_price": 100.0,
                    "current_price": 102.0,
                    "market_value": 2040.0,
                    "pnl_pct": 2.0,
                },
                {
                    "code": "600519",
                    "name": "贵州茅台",
                    "sector": "白酒",
                    "shares": 100,
                    "sellable_shares": 100,
                    "cost_price": 100.0,
                    "current_price": 109.0,
                    "market_value": 10900.0,
                    "pnl_pct": 9.0,
                },
                {
                    "code": "000001",
                    "name": "平安银行",
                    "sector": "银行",
                    "shares": 100,
                    "sellable_shares": 100,
                    "cost_price": 100.0,
                    "current_price": 96.0,
                    "market_value": 9600.0,
                    "pnl_pct": -4.0,
                },
            ],
        }
        holding_quotes = {
            "300750": {"change_pct": -1.2},
            "600519": {"change_pct": 2.8},
            "000001": {"change_pct": -1.1},
        }
        sectors = [
            {"name": "机器人", "change_pct": 2.5, "main_net": 8200},
            {"name": "算力", "change_pct": 1.8, "main_net": 6100},
            {"name": "CPO", "change_pct": 1.2, "main_net": 4200},
        ]
        rotation_candidates = [
            {"code": "002594", "name": "比亚迪", "price": 230.0, "main_net": 5600},
        ]

        advice = build_intraday_portfolio_advice(
            summary=summary,
            portfolio=portfolio,
            holding_quotes=holding_quotes,
            sectors=sectors,
            risk_alerts=[],
            rotation_candidates=rotation_candidates,
        )

        action_by_code = {item["code"]: item for item in advice["actions"]}
        self.assertEqual(action_by_code["300750"]["action"], "buy")
        self.assertEqual(action_by_code["300750"]["strategy"], "strong_pullback_add")
        self.assertEqual(action_by_code["600519"]["action"], "reduce")
        self.assertEqual(action_by_code["600519"]["strategy"], "weak_to_strong_rotation")
        self.assertEqual(action_by_code["000001"]["action"], "hold")
        self.assertEqual(action_by_code["000001"]["strategy"], "wait_for_base")
        self.assertEqual(advice["rotation_candidates"][0]["code"], "002594")

    def test_build_intraday_portfolio_advice_allows_sector_reflow_average_down(self) -> None:
        summary = {
            "bias": "positive",
            "bias_label": "午后偏强",
            "avg_change_pct": 0.86,
            "high_risk_count": 0,
        }
        portfolio = {
            "cash": 18000,
            "total_asset": 60000,
            "actual_position_ratio": 0.32,
            "holdings": [
                {
                    "code": "002594",
                    "name": "比亚迪",
                    "sector": "机器人",
                    "shares": 12,
                    "sellable_shares": 12,
                    "cost_price": 260.0,
                    "current_price": 251.0,
                    "market_value": 3012.0,
                    "pnl_pct": -3.46,
                },
                {
                    "code": "600519",
                    "name": "贵州茅台",
                    "sector": "白酒",
                    "shares": 50,
                    "sellable_shares": 50,
                    "cost_price": 100.0,
                    "current_price": 109.0,
                    "market_value": 5450.0,
                    "pnl_pct": 9.0,
                },
            ],
        }
        holding_quotes = {
            "002594": {"change_pct": -4.3},
            "600519": {"change_pct": 1.6},
        }
        sectors = [
            {"name": "机器人", "change_pct": 2.2, "main_net": 7600},
            {"name": "算力", "change_pct": 1.9, "main_net": 6200},
            {"name": "CPO", "change_pct": 1.4, "main_net": 4100},
        ]

        advice = build_intraday_portfolio_advice(
            summary=summary,
            portfolio=portfolio,
            holding_quotes=holding_quotes,
            sectors=sectors,
            risk_alerts=[],
            rotation_candidates=[],
        )

        action_by_code = {item["code"]: item for item in advice["actions"]}
        self.assertEqual(action_by_code["002594"]["action"], "buy")
        self.assertEqual(action_by_code["002594"]["strategy"], "sector_reflow_average_down")
        self.assertIn("回流补仓", action_by_code["002594"]["ratio"])

    def test_build_intraday_portfolio_advice_allows_leader_mispricing_average_down(self) -> None:
        summary = {
            "bias": "positive",
            "bias_label": "午后偏强",
            "avg_change_pct": 1.02,
            "high_risk_count": 0,
        }
        portfolio = {
            "cash": 18000,
            "total_asset": 70000,
            "actual_position_ratio": 0.46,
            "holdings": [
                {
                    "code": "300750",
                    "name": "宁德时代",
                    "sector": "机器人",
                    "shares": 40,
                    "sellable_shares": 40,
                    "cost_price": 100.0,
                    "current_price": 98.6,
                    "market_value": 3944.0,
                    "pnl_pct": -1.4,
                }
            ],
        }
        holding_quotes = {
            "300750": {"change_pct": -3.4},
        }
        sectors = [
            {"name": "机器人", "change_pct": 2.7, "main_net": 9200},
            {"name": "算力", "change_pct": 2.1, "main_net": 6500},
            {"name": "CPO", "change_pct": 1.6, "main_net": 4300},
        ]

        advice = build_intraday_portfolio_advice(
            summary=summary,
            portfolio=portfolio,
            holding_quotes=holding_quotes,
            sectors=sectors,
            risk_alerts=[],
            rotation_candidates=[],
        )

        action_by_code = {item["code"]: item for item in advice["actions"]}
        self.assertEqual(action_by_code["300750"]["action"], "buy")
        self.assertEqual(action_by_code["300750"]["strategy"], "leader_mispricing_average_down")
        self.assertIn("错杀补仓", action_by_code["300750"]["ratio"])

    def test_build_intraday_portfolio_advice_uses_defensive_lock_profit_in_neutral_market(self) -> None:
        summary = {
            "bias": "neutral",
            "bias_label": "午后震荡",
            "avg_change_pct": 0.12,
            "high_risk_count": 1,
        }
        portfolio = {
            "cash": 9000,
            "total_asset": 55000,
            "actual_position_ratio": 0.58,
            "holdings": [
                {
                    "code": "600519",
                    "name": "贵州茅台",
                    "sector": "白酒",
                    "shares": 80,
                    "sellable_shares": 80,
                    "cost_price": 100.0,
                    "current_price": 109.0,
                    "market_value": 8720.0,
                    "pnl_pct": 9.0,
                }
            ],
        }
        holding_quotes = {
            "600519": {"change_pct": 1.2},
        }
        sectors = [
            {"name": "机器人", "change_pct": 1.8, "main_net": 6800},
            {"name": "算力", "change_pct": 1.4, "main_net": 5200},
            {"name": "CPO", "change_pct": 1.1, "main_net": 3600},
        ]
        rotation_candidates = [
            {"code": "002594", "name": "比亚迪", "price": 230.0, "main_net": 5600},
        ]

        advice = build_intraday_portfolio_advice(
            summary=summary,
            portfolio=portfolio,
            holding_quotes=holding_quotes,
            sectors=sectors,
            risk_alerts=[],
            rotation_candidates=rotation_candidates,
        )

        action_by_code = {item["code"]: item for item in advice["actions"]}
        self.assertEqual(action_by_code["600519"]["action"], "reduce")
        self.assertEqual(action_by_code["600519"]["strategy"], "defensive_lock_profit")
        self.assertIn("先把利润锁住", action_by_code["600519"]["reason"])


    def test_compose_market_decision_context_merges_macro_flow_and_auction_signals(self) -> None:
        summary = {
            "timestamp": "2026-03-30 10:15:00",
            "bias": "positive",
            "bias_label": "上午偏强",
            "avg_change_pct": 0.96,
            "high_risk_count": 0,
            "anomalies": [],
        }
        sectors = [
            {"name": "Power", "change_pct": 2.3, "main_net": 6800},
            {"name": "AI", "change_pct": 1.9, "main_net": 6200},
        ]
        anomaly_result = {
            "anomalies": [],
            "trump_news": [
                {"title": "Tariff headline", "is_sensitive": True},
                {"title": "Trade-war headline", "is_sensitive": True},
            ],
        }
        hot_candidates = [
            {"code": "600001", "name": "Power A", "change_pct": 4.2, "main_net": 3600},
            {"code": "600002", "name": "Power B", "change_pct": 3.1, "main_net": 2800},
        ]
        auction_summary = {
            "watchlist_size": 1,
            "strong": [{"code": "600011"}],
            "weak": [],
            "stocks": [
                {
                    "code": "600011",
                    "name": "Huanneng",
                    "change_pct": 1.4,
                    "auction_amount": 3200,
                    "imbalance_pct": 26.0,
                    "flow_score": 4.6,
                }
            ],
        }
        financial_news = {"sensitive": [{"title": "Macro policy headline"}]}

        context = compose_market_decision_context(
            summary=summary,
            sectors=sectors,
            anomaly_result=anomaly_result,
            hot_candidates=hot_candidates,
            auction_summary=auction_summary,
            financial_news=financial_news,
            stock_code="600011",
            stock_name="Huanneng",
            stock_sector="Power",
        )

        self.assertEqual(context["macro_risk_level"], "high")
        self.assertEqual(context["hot_money_probe"]["signal"], "active")
        self.assertEqual(context["opening_auction"]["direction"], "strong")
        self.assertTrue(context["sector_confirmation"]["confirmed"])
        self.assertEqual(context["sector_confirmation"]["strength"], "strong")


if __name__ == "__main__":
    unittest.main()
