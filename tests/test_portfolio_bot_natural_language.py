# -*- coding: utf-8 -*-
"""Tests for natural-language portfolio bot commands."""

import os
import sys
import unittest
from copy import deepcopy
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import portfolio_bot
from portfolio_bot import handle_portfolio_command, is_portfolio_command


class TestPortfolioBotNaturalLanguage(unittest.TestCase):
    def setUp(self) -> None:
        portfolio_bot._clear_portfolio_llm_cache()

    def test_is_portfolio_command_accepts_name_lot_order(self) -> None:
        self.assertTrue(is_portfolio_command("协鑫集成加5.05加仓3手"))

    def test_is_portfolio_command_does_not_intercept_plain_buy_question(self) -> None:
        self.assertFalse(is_portfolio_command("有没有假如早盘低开买入有可能涨停的龙头股"))

    def test_feedback_text_is_not_misclassified_as_new_clear_order(self) -> None:
        self.assertIsNone(portfolio_bot._detect_trade_action("7.61清仓后立刻被拉升到了7.77"))

    @patch("portfolio_bot.interpret_portfolio_command")
    @patch("portfolio_bot.load_portfolio", return_value={"holdings": [{"code": "002506", "name": "协鑫集成", "shares": 700}]})
    def test_is_portfolio_command_uses_llm_for_free_form_command(
        self,
        _mock_load_portfolio,
        mock_interpret,
    ) -> None:
        mock_interpret.return_value = {
            "is_portfolio_command": True,
            "action": "buy",
            "stock_hint": "协鑫集成",
            "shares": 300,
            "price": 5.05,
            "needs_clarification": False,
            "confidence": 0.98,
        }

        self.assertTrue(is_portfolio_command("帮我把协鑫集成再补三手，挂5块05"))
        mock_interpret.assert_called_once()

    @patch("trade_journal.record_buy")
    @patch("portfolio_bot.save_portfolio")
    @patch("portfolio_bot.load_portfolio")
    @patch("portfolio_bot._get_stock_name", return_value="协鑫集成")
    @patch("portfolio_bot.resolve_name_to_code", return_value=None)
    @patch("portfolio_bot._should_try_llm_portfolio_interpretation", return_value=False)
    def test_handle_buy_by_name_and_lot_updates_holding(
        self,
        _mock_should_try_llm,
        mock_resolve_name,
        mock_get_stock_name,
        mock_load_portfolio,
        mock_save_portfolio,
        mock_record_buy,
    ) -> None:
        mock_load_portfolio.return_value = {
            "cash": 10000.0,
            "holdings": [
                {
                    "code": "002506",
                    "name": "协鑫集成",
                    "shares": 100,
                    "cost_price": 5.0,
                    "current_price": 5.0,
                    "market_value": 500.0,
                    "sector": "光伏",
                    "buy_date": "2026-03-27",
                    "strategy_tag": "短线",
                }
            ],
        }

        result = handle_portfolio_command("协鑫集成加5.05加仓3手")

        self.assertIn("加仓成功", result)
        self.assertIn("300股", result)
        saved_portfolio = mock_save_portfolio.call_args.args[0]
        holding = saved_portfolio["holdings"][0]
        self.assertEqual(holding["shares"], 400)
        self.assertEqual(holding["cost_price"], 5.037)
        self.assertEqual(saved_portfolio["cash"], 8485.0)
        mock_resolve_name.assert_not_called()
        mock_get_stock_name.assert_called_once_with("002506")
        mock_record_buy.assert_called_once()

    @patch("portfolio_bot._should_try_llm_portfolio_interpretation", return_value=False)
    @patch("portfolio_bot._handle_show_portfolio", return_value="mocked portfolio")
    def test_handle_show_portfolio_alias(self, mock_show, _mock_should_try_llm) -> None:
        result = handle_portfolio_command("持仓查看")
        self.assertEqual(result, "mocked portfolio")
        mock_show.assert_called_once()

    @patch("trade_journal.record_sell")
    @patch("portfolio_bot.save_portfolio")
    @patch("portfolio_bot.load_portfolio")
    @patch("portfolio_bot.resolve_name_to_code", return_value=None)
    @patch("portfolio_bot._should_try_llm_portfolio_interpretation", return_value=False)
    def test_handle_sell_by_name_and_lot_updates_holding(
        self,
        _mock_should_try_llm,
        mock_resolve_name,
        mock_load_portfolio,
        mock_save_portfolio,
        mock_record_sell,
    ) -> None:
        mock_load_portfolio.return_value = {
            "cash": 2000.0,
            "holdings": [
                {
                    "code": "002506",
                    "name": "协鑫集成",
                    "shares": 500,
                    "cost_price": 5.0,
                    "current_price": 5.0,
                    "market_value": 2500.0,
                    "sector": "光伏",
                    "buy_date": "2026-03-27",
                    "strategy_tag": "短线",
                }
            ],
        }

        result = handle_portfolio_command("减仓 协鑫集成 1手 5.30")

        self.assertIn("卖出成功", result)
        saved_portfolio = mock_save_portfolio.call_args.args[0]
        holding = saved_portfolio["holdings"][0]
        self.assertEqual(holding["shares"], 400)
        self.assertEqual(saved_portfolio["cash"], 2530.0)
        mock_resolve_name.assert_not_called()
        mock_record_sell.assert_called_once()

    @patch("trade_journal.record_buy")
    @patch("portfolio_bot.save_portfolio")
    @patch("portfolio_bot.load_portfolio")
    @patch("portfolio_bot._get_stock_name", return_value="协鑫集成")
    @patch("portfolio_bot.resolve_name_to_code", return_value=None)
    @patch("portfolio_bot._should_try_llm_portfolio_interpretation", return_value=False)
    def test_handle_buy_by_pinyin_like_name_uses_existing_holding(
        self,
        _mock_should_try_llm,
        mock_resolve_name,
        mock_get_stock_name,
        mock_load_portfolio,
        mock_save_portfolio,
        mock_record_buy,
    ) -> None:
        mock_load_portfolio.return_value = {
            "cash": 10000.0,
            "holdings": [
                {
                    "code": "002506",
                    "name": "协鑫集成",
                    "shares": 700,
                    "cost_price": 5.38,
                    "current_price": 5.02,
                    "market_value": 3514.0,
                    "sector": "光伏",
                    "buy_date": "2026-03-27",
                    "strategy_tag": "短线",
                }
            ],
        }

        result = handle_portfolio_command("写信继承加5.05加仓3手")

        self.assertIn("加仓成功", result)
        saved_portfolio = mock_save_portfolio.call_args.args[0]
        holding = saved_portfolio["holdings"][0]
        self.assertEqual(holding["code"], "002506")
        self.assertEqual(holding["shares"], 1000)
        mock_resolve_name.assert_not_called()
        mock_get_stock_name.assert_called_once_with("002506")
        mock_record_buy.assert_called_once()

    @patch("trade_journal.record_buy")
    @patch("portfolio_bot.save_portfolio")
    @patch("portfolio_bot._get_stock_name", return_value="协鑫集成")
    @patch("portfolio_bot.load_portfolio")
    @patch("portfolio_bot.interpret_portfolio_command")
    def test_handle_buy_via_llm_interpreter(
        self,
        mock_interpret,
        mock_load_portfolio,
        mock_get_stock_name,
        mock_save_portfolio,
        mock_record_buy,
    ) -> None:
        mock_load_portfolio.return_value = {
            "cash": 10000.0,
            "holdings": [
                {
                    "code": "002506",
                    "name": "协鑫集成",
                    "shares": 700,
                    "cost_price": 5.38,
                    "current_price": 5.02,
                    "market_value": 3514.0,
                    "sector": "光伏",
                    "buy_date": "2026-03-27",
                    "strategy_tag": "短线",
                }
            ],
        }
        mock_interpret.return_value = {
            "is_portfolio_command": True,
            "action": "buy",
            "stock_hint": "协鑫集成",
            "shares": 300,
            "price": 5.05,
            "needs_clarification": False,
            "confidence": 0.98,
        }

        result = handle_portfolio_command("帮我把协鑫集成再补三手，挂5块05")

        self.assertIn("加仓成功", result)
        saved_portfolio = mock_save_portfolio.call_args.args[0]
        self.assertEqual(saved_portfolio["holdings"][0]["shares"], 1000)
        mock_get_stock_name.assert_called_once_with("002506")
        mock_record_buy.assert_called_once()

    @patch("portfolio_bot.load_portfolio", return_value={"holdings": []})
    @patch("portfolio_bot.interpret_portfolio_command")
    @patch("src.services.trade_feedback_service.record_trade_feedback")
    def test_handle_feedback_via_llm_interpreter(
        self,
        mock_record_feedback,
        mock_interpret,
        _mock_load_portfolio,
    ) -> None:
        mock_interpret.return_value = {
            "is_portfolio_command": True,
            "action": "feedback",
            "stock_hint": "协鑫集成",
            "price": 7.61,
            "outcome_price": 7.77,
            "reference_action": "clear",
            "feedback_tag": "sold_too_early",
            "summary": "7.61清仓后立刻被拉到7.77，属于卖飞",
            "needs_clarification": False,
            "confidence": 0.96,
        }
        mock_record_feedback.return_value = {
            "saved": True,
            "name": "协鑫集成",
            "code": "002506",
            "feedback_tag": "sold_too_early",
            "summary": "7.61清仓后立刻被拉到7.77，属于卖飞",
            "guidance": "下次若主线和盘口仍强，优先分批止盈并保留底仓做T。",
        }

        result = handle_portfolio_command("7.61清仓后立刻被拉升到了7.77")

        self.assertIn("已记录反馈", result)
        self.assertIn("卖飞", result)
        mock_record_feedback.assert_called_once()

    @patch("trade_journal.record_buy")
    @patch("src.services.trade_feedback_service.record_trade_feedback")
    @patch("portfolio_bot._get_stock_name")
    @patch("portfolio_bot.resolve_name_to_code")
    def test_handle_batch_portfolio_sync_updates_holdings_and_feedback(
        self,
        mock_resolve_name,
        mock_get_stock_name,
        mock_record_feedback,
        mock_record_buy,
    ) -> None:
        state = {
            "cash": 10000.0,
            "holdings": [
                {
                    "code": "002506",
                    "name": "协鑫集成",
                    "shares": 600,
                    "cost_price": 5.075,
                    "current_price": 5.075,
                    "market_value": 3045.0,
                    "sector": "光伏",
                    "buy_date": "2026-03-30",
                    "strategy_tag": "短线",
                }
            ],
        }

        def fake_load():
            return deepcopy(state)

        def fake_save(portfolio):
            state.clear()
            state.update(deepcopy(portfolio))

        name_map = {
            "002506": "协鑫集成",
            "600227": "赤天化",
            "002413": "雷科防务",
        }
        code_map = {
            "赤天化": "600227",
            "雷科防务": "002413",
        }
        mock_get_stock_name.side_effect = lambda code: name_map.get(code, code)
        mock_resolve_name.side_effect = lambda name: code_map.get(name)
        mock_record_feedback.return_value = {
            "saved": True,
            "name": "华银电力",
            "code": "600744",
            "feedback_tag": "sold_too_early",
            "summary": "7.61清仓后拉升到7.76",
            "guidance": "下次主线强化时分批止盈。",
        }

        with patch("portfolio_bot.load_portfolio", side_effect=fake_load), patch(
            "portfolio_bot.save_portfolio", side_effect=fake_save
        ):
            result = handle_portfolio_command(
                "协鑫集成现在是5.24元13手，华银电力7.61清仓后拉升7.76，赤天化4.3加仓3手，雷科防务13.256 2手"
            )

        self.assertIn("已处理批量持仓同步", result)
        holdings_by_code = {item["code"]: item for item in state["holdings"]}
        self.assertEqual(holdings_by_code["002506"]["shares"], 1300)
        self.assertEqual(holdings_by_code["002506"]["current_price"], 5.24)
        self.assertEqual(holdings_by_code["600227"]["shares"], 300)
        self.assertEqual(holdings_by_code["002413"]["shares"], 200)
        mock_record_feedback.assert_called_once()
        mock_record_buy.assert_called_once()

    @patch("portfolio_bot.save_portfolio")
    def test_future_trade_plan_does_not_change_current_holding(self, mock_save_portfolio) -> None:
        with patch(
            "portfolio_bot.load_portfolio",
            return_value={
                "cash": 10000.0,
                "holdings": [
                    {"code": "600227", "name": "赤天化", "shares": 300, "cost_price": 4.3, "current_price": 4.3}
                ],
            },
        ):
            result = handle_portfolio_command("明天赤天化清仓")

        self.assertIn("不会修改当前持仓", result)
        mock_save_portfolio.assert_not_called()


if __name__ == "__main__":
    unittest.main()
