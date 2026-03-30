# -*- coding: utf-8 -*-
import unittest

from portfolio_manager import format_rebalance_report


class TestPortfolioManagerRebalanceReport(unittest.TestCase):
    def test_rebalance_report_includes_agent_discussion_rounds(self):
        rebalance = {
            "overall_position_advice": "当前实际仓位56%，建议调整至35%",
            "market_assessment": "高位震荡偏弱，先降风险。",
            "sector_assessment": "光伏偏强，电力偏弱。",
            "debate_summary": "激进派希望保留弹性，保守派强调电力板块确认不足，最终先降仓位。",
            "agent_discussion": {
                "summary": "本地提案先给出调仓草案，保守派反对继续硬扛电力股，最终由云端仲裁收敛到偏防守结论。",
                "rounds": [
                    {
                        "agent_label": "Agent1 大盘研判",
                        "role_label": "宏观/指数",
                        "model": "ollama/qwen2.5:14b-instruct-q4_K_M",
                        "signal_label": "高位震荡偏弱 | 仓位0.35",
                        "reasoning": "量化抛压仍在，早盘回流不够稳。",
                    },
                    {
                        "agent_label": "Agent4b 保守派质疑",
                        "role_label": "风控审查",
                        "model": "ollama/deepseek-r1:14b",
                        "signal_label": "方案评分 4/10",
                        "reasoning": "华银电力与利欧股份不满足继续持有的胜率条件。",
                    },
                    {
                        "agent_label": "Agent4c 云端仲裁",
                        "role_label": "最终裁决",
                        "model": "azure/gpt-5.4-nano",
                        "signal_label": "当前实际仓位56%，建议调整至35%",
                        "reasoning": "先处理弱线持仓，再等待更强主线回踩。",
                    },
                ],
                "disagreements": [
                    "华银电力: 持有 → 清仓（电力板块确认不足）",
                    "利欧股份: 继续观察 → 先退出（持仓效率衰减）",
                ],
            },
            "actions": [
                {
                    "code": "600744",
                    "name": "华银电力",
                    "action": "sell",
                    "detail": "立即卖出 today sellable 部分",
                    "reason": "亏损扩大且板块确认不足",
                    "target_sell_price": 8.40,
                    "stop_loss_price": 7.80,
                }
            ],
            "new_candidates": [],
            "risk_warning": "先把弱线仓位降下来，再等板块回流。",
        }

        report = format_rebalance_report(rebalance)

        self.assertIn("多模型辩论", report)
        self.assertIn("Agent1 大盘研判", report)
        self.assertIn("ollama/qwen2.5:14b-instruct-q4_K_M", report)
        self.assertIn("Agent4c 云端仲裁", report)
        self.assertIn("azure/gpt-5.4-nano", report)
        self.assertIn("华银电力: 持有 → 清仓", report)

    def test_rebalance_report_without_agent_discussion_keeps_core_sections(self):
        rebalance = {
            "overall_position_advice": "建议仓位30%",
            "market_assessment": "指数弱势震荡。",
            "sector_assessment": "热点轮动加快。",
            "actions": [],
            "new_candidates": [],
            "risk_warning": "谨慎追高。",
        }

        report = format_rebalance_report(rebalance)

        self.assertIn("调仓建议报告", report)
        self.assertIn("仓位建议", report)
        self.assertNotIn("Agent1 大盘研判", report)


if __name__ == "__main__":
    unittest.main()
