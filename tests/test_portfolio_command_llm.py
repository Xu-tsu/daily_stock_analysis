# -*- coding: utf-8 -*-
"""Tests for local-LLM portfolio command interpretation."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()

from src.services.portfolio_command_llm import interpret_portfolio_command


class TestPortfolioCommandLLM(unittest.TestCase):
    @patch.dict(os.environ, {"REBALANCE_LOCAL_MODEL": "ollama/qwen2.5:14b-instruct-q4_K_M", "OLLAMA_BASE_URL": "http://127.0.0.1:11434"}, clear=False)
    @patch("src.services.portfolio_command_llm.detect_local_ollama_models", return_value=[])
    @patch("src.services.portfolio_command_llm.get_config")
    @patch("src.services.portfolio_command_llm.litellm.completion")
    def test_interpret_portfolio_command_prefers_local_model(
        self,
        mock_completion,
        mock_get_config,
        _mock_detect,
    ) -> None:
        mock_get_config.return_value = SimpleNamespace(
            litellm_model="gemini/gemini-2.5-flash",
            litellm_fallback_models=["gemini/gemini-2.5-flash-lite"],
            gemini_api_keys=[],
            anthropic_api_keys=[],
            deepseek_api_keys=[],
            openai_api_keys=[],
            openai_base_url=None,
        )
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"is_portfolio_command":true,"action":"buy","stock_hint":"协鑫集成","shares":300,"price":5.05,"needs_clarification":false,"confidence":0.97,"reason":"用户明确要求加仓3手"}'
                )
            )
        ]
        mock_completion.return_value = response

        result = interpret_portfolio_command(
            "帮我把协鑫集成再补三手，挂5块05",
            holdings=[{"code": "002506", "name": "协鑫集成", "shares": 700}],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "buy")
        self.assertEqual(result["stock_hint"], "协鑫集成")
        self.assertEqual(result["shares"], 300)
        self.assertEqual(result["price"], 5.05)
        self.assertEqual(result["model"], "ollama/qwen2.5:14b-instruct-q4_K_M")
        kwargs = mock_completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "ollama/qwen2.5:14b-instruct-q4_K_M")
        self.assertEqual(kwargs["api_base"], "http://127.0.0.1:11434")
        self.assertEqual(kwargs["api_key"], "ollama")

    @patch("src.services.portfolio_command_llm.detect_local_ollama_models", return_value=[])
    @patch("src.services.portfolio_command_llm.get_config")
    def test_returns_none_when_local_model_unavailable(
        self,
        mock_get_config,
        _mock_detect,
    ) -> None:
        mock_get_config.return_value = SimpleNamespace(
            litellm_model="gemini/gemini-2.5-flash",
            litellm_fallback_models=[],
            gemini_api_keys=[],
            anthropic_api_keys=[],
            deepseek_api_keys=[],
            openai_api_keys=[],
            openai_base_url=None,
        )

        with patch.dict(os.environ, {"REBALANCE_LOCAL_MODEL": ""}, clear=False):
            result = interpret_portfolio_command("看下我现在仓位", holdings=[])

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
