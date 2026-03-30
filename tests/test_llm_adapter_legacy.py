# -*- coding: utf-8 -*-
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from src.agent.llm_adapter import LLMToolAdapter
from src.config import extra_litellm_params


class TestLLMToolAdapterLegacyMode(unittest.TestCase):
    @patch("src.agent.llm_adapter.Router")
    def test_legacy_env_with_ollama_fallback_does_not_use_channel_router(self, mock_router):
        cfg = SimpleNamespace(
            litellm_model="gemini/gemini-2.5-flash",
            litellm_fallback_models=["ollama/qwen2.5:14b-instruct-q4_K_M"],
            llm_model_list=[
                {
                    "model_name": "__legacy_gemini__",
                    "litellm_params": {
                        "model": "__legacy_gemini__",
                        "api_key": "g-12345678",
                    },
                },
                {
                    "model_name": "__legacy_gemini__",
                    "litellm_params": {
                        "model": "__legacy_gemini__",
                        "api_key": "g-87654321",
                    },
                },
                {
                    "model_name": "ollama/qwen2.5:14b-instruct-q4_K_M",
                    "litellm_params": {
                        "model": "ollama/qwen2.5:14b-instruct-q4_K_M",
                        "api_base": "http://localhost:11434/v1",
                        "api_key": "ollama",
                    },
                },
            ],
            llm_models_source="legacy_env",
            llm_channels=[],
            gemini_api_keys=["g-12345678", "g-87654321"],
            anthropic_api_keys=[],
            openai_api_keys=[],
            deepseek_api_keys=[],
            openai_base_url=None,
            llm_temperature=0.2,
        )

        adapter = LLMToolAdapter(config=cfg)

        self.assertFalse(adapter._has_channel_config())
        mock_router.assert_called_once()
        router_model_list = mock_router.call_args.kwargs["model_list"]
        self.assertEqual(len(router_model_list), 2)
        self.assertTrue(
            all(item["litellm_params"]["model"] == "gemini/gemini-2.5-flash" for item in router_model_list)
        )
        self.assertNotIn("__legacy_gemini__", str(router_model_list))


class TestLegacyOllamaParams(unittest.TestCase):
    @patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://127.0.0.1:11434"}, clear=False)
    def test_extra_litellm_params_adds_ollama_base_and_key(self):
        params = extra_litellm_params("ollama/qwen2.5:14b-instruct-q4_K_M", SimpleNamespace(openai_base_url=None))

        self.assertEqual(params["api_base"], "http://127.0.0.1:11434/v1")
        self.assertEqual(params["api_key"], "ollama")


if __name__ == "__main__":
    unittest.main()
