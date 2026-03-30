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
from src.agent import llm_adapter as llm_adapter_module
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


class TestVersionedLiteLLMAliases(unittest.TestCase):
    @patch("src.agent.llm_adapter.litellm.register_model")
    @patch("src.agent.llm_adapter.litellm.get_model_info")
    def test_registers_versioned_azure_model_alias(self, mock_get_model_info, mock_register_model):
        def fake_get_model_info(model_name):
            if model_name == "azure/gpt-5.4-nano":
                return {
                    "key": model_name,
                    "litellm_provider": "azure",
                    "supports_tool_choice": True,
                }
            if model_name == "gpt-5.4-nano":
                return {
                    "key": model_name,
                    "litellm_provider": "openai",
                    "supports_tool_choice": True,
                }
            raise ValueError(model_name)

        mock_get_model_info.side_effect = fake_get_model_info
        llm_adapter_module._REGISTERED_LITELLM_MODEL_ALIASES.clear()

        try:
            llm_adapter_module._register_versioned_model_alias("azure/gpt-5.4-nano-2026-03-17")
        finally:
            llm_adapter_module._REGISTERED_LITELLM_MODEL_ALIASES.clear()

        mock_register_model.assert_called_once()
        registrations = mock_register_model.call_args.args[0]
        self.assertIn("azure/gpt-5.4-nano-2026-03-17", registrations)
        self.assertIn("gpt-5.4-nano-2026-03-17", registrations)
        self.assertEqual(
            registrations["azure/gpt-5.4-nano-2026-03-17"]["litellm_provider"],
            "azure",
        )

    @patch.object(LLMToolAdapter, "_init_litellm", autospec=True, return_value=None)
    @patch.object(LLMToolAdapter, "_parse_litellm_response", return_value="ok")
    @patch("src.agent.llm_adapter.extra_litellm_params", return_value={})
    @patch("src.agent.llm_adapter.get_api_keys_for_model", return_value=[])
    @patch("src.agent.llm_adapter.litellm.completion", return_value=object())
    @patch("src.agent.llm_adapter._register_versioned_model_alias")
    def test_call_path_registers_model_override_alias(
        self,
        mock_register_alias,
        _mock_completion,
        _mock_get_keys,
        _mock_extra_params,
        mock_parse,
        _mock_init,
    ):
        cfg = SimpleNamespace(
            litellm_model="openai/gpt-4o-mini",
            litellm_fallback_models=[],
            llm_model_list=[],
            llm_models_source="legacy_env",
            llm_temperature=0.2,
            openai_base_url=None,
        )
        adapter = LLMToolAdapter(config=cfg)
        adapter._router = None

        result = adapter._call_litellm_model(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            model="azure/gpt-5.4-nano-2026-03-17",
        )

        mock_register_alias.assert_called_once_with("azure/gpt-5.4-nano-2026-03-17")
        mock_parse.assert_called_once()
        self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
