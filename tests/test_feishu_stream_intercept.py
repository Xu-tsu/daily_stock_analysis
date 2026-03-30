# -*- coding: utf-8 -*-
"""Unittest coverage for Feishu Stream portfolio interception."""

import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.models import BotMessage, ChatType
from bot.platforms.feishu_stream import FeishuStreamClient


class TestFeishuStreamIntercept(unittest.TestCase):
    def test_portfolio_intercept_reads_env_without_name_error(self) -> None:
        client = FeishuStreamClient.__new__(FeishuStreamClient)
        handler = client._create_message_handler()
        message = BotMessage(
            platform="feishu",
            message_id="msg_1",
            user_id="user_1",
            user_name="tester",
            chat_id="chat_1",
            chat_type=ChatType.GROUP,
            content="持仓",
        )

        fake_dispatcher_module = types.SimpleNamespace(
            get_dispatcher=lambda: types.SimpleNamespace(
                dispatch=lambda _message: None
            )
        )

        with patch.dict(os.environ, {"FEISHU_PORTFOLIO_COMMAND_INTERCEPT": "true"}, clear=False):
            with patch.dict(
                sys.modules,
                {
                    "portfolio_bot": types.SimpleNamespace(
                        is_portfolio_command=lambda text: True,
                        handle_portfolio_command=lambda text: "拦截成功",
                    ),
                    "bot.dispatcher": fake_dispatcher_module,
                },
                clear=False,
            ):
                response = handler(message)

        self.assertEqual(response.text, "拦截成功")
        self.assertFalse(response.at_user)


if __name__ == "__main__":
    unittest.main()
