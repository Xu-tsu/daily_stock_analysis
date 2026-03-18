# -*- coding: utf-8 -*-
"""
Agent tools package.

Provides ToolRegistry, @tool decorator, and wrapped tools
for the stock analysis agent.
"""

from src.agent.tools.registry import ToolRegistry, ToolDefinition, ToolParameter, tool
from src.agent.tools.scanner_tools import scan_strong_stocks

__all__ = ["ToolRegistry", "ToolDefinition", "ToolParameter", "tool", "scan_strong_stocks"]