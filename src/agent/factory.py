# -*- coding: utf-8 -*-
"""
Shared factory for building fully-configured AgentExecutor instances.

Centralises construction to eliminate boilerplate duplicated across
api/v1/endpoints/agent.py, bot/commands/chat.py, bot/commands/ask.py,
and src/core/pipeline.py.

Performance notes
-----------------
* ``ToolRegistry`` is built once and cached at module level — tool
  registrations are immutable after setup so the object is safe to share
  across every request.
* ``SkillManager`` is expensive to create (loads YAML files from disk).
  A prototype is built on first use and cheap ``deepcopy`` clones are
  returned for each request, preserving thread-safety (``activate()``
  mutates internal state).

Usage::

    from src.agent.factory import build_agent_executor

    executor = build_agent_executor(config, skills=["bull_trend", "shrink_pullback"])
    result   = executor.chat(message="...", session_id="...")
"""

import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_TOOL_REGISTRY = None
_SKILL_MANAGER_PROTOTYPE = None
# Sentinel used as initial value so None (i.e. no custom dir) compares as "changed"
# on the very first call, forcing a build rather than accidentally skipping it.
_SENTINEL = object()
# Track which custom_dir the prototype was built with so we can invalidate
# the cache if AGENT_STRATEGY_DIR changes at runtime (e.g. via config reload).
_SKILL_MANAGER_CUSTOM_DIR: object = _SENTINEL

DEFAULT_AGENT_SKILLS = [
    "bull_trend",
    "ma_golden_cross",
    "volume_breakout",
    "shrink_pullback",
]


def _build_router_context(context: Optional[Dict[str, Any]]):
    """Convert caller context into an AgentContext for strategy routing."""
    if not isinstance(context, dict) or not context:
        return None

    from src.agent.protocols import AgentContext

    ctx = AgentContext(
        query=str(context.get("query") or context.get("task") or ""),
        stock_code=str(context.get("stock_code") or ""),
        stock_name=str(context.get("stock_name") or ""),
    )

    for data_key in (
        "realtime_quote",
        "daily_history",
        "chip_distribution",
        "trend_result",
        "news_context",
        "market_context",
        "current_holding",
        "adaptive_trading_rules",
        "fundamental_context",
    ):
        value = context.get(data_key)
        if value is not None:
            ctx.set_data(data_key, value)

    strategies = context.get("strategies")
    if isinstance(strategies, list):
        ctx.meta["strategies_requested"] = strategies

    market_context = context.get("market_context") or {}
    if isinstance(market_context, dict):
        ctx.meta["market_bias"] = market_context.get("bias", "")
        ctx.meta["market_score"] = market_context.get("market_score")
        ctx.meta["macro_risk_level"] = market_context.get("macro_risk_level", "")
        ctx.meta["quant_pressure_signal"] = (
            (market_context.get("quant_pressure") or {}).get("signal", "")
        )
        ctx.meta["hot_money_signal"] = (
            (market_context.get("hot_money_probe") or {}).get("signal", "")
        )
        ctx.meta["auction_direction"] = (
            (market_context.get("opening_auction") or {}).get("direction", "")
        )
        sector_confirmation = market_context.get("sector_confirmation") or {}
        ctx.meta["sector_hot"] = bool(sector_confirmation.get("confirmed"))

    return ctx


def _resolve_skills_to_activate(
    config,
    skills: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Resolve active skills for this executor build.

    Priority:
    1. Explicit caller-provided skills
    2. Auto routing from runtime context when enabled
    3. Configured AGENT_SKILLS / default fallback
    """
    explicit_skills = [skill for skill in (skills or []) if isinstance(skill, str) and skill]
    if explicit_skills:
        return explicit_skills

    routing_mode = getattr(config, "agent_strategy_routing", "auto")
    if routing_mode == "auto":
        try:
            from src.agent.strategies.router import StrategyRouter

            router_ctx = _build_router_context(context)
            if router_ctx is not None:
                selected = StrategyRouter().select_strategies(router_ctx)
                if selected:
                    logger.info("[AgentFactory] Auto-routed strategies from runtime context: %s", selected)
                    return selected
        except Exception as exc:
            logger.warning("[AgentFactory] Failed to auto-route strategies from context: %s", exc)

    configured = [
        skill
        for skill in (getattr(config, "agent_skills", None) or [])
        if isinstance(skill, str) and skill
    ]
    if configured:
        return configured
    return list(DEFAULT_AGENT_SKILLS)


def get_tool_registry():
    """Return a cached ToolRegistry (built once, shared across requests)."""
    global _TOOL_REGISTRY
    if _TOOL_REGISTRY is not None:
        return _TOOL_REGISTRY

    from src.agent.tools.registry import ToolRegistry
    from src.agent.tools.data_tools import ALL_DATA_TOOLS
    from src.agent.tools.analysis_tools import ALL_ANALYSIS_TOOLS
    from src.agent.tools.search_tools import ALL_SEARCH_TOOLS
    from src.agent.tools.market_tools import ALL_MARKET_TOOLS
    from src.agent.tools.backtest_tools import ALL_BACKTEST_TOOLS
    from src.agent.tools.scanner_tools import ALL_SCANNER_TOOLS

    registry = ToolRegistry()
    for tool_fn in ALL_DATA_TOOLS + ALL_ANALYSIS_TOOLS + ALL_SEARCH_TOOLS + ALL_MARKET_TOOLS + ALL_BACKTEST_TOOLS + ALL_SCANNER_TOOLS:
        registry.register(tool_fn)

    _TOOL_REGISTRY = registry
    logger.info("[AgentFactory] ToolRegistry cached (%d tools)", len(registry._tools) if hasattr(registry, "_tools") else -1)
    return _TOOL_REGISTRY


def get_skill_manager(config=None):
    """Return a deepcopy-clone of the cached SkillManager prototype.

    The prototype is initialised from disk on first call; subsequent calls
    return ``copy.deepcopy(prototype)`` which is ~10× faster than re-reading
    YAML files.  Each clone is independent so ``.activate()`` calls do not
    bleed between requests.

    Cache invalidation: if ``config.agent_strategy_dir`` changes at runtime
    (e.g. via the web settings reload), the prototype is rebuilt automatically.
    """
    global _SKILL_MANAGER_PROTOTYPE, _SKILL_MANAGER_CUSTOM_DIR

    if config is None:
        from src.config import get_config
        config = get_config()

    current_custom_dir = getattr(config, "agent_strategy_dir", None)
    if _SKILL_MANAGER_PROTOTYPE is not None and current_custom_dir == _SKILL_MANAGER_CUSTOM_DIR:
        return copy.deepcopy(_SKILL_MANAGER_PROTOTYPE)

    from src.agent.skills.base import SkillManager

    if _SKILL_MANAGER_PROTOTYPE is not None:
        logger.info("[AgentFactory] SkillManager prototype invalidated (agent_strategy_dir changed: %r → %r)",
                    _SKILL_MANAGER_CUSTOM_DIR, current_custom_dir)

    skill_manager = SkillManager()
    skill_manager.load_builtin_strategies()

    if current_custom_dir:
        try:
            skill_manager.load_custom_strategies(current_custom_dir)
        except Exception as exc:
            logger.warning("[AgentFactory] Failed to load custom strategies from %s: %s", current_custom_dir, exc)

    _SKILL_MANAGER_PROTOTYPE = skill_manager
    _SKILL_MANAGER_CUSTOM_DIR = current_custom_dir
    logger.info("[AgentFactory] SkillManager prototype cached (%d strategies)", len(skill_manager._skills))
    return copy.deepcopy(_SKILL_MANAGER_PROTOTYPE)


def build_agent_executor(
    config=None,
    skills: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
):
    """Build and return a configured AgentExecutor (or future orchestrator).

    When ``AGENT_ARCH=multi``, this returns an orchestrator that manages
    multiple specialised agents. Otherwise it returns the legacy single-agent
    executor.

    Args:
        config: Application config object.  When *None*, ``get_config()`` is
                called automatically.
        skills: Strategy ids to activate. When provided, this bypasses
                environment-aware auto routing.
        context: Optional runtime context used for auto routing in ``auto``
                 mode (e.g. pre-fetched ``trend_result`` / ``market_context``).

    Returns:
        A ready-to-call :class:`src.agent.executor.AgentExecutor` instance.
    """
    if config is None:
        from src.config import get_config
        config = get_config()

    arch = getattr(config, "agent_arch", "single")

    from src.agent.llm_adapter import LLMToolAdapter

    registry = get_tool_registry()
    skill_manager = get_skill_manager(config)

    skills_to_activate = _resolve_skills_to_activate(config, skills=skills, context=context)
    skill_manager.activate(skills_to_activate if skills_to_activate else ["all"])
    logger.info("[AgentFactory] Activated strategies: %s (arch=%s)", skills_to_activate, arch)

    llm_adapter = LLMToolAdapter(config)

    if arch == "multi":
        return _build_orchestrator(config, registry, llm_adapter, skill_manager)

    from src.agent.executor import AgentExecutor
    return AgentExecutor(
        tool_registry=registry,
        llm_adapter=llm_adapter,
        skill_instructions=skill_manager.get_skill_instructions(),
        max_steps=getattr(config, "agent_max_steps", 10),
    )


def _build_orchestrator(config, registry, llm_adapter, skill_manager):
    """Build and return an :class:`AgentOrchestrator` (multi-agent mode).

    The orchestrator presents the same ``run()`` / ``chat()`` interface as
    :class:`AgentExecutor` so callers need no changes.
    """
    from src.agent.orchestrator import AgentOrchestrator

    mode = getattr(config, "agent_orchestrator_mode", "standard")
    logger.info("[AgentFactory] Building AgentOrchestrator (mode=%s)", mode)

    return AgentOrchestrator(
        tool_registry=registry,
        llm_adapter=llm_adapter,
        skill_instructions=skill_manager.get_skill_instructions(),
        max_steps=getattr(config, "agent_max_steps", 10),
        mode=mode,
        skill_manager=skill_manager,
        config=config,
    )


# Keep legacy alias so any external callers using the old name still work.
build_executor = build_agent_executor
