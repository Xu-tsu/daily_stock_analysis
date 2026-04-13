# -*- coding: utf-8 -*-
"""
StrategyRouter — rule-based strategy selection.

Selects which strategies to apply based on:
1. User-explicit request (highest priority — bypass router)
2. Market regime detection from technical data in ``AgentContext``
3. Default fallback set

No LLM calls — pure rule evaluation for speed and predictability.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.agent.protocols import AgentContext

logger = logging.getLogger(__name__)

# Mapping from detected market regime → preferred strategy IDs.
# Multiple strategies per regime to allow aggregation.
_REGIME_STRATEGIES: Dict[str, List[str]] = {
    "trending_up": ["bull_trend", "volume_breakout", "ma_golden_cross"],
    "trending_down": ["shrink_pullback", "bottom_volume"],
    "sideways": ["box_oscillation", "shrink_pullback"],
    "volatile": ["chan_theory", "wave_theory"],
    "sector_hot": ["dragon_head", "emotion_cycle"],
}

# Fallback when regime can't be determined
_DEFAULT_STRATEGIES = ["bull_trend", "shrink_pullback"]


class StrategyRouter:
    """Select applicable strategies for a given analysis context.

    Priority order:
    1. ``ctx.meta["strategies_requested"]`` — user explicitly chosen
    2. Market-regime based selection from technical opinions
    3. Default fallback
    """

    def select_strategies(
        self,
        ctx: AgentContext,
        max_count: int = 3,
    ) -> List[str]:
        """Return a list of strategy IDs to evaluate.

        Args:
            ctx: The shared pipeline context (with opinions from prior stages).
            max_count: Maximum number of strategies to return.

        Returns:
            Ordered list of strategy IDs.
        """
        # Priority 1: User-explicit
        user_requested = ctx.meta.get("strategies_requested", [])
        if user_requested:
            logger.info("[StrategyRouter] user-requested strategies: %s", user_requested)
            return user_requested[:max_count]

        # If routing mode is "manual", only use AGENT_SKILLS (already in user_requested);
        # since no explicit request was made, fall back to defaults without regime detection.
        routing_mode = self._get_routing_mode()
        if routing_mode == "manual":
            selected = self._get_manual_strategies(max_count=max_count)
            logger.info("[StrategyRouter] manual mode — using strategies: %s", selected)
            return selected

        # Priority 2: Infer from technical opinion (auto mode)
        regime = self._detect_regime(ctx)
        if regime:
            candidates = _REGIME_STRATEGIES.get(regime, _DEFAULT_STRATEGIES)
            # Filter to only available strategies
            available = self._get_available_ids()
            selected = [s for s in candidates if s in available][:max_count]
            if selected:
                logger.info("[StrategyRouter] regime=%s → strategies: %s", regime, selected)
                return selected

        # Fallback
        logger.info("[StrategyRouter] using default strategies")
        return _DEFAULT_STRATEGIES[:max_count]

    def _detect_regime(self, ctx: AgentContext) -> Optional[str]:
        """Infer market regime from technical agent's opinion data."""
        quant_pressure = str(ctx.meta.get("quant_pressure_signal", "") or "").lower()
        market_bias = str(ctx.meta.get("market_bias", "") or "").lower()
        hot_money_signal = str(ctx.meta.get("hot_money_signal", "") or "").lower()

        for op in ctx.opinions:
            if op.agent_name != "technical":
                continue
            regime = self._detect_regime_from_payload(op.raw_data or {})
            if regime:
                if self._should_prefer_sector_hot(ctx, regime, market_bias, quant_pressure, hot_money_signal):
                    return "sector_hot"
                return regime

        # Single-agent and report pipelines often pre-fetch trend_result before
        # the executor is built, so route from that deterministic snapshot too.
        prefetched_trend = ctx.get_data("trend_result")
        if isinstance(prefetched_trend, dict):
            regime = self._detect_regime_from_payload(prefetched_trend)
            if regime:
                if self._should_prefer_sector_hot(ctx, regime, market_bias, quant_pressure, hot_money_signal):
                    return "sector_hot"
                return regime

        if quant_pressure == "high":
            return "volatile"

        if ctx.meta.get("sector_hot") and hot_money_signal in {"active", "constructive"}:
            return "sector_hot"

        if market_bias == "negative":
            return "trending_down"
        if market_bias == "positive" and ctx.meta.get("sector_hot"):
            return "sector_hot"

        # Check sector context in meta
        if ctx.meta.get("sector_hot"):
            return "sector_hot"

        return None

    def _detect_regime_from_payload(self, raw: Dict[str, object]) -> Optional[str]:
        """Infer regime from a technical payload.

        Supports both:
        - multi-agent technical JSON (`bullish|neutral|bearish`, `heavy|normal|light`)
        - pre-fetched `TrendAnalysisResult.to_dict()` payloads from the report path
          (`多头排列`, `空头排列`, `缩量回调`, `放量下跌`, ...)
        """
        if not isinstance(raw, dict):
            return None

        ma_alignment = self._normalize_ma_alignment(raw.get("ma_alignment"))
        trend_score = self._coerce_trend_score(raw)
        volume_status = self._normalize_volume_status(raw.get("volume_status"))

        if ma_alignment == "bullish" and trend_score >= 70:
            return "trending_up"
        if ma_alignment == "bearish" and trend_score <= 30:
            return "trending_down"
        if volume_status == "heavy" and 30 < trend_score < 70:
            return "volatile"
        if ma_alignment == "neutral" or 35 <= trend_score <= 65:
            return "sideways"
        return None

    @staticmethod
    def _coerce_trend_score(raw: Dict[str, object]) -> float:
        for key in ("trend_score", "signal_score", "trend_strength"):
            try:
                value = raw.get(key)
                if value is None:
                    continue
                return float(value)
            except (TypeError, ValueError):
                continue
        return 50.0

    @staticmethod
    def _normalize_ma_alignment(value: object) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        if "bullish" in text or "多头" in text:
            return "bullish"
        if "bearish" in text or "空头" in text:
            return "bearish"
        if "neutral" in text or "缠绕" in text or "盘整" in text:
            return "neutral"
        return text

    @staticmethod
    def _normalize_volume_status(value: object) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        if "heavy" in text or "放量" in text:
            return "heavy"
        if "light" in text or "shrink" in text or "缩量" in text:
            return "light"
        if "normal" in text or "正常" in text:
            return "normal"
        return text

    @staticmethod
    def _should_prefer_sector_hot(
        ctx: AgentContext,
        regime: str,
        market_bias: str,
        quant_pressure: str,
        hot_money_signal: str,
    ) -> bool:
        return (
            regime == "sideways"
            and bool(ctx.meta.get("sector_hot"))
            and hot_money_signal in {"active", "constructive"}
            and market_bias != "negative"
            and quant_pressure != "high"
        )

    @staticmethod
    def _get_routing_mode() -> str:
        """Read the strategy routing mode from config (default: 'auto')."""
        try:
            from src.config import get_config
            config = get_config()
            return getattr(config, "agent_strategy_routing", "auto")
        except Exception:
            return "auto"

    @staticmethod
    def _get_available_ids() -> set:
        """Get the set of strategy IDs available from SkillManager.

        Reads from the cached prototype directly to avoid an unnecessary
        ``deepcopy`` — we only need the skill names (read-only).
        """
        try:
            from src.agent.factory import _SKILL_MANAGER_PROTOTYPE
            if _SKILL_MANAGER_PROTOTYPE is not None:
                return {s.name for s in _SKILL_MANAGER_PROTOTYPE.list_skills()}
            # Prototype not yet initialised — build via get_skill_manager
            from src.agent.factory import get_skill_manager
            sm = get_skill_manager()
            return {s.name for s in sm.list_skills()}
        except Exception:
            return set(_DEFAULT_STRATEGIES)

    @classmethod
    def _get_manual_strategies(cls, max_count: int) -> List[str]:
        """Return strategies configured for manual routing mode."""
        configured: List[str] = []
        try:
            from src.config import get_config
            config = get_config()
            configured = [
                strategy_id
                for strategy_id in getattr(config, "agent_skills", []) or []
                if isinstance(strategy_id, str) and strategy_id
            ]
        except Exception:
            configured = []

        available = cls._get_available_ids()
        selected = [strategy_id for strategy_id in configured if strategy_id in available][:max_count]
        if selected:
            return selected

        fallback = [strategy_id for strategy_id in _DEFAULT_STRATEGIES if strategy_id in available][:max_count]
        return fallback or _DEFAULT_STRATEGIES[:max_count]
