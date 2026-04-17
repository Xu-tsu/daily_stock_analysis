"""实盘 V2 策略参数（来自 backtest_adaptive_v2.py，回测 +10.58% / DD 9.61%）。

用途：将"收益最高"的 V2 配置固化成实盘单一来源（Single Source of Truth）。
- _auto_build_positions 读 get_position_plan(regime, adaptive_coeff)
- 风控止盈/止损沿用 risk_control.py 的 env 变量，由 .env 写入 V2 默认值
- 只负责参数，不负责选股（选股仍走 dual_trader 的龙头打板）
"""

from __future__ import annotations
from typing import Dict


# 直接从 backtest_adaptive_v2.REGIME_CONFIG_V2 抄录（去掉 score_fn 等回测专用字段）
REGIME_CONFIG_V2: Dict[str, Dict] = {
    "BULL": {
        "max_positions": 2, "single_pct": 0.50, "total_pct": 0.90,
        "stop_loss": -4.0, "tp_half": 6.0, "tp_full": 15.0,
        "trail_trigger": 8.0, "trail_dd": 3.0,
        "hold_max": 5, "price_range": (3, 50),
    },
    "SIDE": {
        "max_positions": 2, "single_pct": 0.35, "total_pct": 0.60,
        "stop_loss": -3.0, "tp_half": 4.0, "tp_full": 8.0,
        "trail_trigger": 4.0, "trail_dd": 2.0,
        "hold_max": 4, "price_range": (3, 30),
    },
    "BEAR": {
        "max_positions": 1, "single_pct": 0.25, "total_pct": 0.25,
        "stop_loss": -3.0, "tp_half": 3.0, "tp_full": 6.0,
        "trail_trigger": 3.0, "trail_dd": 1.5,
        "hold_max": 2, "price_range": (3, 20),
    },
    "CRASH": {
        "max_positions": 1, "single_pct": 0.15, "total_pct": 0.15,
        "stop_loss": -2.5, "tp_half": 2.0, "tp_full": 5.0,
        "trail_trigger": 2.0, "trail_dd": 1.5,
        "hold_max": 2, "price_range": (3, 15),
    },
}


_REGIME_ALIAS = {
    "bull": "BULL", "bullish": "BULL", "牛": "BULL",
    "sideways": "SIDE", "side": "SIDE", "震荡": "SIDE",
    "bear": "BEAR", "bearish": "BEAR", "熊": "BEAR",
    "crash": "CRASH", "崩盘": "CRASH", "panic": "CRASH",
}


def normalize_regime(regime) -> str:
    """兼容实盘 regime 字符串 / dict 两种输入，统一成 V2 key。"""
    if isinstance(regime, dict):
        regime = regime.get("regime") or regime.get("market") or "SIDE"
    if not isinstance(regime, str):
        return "SIDE"
    key = regime.strip().lower()
    return _REGIME_ALIAS.get(key, regime.strip().upper() if regime.strip().upper() in REGIME_CONFIG_V2 else "SIDE")


def get_strategy_params(regime) -> Dict:
    """按 regime 返回 V2 策略参数。"""
    return REGIME_CONFIG_V2[normalize_regime(regime)]


def get_position_plan(regime, adaptive_coeff: float = 1.0, event_defense: bool = False) -> Dict:
    """实盘建仓计划。

    V2 规则 × 自适应安全系数 × 事件防御：
      - single_pct / total_pct 取自 V2 当前 regime
      - 再乘以 adaptive_coeff（0.1~1.0，由 AdaptiveTradeState 提供）
      - 如果事件防御生效，再乘 0.5
    """
    p = get_strategy_params(regime)
    coeff = max(0.1, min(1.0, float(adaptive_coeff or 1.0)))
    defense = 0.5 if event_defense else 1.0
    multiplier = coeff * defense
    return {
        "regime": normalize_regime(regime),
        "max_positions": p["max_positions"],
        "single_pct": round(p["single_pct"] * multiplier, 4),
        "total_pct": round(p["total_pct"] * multiplier, 4),
        "base_single_pct": p["single_pct"],
        "base_total_pct": p["total_pct"],
        "adaptive_coeff": coeff,
        "event_defense": event_defense,
        # 透传给风控用
        "stop_loss": p["stop_loss"],
        "tp_half": p["tp_half"],
        "tp_full": p["tp_full"],
        "trail_trigger": p["trail_trigger"],
        "trail_dd": p["trail_dd"],
        "hold_max": p["hold_max"],
        "price_range": p["price_range"],
    }
