"""
Blind agent V4c: environment-routed strategy backtest.

Key rules:
1. No future labels, no pre-fed bull/bear tags.
2. Use only same-day cross-sectional breadth from price data.
3. Reuse StrategyRouter to route candidate scoring by environment.
4. Keep the stronger V4b structural filters for weak/hot environments.

Usage:
    python backtest_blind_agent_v4c.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import backtest_blind_agent as blind
import backtest_blind_agent_v4b as v4b
import train_blind_agent as trainer
from backtest_2025 import fetch_all_a_share_daily
from market_scanner import analyze_kline
from src.agent.protocols import AgentContext
from src.agent.strategies.router import StrategyRouter


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

OUT_DIR = Path("data/backtest_blind_agent_v4c")
ROUTING_PARAMS = {
    "consensus_bonus": 4.0,
    "dragon_bonus": 6.0,
    "bottom_bonus": 4.0,
    "breakout_bonus": 3.0,
}

SKILL_MODE_MAP = {
    "bull_trend": "trend",
    "ma_golden_cross": "trend",
    "shrink_pullback": "trend",
    "box_oscillation": "trend",
    "volume_breakout": "breakout",
    "chan_theory": "breakout",
    "wave_theory": "breakout",
    "bottom_volume": "oversold",
    "dragon_head": "dragon",
    "emotion_cycle": "dragon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind agent V4c backtest")
    parser.add_argument("--start", default="2026-01-05", help="Backtest start date")
    parser.add_argument("--end", default="2026-04-10", help="Backtest end date")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    return parser.parse_args()


def _trend_payload(ind: Dict[str, float]) -> Dict[str, object]:
    if ind["ma5"] > ind["ma10"] > ind["ma20"]:
        ma_alignment = "bullish"
    elif ind["ma5"] < ind["ma10"] < ind["ma20"]:
        ma_alignment = "bearish"
    else:
        ma_alignment = "neutral"

    if ind["vol_ratio"] > 1.5:
        volume_status = "heavy"
    elif ind["vol_ratio"] < 0.8:
        volume_status = "light"
    else:
        volume_status = "normal"

    trend_score = 50.0
    if ma_alignment == "bullish":
        trend_score += 20
    elif ma_alignment == "bearish":
        trend_score -= 20
    if ind["ma5_slope"] > 0.6:
        trend_score += 8
    elif ind["ma5_slope"] < -0.6:
        trend_score -= 8
    if ind["dif"] > ind["dea"]:
        trend_score += 6
    else:
        trend_score -= 4
    if ind["bias20"] > 3:
        trend_score += 6
    elif ind["bias20"] < -5:
        trend_score -= 10
    if ind["today_chg"] > 3:
        trend_score += 5
    elif ind["today_chg"] < -3:
        trend_score -= 8

    return {
        "ma_alignment": ma_alignment,
        "trend_score": round(max(0.0, min(100.0, trend_score)), 2),
        "volume_status": volume_status,
    }


def _market_meta(
    day_ctx: v4b.MarketContext,
    ind: Dict[str, float],
    structure_params: Dict[str, object],
) -> Dict[str, object]:
    hot = v4b.is_hot_context(day_ctx, structure_params)
    cold = v4b.is_cold_block(day_ctx, structure_params)

    if cold:
        market_bias = "negative"
    elif hot:
        market_bias = "positive"
    else:
        market_bias = "neutral"

    if cold or day_ctx.weak_ratio >= 0.72:
        quant_pressure = "high"
    elif day_ctx.weak_ratio >= 0.55:
        quant_pressure = "medium"
    else:
        quant_pressure = "low"

    stock_is_hot = bool(
        hot
        and (
            ind["today_chg"] >= 2
            or ind["vol_breakout"]
            or ind["yest_limit"]
            or ind["consec_up"] >= 2
        )
    )

    if stock_is_hot:
        hot_money_signal = "active"
    elif hot or day_ctx.breakout_count >= 4:
        hot_money_signal = "constructive"
    else:
        hot_money_signal = "quiet"

    return {
        "market_bias": market_bias,
        "quant_pressure_signal": quant_pressure,
        "hot_money_signal": hot_money_signal,
        "sector_hot": stock_is_hot,
    }


def _route_modes_for_candidate(
    day_ctx: v4b.MarketContext,
    ind: Dict[str, float],
    structure_params: Dict[str, object],
    router: StrategyRouter,
) -> Tuple[List[str], List[str], str]:
    ctx = AgentContext(stock_code="", stock_name="")
    ctx.set_data("trend_result", _trend_payload(ind))
    meta = _market_meta(day_ctx, ind, structure_params)
    ctx.meta.update(meta)
    skills = router.select_strategies(ctx, max_count=3)
    modes = []
    for skill in skills:
        mode = SKILL_MODE_MAP.get(skill)
        if mode and mode not in modes:
            modes.append(mode)
    if not modes:
        modes = ["trend"]
    regime = router._detect_regime(ctx) or "fallback"
    return skills, modes, regime


def _routed_score(
    df: pd.DataFrame,
    idx: int,
    ind: Dict[str, float],
    day_ctx: v4b.MarketContext,
    structure_params: Dict[str, object],
    router: StrategyRouter,
) -> Tuple[float, List[str], str, Dict[str, float]]:
    window = df.iloc[: idx + 1].copy()
    skills, modes, regime = _route_modes_for_candidate(day_ctx, ind, structure_params, router)

    mode_scores: Dict[str, float] = {}
    for mode in modes:
        try:
            ta = analyze_kline(window, mode=mode)
            mode_scores[mode] = float(ta.get("score", 0) or 0)
        except Exception:
            mode_scores[mode] = 0.0

    best = max(mode_scores.values()) if mode_scores else 0.0
    routed = best
    if len(modes) >= 2:
        routed += ROUTING_PARAMS["consensus_bonus"] * (len(modes) - 1)
    if "dragon_head" in skills and mode_scores.get("dragon", 0) > 0:
        routed += ROUTING_PARAMS["dragon_bonus"]
    if "emotion_cycle" in skills and mode_scores.get("dragon", 0) > 0:
        routed += ROUTING_PARAMS["dragon_bonus"] * 0.5
    if "bottom_volume" in skills and mode_scores.get("oversold", 0) > 0:
        routed += ROUTING_PARAMS["bottom_bonus"]
    if "volume_breakout" in skills and mode_scores.get("breakout", 0) > 0:
        routed += ROUTING_PARAMS["breakout_bonus"]

    return round(max(0.0, min(100.0, routed)), 2), skills, regime, mode_scores


class RoutedBlindAgentBacktest(v4b.StructuredBlindAgentBacktest):
    def __init__(
        self,
        all_data: Dict[str, pd.DataFrame],
        market_contexts: Dict[str, v4b.MarketContext],
        structure_params: Optional[Dict[str, object]] = None,
    ):
        super().__init__(all_data, market_contexts, structure_params=structure_params)
        self.router = StrategyRouter()
        self.route_counter: Counter[str] = Counter()
        self.skill_counter: Counter[str] = Counter()
        self.entry_meta: Dict[str, Dict[str, object]] = {}

    def _scan_buys(self, date, day_idx):
        if len(self.holdings) >= blind.RISK_PARAMS["max_positions"]:
            return
        if self._pending_buys:
            return
        if self._state.in_cooldown:
            return
        if self._state.consecutive_losses >= 4:
            self._state.trigger_cooldown()
            return

        ctx = self._ctx(date)
        sp = self.structure_params
        if v4b.is_cold_block(ctx, sp):
            self.context_stats["cold_block_days"] += 1
            self.context_stats["blocked_buy_days"] += 1
            return

        caution = v4b.is_caution_context(ctx, sp)
        if caution:
            self.context_stats["caution_days"] += 1

        threshold = self._state.buy_threshold + (sp["caution_threshold_add"] if caution else 0)
        extra_score_needed = sp["caution_score_add"] if caution else 0
        pmin, pmax = blind.RISK_PARAMS["price_range"]
        min_to = blind.RISK_PARAMS["min_turnover"]

        cands = []
        for code, df in self.all_data.items():
            if code in self.holdings:
                continue
            ri = df.index[df["date"] == date]
            if len(ri) == 0:
                continue
            idx = int(ri[0])
            row = df.iloc[idx]
            price = float(row["close"])
            if price < pmin or price > pmax:
                continue
            turnover = float(row.get("turnover_rate", 0) or 0)
            if turnover < min_to:
                continue
            ind = blind.calc_indicators(df, idx)
            if ind is None:
                continue

            score, skills, regime, mode_scores = _routed_score(
                df,
                idx,
                ind,
                ctx,
                sp,
                self.router,
            )
            if score < threshold + extra_score_needed:
                continue

            self.route_counter[regime] += 1
            for skill in skills:
                self.skill_counter[skill] += 1

            cands.append(
                {
                    "code": code,
                    "name": str(row.get("stock_name", code)),
                    "signal_close": price,
                    "score": score,
                    "skills": skills,
                    "route_regime": regime,
                    "mode_scores": mode_scores,
                }
            )

        cands.sort(key=lambda x: x["score"], reverse=True)
        slots = blind.RISK_PARAMS["max_positions"] - len(self.holdings)
        if caution:
            slots = min(slots, sp["caution_slots"])
        for candidate in cands[: max(1, slots)]:
            self._pending_buys.append(candidate)

    def _exec_pending_buys(self, date, day_idx):
        buys = self._pending_buys[:]
        self._pending_buys = []
        if not buys:
            return

        for pb in buys:
            code = pb["code"]
            row = self._get_row(code, date)
            if row is None:
                continue
            op = float(row["open"])
            if op <= 0:
                continue
            sc = pb.get("signal_close", 0)
            if sc > 0 and op >= sc * 1.098:
                continue
            if sc > 0 and op > sc * 1.04:
                continue

            cost = op * (1 + blind.SLIPPAGE_PCT)
            pc = self._state.position_coeff
            total_asset = self.cash + sum(h.market_value for h in self.holdings.values())
            mv_now = sum(h.market_value for h in self.holdings.values())
            max_mv = total_asset * pc
            remaining = max_mv - mv_now
            if remaining <= 0:
                continue

            buy_amount = min(
                self.cash * 0.95,
                total_asset * blind.RISK_PARAMS["single_pct"],
                remaining,
            )
            shares = int(buy_amount / cost / 100) * 100
            if shares < 100:
                continue
            amount = cost * shares
            comm = max(amount * blind.COMMISSION_RATE, 5)
            if amount + comm > self.cash:
                continue

            self.cash -= (amount + comm)
            self.holdings[code] = blind.Holding(
                code=code,
                name=pb.get("name", code),
                shares=shares,
                cost_price=cost,
                buy_date=date,
                buy_day_idx=day_idx,
                current_price=op,
                peak_price=op,
            )
            self.entry_scores[code] = float(pb.get("score", 0.0))
            self.entry_meta[code] = {
                "skills": list(pb.get("skills", [])),
                "route_regime": pb.get("route_regime", ""),
            }
            self.trades.append(
                blind.Trade(
                    date,
                    code,
                    pb.get("name", code),
                    "buy",
                    shares,
                    cost,
                    amount,
                    comm,
                    f"s={pb.get('score', 0)}|r={pb.get('route_regime', '')}|skills={','.join(pb.get('skills', []))}",
                )
            )

    def _sell(self, code, shares, price, date, reason):
        super()._sell(code, shares, price, date, reason)
        if code not in self.holdings:
            self.entry_meta.pop(code, None)


class SilentRoutedBlindAgentBacktest(RoutedBlindAgentBacktest):
    def _print_report(self):
        return


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    market_contexts: Dict[str, v4b.MarketContext],
    start_date: str,
    end_date: str,
    structure_params: Optional[Dict[str, object]] = None,
    silent: bool = True,
) -> Tuple[RoutedBlindAgentBacktest, trainer.Metrics]:
    previous_level = blind.logger.level
    blind.logger.setLevel(logging.WARNING)
    try:
        engine_cls = SilentRoutedBlindAgentBacktest if silent else RoutedBlindAgentBacktest
        engine = engine_cls(all_data, market_contexts, structure_params=structure_params)
        engine.run(start_date, end_date)
        metrics = trainer._summarize_engine(engine)
        return engine, metrics
    finally:
        blind.logger.setLevel(previous_level)


def save_outputs(
    engine: RoutedBlindAgentBacktest,
    metrics: trainer.Metrics,
    out_dir: Path,
    structure_params: Dict[str, object],
) -> None:
    trainer.save_backtest_outputs(
        engine,
        metrics,
        out_dir,
        extra={
            "strategy": "blind_agent_v4c_routed",
            "selected_params": copy.deepcopy(blind.RISK_PARAMS),
            "structure_params": copy.deepcopy(structure_params),
            "routing_params": copy.deepcopy(ROUTING_PARAMS),
            "route_counts": dict(engine.route_counter),
            "skill_counts": dict(engine.skill_counter),
            **engine.context_stats,
        },
    )


def main() -> None:
    args = parse_args()
    logger.info("Loading data cache...")
    all_data = fetch_all_a_share_daily("20251001", "20260410", cache_name="backtest_cache_2026ytd.pkl")
    logger.info("Building market contexts...")
    market_contexts = v4b.build_market_contexts(all_data)
    engine, metrics = run_backtest(
        all_data,
        market_contexts,
        args.start,
        args.end,
        structure_params=v4b.STRUCTURE_PARAMS,
        silent=False,
    )
    out_dir = Path(args.out_dir)
    save_outputs(engine, metrics, out_dir, v4b.STRUCTURE_PARAMS)
    logger.info(
        "V4c result | return=%+.2f%% max_dd=%.2f%% sharpe=%.2f avg_month=%+.2f%%",
        metrics.ret,
        metrics.max_dd,
        metrics.sharpe,
        metrics.avg_month,
    )
    logger.info("Results saved to %s", out_dir)
    with open(out_dir / "route_preview.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "route_counts": dict(engine.route_counter),
                "skill_counts": dict(engine.skill_counter),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
