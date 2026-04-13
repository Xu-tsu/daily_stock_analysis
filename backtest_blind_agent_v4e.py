"""
Blind agent V4e: adaptive strategy selection on top of V4b structure.

Key rules:
1. No future labels, no bull/bear regime tags.
2. Keep the V4b market-structure layer unchanged.
3. Score four native entry styles independently:
   - oversold
   - breakout
   - pullback
   - relay
4. Select the final strategy adaptively from:
   - same-day market context
   - candidate state
   - recent strategy-specific trade results

Usage:
    python backtest_blind_agent_v4e.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import backtest_blind_agent as blind
import backtest_blind_agent_v4b as v4b
import train_blind_agent as trainer
from backtest_2025 import fetch_all_a_share_daily


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

OUT_DIR = Path("data/backtest_blind_agent_v4e")
MODE_NAMES = ("oversold", "breakout", "pullback", "relay")
ADAPTIVE_PARAMS = {
    "hot_breakout_bonus": 6.0,
    "hot_relay_bonus": 8.0,
    "hot_pullback_bonus": 1.5,
    "hot_oversold_penalty": 6.0,
    "caution_oversold_bonus": 6.0,
    "caution_pullback_bonus": 4.0,
    "caution_breakout_penalty": 7.0,
    "caution_relay_penalty": 11.0,
    "normal_pullback_bonus": 4.0,
    "normal_breakout_bonus": 2.5,
    "normal_relay_penalty": 3.0,
    "mode_recent_loss_penalty": 5.0,
    "mode_recent_gain_cap": 4.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind agent V4e backtest")
    parser.add_argument("--start", default="2026-01-05", help="Backtest start date")
    parser.add_argument("--end", default="2026-04-10", help="Backtest end date")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    return parser.parse_args()


def mode_scores(ind: Dict[str, float]) -> Dict[str, float]:
    """Split the original unified score into four native strategies."""
    scores: Dict[str, float] = {}

    oversold = 0.0
    if ind["price_pos"] < 0.12:
        oversold += 25
    elif ind["price_pos"] < 0.20:
        oversold += 20
    elif ind["price_pos"] < 0.30:
        oversold += 15
    elif ind["price_pos"] < 0.40:
        oversold += 8
    elif ind["price_pos"] > 0.60:
        oversold -= 999
    if oversold > -100:
        if ind["bias20"] < -8:
            oversold += 18
        elif ind["bias20"] < -5:
            oversold += 13
        elif ind["bias20"] < -3:
            oversold += 8
        elif ind["bias20"] < 0:
            oversold += 3
        if ind["vol_ratio"] < 0.6:
            oversold += 12
        elif ind["vol_ratio"] < 0.8:
            oversold += 8
        elif ind["vol_ratio"] > 2.0:
            oversold -= 8
        if ind["macd_cross"]:
            oversold += 10
        elif ind["dif"] < 0 and ind["dif"] > ind["dea"]:
            oversold += 5
        if ind["rsi"] < 25:
            oversold += 10
        elif ind["rsi"] < 35:
            oversold += 5
        if 0 < ind["today_chg"] < 2:
            oversold += 5
        if ind["today_chg"] < -5:
            oversold -= 10
    scores["oversold"] = max(0.0, oversold)

    breakout = 0.0
    if ind["vol_breakout"]:
        breakout += 20
        if ind["ma5"] > ind["ma10"] > ind["ma20"]:
            breakout += 15
        elif ind["ma5"] > ind["ma10"]:
            breakout += 8
        if 2 < ind["today_chg"] < 7:
            breakout += 10
        if 5 < ind["turnover"] < 15:
            breakout += 8
        if ind["upper_ratio"] < 0.3:
            breakout += 5
        if ind["rsi"] > 75:
            breakout -= 15
        if ind["chg_5d"] > 20:
            breakout -= 15
    scores["breakout"] = max(0.0, breakout)

    pullback = 0.0
    if ind["ma5"] > ind["ma10"]:
        if ind["vol_pullback"]:
            pullback += 15
        if ind["macd_cross"]:
            pullback += 12
        elif ind["dif"] > ind["dea"]:
            pullback += 5
        if pullback > 0:
            if ind["ma5"] > ind["ma10"] > ind["ma20"]:
                pullback += 10
            if 0.25 < ind["price_pos"] < 0.55:
                pullback += 8
            if 30 < ind["rsi"] < 50:
                pullback += 8
            if 2 <= ind["consec_up"] <= 4:
                pullback += 5
            if ind["upper_ratio"] < 0.25:
                pullback += 5
    scores["pullback"] = max(0.0, pullback)

    relay = 0.0
    if ind["yest_limit"] and ind["consec_limit"] <= 2:
        relay += 20
        if -2 < ind["today_chg"] < 3:
            relay += 15
        if 5 < ind["turnover"] < 15:
            relay += 10
        if ind["upper_ratio"] < 0.25:
            relay += 8
        if ind["ma5"] > ind["ma20"]:
            relay += 5
        if ind["price_pos"] > 0.8:
            relay -= 20
        if ind["turnover"] > 20:
            relay -= 15
    scores["relay"] = max(0.0, relay)

    return scores


def _common_mode_penalty(ind: Dict[str, float], mode: str) -> float:
    penalty = 0.0
    if ind["today_limit"]:
        penalty -= 10.0 if mode == "relay" else 20.0
    if ind["chg_5d"] > 25:
        penalty -= 6.0 if mode == "relay" else 10.0
    return penalty


def _env_bonus(
    mode: str,
    ind: Dict[str, float],
    ctx: v4b.MarketContext,
    structure_params: Dict[str, object],
) -> float:
    hot = v4b.is_hot_context(ctx, structure_params)
    caution = v4b.is_caution_context(ctx, structure_params)
    bonus = 0.0

    if hot:
        if mode == "breakout":
            bonus += ADAPTIVE_PARAMS["hot_breakout_bonus"]
            if ind["vol_breakout"]:
                bonus += 2.0
        elif mode == "relay":
            bonus += ADAPTIVE_PARAMS["hot_relay_bonus"]
            if ind["yest_limit"]:
                bonus += 4.0
        elif mode == "pullback":
            bonus += ADAPTIVE_PARAMS["hot_pullback_bonus"]
        elif mode == "oversold":
            bonus -= ADAPTIVE_PARAMS["hot_oversold_penalty"]
    elif caution:
        if mode == "oversold":
            bonus += ADAPTIVE_PARAMS["caution_oversold_bonus"]
            if ind["price_pos"] < 0.2:
                bonus += 2.0
        elif mode == "pullback":
            bonus += ADAPTIVE_PARAMS["caution_pullback_bonus"]
        elif mode == "breakout":
            bonus -= ADAPTIVE_PARAMS["caution_breakout_penalty"]
        elif mode == "relay":
            bonus -= ADAPTIVE_PARAMS["caution_relay_penalty"]
    else:
        if mode == "pullback":
            bonus += ADAPTIVE_PARAMS["normal_pullback_bonus"]
            if ind["vol_pullback"]:
                bonus += 2.0
        elif mode == "breakout":
            bonus += ADAPTIVE_PARAMS["normal_breakout_bonus"]
        elif mode == "relay":
            bonus -= ADAPTIVE_PARAMS["normal_relay_penalty"]
        elif mode == "oversold" and ind["price_pos"] < 0.25:
            bonus += 1.5

    strong_edge = ctx.strong_count - ctx.weak_count
    if strong_edge >= 5 and mode in {"breakout", "relay"}:
        bonus += 2.0
    if ctx.weak_ratio >= 0.70 and mode in {"breakout", "relay"}:
        bonus -= 4.0
    if ctx.weak_ratio >= 0.70 and mode == "oversold":
        bonus += 2.0

    if mode == "breakout" and not (ind["ma5"] > ind["ma10"] > ind["ma20"]):
        bonus -= 4.0
    if mode == "pullback" and not (ind["ma5"] > ind["ma10"]):
        bonus -= 4.0
    if mode == "relay" and not ind["yest_limit"]:
        bonus -= 6.0

    return bonus


def _mode_perf_bonus(history: Dict[str, List[float]], mode: str) -> float:
    recent = history.get(mode, [])[-6:]
    if len(recent) < 2:
        return 0.0

    win_rate = sum(1 for value in recent if value > 0) / len(recent)
    avg_pnl = float(np.mean(recent))
    bonus = (win_rate - 0.5) * 8.0 + max(-ADAPTIVE_PARAMS["mode_recent_gain_cap"], min(ADAPTIVE_PARAMS["mode_recent_gain_cap"], avg_pnl))

    if len(recent) >= 2 and recent[-1] < 0 and recent[-2] < 0:
        bonus -= ADAPTIVE_PARAMS["mode_recent_loss_penalty"]
    return bonus


def adaptive_mode_decision(
    ind: Dict[str, float],
    ctx: v4b.MarketContext,
    structure_params: Dict[str, object],
    mode_history: Dict[str, List[float]],
) -> Tuple[str, float, Dict[str, float], Dict[str, float]]:
    raw_scores = mode_scores(ind)
    adjusted_scores: Dict[str, float] = {}
    for mode, raw in raw_scores.items():
        adjusted_scores[mode] = round(
            raw
            + _env_bonus(mode, ind, ctx, structure_params)
            + _mode_perf_bonus(mode_history, mode)
            + _common_mode_penalty(ind, mode),
            2,
        )

    best_mode = max(adjusted_scores.items(), key=lambda item: item[1])[0]
    return best_mode, max(0.0, adjusted_scores[best_mode]), raw_scores, adjusted_scores


class AdaptiveStrategyBlindAgentBacktest(v4b.StructuredBlindAgentBacktest):
    def __init__(
        self,
        all_data: Dict[str, pd.DataFrame],
        market_contexts: Dict[str, v4b.MarketContext],
        structure_params: Optional[Dict[str, object]] = None,
    ):
        super().__init__(all_data, market_contexts, structure_params=structure_params)
        self.mode_history: Dict[str, List[float]] = defaultdict(list)
        self.entry_modes: Dict[str, str] = {}
        self.entry_adjusted_scores: Dict[str, float] = {}
        self.mode_pick_counter: Counter[str] = Counter()
        self.mode_win_counter: Counter[str] = Counter()
        self.mode_loss_counter: Counter[str] = Counter()

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

        candidates = []
        for code, df in self.all_data.items():
            if code in self.holdings:
                continue
            rows = df.index[df["date"] == date]
            if len(rows) == 0:
                continue
            idx = int(rows[0])
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

            mode, adjusted_score, raw_scores, adjusted_scores = adaptive_mode_decision(
                ind,
                ctx,
                sp,
                self.mode_history,
            )
            if adjusted_score < threshold + extra_score_needed:
                continue
            candidates.append(
                {
                    "code": code,
                    "name": str(row.get("stock_name", code)),
                    "signal_close": price,
                    "score": adjusted_score,
                    "mode": mode,
                    "raw_mode_scores": raw_scores,
                    "adjusted_mode_scores": adjusted_scores,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        slots = blind.RISK_PARAMS["max_positions"] - len(self.holdings)
        if caution:
            slots = min(slots, sp["caution_slots"])
        for candidate in candidates[: max(1, slots)]:
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
            self.entry_modes[code] = str(pb.get("mode", "unknown"))
            self.entry_adjusted_scores[code] = float(pb.get("score", 0.0))
            self.mode_pick_counter[self.entry_modes[code]] += 1
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
                    f"mode={self.entry_modes[code]}|s={pb.get('score', 0):.1f}|pc={pc:.0%}",
                )
            )

    def _sell(self, code, shares, price, date, reason):
        holding = self.holdings.get(code)
        if not holding:
            return
        entry_mode = self.entry_modes.get(code, "unknown")
        super()._sell(code, shares, price, date, reason)
        if code not in self.holdings:
            pnl = (price * (1 - blind.SLIPPAGE_PCT) - holding.cost_price) / holding.cost_price * 100 if holding.cost_price > 0 else 0.0
            if entry_mode in MODE_NAMES:
                self.mode_history[entry_mode].append(float(pnl))
                if len(self.mode_history[entry_mode]) > 20:
                    self.mode_history[entry_mode].pop(0)
                if pnl > 0:
                    self.mode_win_counter[entry_mode] += 1
                else:
                    self.mode_loss_counter[entry_mode] += 1
            self.entry_modes.pop(code, None)
            self.entry_adjusted_scores.pop(code, None)


class SilentAdaptiveStrategyBlindAgentBacktest(AdaptiveStrategyBlindAgentBacktest):
    def _print_report(self):
        return


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    market_contexts: Dict[str, v4b.MarketContext],
    start_date: str,
    end_date: str,
    structure_params: Optional[Dict[str, object]] = None,
    silent: bool = True,
) -> Tuple[AdaptiveStrategyBlindAgentBacktest, trainer.Metrics]:
    previous_level = blind.logger.level
    blind.logger.setLevel(logging.WARNING)
    try:
        engine_cls = SilentAdaptiveStrategyBlindAgentBacktest if silent else AdaptiveStrategyBlindAgentBacktest
        engine = engine_cls(all_data, market_contexts, structure_params=structure_params)
        engine.run(start_date, end_date)
        metrics = trainer._summarize_engine(engine)
        return engine, metrics
    finally:
        blind.logger.setLevel(previous_level)


def save_outputs(
    engine: AdaptiveStrategyBlindAgentBacktest,
    metrics: trainer.Metrics,
    out_dir: Path,
    structure_params: Dict[str, object],
) -> None:
    trainer.save_backtest_outputs(
        engine,
        metrics,
        out_dir,
        extra={
            "strategy": "blind_agent_v4e_adaptive_mode",
            "selected_params": copy.deepcopy(blind.RISK_PARAMS),
            "structure_params": copy.deepcopy(structure_params),
            "adaptive_params": copy.deepcopy(ADAPTIVE_PARAMS),
            "mode_picks": dict(engine.mode_pick_counter),
            "mode_wins": dict(engine.mode_win_counter),
            "mode_losses": dict(engine.mode_loss_counter),
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
        "V4e result | return=%+.2f%% max_dd=%.2f%% sharpe=%.2f avg_month=%+.2f%%",
        metrics.ret,
        metrics.max_dd,
        metrics.sharpe,
        metrics.avg_month,
    )
    logger.info("Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
