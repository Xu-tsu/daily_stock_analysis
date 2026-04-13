"""
Blind agent V4f: light adaptive strategy overlay on top of V4b.

This variant keeps V4b as the base alpha source and only uses adaptive
strategy selection as a small overlay, so the selector cannot fully overturn
the stronger base model.

Usage:
    python backtest_blind_agent_v4f.py
"""

from __future__ import annotations

import argparse
import copy
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import backtest_blind_agent as blind
import backtest_blind_agent_v4b as v4b
import backtest_blind_agent_v4e as v4e
import train_blind_agent as trainer
from backtest_2025 import fetch_all_a_share_daily


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

OUT_DIR = Path("data/backtest_blind_agent_v4f")
OVERLAY_PARAMS = {
    "env_scale": 0.55,
    "perf_scale": 0.45,
    "overlay_pos_cap": 6.0,
    "overlay_neg_cap": -2.5,
    "weak_raw_penalty": 1.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind agent V4f backtest")
    parser.add_argument("--start", default="2026-01-05", help="Backtest start date")
    parser.add_argument("--end", default="2026-04-10", help="Backtest end date")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    return parser.parse_args()


def adaptive_overlay_decision(
    ind: Dict[str, float],
    ctx: v4b.MarketContext,
    structure_params: Dict[str, object],
    mode_history: Dict[str, List[float]],
) -> Tuple[str, float, Dict[str, float], Dict[str, float]]:
    raw_scores = v4e.mode_scores(ind)
    base_raw_best = max(raw_scores.values()) if raw_scores else 0.0
    adjusted_scores: Dict[str, float] = {}

    for mode, raw in raw_scores.items():
        adjusted = raw
        adjusted += v4e._env_bonus(mode, ind, ctx, structure_params) * OVERLAY_PARAMS["env_scale"]
        adjusted += v4e._mode_perf_bonus(mode_history, mode) * OVERLAY_PARAMS["perf_scale"]
        adjusted += v4e._common_mode_penalty(ind, mode)
        if raw < 15:
            adjusted -= OVERLAY_PARAMS["weak_raw_penalty"]
        adjusted_scores[mode] = round(adjusted, 2)

    best_mode = max(adjusted_scores.items(), key=lambda item: item[1])[0]
    overlay = adjusted_scores[best_mode] - base_raw_best
    overlay = max(OVERLAY_PARAMS["overlay_neg_cap"], min(OVERLAY_PARAMS["overlay_pos_cap"], overlay))

    final_score = round(max(0.0, min(100.0, blind.unified_score(ind) + overlay)), 2)
    return best_mode, final_score, raw_scores, adjusted_scores


class AdaptiveOverlayBlindAgentBacktest(v4b.StructuredBlindAgentBacktest):
    def __init__(
        self,
        all_data: Dict[str, pd.DataFrame],
        market_contexts: Dict[str, v4b.MarketContext],
        structure_params: Optional[Dict[str, object]] = None,
    ):
        super().__init__(all_data, market_contexts, structure_params=structure_params)
        self.mode_history: Dict[str, List[float]] = defaultdict(list)
        self.entry_modes: Dict[str, str] = {}
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

            mode, score, raw_scores, adjusted_scores = adaptive_overlay_decision(
                ind, ctx, sp, self.mode_history
            )
            if score < threshold + extra_score_needed:
                continue

            candidates.append(
                {
                    "code": code,
                    "name": str(row.get("stock_name", code)),
                    "signal_close": price,
                    "score": score,
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
            if entry_mode in v4e.MODE_NAMES:
                self.mode_history[entry_mode].append(float(pnl))
                if len(self.mode_history[entry_mode]) > 20:
                    self.mode_history[entry_mode].pop(0)
                if pnl > 0:
                    self.mode_win_counter[entry_mode] += 1
                else:
                    self.mode_loss_counter[entry_mode] += 1
            self.entry_modes.pop(code, None)


class SilentAdaptiveOverlayBlindAgentBacktest(AdaptiveOverlayBlindAgentBacktest):
    def _print_report(self):
        return


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    market_contexts: Dict[str, v4b.MarketContext],
    start_date: str,
    end_date: str,
    structure_params: Optional[Dict[str, object]] = None,
    silent: bool = True,
) -> Tuple[AdaptiveOverlayBlindAgentBacktest, trainer.Metrics]:
    previous_level = blind.logger.level
    blind.logger.setLevel(logging.WARNING)
    try:
        engine_cls = SilentAdaptiveOverlayBlindAgentBacktest if silent else AdaptiveOverlayBlindAgentBacktest
        engine = engine_cls(all_data, market_contexts, structure_params=structure_params)
        engine.run(start_date, end_date)
        metrics = trainer._summarize_engine(engine)
        return engine, metrics
    finally:
        blind.logger.setLevel(previous_level)


def save_outputs(
    engine: AdaptiveOverlayBlindAgentBacktest,
    metrics: trainer.Metrics,
    out_dir: Path,
    structure_params: Dict[str, object],
) -> None:
    trainer.save_backtest_outputs(
        engine,
        metrics,
        out_dir,
        extra={
            "strategy": "blind_agent_v4f_adaptive_overlay",
            "selected_params": copy.deepcopy(blind.RISK_PARAMS),
            "structure_params": copy.deepcopy(structure_params),
            "overlay_params": copy.deepcopy(OVERLAY_PARAMS),
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
        "V4f result | return=%+.2f%% max_dd=%.2f%% sharpe=%.2f avg_month=%+.2f%%",
        metrics.ret,
        metrics.max_dd,
        metrics.sharpe,
        metrics.avg_month,
    )
    logger.info("Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
