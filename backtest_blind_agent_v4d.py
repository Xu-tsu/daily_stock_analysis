"""
Blind agent V4d: multi-timeframe stock analysis on top of V4b structure.

Key rules:
1. No future labels, no bull/bear regime tags.
2. Keep the V4b market-structure layer unchanged.
3. Upgrade stock analysis from pure daily signals to:
   - MA5 short-term execution
   - MA30 medium-term trend
   - monthly trend/position

Usage:
    python backtest_blind_agent_v4d.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import backtest_blind_agent as blind
import backtest_blind_agent_v4b as v4b
import train_blind_agent as trainer
from backtest_2025 import fetch_all_a_share_daily


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

OUT_DIR = Path("data/backtest_blind_agent_v4d")
MTF_PARAMS = {
    "ma5_over_ma30_bonus": 2.0,
    "ma30_slope_bonus": 2.0,
    "ma30_down_penalty": 6.0,
    "monthly_trend_bonus": 3.0,
    "monthly_breakout_bonus": 2.0,
    "monthly_bear_penalty": 8.0,
    "monthly_too_hot_penalty": 3.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind agent V4d backtest")
    parser.add_argument("--start", default="2026-01-05", help="Backtest start date")
    parser.add_argument("--end", default="2026-04-10", help="Backtest end date")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    return parser.parse_args()


def calc_multiframe_indicators(df: pd.DataFrame, idx: int) -> Optional[Dict[str, float]]:
    """Daily + MA30 + monthly features using data up to idx only."""
    base = blind.calc_indicators(df, idx)
    if base is None:
        return None

    closes = df.iloc[: idx + 1]["close"].astype(float).values
    price = float(closes[-1])

    ma30_window = closes[-30:] if len(closes) >= 30 else closes
    ma30 = float(np.mean(ma30_window))
    if len(closes) >= 31:
        ma30_prev = float(np.mean(closes[-31:-1]))
    else:
        ma30_prev = ma30
    ma30_slope = (ma30 - ma30_prev) / ma30_prev * 100 if ma30_prev > 0 else 0.0
    bias30 = (price - ma30) / ma30 * 100 if ma30 > 0 else 0.0

    sub = df.iloc[: idx + 1][["date", "open", "high", "low", "close", "volume"]].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    monthly = (
        sub.groupby(sub["date"].dt.to_period("M"))
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .reset_index(drop=True)
    )

    mc = monthly["close"].astype(float).values
    mh = monthly["high"].astype(float).values
    ml = monthly["low"].astype(float).values

    mma3 = float(np.mean(mc[-3:])) if len(mc) >= 3 else float(np.mean(mc))
    mma6 = float(np.mean(mc[-6:])) if len(mc) >= 6 else float(np.mean(mc))
    month_chg = (mc[-1] / mc[-2] - 1) * 100 if len(mc) >= 2 and mc[-2] > 0 else 0.0
    prev_month_chg = (mc[-2] / mc[-3] - 1) * 100 if len(mc) >= 3 and mc[-3] > 0 else 0.0
    month_2up = month_chg > 0 and prev_month_chg > 0

    month_trend_up = bool(mc[-1] >= mma3 and mma3 >= mma6 * 0.995)
    month_bearish_stack = bool(mc[-1] < mma3 and mma3 < mma6 * 0.995)

    if len(mh) >= 2:
        prior_highs = mh[:-1]
    else:
        prior_highs = mh
    month_breakout = bool(len(prior_highs) > 0 and mc[-1] >= float(np.max(prior_highs[-3:])) * 0.985)

    high6 = float(np.max(mh[-6:]))
    low6 = float(np.min(ml[-6:]))
    month_pos6 = (mc[-1] - low6) / (high6 - low6) if high6 > low6 else 0.5

    month_above_prev = bool(len(mc) >= 2 and mc[-1] >= mc[-2])
    ma5_over_ma30 = bool(base["ma5"] > ma30)

    ind = dict(base)
    ind.update(
        {
            "ma30": ma30,
            "ma30_slope": ma30_slope,
            "bias30": bias30,
            "ma5_over_ma30": ma5_over_ma30,
            "month_ma3": mma3,
            "month_ma6": mma6,
            "month_chg": month_chg,
            "prev_month_chg": prev_month_chg,
            "month_2up": month_2up,
            "month_trend_up": month_trend_up,
            "month_bearish_stack": month_bearish_stack,
            "month_breakout": month_breakout,
            "month_pos6": month_pos6,
            "month_above_prev": month_above_prev,
        }
    )
    return ind


def unified_score_multiframe(ind: Dict[str, float]) -> float:
    """Use MA30/monthly signals as a quality filter, not a separate alpha source."""
    base = float(blind.unified_score(ind))
    if base <= 0:
        return 0.0

    breakout_setup = bool(ind["vol_breakout"])
    pullback_setup = bool(ind["vol_pullback"] or (ind["ma5"] > ind["ma10"] and ind["macd_cross"]))
    oversold_setup = bool(ind["price_pos"] < 0.30 and ind["bias20"] < -3.0)

    if (
        not oversold_setup
        and ind["ma5"] < ind["ma30"]
        and ind["ma30_slope"] < -0.4
        and ind["month_bearish_stack"]
        and ind["month_chg"] < 0
    ):
        return max(0.0, round(base - 18.0, 2))

    overlay = 0.0

    if ind["ma5_over_ma30"]:
        overlay += MTF_PARAMS["ma5_over_ma30_bonus"]
    elif breakout_setup:
        overlay -= 5.0

    if ind["ma30_slope"] > 0.4:
        overlay += MTF_PARAMS["ma30_slope_bonus"]
    elif ind["ma30_slope"] < -0.4:
        overlay -= MTF_PARAMS["ma30_down_penalty"]

    if -2.0 <= ind["bias30"] <= 12.0:
        overlay += 3.0
    elif ind["bias30"] < -8.0:
        overlay -= 6.0

    if ind["month_trend_up"]:
        overlay += MTF_PARAMS["monthly_trend_bonus"]
    elif ind["month_bearish_stack"]:
        overlay -= MTF_PARAMS["monthly_bear_penalty"]

    if ind["month_breakout"]:
        overlay += MTF_PARAMS["monthly_breakout_bonus"]
    if ind["month_2up"]:
        overlay += 1.5
    if ind["month_chg"] < -4.0:
        overlay -= 5.0

    if ind["month_pos6"] > 0.88 and ind["month_chg"] > 8.0:
        overlay -= MTF_PARAMS["monthly_too_hot_penalty"]

    if breakout_setup and not (ind["ma5_over_ma30"] and ind["month_trend_up"]):
        overlay -= 8.0
    if pullback_setup and not (ind["ma5_over_ma30"] or ind["month_trend_up"]):
        overlay -= 6.0
    if oversold_setup and ind["month_bearish_stack"] and ind["month_chg"] < -5.0:
        overlay -= 7.0

    return round(max(0.0, min(100.0, base + overlay)), 2)


def build_market_contexts(all_data: Dict[str, pd.DataFrame]) -> Dict[str, v4b.MarketContext]:
    # Keep the V4b structure layer unchanged; only upgrade stock-level analysis.
    return v4b.build_market_contexts(all_data)


class MultiTimeframeBlindAgentBacktest(v4b.StructuredBlindAgentBacktest):
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
            ind = calc_multiframe_indicators(df, idx)
            if ind is None:
                continue
            score = unified_score_multiframe(ind)
            if score < threshold + extra_score_needed:
                continue
            cands.append(
                {
                    "code": code,
                    "name": str(row.get("stock_name", code)),
                    "signal_close": price,
                    "score": score,
                }
            )

        cands.sort(key=lambda item: item["score"], reverse=True)
        slots = blind.RISK_PARAMS["max_positions"] - len(self.holdings)
        if caution:
            slots = min(slots, sp["caution_slots"])
        for candidate in cands[: max(1, slots)]:
            self._pending_buys.append(candidate)


class SilentMultiTimeframeBlindAgentBacktest(MultiTimeframeBlindAgentBacktest):
    def _print_report(self):
        return


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    market_contexts: Dict[str, v4b.MarketContext],
    start_date: str,
    end_date: str,
    structure_params: Optional[Dict[str, object]] = None,
    silent: bool = True,
) -> Tuple[MultiTimeframeBlindAgentBacktest, trainer.Metrics]:
    previous_level = blind.logger.level
    blind.logger.setLevel(logging.WARNING)
    try:
        engine_cls = SilentMultiTimeframeBlindAgentBacktest if silent else MultiTimeframeBlindAgentBacktest
        engine = engine_cls(all_data, market_contexts, structure_params=structure_params)
        engine.run(start_date, end_date)
        metrics = trainer._summarize_engine(engine)
        return engine, metrics
    finally:
        blind.logger.setLevel(previous_level)


def save_outputs(
    engine: MultiTimeframeBlindAgentBacktest,
    metrics: trainer.Metrics,
    out_dir: Path,
    structure_params: Dict[str, object],
) -> None:
    trainer.save_backtest_outputs(
        engine,
        metrics,
        out_dir,
        extra={
            "strategy": "blind_agent_v4d_multitimeframe",
            "selected_params": copy.deepcopy(blind.RISK_PARAMS),
            "structure_params": copy.deepcopy(structure_params),
            "multiframe_params": copy.deepcopy(MTF_PARAMS),
            **engine.context_stats,
        },
    )


def main() -> None:
    args = parse_args()
    logger.info("Loading data cache...")
    all_data = fetch_all_a_share_daily("20251001", "20260410", cache_name="backtest_cache_2026ytd.pkl")
    logger.info("Building multi-timeframe market contexts...")
    market_contexts = build_market_contexts(all_data)
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
        "V4d result | return=%+.2f%% max_dd=%.2f%% sharpe=%.2f avg_month=%+.2f%%",
        metrics.ret,
        metrics.max_dd,
        metrics.sharpe,
        metrics.avg_month,
    )
    logger.info("Results saved to %s", out_dir)
    with open(out_dir / "market_context_preview.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                date: {
                    "cnt50": ctx.cnt50,
                    "top3_avg": ctx.top3_avg,
                    "strong_count": ctx.strong_count,
                    "weak_count": ctx.weak_count,
                    "weak_ratio": ctx.weak_ratio,
                }
                for date, ctx in market_contexts.items()
                if args.start <= date <= args.end
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
