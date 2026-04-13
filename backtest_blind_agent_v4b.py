"""
Blind agent V4b: real-trading-like structural upgrade.

Key rules:
1. No future labels, no bull/bear regime tags.
2. Use only same-day cross-sectional breadth as environment context.
3. Weak environments reduce trade frequency automatically.
4. Strong environments let profitable winners run a bit longer.

Usage:
    python backtest_blind_agent_v4b.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import backtest_blind_agent as blind
import train_blind_agent as trainer
from backtest_2025 import fetch_all_a_share_daily


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

OUT_DIR = Path("data/backtest_blind_agent_v4b")
STRUCTURE_PARAMS = {
    "cold_weak_ratio": 0.83,
    "cold_cnt50_max": 12,
    "cold_top3_max": 56.0,
    "panic_cnt50_override": 22,
    "panic_top3_override": 60.0,
    "caution_weak_ratio": 0.99,
    "caution_cnt50_max": 0,
    "caution_threshold_add": 0,
    "caution_score_add": 0,
    "caution_slots": 1,
    "hot_cnt50": 35,
    "hot_strong_minus_weak_min": 5,
    "hot_hold_bonus": 1,
    "hot_trail_dd_bonus": 0.5,
    "hot_tp_full_bonus": 2.0,
    "strong_entry_score": 58,
}


@dataclass
class MarketContext:
    date: str
    cnt40: int
    cnt50: int
    cnt60: int
    top3_avg: float
    top5_avg: float
    strong_count: int
    weak_count: int
    breakout_count: int
    pullback_count: int
    weak_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind agent V4b backtest")
    parser.add_argument("--start", default="2026-01-05", help="Backtest start date")
    parser.add_argument("--end", default="2026-04-10", help="Backtest end date")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    return parser.parse_args()


def build_market_contexts(all_data: Dict[str, pd.DataFrame]) -> Dict[str, MarketContext]:
    dates_set = set()
    for df in all_data.values():
        dates_set.update(df["date"].tolist())

    pmin, pmax = blind.RISK_PARAMS["price_range"]
    min_turnover = blind.RISK_PARAMS["min_turnover"]
    contexts: Dict[str, MarketContext] = {}

    for date in sorted(dates_set):
        scores: List[float] = []
        cnt40 = cnt50 = cnt60 = 0
        strong_count = weak_count = breakout_count = pullback_count = 0

        for code, df in all_data.items():
            rows = df.index[df["date"] == date]
            if len(rows) == 0:
                continue
            idx = int(rows[0])
            row = df.iloc[idx]
            ind = blind.calc_indicators(df, idx)
            if ind is None:
                continue

            if ind["today_chg"] >= 3:
                strong_count += 1
            if ind["today_chg"] <= -3:
                weak_count += 1
            if ind["vol_breakout"]:
                breakout_count += 1
            if ind["vol_pullback"]:
                pullback_count += 1

            price = float(row["close"])
            turnover = float(row.get("turnover_rate", 0) or 0)
            if price < pmin or price > pmax or turnover < min_turnover:
                continue

            score = blind.unified_score(ind)
            scores.append(score)
            if score >= 40:
                cnt40 += 1
            if score >= 50:
                cnt50 += 1
            if score >= 60:
                cnt60 += 1

        scores.sort(reverse=True)
        top3 = scores[:3]
        top5 = scores[:5]
        strong_weak_total = strong_count + weak_count
        contexts[date] = MarketContext(
            date=date,
            cnt40=cnt40,
            cnt50=cnt50,
            cnt60=cnt60,
            top3_avg=round(sum(top3) / len(top3), 2) if top3 else 0.0,
            top5_avg=round(sum(top5) / len(top5), 2) if top5 else 0.0,
            strong_count=strong_count,
            weak_count=weak_count,
            breakout_count=breakout_count,
            pullback_count=pullback_count,
            weak_ratio=round(weak_count / strong_weak_total, 4) if strong_weak_total else 0.0,
        )

    return contexts


def is_hot_context(ctx: MarketContext, structure_params: Dict[str, object]) -> bool:
    return (
        ctx.cnt50 >= structure_params["hot_cnt50"]
        or (ctx.strong_count - ctx.weak_count >= structure_params["hot_strong_minus_weak_min"] and ctx.cnt60 >= 2)
    )


def is_panic_override(ctx: MarketContext, structure_params: Dict[str, object]) -> bool:
    return (
        ctx.cnt50 >= structure_params["panic_cnt50_override"]
        and ctx.top3_avg >= structure_params["panic_top3_override"]
    )


def is_cold_block(ctx: MarketContext, structure_params: Dict[str, object]) -> bool:
    return (
        ctx.weak_ratio >= structure_params["cold_weak_ratio"]
        and ctx.cnt50 <= structure_params["cold_cnt50_max"]
        and ctx.top3_avg <= structure_params["cold_top3_max"]
        and not is_panic_override(ctx, structure_params)
    )


def is_caution_context(ctx: MarketContext, structure_params: Dict[str, object]) -> bool:
    return (
        ctx.weak_ratio >= structure_params["caution_weak_ratio"]
        and ctx.cnt50 <= structure_params["caution_cnt50_max"]
        and not is_hot_context(ctx, structure_params)
        and not is_panic_override(ctx, structure_params)
    )


class StructuredBlindAgentBacktest(blind.BlindAgentBacktest):
    def __init__(
        self,
        all_data: Dict[str, pd.DataFrame],
        market_contexts: Dict[str, MarketContext],
        structure_params: Optional[Dict[str, object]] = None,
    ):
        super().__init__(all_data)
        self.market_contexts = market_contexts
        self.structure_params = copy.deepcopy(STRUCTURE_PARAMS)
        if structure_params:
            self.structure_params.update(copy.deepcopy(structure_params))
        self.entry_scores: Dict[str, float] = {}
        self.context_stats = {
            "cold_block_days": 0,
            "caution_days": 0,
            "hot_days": 0,
            "blocked_buy_days": 0,
        }

    def _ctx(self, date: str) -> MarketContext:
        return self.market_contexts[date]

    def _review_sells(self, date, day_idx):
        rp = blind.RISK_PARAMS
        sp = self.structure_params
        ctx = self._ctx(date)
        market_hot = is_hot_context(ctx, sp)
        if market_hot:
            self.context_stats["hot_days"] += 1

        for code, h in list(self.holdings.items()):
            if h.buy_day_idx == day_idx:
                continue

            pnl = h.pnl_pct
            hold = day_idx - h.buy_day_idx
            pk = (h.peak_price - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            dd = (h.peak_price - h.current_price) / h.peak_price * 100 if h.peak_price > 0 else 0

            entry_score = self.entry_scores.get(code, 0.0)
            trail_dd = rp["trail_dd"]
            tp_full = rp["tp_full"]
            max_hold = rp["hold_max"]

            if market_hot and entry_score >= sp["strong_entry_score"] and pnl > 0:
                max_hold += sp["hot_hold_bonus"]
                trail_dd += sp["hot_trail_dd_bonus"]
                tp_full += sp["hot_tp_full_bonus"]

            if pnl <= rp["stop_loss"]:
                self._pending_sells.append((code, h.shares, f"SL{pnl:.1f}%"))
                continue

            if pk >= rp["trail_trigger"] and dd >= trail_dd:
                self._pending_sells.append((code, h.shares, f"TRAIL pk{pk:.1f}% dd{dd:.1f}%"))
                continue

            if not h.partial_sold and pnl >= rp["tp_half"]:
                half = max(100, (h.shares // 200) * 100)
                if half < h.shares:
                    self._pending_sells.append((code, half, f"TP_HALF{pnl:.1f}%"))
                    h.partial_sold = True
                    continue

            if pnl >= tp_full:
                self._pending_sells.append((code, h.shares, f"TP_FULL{pnl:.1f}%"))
                continue

            if pnl > 2:
                max_hold += 2
            if hold >= max_hold and pnl < 1:
                self._pending_sells.append((code, h.shares, f"EXPIRE{hold}d"))

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
        if is_cold_block(ctx, sp):
            self.context_stats["cold_block_days"] += 1
            self.context_stats["blocked_buy_days"] += 1
            return

        caution = is_caution_context(ctx, sp)
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
            score = blind.unified_score(ind)
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
                    f"s={pb.get('score', 0)}|pc={pc:.0%}",
                )
            )

    def _sell(self, code, shares, price, date, reason):
        super()._sell(code, shares, price, date, reason)
        if code not in self.holdings:
            self.entry_scores.pop(code, None)


class SilentStructuredBlindAgentBacktest(StructuredBlindAgentBacktest):
    def _print_report(self):
        return


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    market_contexts: Dict[str, MarketContext],
    start_date: str,
    end_date: str,
    structure_params: Optional[Dict[str, object]] = None,
    silent: bool = True,
) -> Tuple[StructuredBlindAgentBacktest, trainer.Metrics]:
    previous_level = blind.logger.level
    blind.logger.setLevel(logging.WARNING)
    try:
        engine_cls = SilentStructuredBlindAgentBacktest if silent else StructuredBlindAgentBacktest
        engine = engine_cls(all_data, market_contexts, structure_params=structure_params)
        engine.run(start_date, end_date)
        metrics = trainer._summarize_engine(engine)
        return engine, metrics
    finally:
        blind.logger.setLevel(previous_level)


def save_outputs(
    engine: StructuredBlindAgentBacktest,
    metrics: trainer.Metrics,
    out_dir: Path,
    structure_params: Dict[str, object],
) -> None:
    trainer.save_backtest_outputs(
        engine,
        metrics,
        out_dir,
        extra={
            "strategy": "blind_agent_v4b_structured",
            "selected_params": copy.deepcopy(blind.RISK_PARAMS),
            "structure_params": copy.deepcopy(structure_params),
            **engine.context_stats,
        },
    )


def main() -> None:
    args = parse_args()
    logger.info("Loading data cache...")
    all_data = fetch_all_a_share_daily("20251001", "20260410", cache_name="backtest_cache_2026ytd.pkl")
    logger.info("Building market contexts...")
    market_contexts = build_market_contexts(all_data)
    engine, metrics = run_backtest(
        all_data,
        market_contexts,
        args.start,
        args.end,
        structure_params=STRUCTURE_PARAMS,
        silent=False,
    )
    out_dir = Path(args.out_dir)
    save_outputs(engine, metrics, out_dir, STRUCTURE_PARAMS)
    logger.info(
        "V4b result | return=%+.2f%% max_dd=%.2f%% sharpe=%.2f avg_month=%+.2f%%",
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
