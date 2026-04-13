"""
Blind agent backtest with online distilled expert-skill bias.

Rule of use:
1. Keep the same no-future blind-agent engine and tuned risk params.
2. Only use expert trades completed before the current backtest date.
3. Convert those historical expert trades into archetype/sector patterns.
4. Use the patterns as score bonuses, not unconditional buy signals.

Usage:
    python backtest_blind_agent_distilled.py
    python backtest_blind_agent_distilled.py --min-samples 2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import backtest_blind_agent as blind
import train_blind_agent as trainer
from backtest_2025 import fetch_all_a_share_daily
from distill_expert_skills import (
    DEFAULT_REPORT_DIR,
    derive_feature_labels,
    enrich_completed_trades,
    load_price_history,
    load_trade_rows,
    pair_completed_trades,
    rank_patterns,
    classify_archetype,
)
from event_signal import get_stock_sector


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

OUT_DIR = Path("data/backtest_blind_agent_distilled")
TEST_WINDOW = ("2026-01-05", "2026-04-10")
EXPERT_SOURCE = "broker_export"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind agent + online expert skill bias")
    parser.add_argument("--start", default=TEST_WINDOW[0], help="Backtest start date, default 2026-01-05")
    parser.add_argument("--end", default=TEST_WINDOW[1], help="Backtest end date, default 2026-04-10")
    parser.add_argument("--expert-source", default=EXPERT_SOURCE, help="trade_log expert source, default broker_export")
    parser.add_argument("--min-samples", type=int, default=3, help="Minimum samples to keep an expert pattern")
    parser.add_argument("--top-n", type=int, default=4, help="Maximum active expert patterns")
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output directory for summary/trades/daily",
    )
    return parser.parse_args()


@dataclass
class SkillMatch:
    bonus: float
    effective_score: float
    base_score: float
    archetype: str
    sector: str
    pattern_name: str
    pattern_stats: Dict[str, object]


class ExpertSkillOverlay:
    def __init__(
        self,
        all_data: Dict[str, pd.DataFrame],
        expert_source: str,
        min_samples: int,
        top_n: int,
    ) -> None:
        self.all_data = all_data
        self.expert_source = expert_source
        self.min_samples = min_samples
        self.top_n = top_n
        self.pattern_days: Dict[str, List[Dict[str, object]]] = {}
        self.available_from: Optional[str] = None
        self.expert_trade_count = 0
        self._bootstrap()

    def _bootstrap(self) -> None:
        rows = load_trade_rows(self.expert_source)
        completed = pair_completed_trades(rows)
        expert_history = load_price_history(completed, "expert_trade_history.pkl", DEFAULT_REPORT_DIR)
        enriched = enrich_completed_trades(completed, expert_history)
        self.expert_trade_count = len(enriched)
        sell_dates = sorted({trade.sell_date for trade in enriched})
        self.available_from = sell_dates[0] if sell_dates else None

        logger.info(
            "Expert overlay bootstrap | source=%s enriched=%s first_skill_day=%s",
            self.expert_source,
            len(enriched),
            self.available_from or "n/a",
        )

        for date in sell_dates:
            available_trades = [trade for trade in enriched if trade.sell_date < date]
            if len(available_trades) < self.min_samples:
                self.pattern_days[date] = []
                continue
            self.pattern_days[date] = rank_patterns(
                available_trades,
                min_samples=self.min_samples,
                top_n=self.top_n,
            )

    def get_patterns(self, date: str) -> List[Dict[str, object]]:
        eligible = [day for day in self.pattern_days if day < date]
        if not eligible:
            return []
        latest_day = eligible[-1]
        return self.pattern_days.get(latest_day, [])

    @staticmethod
    def _bonus_from_pattern(pattern: Dict[str, object], exact_sector: bool) -> float:
        avg_pnl = max(0.0, float(pattern.get("avg_pnl", 0.0)))
        win_edge = max(0.0, float(pattern.get("win_rate", 0.0)) - 55.0)
        pf_edge = max(0.0, float(pattern.get("profit_factor", 0.0)) - 1.0)
        sample_edge = min(float(pattern.get("samples", 0.0)), 10.0)
        bonus = avg_pnl * 1.1 + win_edge * 0.07 + pf_edge * 0.30 + sample_edge * 0.20
        if exact_sector and pattern.get("sector") not in {None, "", "未知"}:
            bonus += 1.0
        return round(min(10.0, bonus), 2)

    def match(self, code: str, ind: Dict[str, float], base_score: float, date: str) -> Optional[SkillMatch]:
        patterns = self.get_patterns(date)
        if not patterns:
            return None

        sector = get_stock_sector(code) or "未知"
        labels = derive_feature_labels(ind, sector)
        archetype = classify_archetype(ind, labels)

        best_match = None
        for pattern in patterns:
            if pattern["archetype"] != archetype:
                continue
            exact_sector = pattern["sector"] == sector and sector != "未知"
            broad_sector = pattern["sector"] == "未知"
            if not exact_sector and not broad_sector:
                continue

            bonus = self._bonus_from_pattern(pattern, exact_sector=exact_sector)
            if bonus <= 0:
                continue
            candidate = SkillMatch(
                bonus=bonus,
                effective_score=base_score + bonus,
                base_score=base_score,
                archetype=archetype,
                sector=sector,
                pattern_name=f"{pattern['archetype']}@{pattern['sector']}",
                pattern_stats=pattern,
            )
            if best_match is None or candidate.bonus > best_match.bonus:
                best_match = candidate

        return best_match


class DistilledBlindAgentBacktest(blind.BlindAgentBacktest):
    def __init__(self, all_data: Dict[str, pd.DataFrame], overlay: ExpertSkillOverlay):
        super().__init__(all_data)
        self.overlay = overlay
        self.skill_match_days = 0
        self.skill_match_candidates = 0
        self.skill_buy_count = 0
        self.skill_pattern_counter: Counter[str] = Counter()

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

        threshold = self._state.buy_threshold
        pmin, pmax = blind.RISK_PARAMS["price_range"]
        min_to = blind.RISK_PARAMS["min_turnover"]

        cands = []
        matched_today = 0
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
            to = float(row.get("turnover_rate", 0) or 0)
            if to < min_to:
                continue
            ind = blind.calc_indicators(df, idx)
            if ind is None:
                continue

            base_score = blind.unified_score(ind)
            skill_match = self.overlay.match(code, ind, base_score, date)
            effective_score = base_score
            if skill_match is not None:
                # Skills are a bias layer, not a free pass for weak setups.
                if base_score < max(32, threshold - 8):
                    continue
                effective_score = skill_match.effective_score
                matched_today += 1
            if effective_score < threshold:
                continue

            cands.append(
                {
                    "code": code,
                    "name": str(row.get("stock_name", code)),
                    "signal_close": price,
                    "score": round(effective_score, 2),
                    "base_score": round(base_score, 2),
                    "skill_match": skill_match,
                }
            )

        if matched_today:
            self.skill_match_days += 1
            self.skill_match_candidates += matched_today

        cands.sort(key=lambda x: (x["score"], x["base_score"]), reverse=True)
        slots = blind.RISK_PARAMS["max_positions"] - len(self.holdings)
        for candidate in cands[: max(1, slots)]:
            skill_match = candidate.get("skill_match")
            if skill_match is not None:
                self.skill_buy_count += 1
                self.skill_pattern_counter[skill_match.pattern_name] += 1
            self._pending_buys.append(candidate)

    def _exec_pending_buys(self, date, day_idx):
        buys = self._pending_buys[:]
        self._pending_buys = []
        if not buys:
            return
        slots = blind.RISK_PARAMS["max_positions"] - len(self.holdings)
        if slots <= 0:
            return

        pc = self._state.position_coeff
        total_target = self.cash * pc
        each = min(total_target / max(1, slots), (self.cash + sum(h.market_value for h in self.holdings.values())) * blind.RISK_PARAMS["single_pct"])

        for pb in buys[:slots]:
            code = pb["code"]
            row = self._get_row(code, date)
            if row is None:
                continue
            op = float(row["open"])
            if op <= 0:
                continue
            cost = op * (1 + blind.SLIPPAGE_PCT)
            remaining = blind.RISK_PARAMS["max_positions"] - len(self.holdings)
            buy_amount = min(each, self.cash * 0.98 / max(1, remaining))
            shares = int(buy_amount / cost / 100) * 100
            if shares < 100:
                continue
            amount = cost * shares
            comm = max(amount * blind.COMMISSION_RATE, 5)
            if amount + comm > self.cash:
                continue

            skill_match: Optional[SkillMatch] = pb.get("skill_match")
            reason = f"s={pb.get('score', 0)}|base={pb.get('base_score', 0)}|pc={pc:.0%}"
            if skill_match is not None:
                reason += f"|skill={skill_match.pattern_name}+{skill_match.bonus}"

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
                    reason,
                )
            )


def _restore_risk_params(snapshot: Dict[str, object]) -> None:
    blind.RISK_PARAMS.clear()
    blind.RISK_PARAMS.update(snapshot)


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    overlay: ExpertSkillOverlay,
    start_date: str,
    end_date: str,
) -> Tuple[DistilledBlindAgentBacktest, trainer.Metrics]:
    snapshot = copy.deepcopy(blind.RISK_PARAMS)
    previous_level = blind.logger.level
    blind.logger.setLevel(logging.WARNING)
    try:
        engine = DistilledBlindAgentBacktest(all_data, overlay)
        engine.run(start_date, end_date)
        metrics = trainer._summarize_engine(engine)
        return engine, metrics
    finally:
        blind.logger.setLevel(previous_level)
        _restore_risk_params(snapshot)


def main() -> None:
    args = parse_args()
    logger.info("Loading backtest data cache...")
    all_data = fetch_all_a_share_daily("20251001", "20260410", cache_name="backtest_cache_2026ytd.pkl")

    logger.info("Bootstrapping online expert overlay...")
    overlay = ExpertSkillOverlay(
        all_data=all_data,
        expert_source=args.expert_source,
        min_samples=args.min_samples,
        top_n=args.top_n,
    )

    engine, metrics = run_backtest(all_data, overlay, args.start, args.end)
    out_dir = Path(args.out_dir)
    trainer.save_backtest_outputs(
        engine,
        metrics,
        out_dir,
        extra={
            "strategy": "blind_agent_v4_online_distilled",
            "expert_source": args.expert_source,
            "min_samples": args.min_samples,
            "top_n": args.top_n,
            "expert_trade_count": overlay.expert_trade_count,
            "skill_available_from": overlay.available_from,
            "skill_match_days": engine.skill_match_days,
            "skill_match_candidates": engine.skill_match_candidates,
            "skill_buy_count": engine.skill_buy_count,
            "skill_patterns_used": dict(engine.skill_pattern_counter),
            "selected_params": copy.deepcopy(blind.RISK_PARAMS),
        },
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    logger.info(
        "Distilled result | return=%+.2f%% max_dd=%.2f%% sharpe=%.2f avg_month=%+.2f%% skill_buys=%s",
        summary["return"],
        summary["max_dd"],
        summary["sharpe"],
        summary["avg_month"],
        summary["skill_buy_count"],
    )


if __name__ == "__main__":
    main()
