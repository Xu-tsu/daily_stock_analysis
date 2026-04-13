"""
Walk-forward trainer for the blind agent backtest.

This script keeps the "no future data" rule:
1. Search and score candidate risk parameters on 2025 only.
2. Select one profile using train-only windows.
3. Run an out-of-sample backtest on 2026 YTD with the selected profile.

Usage:
    python train_blind_agent.py
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import backtest_blind_agent as blind
from backtest_2025 import fetch_all_a_share_daily


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

TRAIN_OUT_DIR = Path("data/blind_agent_training")
TEST_OUT_DIR = Path("data/backtest_blind_agent_trained")

# Keep the original V4 baseline as the search anchor.
BASE_RISK_PARAMS = {
    "stop_loss": -3.0,
    "tp_half": 5.0,
    "tp_full": 12.0,
    "trail_trigger": 6.0,
    "trail_dd": 2.5,
    "hold_max": 5,
    "emergency_stop": -5.0,
    "max_positions": 3,
    "single_pct": 0.35,
    "price_range": (3, 50),
    "min_turnover": 2.0,
}

SEARCH_SPACE = {
    "trail_trigger": [5.5, 5.8, 6.0],
    "trail_dd": [1.8, 2.0, 2.2, 2.5],
    "min_turnover": [3.0, 3.2, 3.4, 3.6],
}

COARSE_KEEP = 10
TRAIN_WINDOWS = [
    ("2025_full", "2025-01-06", "2025-12-31", 0.60),
    ("2025_h1", "2025-01-06", "2025-06-30", 0.20),
    ("2025_h2", "2025-07-01", "2025-12-31", 0.20),
]
TEST_WINDOW = ("2026_ytd", "2026-01-05", "2026-04-10")


@dataclass
class Metrics:
    ret: float
    annual: float
    max_dd: float
    sharpe: float
    win_rate: float
    profit_factor: float
    trades: int
    final: float
    avg_month: float
    monthly: Dict[str, float]


class SilentBlindAgentBacktest(blind.BlindAgentBacktest):
    def _print_report(self):
        return


def _restore_risk_params(snapshot: Dict[str, object]) -> None:
    blind.RISK_PARAMS.clear()
    blind.RISK_PARAMS.update(snapshot)


def _apply_risk_params(params: Dict[str, object]) -> Dict[str, object]:
    snapshot = copy.deepcopy(blind.RISK_PARAMS)
    blind.RISK_PARAMS.clear()
    blind.RISK_PARAMS.update(copy.deepcopy(BASE_RISK_PARAMS))
    blind.RISK_PARAMS.update(copy.deepcopy(params))
    return snapshot


def _summarize_engine(engine: blind.BlindAgentBacktest) -> Metrics:
    if not engine.snapshots:
        raise RuntimeError("Backtest produced no snapshots")

    final = engine.snapshots[-1].total_asset
    ret = (final - blind.INITIAL_CAPITAL) / blind.INITIAL_CAPITAL * 100

    peak = blind.INITIAL_CAPITAL
    max_dd = 0.0
    daily_returns: List[float] = []
    monthly = defaultdict(float)

    for snapshot in engine.snapshots:
        peak = max(peak, snapshot.total_asset)
        dd = (peak - snapshot.total_asset) / peak * 100
        max_dd = max(max_dd, dd)
        daily_returns.append(snapshot.daily_return_pct)
        monthly[snapshot.date[:7]] += snapshot.daily_return_pct

    completed = defaultdict(list)
    for trade in engine.trades:
        completed[trade.code].append(trade)

    wins = 0
    losses = 0
    win_list: List[float] = []
    loss_list: List[float] = []
    for trade_list in completed.values():
        buys = [trade for trade in trade_list if trade.direction == "buy"]
        sells = [trade for trade in trade_list if trade.direction == "sell"]
        if not buys or not sells:
            continue
        avg_buy = sum(trade.price * trade.shares for trade in buys) / sum(trade.shares for trade in buys)
        avg_sell = sum(trade.price * trade.shares for trade in sells) / sum(trade.shares for trade in sells)
        pnl = (avg_sell - avg_buy) / avg_buy * 100
        if pnl > 0:
            wins += 1
            win_list.append(pnl)
        else:
            losses += 1
            loss_list.append(pnl)

    trade_count = wins + losses
    win_rate = wins / trade_count * 100 if trade_count else 0.0
    profit_factor = abs(sum(win_list) / sum(loss_list)) if loss_list and sum(loss_list) != 0 else 0.0

    days = len(engine.snapshots)
    annual = ret * (252 / days) if days else 0.0
    daily_std = np.std(daily_returns) if len(daily_returns) > 1 else 0.0
    sharpe = np.mean(daily_returns) / daily_std * np.sqrt(252) if daily_std > 0 else 0.0
    avg_month = float(np.mean(list(monthly.values()))) if monthly else 0.0

    return Metrics(
        ret=round(ret, 2),
        annual=round(annual, 2),
        max_dd=round(max_dd, 2),
        sharpe=round(float(sharpe), 2),
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        trades=len(engine.trades),
        final=round(final, 0),
        avg_month=round(avg_month, 2),
        monthly={month: round(value, 2) for month, value in sorted(monthly.items())},
    )


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    params: Dict[str, object],
) -> Tuple[SilentBlindAgentBacktest, Metrics]:
    snapshot = _apply_risk_params(params)
    previous_level = blind.logger.level
    blind.logger.setLevel(logging.WARNING)
    try:
        engine = SilentBlindAgentBacktest(all_data)
        engine.run(start_date, end_date)
        metrics = _summarize_engine(engine)
        return engine, metrics
    finally:
        blind.logger.setLevel(previous_level)
        _restore_risk_params(snapshot)


def build_candidates() -> Iterable[Dict[str, object]]:
    for trail_trigger, trail_dd, min_turnover in product(
        SEARCH_SPACE["trail_trigger"],
        SEARCH_SPACE["trail_dd"],
        SEARCH_SPACE["min_turnover"],
    ):
        yield {
            "trail_trigger": trail_trigger,
            "trail_dd": trail_dd,
            "min_turnover": min_turnover,
        }


def coarse_score(metrics: Metrics) -> float:
    return metrics.ret + metrics.sharpe * 6.0 - metrics.max_dd * 1.2


def train_score(window_metrics: Dict[str, Metrics]) -> float:
    full = window_metrics["2025_full"]
    h1 = window_metrics["2025_h1"]
    h2 = window_metrics["2025_h2"]
    return (
        full.ret * 0.60
        + h1.ret * 0.20
        + h2.ret * 0.20
        + full.sharpe * 6.0
        - full.max_dd * 1.2
        - max(0.0, -h2.ret) * 1.5
    )


def save_training_artifacts(
    coarse_rows: List[Dict[str, object]],
    final_rows: List[Dict[str, object]],
    best_profile: Dict[str, object],
) -> None:
    TRAIN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coarse_rows).sort_values("coarse_score", ascending=False).to_csv(
        TRAIN_OUT_DIR / "coarse_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(final_rows).sort_values("train_score", ascending=False).to_csv(
        TRAIN_OUT_DIR / "walkforward_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(TRAIN_OUT_DIR / "best_profile.json", "w", encoding="utf-8") as file:
        json.dump(best_profile, file, ensure_ascii=False, indent=2)


def save_backtest_outputs(
    engine: blind.BlindAgentBacktest,
    metrics: Metrics,
    out_dir: Path,
    extra: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "return": metrics.ret,
        "annual": metrics.annual,
        "max_dd": metrics.max_dd,
        "sharpe": metrics.sharpe,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "trades": metrics.trades,
        "final": metrics.final,
        "avg_month": metrics.avg_month,
        "monthly": metrics.monthly,
    }
    summary.update(extra)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    trade_rows = [
        {
            "date": trade.date,
            "code": trade.code,
            "name": trade.name,
            "dir": trade.direction,
            "shares": trade.shares,
            "price": round(trade.price, 3),
            "amount": round(trade.amount, 0),
            "reason": trade.reason,
        }
        for trade in engine.trades
    ]
    pd.DataFrame(trade_rows).to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")

    daily_rows = [
        {
            "date": snapshot.date,
            "total": round(snapshot.total_asset, 0),
            "cash": round(snapshot.cash, 0),
            "mv": round(snapshot.market_value, 0),
            "hold": snapshot.holdings_count,
            "ret": round(snapshot.daily_return_pct, 4),
            "pos_coeff": round(snapshot.position_coeff, 3),
            "buy_thr": snapshot.buy_threshold,
        }
        for snapshot in engine.snapshots
    ]
    pd.DataFrame(daily_rows).to_csv(out_dir / "daily.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    logger.info("Loading training data cache...")
    data_2025 = fetch_all_a_share_daily("20250101", "20251231", cache_name="backtest_cache_2025.pkl")
    logger.info("Loading test data cache...")
    data_2026 = fetch_all_a_share_daily("20251001", "20260410", cache_name="backtest_cache_2026ytd.pkl")

    coarse_rows: List[Dict[str, object]] = []
    coarse_candidates: List[Dict[str, object]] = []

    logger.info("Stage 1/2: coarse search on 2025 full-year")
    for params in build_candidates():
        _, metrics = run_backtest(data_2025, "2025-01-06", "2025-12-31", params)
        score = round(coarse_score(metrics), 4)
        row = {
            **params,
            "coarse_score": score,
            "train_return": metrics.ret,
            "train_sharpe": metrics.sharpe,
            "train_max_dd": metrics.max_dd,
            "train_avg_month": metrics.avg_month,
            "train_trades": metrics.trades,
        }
        coarse_rows.append(row)
        coarse_candidates.append({"params": params, "metrics": metrics, "coarse_score": score})

    coarse_candidates.sort(
        key=lambda item: (
            item["coarse_score"],
            item["metrics"].ret,
            -item["metrics"].max_dd,
            -item["params"]["min_turnover"],
        ),
        reverse=True,
    )
    top_candidates = coarse_candidates[:COARSE_KEEP]

    logger.info("Stage 2/2: walk-forward scoring on 2025 train-only windows")
    final_rows: List[Dict[str, object]] = []
    best_choice = None
    best_windows = None

    for candidate in top_candidates:
        params = candidate["params"]
        window_metrics = {"2025_full": candidate["metrics"]}
        for window_name, start_date, end_date, _weight in TRAIN_WINDOWS[1:]:
            _, metrics = run_backtest(data_2025, start_date, end_date, params)
            window_metrics[window_name] = metrics

        score = round(train_score(window_metrics), 4)
        row = {
            **params,
            "train_score": score,
            "full_return": window_metrics["2025_full"].ret,
            "full_max_dd": window_metrics["2025_full"].max_dd,
            "full_sharpe": window_metrics["2025_full"].sharpe,
            "h1_return": window_metrics["2025_h1"].ret,
            "h2_return": window_metrics["2025_h2"].ret,
            "h2_max_dd": window_metrics["2025_h2"].max_dd,
            "h2_sharpe": window_metrics["2025_h2"].sharpe,
        }
        final_rows.append(row)

        rank_key = (
            score,
            window_metrics["2025_full"].ret,
            -window_metrics["2025_full"].max_dd,
            -params["min_turnover"],
        )
        if best_choice is None or rank_key > best_choice["rank_key"]:
            best_choice = {"params": copy.deepcopy(params), "rank_key": rank_key}
            best_windows = window_metrics

    if best_choice is None or best_windows is None:
        raise RuntimeError("Training failed to select a parameter profile")

    selected_params = best_choice["params"]
    logger.info("Selected params: %s", selected_params)

    logger.info("Running out-of-sample backtest on 2026 YTD")
    test_engine, test_metrics = run_backtest(
        data_2026,
        TEST_WINDOW[1],
        TEST_WINDOW[2],
        selected_params,
    )

    best_profile = {
        "strategy": "blind_agent_v4_walkforward",
        "selection_method": "two_stage_walk_forward",
        "train_windows": [
            {"name": window_name, "start": start_date, "end": end_date, "weight": weight}
            for window_name, start_date, end_date, weight in TRAIN_WINDOWS
        ],
        "test_window": {"name": TEST_WINDOW[0], "start": TEST_WINDOW[1], "end": TEST_WINDOW[2]},
        "search_space": SEARCH_SPACE,
        "selected_params": selected_params,
        "training_metrics": {
            window_name: {
                "return": metrics.ret,
                "annual": metrics.annual,
                "max_dd": metrics.max_dd,
                "sharpe": metrics.sharpe,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "trades": metrics.trades,
                "avg_month": metrics.avg_month,
            }
            for window_name, metrics in best_windows.items()
        },
        "test_metrics": {
            "return": test_metrics.ret,
            "annual": test_metrics.annual,
            "max_dd": test_metrics.max_dd,
            "sharpe": test_metrics.sharpe,
            "win_rate": test_metrics.win_rate,
            "profit_factor": test_metrics.profit_factor,
            "trades": test_metrics.trades,
            "avg_month": test_metrics.avg_month,
        },
    }

    save_training_artifacts(coarse_rows, final_rows, best_profile)
    save_backtest_outputs(
        test_engine,
        test_metrics,
        TEST_OUT_DIR,
        extra={
            "selection_method": "two_stage_walk_forward",
            "selected_params": selected_params,
            "trained_on": "2025-only",
            "tested_on": TEST_WINDOW[0],
        },
    )

    logger.info("Training artifacts saved to %s", TRAIN_OUT_DIR)
    logger.info("Out-of-sample backtest saved to %s", TEST_OUT_DIR)
    logger.info(
        "OOS result | return=%+.2f%% max_dd=%.2f%% sharpe=%.2f avg_month=%+.2f%%",
        test_metrics.ret,
        test_metrics.max_dd,
        test_metrics.sharpe,
        test_metrics.avg_month,
    )


if __name__ == "__main__":
    main()
