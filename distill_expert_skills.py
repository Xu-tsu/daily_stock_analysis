"""
Distill expert A-share trade records into YAML strategy skills.

The script reads completed trades from ``trade_log`` (default source: broker_export),
rebuilds entry-day features using historical market data only, and exports the best
expert patterns as YAML strategy files under ``strategies/distilled``.

Usage:
    python distill_expert_skills.py
    python distill_expert_skills.py --source broker_export --top-n 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

import backtest_blind_agent as blind
from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.base import normalize_stock_code
from event_signal import get_stock_sector


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

DB_PATH = Path(os.getenv("SCANNER_DB_PATH", "data/scanner_history.db"))
DEFAULT_OUT_DIR = Path("strategies/distilled")
DEFAULT_REPORT_DIR = Path("data/expert_skill_distillation")
DEFAULT_CACHE = "expert_trade_history.pkl"


@dataclass
class CompletedTrade:
    code: str
    name: str
    shares: int
    buy_date: str
    sell_date: str
    buy_price: float
    sell_price: float
    pnl_pct: float
    hold_days: int
    source: str
    sector: str
    indicators: Dict[str, float]
    features: Dict[str, object]
    archetype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="蒸馏高手交易记录为 Agent YAML skills")
    parser.add_argument("--source", default="broker_export", help="trade_log.source 过滤值，默认 broker_export")
    parser.add_argument("--top-n", type=int, default=4, help="最多导出多少个 skill，默认 4")
    parser.add_argument("--min-samples", type=int, default=3, help="单个模式的最少样本数，默认 3")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="导出 YAML skill 目录，默认 strategies/distilled",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="导出摘要报告目录，默认 data/expert_skill_distillation",
    )
    parser.add_argument(
        "--cache-name",
        default=DEFAULT_CACHE,
        help="定向历史行情缓存文件名，默认 expert_trade_history.pkl",
    )
    return parser.parse_args()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_trade_rows(source: str) -> List[sqlite3.Row]:
    conn = _conn()
    try:
        rows = conn.execute(
            """
            SELECT id, trade_date, trade_type, code, name, shares, price, pnl_pct, hold_days, source
            FROM trade_log
            WHERE source = ?
            ORDER BY trade_date, id
            """,
            (source,),
        ).fetchall()
        return rows
    finally:
        conn.close()


def pair_completed_trades(rows: Iterable[sqlite3.Row]) -> List[dict]:
    buy_queues: Dict[str, Deque[dict]] = defaultdict(deque)
    completed: List[dict] = []

    for row in rows:
        code = str(row["code"]).strip()
        shares = int(row["shares"] or 0)
        price = float(row["price"] or 0)
        if not code or shares <= 0 or price <= 0:
            continue

        if row["trade_type"] == "buy":
            buy_queues[code].append(
                {
                    "buy_date": row["trade_date"],
                    "buy_price": price,
                    "remaining_shares": shares,
                    "name": row["name"] or code,
                    "source": row["source"],
                }
            )
            continue

        if row["trade_type"] != "sell":
            continue

        remaining_to_match = shares
        hold_days = int(row["hold_days"] or 0)
        sell_price = price
        pnl_pct = float(row["pnl_pct"] or 0)

        queue = buy_queues[code]
        while remaining_to_match > 0 and queue:
            buy_leg = queue[0]
            matched_shares = min(remaining_to_match, buy_leg["remaining_shares"])
            leg_pnl_pct = (sell_price - buy_leg["buy_price"]) / buy_leg["buy_price"] * 100
            completed.append(
                {
                    "code": code,
                    "name": buy_leg["name"],
                    "shares": matched_shares,
                    "buy_date": buy_leg["buy_date"],
                    "sell_date": row["trade_date"],
                    "buy_price": buy_leg["buy_price"],
                    "sell_price": sell_price,
                    "pnl_pct": round(leg_pnl_pct if math.isfinite(leg_pnl_pct) else pnl_pct, 4),
                    "hold_days": hold_days,
                    "source": buy_leg["source"],
                }
            )
            remaining_to_match -= matched_shares
            buy_leg["remaining_shares"] -= matched_shares
            if buy_leg["remaining_shares"] <= 0:
                queue.popleft()

    return completed


def _resolve_cache_file(cache_name: str, report_dir: Path) -> Path:
    candidate = Path(cache_name)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate
    return report_dir / cache_name


def _normalize_history_frame(raw_df: pd.DataFrame, code: str) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "换手率": "turnover_rate",
            "turnover": "turnover_rate",
        }
    )
    if "date" not in df.columns:
        raise ValueError(f"{code} history is missing date column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns and "amount" in df.columns:
        # Tencent fallback occasionally exposes only amount; use it as a stable liquidity proxy
        # so relative volume features can still be reconstructed without dropping the trade sample.
        df["volume"] = df["amount"]
    if "pct_chg" not in df.columns and "close" in df.columns:
        df["pct_chg"] = df["close"].pct_change() * 100
    if "turnover_rate" not in df.columns:
        df["turnover_rate"] = 0.0

    keep_cols = ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover_rate"]
    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    df["code"] = code
    return df


def load_price_history(
    completed_trades: List[dict],
    cache_name: str,
    report_dir: Path,
) -> Dict[str, pd.DataFrame]:
    if not completed_trades:
        return {}

    min_buy_date = min(datetime.strptime(item["buy_date"], "%Y-%m-%d") for item in completed_trades)
    max_sell_date = max(datetime.strptime(item["sell_date"], "%Y-%m-%d") for item in completed_trades)
    start = (min_buy_date - timedelta(days=95)).strftime("%Y-%m-%d")
    end = (max_sell_date + timedelta(days=10)).strftime("%Y-%m-%d")
    logger.info("Loading targeted price history for distillation window: %s -> %s", start, end)

    cache_file = _resolve_cache_file(cache_name, report_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cached: Dict[str, pd.DataFrame] = {}
    if cache_file.exists():
        cached = pd.read_pickle(cache_file)
        logger.info("Loaded distilled history cache: %s (%s codes)", cache_file, len(cached))

    requested_codes = sorted({normalize_stock_code(item["code"]) for item in completed_trades})
    fetcher = AkshareFetcher(sleep_min=0.2, sleep_max=0.5)

    refreshed: Dict[str, pd.DataFrame] = {}
    failed_codes: List[str] = []
    for code in requested_codes:
        cached_df = cached.get(code)
        if cached_df is not None and not cached_df.empty:
            if "volume" not in cached_df.columns and "amount" in cached_df.columns:
                cached_df = cached_df.copy()
                cached_df["volume"] = cached_df["amount"]
            if "turnover_rate" not in cached_df.columns:
                cached_df = cached_df.copy()
                cached_df["turnover_rate"] = 0.0
            has_start = str(cached_df["date"].min()) <= start
            has_end = str(cached_df["date"].max()) >= end
            if has_start and has_end:
                refreshed[code] = cached_df
                continue

        try:
            raw_df = fetcher._fetch_raw_data(code, start, end)
            normalized_df = _normalize_history_frame(raw_df, code)
            if len(normalized_df) >= 20:
                refreshed[code] = normalized_df
            else:
                failed_codes.append(code)
        except Exception as exc:
            logger.warning("Failed to fetch history for %s: %s", code, exc)
            failed_codes.append(code)

    pd.to_pickle(refreshed, cache_file)
    logger.info("Prepared history for %s/%s trade codes", len(refreshed), len(requested_codes))
    if failed_codes:
        logger.warning("History unavailable for %s codes: %s", len(failed_codes), ", ".join(failed_codes[:12]))

    return refreshed


def derive_feature_labels(ind: Dict[str, float], sector: str) -> Dict[str, object]:
    trend = "bull_align" if ind["ma5"] > ind["ma10"] > ind["ma20"] else "up_cross" if ind["ma5"] > ind["ma10"] else "weak"
    rsi_bucket = "oversold" if ind["rsi"] < 35 else "balanced" if ind["rsi"] < 60 else "hot"
    pos_bucket = "low_zone" if ind["price_pos"] < 0.35 else "mid_zone" if ind["price_pos"] < 0.7 else "high_zone"
    turnover_bucket = "active" if ind["turnover"] >= 3.4 else "normal" if ind["turnover"] >= 2.0 else "cold"
    vol_bucket = "breakout" if ind["vol_breakout"] else "pullback" if ind["vol_pullback"] else "neutral"
    macd_bucket = "cross_up" if ind["macd_cross"] else "above_dea" if ind["dif"] > ind["dea"] else "weak"
    day_bucket = "strong_day" if ind["today_chg"] >= 3 else "soft_day" if ind["today_chg"] >= 0 else "down_day"
    hold_bucket = "quick" if ind.get("hold_days", 0) <= 2 else "swing"

    return {
        "trend": trend,
        "rsi_bucket": rsi_bucket,
        "pos_bucket": pos_bucket,
        "turnover_bucket": turnover_bucket,
        "vol_bucket": vol_bucket,
        "macd_bucket": macd_bucket,
        "day_bucket": day_bucket,
        "sector": sector or "未知",
        "hold_bucket": hold_bucket,
    }


def classify_archetype(ind: Dict[str, float], labels: Dict[str, object]) -> str:
    if ind["yest_limit"] and -2.5 <= ind["today_chg"] <= 5.5:
        return "board_relay"
    if ind["vol_breakout"] and labels["trend"] in {"bull_align", "up_cross"}:
        return "momentum_breakout"
    if labels["vol_bucket"] == "pullback" and labels["trend"] in {"bull_align", "up_cross"}:
        return "trend_pullback"
    if labels["rsi_bucket"] == "oversold" and labels["pos_bucket"] == "low_zone":
        return "oversold_rebound"
    if labels["macd_bucket"] == "cross_up" and labels["day_bucket"] != "down_day":
        return "macd_reversal"
    return "mixed_setup"


def enrich_completed_trades(
    completed_trades: List[dict],
    all_data: Dict[str, pd.DataFrame],
) -> List[CompletedTrade]:
    enriched: List[CompletedTrade] = []
    skipped = 0

    for item in completed_trades:
        code = normalize_stock_code(item["code"])
        df = all_data.get(code)
        if df is None:
            skipped += 1
            continue

        match_idx = df.index[df["date"] == item["buy_date"]]
        if len(match_idx) == 0:
            skipped += 1
            continue

        ind = blind.calc_indicators(df, int(match_idx[0]))
        if ind is None:
            skipped += 1
            continue

        sector = get_stock_sector(code) or "未知"
        ind["hold_days"] = item["hold_days"]
        labels = derive_feature_labels(ind, sector)
        archetype = classify_archetype(ind, labels)

        enriched.append(
            CompletedTrade(
                code=code,
                name=item["name"],
                shares=item["shares"],
                buy_date=item["buy_date"],
                sell_date=item["sell_date"],
                buy_price=item["buy_price"],
                sell_price=item["sell_price"],
                pnl_pct=float(item["pnl_pct"]),
                hold_days=int(item["hold_days"]),
                source=item["source"],
                sector=sector,
                indicators=ind,
                features=labels,
                archetype=archetype,
            )
        )

    logger.info("Enriched %s completed trades, skipped %s", len(enriched), skipped)
    return enriched


def summarize_group(trades: List[CompletedTrade]) -> Dict[str, object]:
    returns = [trade.pnl_pct for trade in trades]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    return {
        "samples": len(trades),
        "win_rate": round(len(wins) * 100 / len(trades), 1) if trades else 0.0,
        "avg_pnl": round(sum(returns) / len(returns), 2) if trades else 0.0,
        "profit_factor": round(abs(sum(wins) / sum(losses)), 2) if losses and sum(losses) != 0 else 0.0,
        "median_hold_days": round(float(pd.Series([trade.hold_days for trade in trades]).median()), 1) if trades else 0.0,
        "median_turnover": round(float(pd.Series([trade.indicators["turnover"] for trade in trades]).median()), 2) if trades else 0.0,
        "median_rsi": round(float(pd.Series([trade.indicators["rsi"] for trade in trades]).median()), 1) if trades else 0.0,
        "median_vol_ratio": round(float(pd.Series([trade.indicators["vol_ratio"] for trade in trades]).median()), 2) if trades else 0.0,
        "median_day_chg": round(float(pd.Series([trade.indicators["today_chg"] for trade in trades]).median()), 2) if trades else 0.0,
        "median_price_pos": round(float(pd.Series([trade.indicators["price_pos"] for trade in trades]).median()) * 100, 1) if trades else 0.0,
    }


def archetype_meta(archetype: str) -> Dict[str, object]:
    mapping = {
        "board_relay": {"display": "高手蒸馏-涨停接力", "category": "trend", "core_rules": [2, 7]},
        "momentum_breakout": {"display": "高手蒸馏-放量突破", "category": "trend", "core_rules": [2, 3, 6]},
        "trend_pullback": {"display": "高手蒸馏-趋势回踩", "category": "pattern", "core_rules": [2, 4, 6]},
        "oversold_rebound": {"display": "高手蒸馏-超跌反弹", "category": "reversal", "core_rules": [4, 5]},
        "macd_reversal": {"display": "高手蒸馏-MACD反转", "category": "reversal", "core_rules": [2, 4, 6]},
        "mixed_setup": {"display": "高手蒸馏-混合形态", "category": "framework", "core_rules": [2, 4, 6]},
    }
    return mapping.get(archetype, mapping["mixed_setup"])


def build_strategy_yaml(
    rank: int,
    archetype: str,
    sector: str,
    stats: Dict[str, object],
    source: str,
) -> Dict[str, object]:
    meta = archetype_meta(archetype)
    sector_suffix = f"-{sector}" if sector and sector != "未知" else ""
    safe_suffix = re.sub(r"[^a-z0-9_]+", "_", archetype.lower()).strip("_")
    name = f"expert_{rank:02d}_{safe_suffix}"
    display_name = f"{meta['display']}{sector_suffix}"

    description = (
        f"从 {source} 蒸馏出的高手交易模式，样本{stats['samples']}笔，"
        f"胜率{stats['win_rate']}%，均收益{stats['avg_pnl']}%。"
    )
    if sector and sector != "未知":
        description += f" 该模式主要在 {sector} 板块内出现。"

    sector_line = (
        f"   - 优先在 `{sector}` 板块或强相关细分方向内使用，避免跨板块硬套。\n"
        if sector and sector != "未知"
        else ""
    )

    instructions = (
        f"**{display_name}**\n\n"
        f"这是一条从真实高手成交记录中蒸馏出的经验型 strategy，不是拍脑袋规则。\n"
        f"当前样本: {stats['samples']}笔，胜率 {stats['win_rate']}%，均收益 {stats['avg_pnl']}%，"
        f"中位持有 {stats['median_hold_days']} 天。\n\n"
        "执行要点：\n\n"
        "1. **先确认环境匹配**：\n"
        f"{sector_line}"
        f"   - 仅在买点当日形态接近这条模式时加分，不要因为名字相似就强行套用。\n\n"
        "2. **技术面过滤**：\n"
        f"   - 中位换手率约 `{stats['median_turnover']}%`，量比中位数约 `{stats['median_vol_ratio']}`。\n"
        f"   - RSI 中位数约 `{stats['median_rsi']}`，20日位置中位数约 `{stats['median_price_pos']}%`。\n"
        f"   - 当日涨跌幅中位数约 `{stats['median_day_chg']}%`，只有在承接与量能接近样本特征时才提高信心。\n\n"
        "3. **仓位与纪律**：\n"
        "   - 这是加分型 skill，不是无条件开仓指令。\n"
        "   - 若大盘弱、板块不跟、开盘承接差，即使形态相似也要降级处理。\n\n"
        "4. **输出要求**：\n"
        f"   - 在 `buy_reason` 中明确标注“{display_name}”。\n"
        "   - 如果你认为当前个股与该模式不匹配，要明确写出不匹配原因，而不是模糊带过。\n"
    )

    return {
        "name": name,
        "display_name": display_name,
        "description": description,
        "category": meta["category"],
        "core_rules": meta["core_rules"],
        "required_tools": ["get_daily_history", "analyze_trend", "get_realtime_quote", "get_sector_rankings"],
        "instructions": instructions,
    }


def rank_patterns(
    trades: List[CompletedTrade],
    min_samples: int,
    top_n: int,
) -> List[Dict[str, object]]:
    overall = summarize_group(trades)
    grouped: Dict[Tuple[str, str], List[CompletedTrade]] = defaultdict(list)

    for trade in trades:
        grouped[(trade.archetype, trade.sector)].append(trade)
        grouped[(trade.archetype, "未知")].append(trade)

    rows: List[Dict[str, object]] = []
    fallback_rows: List[Dict[str, object]] = []
    for (archetype, sector), items in grouped.items():
        stats = summarize_group(items)
        if stats["samples"] < min_samples:
            continue

        score = round(
            (stats["avg_pnl"] - overall["avg_pnl"]) * 2.8
            + (stats["win_rate"] - overall["win_rate"]) * 0.18
            + stats["profit_factor"] * 1.5
            + min(stats["samples"], 8) * 0.4,
            4,
        )
        row = {
            "archetype": archetype,
            "sector": sector,
            "score": score,
            **stats,
        }
        fallback_rows.append(row)
        if stats["avg_pnl"] > overall["avg_pnl"] and stats["win_rate"] >= max(35.0, overall["win_rate"]):
            rows.append(row)

    rows.sort(key=lambda item: (item["score"], item["avg_pnl"], item["win_rate"], item["samples"]), reverse=True)
    fallback_rows.sort(key=lambda item: (item["score"], item["avg_pnl"], item["win_rate"], item["samples"]), reverse=True)

    selected: List[Dict[str, object]] = []
    used_archetypes = set()
    candidate_rows = rows or fallback_rows
    for row in candidate_rows:
        if row["sector"] != "未知" and row["archetype"] in used_archetypes:
            continue
        selected.append(row)
        used_archetypes.add(row["archetype"])
        if len(selected) >= top_n:
            break

    return selected


def write_outputs(
    selected_patterns: List[Dict[str, object]],
    source: str,
    out_dir: Path,
    report_dir: Path,
    overall_stats: Optional[Dict[str, object]] = None,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    created_files: List[Path] = []
    summary_rows: List[Dict[str, object]] = []

    # clean previously generated yaml files for deterministic output
    for old_file in out_dir.glob("expert_*.yaml"):
        old_file.unlink()

    for idx, pattern in enumerate(selected_patterns, start=1):
        payload = build_strategy_yaml(idx, pattern["archetype"], pattern["sector"], pattern, source)
        target = out_dir / f"{payload['name']}.yaml"
        with open(target, "w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)
        created_files.append(target)
        summary_rows.append({**pattern, "name": payload["name"], "display_name": payload["display_name"]})

    with open(report_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "source": source,
                "overall_stats": overall_stats or {},
                "generated_skills": summary_rows,
                "out_dir": str(out_dir),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    pd.DataFrame(summary_rows).to_csv(report_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return created_files


def main() -> None:
    args = parse_args()
    rows = load_trade_rows(args.source)
    completed = pair_completed_trades(rows)
    if not completed:
        raise RuntimeError(f"未找到 source={args.source} 的完整买卖配对交易")

    report_dir = Path(args.report_dir)
    all_data = load_price_history(completed, args.cache_name, report_dir)
    enriched = enrich_completed_trades(completed, all_data)
    if not enriched:
        raise RuntimeError("高手交易样本无法补全历史技术特征，蒸馏中止")

    selected = rank_patterns(enriched, min_samples=args.min_samples, top_n=args.top_n)
    if not selected:
        raise RuntimeError("当前样本未筛出稳定优势模式，请增加样本或降低门槛")

    requested_codes = sorted({normalize_stock_code(item["code"]) for item in completed})
    missing_history_codes = sorted(set(requested_codes) - set(all_data.keys()))
    proxy_volume_codes = sorted(
        code
        for code, df in all_data.items()
        if "volume" in df.columns
        and "amount" in df.columns
        and df["volume"].fillna(0).equals(df["amount"].fillna(0))
    )
    overall_stats = {
        "raw_rows": len(rows),
        "completed_trades": len(completed),
        "enriched_trades": len(enriched),
        "requested_codes": len(requested_codes),
        "history_ready_codes": len(all_data),
        "missing_history_codes": missing_history_codes,
        "proxy_volume_codes": proxy_volume_codes,
        **summarize_group(enriched),
    }
    created = write_outputs(selected, args.source, Path(args.out_dir), report_dir, overall_stats=overall_stats)
    logger.info("Generated %s distilled strategy skills", len(created))
    for path in created:
        logger.info("  -> %s", path)


if __name__ == "__main__":
    main()
