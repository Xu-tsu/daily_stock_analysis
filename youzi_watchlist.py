# -*- coding: utf-8 -*-
"""游资选股 watchlist 持久化。

每一次 youzi_pipeline 跑完，都把：
  - final_picks   : 真实买入 / 强 buy 的票
  - watch         : 中间态（watch/skip），未下单但也想观察
  - vetoed        : 被某个风格否决的票（也要跟踪，看是否救了我们）

全部存成单条 "snapshot"，10 天后自动剪枝。
复盘模块会读取这个文件，对每条做 T+1~T+N 的行情回访 + 胜率归因。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_DATA_DIR.mkdir(exist_ok=True)
WATCHLIST_PATH = _DATA_DIR / "youzi_watchlist.json"

KEEP_DAYS = 15  # 保留最近 15 天的 snapshot 用于复盘


def _load() -> Dict:
    if not WATCHLIST_PATH.exists():
        return {"snapshots": []}
    try:
        return json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[youzi_watchlist] load failed, reset: {e}")
        return {"snapshots": []}


def _save(data: Dict) -> None:
    tmp = WATCHLIST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(WATCHLIST_PATH)


def _prune(data: Dict) -> Dict:
    """剪掉 >KEEP_DAYS 天的快照。"""
    cutoff = datetime.now() - timedelta(days=KEEP_DAYS)
    keep = []
    for s in data.get("snapshots", []):
        try:
            ts = datetime.strptime(s.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
            if ts >= cutoff:
                keep.append(s)
        except Exception:
            continue
    data["snapshots"] = keep
    return data


def _simplify_candidate(c: Dict) -> Dict:
    """只存复盘必须的字段，避免 JSON 膨胀。"""
    return {
        "code": str(c.get("code", "")).replace("sh", "").replace("sz", "")[-6:],
        "name": c.get("name", ""),
        "entry_price": float(c.get("price", 0) or 0),
        "entry_change_pct": float(c.get("change_pct", 0) or 0),
        "tech_score": int(c.get("tech_score", 0) or 0),
        "signal_type": c.get("signal_type", ""),
        "market_cap": float(c.get("market_cap", 0) or 0),
        "youzi_verdict": c.get("youzi_verdict"),
        "youzi_buy_votes": c.get("youzi_buy_votes", []),
        "youzi_weighted_score": c.get("youzi_weighted_score", 0),
        "youzi_vetoed_by": (c.get("youzi_aggregate", {}) or {}).get("vetoed_by", []),
        "hot_concepts": (c.get("news_ctx", {}) or {}).get("hot_concept_names", []),
        "hot_concept_rank": (c.get("news_ctx", {}) or {}).get("hot_concept_rank", 0),
        "per_style_verdicts": {
            k: v.get("verdict", "") for k, v in (c.get("youzi_results", {}) or {}).items()
        },
        "per_style_scores": {
            k: v.get("final_score", 0) for k, v in (c.get("youzi_results", {}) or {}).items()
        },
    }


def save_snapshot(
    enriched: List[Dict],
    buy_list: List[Dict],
    regime: str,
    v2_plan: Dict,
) -> str:
    """保存一次扫描快照。

    buy_list 与 enriched 可以重叠；buy_list 里的 code 会被标记 acted=True。
    """
    data = _prune(_load())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    buy_codes = {
        str(b.get("code", "")).replace("sh", "").replace("sz", "")[-6:]
        for b in buy_list
    }

    candidates = []
    for c in enriched:
        item = _simplify_candidate(c)
        item["acted"] = item["code"] in buy_codes
        candidates.append(item)

    snapshot = {
        "timestamp": ts,
        "trade_date": datetime.now().strftime("%Y-%m-%d"),
        "regime": regime,
        "v2_plan": v2_plan,
        "buy_codes": list(buy_codes),
        "candidates": candidates,
    }
    data["snapshots"].append(snapshot)
    _save(data)
    logger.info(f"[youzi_watchlist] snapshot saved ({len(candidates)} candidates, {len(buy_codes)} bought)")
    return ts


def list_snapshots(days: int = KEEP_DAYS) -> List[Dict]:
    data = _prune(_load())
    cutoff = datetime.now() - timedelta(days=days)
    out = []
    for s in data["snapshots"]:
        try:
            ts = datetime.strptime(s["timestamp"], "%Y-%m-%d %H:%M:%S")
            if ts >= cutoff:
                out.append(s)
        except Exception:
            continue
    return out


def snapshots_needing_review(horizon_days: int = 5) -> List[Dict]:
    """返回距今 >= horizon_days 的快照（可以安全做 T+N 回访）。"""
    cutoff = datetime.now() - timedelta(days=horizon_days)
    out = []
    for s in list_snapshots():
        try:
            ts = datetime.strptime(s["timestamp"], "%Y-%m-%d %H:%M:%S")
            if ts <= cutoff:
                out.append(s)
        except Exception:
            continue
    return out
