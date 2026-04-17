# -*- coding: utf-8 -*-
"""Oracle 校准版回测（2026 YTD）。

基于 tools/oracle_diagnostic.py 产出的"事后龙头特征中位数/四分位"
反推出的硬过滤 + 重打分：

校准点：
  1. today_chg 过滤：[-3%, +7%]（原来无上限）
     Oracle P75 仅 +4.4%，实际买入中位 +10% → 彻底压住追高。
  2. chg_5d 过滤：≤ 15%（原 <= 50）
     Oracle P75 仅 +10.9%，实际买入中位 +20% → 避开已涨多。
  3. RSI 过滤：≤ 72
     Oracle P75 65 → 超热不碰。
  4. vol_ratio 过滤：≤ 3.0
     Oracle P75 1.7 → 爆量尾部不追。
  5. turnover：[3%, 15%]（原 >=1%）
     Oracle 甜蜜区。
  6. Signal 偏好：trend / momentum / relay / breakout 优先，board_hit 只在 consec_limit==0 首板时参与
     Oracle 83% 是 trend，涨停只占 10%。
  7. 允许空头排列 + MA 不要求多头：Oracle 43% 样本非多头排列。
  8. Regime 适配：CRASH regime 完全跳过买入（保护资金）
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from backtest_adaptive_v2 import (
    INITIAL_CAPITAL, COMMISSION_RATE, STAMP_TAX_RATE, SLIPPAGE_PCT,
    Holding, Trade, Snapshot,
    calc_indicators, MarketRegimeDetector, REGIME_CONFIG_V2,
)
from youzi_styles import (
    NewsContext, STYLE_REGISTRY,
    score_all_styles, aggregate_verdict, list_styles,
)
from youzi_timeframe import TimeframeContext, build_timeframe_context

from backtest_youzi_2026 import YouziBacktest, REGIME_WEIGHTS


# ── Oracle 校准阈值（来自 oracle_diagnostic） ──
CAL_TODAY_CHG_MAX = 7.0
CAL_TODAY_CHG_MIN = -3.5
CAL_CHG5D_MAX     = 15.0
CAL_RSI_MAX       = 72.0
CAL_VOL_MAX       = 3.0
CAL_VOL_MIN       = 0.6
CAL_TURN_MIN      = 3.0
CAL_TURN_MAX      = 15.0
CAL_PRICE_MIN     = 5.0
CAL_PRICE_MAX     = 35.0
CAL_SKIP_CRASH    = True


def _derive_signal_calib(ind: Dict) -> str:
    # 弱化涨停归类：首板可以，连板直接忽略
    if ind["today_limit"] and ind.get("consec_limit", 0) == 1:
        return "board_hit"
    if ind["today_limit"] and ind.get("consec_limit", 0) >= 2:
        return "board_hit_late"  # 标记，会被过滤
    if ind["yest_limit"] and 2 < ind["today_chg"] < 6:
        return "relay"
    if ind.get("vol_breakout") and ind["today_chg"] < 7:
        return "breakout"
    if 3 <= ind["chg_3d"] <= 12 and 0 < ind["today_chg"] < 5:
        return "momentum"
    return "trend"


def _base_tech_score_calib(ind: Dict) -> int:
    score = 45
    # 温和上涨加分（不是越涨越好）
    tc = ind["today_chg"]
    if 0.5 <= tc <= 4: score += 12
    elif 4 < tc <= 6: score += 5
    elif tc > 7: score -= 15             # 追高扣
    elif tc < -3: score -= 10

    # 5 日涨幅：中位数区加分
    if -3 <= ind["chg_5d"] <= 12: score += 10
    elif ind["chg_5d"] > 20: score -= 25

    # 换手：甜蜜区加分
    if 5 <= ind["turnover"] <= 13: score += 12
    elif ind["turnover"] < 2 or ind["turnover"] > 18: score -= 10

    # 温和放量
    vr = ind["vol_ratio"]
    if 1.0 <= vr <= 2.2: score += 8
    elif vr > 3: score -= 10

    # 均线不要求多头（oracle 43% 样本非多头），但要求不是崩溃
    if ind["ma5_slope"] > -1 and ind["ma10_slope"] > -1.5: score += 5

    # RSI
    if ind["rsi"] > 72: score -= 10
    elif 45 <= ind["rsi"] <= 65: score += 5

    # 涨停（首板）温和加分
    if ind["today_limit"] and ind.get("consec_limit", 0) == 1: score += 5

    return max(0, min(100, score))


def _build_candidate_calib(code: str, name: str, df: pd.DataFrame, idx: int) -> Optional[Dict]:
    ind = calc_indicators(df, idx)
    if ind is None: return None
    row = df.iloc[idx]
    price = float(ind["price"])
    if price < CAL_PRICE_MIN or price > CAL_PRICE_MAX: return None
    if ind["turnover"] < CAL_TURN_MIN or ind["turnover"] > CAL_TURN_MAX: return None
    if ind["today_chg"] < CAL_TODAY_CHG_MIN or ind["today_chg"] > CAL_TODAY_CHG_MAX: return None
    if ind["chg_5d"] > CAL_CHG5D_MAX: return None
    if ind["rsi"] > CAL_RSI_MAX: return None
    vr = ind["vol_ratio"]
    if vr < CAL_VOL_MIN or vr > CAL_VOL_MAX: return None

    sig = _derive_signal_calib(ind)
    if sig == "board_hit_late": return None  # 连板不追

    mc = 0.0
    if "outstanding_share" in df.columns:
        mc = float(row.get("outstanding_share", 0) or 0) * price

    return {
        "code": code,
        "name": str(row.get("stock_name", name or code)),
        "price": price,
        "change_pct": ind["today_chg"],
        "turnover_rate": ind["turnover"],
        "market_cap": mc,
        "vol_ratio": ind["vol_ratio"],
        "signal_type": sig,
        "tech_score": _base_tech_score_calib(ind),
        "ma_trend": "bull" if ind["ma5_slope"] > 0 and ind["ma10_slope"] > 0
                    else ("bear" if ind["ma5_slope"] < 0 and ind["ma10_slope"] < 0 else "neutral"),
        "rsi": ind["rsi"],
    }


class YouziBacktestCalibrated(YouziBacktest):
    """使用 oracle 校准的候选构造 + 打分。"""

    def _scan_buys_youzi(self, date, day_idx, cfg):
        if len(self.holdings) >= cfg["max_positions"]: return
        if self._pending_buys: return

        # CRASH 日跳过买入
        if CAL_SKIP_CRASH and self._current_regime == "CRASH":
            return

        weights = REGIME_WEIGHTS.get(self._current_regime, REGIME_WEIGHTS["SIDE"])

        pool = []
        for code, df in self.all_data.items():
            if code in self.holdings: continue
            idx = self._get_idx(code, date)
            if idx < 30: continue
            c = _build_candidate_calib(code, self.name_map.get(code, code), df, idx)
            if c is None: continue
            pool.append((c, df, idx))

        pool.sort(key=lambda t: t[0]["tech_score"], reverse=True)
        pool = pool[:self.top_universe]

        scored = []
        for cand, df, idx in pool:
            try:
                from backtest_youzi_2026 import _build_tf_no_lookahead
                tf = _build_tf_no_lookahead(cand["code"], df, idx)
            except Exception:
                continue
            news = NewsContext()
            per = score_all_styles(cand, tf, news,
                                    regime=self._current_regime, active=list_styles())
            agg = aggregate_verdict(per, weights=weights)
            total_delta, ws = 0.0, 0.0
            for sn, r in per.items():
                if r.verdict == "veto": continue
                w = weights.get(sn, 1.0)
                total_delta += r.score_delta * w; ws += w
            avg_delta = total_delta / ws if ws > 0 else 0
            final_score = cand["tech_score"] + avg_delta

            # 放宽阈值：校准后整体门槛调低，但要求 buy_votes ≥ 1
            vetoed_all = bool(agg["vetoed_by"]) and not agg["buy_votes"]
            if vetoed_all: continue
            if not agg["buy_votes"]: continue  # 必须至少 1 票 buy

            scored.append({
                "code": cand["code"], "name": cand["name"],
                "signal_close": cand["price"],
                "final_score": final_score,
                "weighted_score": agg["weighted_score"],
                "buy_votes": agg["buy_votes"],
                "per_style_verdicts": {k: v.verdict for k, v in per.items()},
                "reason": f"CALIB|{self._current_regime}|sig={cand['signal_type']}|ws={agg['weighted_score']:.0f}",
                "strategy": self._current_regime,
                "_sig": cand["signal_type"],
                "_today_chg": cand["change_pct"],
                "_chg_5d": 0,
            })

        if not scored: return
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        slots = cfg["max_positions"] - len(self.holdings)
        for c in scored[:max(1, slots)]:
            self._pending_buys.append(c)


def main(start: str = "2026-01-05", end: str = "2026-04-09"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")

    cache = Path("data/backtest_cache_2026ytd.pkl")
    all_data = pickle.loads(cache.read_bytes())
    logger.info(f"loaded {len(all_data)} stocks")

    engine = YouziBacktestCalibrated(all_data, top_universe=60)
    engine.run(start, end)

    out = {
        "snapshots": [s.__dict__ for s in engine.snapshots],
        "trades": [t.__dict__ for t in engine.trades],
        "style_attribution": {
            sn: {
                "buys": engine._style_buy_count.get(sn, 0),
                "wins": engine._style_win_count.get(sn, 0),
                "pnl_sum": engine._style_pnl_sum.get(sn, 0),
            } for sn in list_styles()
        },
    }
    import json
    out_path = Path(f"reports/youzi_backtest_2026ytd_CALIB_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    logger.info(f"saved: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-01-05")
    p.add_argument("--end", default="2026-04-09")
    args = p.parse_args()
    main(args.start, args.end)
