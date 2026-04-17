# -*- coding: utf-8 -*-
"""游资·龙头板 v2（2026 YTD）—— 让龙头跑满
=====================================================
在 v1 dragon 基础上的三大升级：

(A) 入场增强：封板压单强度（seal_strength）
    - ULTRA / STRONG 封板 → 入场 +30~+50 分
    - WEAK / FRAGILE → 直接过滤
    - 预判次日接力概率（连板衰减曲线）

(B) 龙头专属止盈表（彻底放开主升空间）
    BULL regime:
      stop_loss     = -6.0   (原 -5.0，给龙头日内晃动空间)
      tp_half       = +15.0  (原 +6.0，等主升确认)
      tp_full       = +35.0  (原 +15.0，博连板加速)
      trail_trigger = +20.0  (原 +8.0)
      trail_dd      = 10.0   (原 5.0，连板正常回踩不砍)
      hold_max      = 8d     (原 4d)
    SIDE/BEAR 保守一点但也放宽。

(C) ULTRA 封 × 1→2 → ALL-IN 单票 80%
    （其他仓位配置保持不变）

连板天花板规则保留：5 板全清 / 4 板减半 / 3 板+RSI>78 减半 / 炸板清仓。
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from collections import defaultdict
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
from youzi_styles import NewsContext, score_all_styles, aggregate_verdict, list_styles
from youzi_timeframe import build_timeframe_context
from backtest_youzi_2026 import YouziBacktest
from backtest_youzi_2026_dragon import (
    _classify_entry, _score_by_entry, REGIME_WEIGHTS_DRAGON,
    CEILING_FULL_CLEAR, CEILING_HALF_CLEAR, CEILING_HALF_RSI,
    BOARD_BREAK_DROP,
)
from seal_strength import estimate_seal_from_daily, estimate_next_day_continuation_prob


# ── 龙头专属止盈表（彻底重写 REGIME_CONFIG 的退出阈值） ──
DRAGON_EXIT_CONFIG = {
    "BULL": {
        "stop_loss":     -6.0,
        "tp_half":       15.0,
        "tp_full":       35.0,
        "trail_trigger": 20.0,
        "trail_dd":      10.0,
        "hold_max":      8,
    },
    "SIDE": {
        "stop_loss":     -5.0,
        "tp_half":       10.0,
        "tp_full":       25.0,
        "trail_trigger": 15.0,
        "trail_dd":      8.0,
        "hold_max":      6,
    },
    "BEAR": {
        "stop_loss":     -4.0,
        "tp_half":       6.0,
        "tp_full":       15.0,
        "trail_trigger": 8.0,
        "trail_dd":      5.0,
        "hold_max":      4,
    },
    "CRASH": REGIME_CONFIG_V2["CRASH"],
}


def _build_candidate_with_seal(code: str, name: str, df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """在 dragon v1 候选基础上附加封板强度。"""
    ind = calc_indicators(df, idx)
    if ind is None: return None
    row = df.iloc[idx]
    price = float(ind["price"])
    if price < 5 or price > 40: return None
    if ind["turnover"] < 2: return None

    entry = _classify_entry(ind)
    if entry is None: return None

    # 对涨停类候选评估封板强度
    seal = None
    if entry in ("1to2", "2to3"):
        seal = estimate_seal_from_daily(df, idx)
        # WEAK / FRAGILE 原本一刀切 veto，误伤高换手游资龙头
        # （如圣阳 002580 4/9 FRAGILE(22) 但随后连拉 6 板）
        # 规则放宽：仅在"真·弱封"时否决 —— 成交低迷 或 1→2 低换手
        turnover = float(ind.get("turnover", 0))
        if seal.tier == "WEAK" and turnover < 4.0:
            return None
        if seal.tier == "FRAGILE":
            # FRAGILE 但高换手（>6%）且 1→2 首板，允许进入（打分时再扣分）
            if not (entry == "1to2" and turnover >= 6.0):
                return None
    elif entry == "pre_limit":
        # 预涨停不是涨停，但衡量"收盘离封板差距"
        seal = None

    if entry == "pre_limit":
        high = float(row.get("high", price))
        if high > 0 and (high - price) / price > 0.012:
            return None

    # 基础分
    base = _score_by_entry(entry, ind)

    # 封板强度加分
    seal_bonus = 0
    if seal is not None:
        bonus_map = {"ULTRA": 30, "STRONG": 18, "MEDIUM": 8, "WEAK": -5, "FRAGILE": -15}
        seal_bonus = bonus_map.get(seal.tier, 0)

    # 次日接力概率调整
    next_prob = 0.0
    if seal is not None:
        next_prob = estimate_next_day_continuation_prob(seal, int(ind.get("consec_limit", 0)))
        if next_prob >= 0.55:
            seal_bonus += 8
        elif next_prob < 0.25:
            seal_bonus -= 8

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
        "signal_type": entry,
        "tech_score": max(0, min(100, base + seal_bonus)),
        "ma_trend": "bull" if ind["ma5_slope"] > 0 and ind["ma10_slope"] > 0
                    else ("bear" if ind["ma5_slope"] < 0 and ind["ma10_slope"] < 0 else "neutral"),
        "rsi": ind["rsi"],
        "_consec_limit": int(ind.get("consec_limit", 0)),
        "_today_limit": bool(ind["today_limit"]),
        "_chg_5d": ind["chg_5d"],
        "_entry_type": entry,
        "_seal_tier": seal.tier if seal else "NONE",
        "_seal_score": seal.score if seal else 0,
        "_next_prob": next_prob,
    }


class YouziBacktestDragonV2(YouziBacktest):
    """龙头 v2：封板强度 + 专属止盈 + ULTRA ALL-IN。"""

    def _exec_pending_buys(self, date, day_idx):
        buys = self._pending_buys[:]; self._pending_buys = []
        cfg = REGIME_CONFIG_V2[self._current_regime]
        pm = self._position_multiplier()

        for pb in buys:
            code = pb["code"]
            if code in self.holdings or len(self.holdings) >= cfg["max_positions"]:
                continue
            row = self._get_row(code, date)
            if row is None: continue
            op = float(row["open"])
            if op <= 0: continue
            sc = pb.get("signal_close", 0)
            if sc > 0 and op >= sc * 1.098: continue
            if sc > 0 and op > sc * 1.05: continue

            cost = op * (1 + SLIPPAGE_PCT)
            total_asset = self.cash + sum(h.market_value for h in self.holdings.values())
            mv_now = sum(h.market_value for h in self.holdings.values())

            entry = pb.get("_entry_type", "ignition")
            seal_tier = pb.get("_seal_tier", "NONE")

            # ── ALL-IN 规则：ULTRA 封 + 1→2 → 吃满 85%
            if seal_tier == "ULTRA" and entry == "1to2":
                single_pct = 0.85
            elif seal_tier == "STRONG" and entry == "1to2":
                single_pct = 0.60
            elif seal_tier == "STRONG" and entry == "2to3":
                single_pct = 0.50
            elif entry == "1to2":
                single_pct = 0.45
            elif entry == "2to3":
                single_pct = 0.40
            elif entry == "pre_limit":
                single_pct = 0.35
            else:
                single_pct = 0.30

            max_mv = total_asset * cfg["total_pct"] * pm
            remaining = max_mv - mv_now
            if remaining <= 0: continue
            buy_amount = min(self.cash * 0.95, total_asset * single_pct * pm, remaining)
            shares = int(buy_amount / cost / 100) * 100
            if shares < 100: continue
            amount = cost * shares; comm = max(amount * COMMISSION_RATE, 5)
            if amount + comm > self.cash: continue

            self.cash -= (amount + comm)
            h = Holding(
                code=code, name=pb.get("name", code), shares=shares,
                cost_price=cost, buy_date=date, buy_day_idx=day_idx,
                strategy=pb.get("strategy", self._current_regime),
                current_price=op, peak_price=op,
            )
            h.entry_type = entry
            h.entry_boards = pb.get("_consec_limit", 0)
            h.current_boards = pb.get("_consec_limit", 0)
            h.board_broken = False
            h.entry_seal = seal_tier
            self.holdings[code] = h

            self.trades.append(Trade(
                date, code, pb.get("name", code), "buy", shares, cost, amount, comm,
                f"DRAGONv2|{entry}|b{h.entry_boards}|seal={seal_tier}|allin={single_pct*100:.0f}%",
                self._current_regime,
            ))
            self._style_attribution[code] = {
                "entry_date": date, "cost": cost,
                "buy_votes": pb.get("buy_votes", []),
                "weighted_score": pb.get("weighted_score", 0),
                "per_style_verdicts": pb.get("per_style_verdicts", {}),
                "entry_type": entry,
                "seal": seal_tier,
            }
            for sn in pb.get("buy_votes", []):
                self._style_buy_count[sn] += 1

    def _update_close(self, date):
        for code, h in self.holdings.items():
            row = self._get_row(code, date)
            if row is None: continue
            h.current_price = float(row["close"])
            h.peak_price = max(h.peak_price, float(row["high"]))
            idx = self._get_idx(code, date)
            if idx >= 0:
                df = self.all_data.get(code)
                if df is not None:
                    ind = calc_indicators(df, idx)
                    if ind is not None:
                        new_boards = int(ind.get("consec_limit", 0))
                        opn = float(row.get("open", h.current_price))
                        high = float(row.get("high", h.current_price))
                        if high >= opn * 1.095 and (high - h.current_price) / high * 100 > BOARD_BREAK_DROP:
                            h.board_broken = True
                        h.current_boards = new_boards

    # ── 出场：龙头专属止盈表 ──
    def _review_sells(self, date, day_idx, cfg):
        # 用 DRAGON_EXIT_CONFIG 覆盖退出阈值
        dcfg = DRAGON_EXIT_CONFIG.get(self._current_regime, cfg)
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx == day_idx: continue
            pnl = h.pnl_pct
            hold = day_idx - h.buy_day_idx
            pk = (h.peak_price - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            dd = (h.peak_price - h.current_price) / h.peak_price * 100 if h.peak_price > 0 else 0
            boards = getattr(h, "current_boards", 0)
            broken = getattr(h, "board_broken", False)

            # 连板天花板
            if boards >= CEILING_FULL_CLEAR:
                self._pending_sells.append((code, h.shares, f"CEIL{boards}板全清"))
                continue
            if broken:
                self._pending_sells.append((code, h.shares, f"BOARDBREAK@{boards}板"))
                continue
            if boards == CEILING_HALF_CLEAR and not h.partial_sold:
                half = max(100, (h.shares // 200) * 100)
                if half < h.shares:
                    self._pending_sells.append((code, half, f"CEIL{boards}板减半"))
                    h.partial_sold = True
                    continue
            if boards == 3 and not h.partial_sold:
                idx = self._get_idx(code, date)
                if idx >= 0:
                    ind = calc_indicators(self.all_data[code], idx) or {}
                    if ind.get("rsi", 0) > CEILING_HALF_RSI:
                        half = max(100, (h.shares // 200) * 100)
                        if half < h.shares:
                            self._pending_sells.append((code, half, f"CEIL3板RSI{ind.get('rsi',0):.0f}"))
                            h.partial_sold = True
                            continue

            # 龙头专属止盈
            if pnl <= dcfg["stop_loss"]:
                self._pending_sells.append((code, h.shares, f"SL{pnl:.1f}%")); continue
            if pk >= dcfg["trail_trigger"] and dd >= dcfg["trail_dd"]:
                self._pending_sells.append((code, h.shares, f"TRAIL pk{pk:.1f}%dd{dd:.1f}%")); continue
            if not h.partial_sold and pnl >= dcfg["tp_half"]:
                half = max(100, (h.shares // 200) * 100)
                if half < h.shares:
                    self._pending_sells.append((code, half, f"TP_HALF{pnl:.1f}%"))
                    h.partial_sold = True; continue
            if pnl >= dcfg["tp_full"]:
                self._pending_sells.append((code, h.shares, f"TP_FULL{pnl:.1f}%")); continue
            max_hold = dcfg["hold_max"]
            if pnl > 2: max_hold += 2
            if hold >= max_hold and pnl < 1:
                self._pending_sells.append((code, h.shares, f"EXPIRE{hold}d"))

    def _scan_buys_youzi(self, date, day_idx, cfg):
        if len(self.holdings) >= cfg["max_positions"]: return
        if self._pending_buys: return
        if self._current_regime == "CRASH": return

        weights = REGIME_WEIGHTS_DRAGON.get(self._current_regime, REGIME_WEIGHTS_DRAGON["SIDE"])

        pool = []
        for code, df in self.all_data.items():
            if code in self.holdings: continue
            idx = self._get_idx(code, date)
            if idx < 30: continue
            c = _build_candidate_with_seal(code, self.name_map.get(code, code), df, idx)
            if c is None: continue
            pool.append((c, df, idx))

        ENTRY_PRIORITY = {"1to2": 0, "2to3": 1, "pre_limit": 2, "ignition": 3}
        pool.sort(key=lambda t: (ENTRY_PRIORITY.get(t[0]["_entry_type"], 9),
                                   -t[0]["tech_score"]))
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

            vetoed_all = bool(agg["vetoed_by"]) and not agg["buy_votes"]
            if vetoed_all: continue
            if not agg["buy_votes"]: continue
            if self._current_regime == "BEAR" and cand["_entry_type"] in ("2to3", "pre_limit"):
                continue

            scored.append({
                "code": cand["code"], "name": cand["name"],
                "signal_close": cand["price"],
                "final_score": final_score,
                "weighted_score": agg["weighted_score"],
                "buy_votes": agg["buy_votes"],
                "per_style_verdicts": {k: v.verdict for k, v in per.items()},
                "reason": (f"DRAGONv2|{self._current_regime}|{cand['_entry_type']}|"
                           f"seal={cand['_seal_tier']}|np={cand['_next_prob']:.2f}"),
                "strategy": self._current_regime,
                "_entry_type": cand["_entry_type"],
                "_consec_limit": cand["_consec_limit"],
                "_today_limit": cand["_today_limit"],
                "_seal_tier": cand["_seal_tier"],
                "_next_prob": cand["_next_prob"],
            })

        if not scored: return
        # ULTRA 封 1→2 单独最高优先级
        def _prio(x):
            base_p = ENTRY_PRIORITY.get(x["_entry_type"], 9)
            if x["_seal_tier"] == "ULTRA": base_p -= 0.5
            if x["_seal_tier"] == "STRONG": base_p -= 0.2
            return (base_p, -x["final_score"])
        scored.sort(key=_prio)
        slots = cfg["max_positions"] - len(self.holdings)
        # 如果头号候选是 ULTRA 1→2 且 regime=BULL，只用 1 个仓位（ALL-IN）
        top = scored[0]
        if top["_seal_tier"] == "ULTRA" and top["_entry_type"] == "1to2" \
                and self._current_regime == "BULL":
            self._pending_buys.append(top)
        else:
            for c in scored[:max(1, slots)]:
                self._pending_buys.append(c)


def main(start: str = "2026-01-05", end: str = "2026-04-09"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")
    cache = Path("data/backtest_cache_2026ytd.pkl")
    all_data = pickle.loads(cache.read_bytes())
    logger.info(f"loaded {len(all_data)} stocks")

    engine = YouziBacktestDragonV2(all_data, top_universe=60)
    engine.run(start, end)

    # 汇总 seal tier 分布
    seal_counts = defaultdict(int)
    seal_wins = defaultdict(int)
    seal_pnl = defaultdict(float)
    by_code = defaultdict(list)
    for t in engine.trades: by_code[t.code].append(t)
    for code, lst in by_code.items():
        buys = [t for t in lst if t.direction == "buy"]
        sells = [t for t in lst if t.direction == "sell"]
        if not buys or not sells: continue
        bp = sum(t.price*t.shares for t in buys) / max(1, sum(t.shares for t in buys))
        sp = sum(t.price*t.shares for t in sells) / max(1, sum(t.shares for t in sells))
        pnl = (sp-bp)/bp*100
        seal = "NONE"
        for r in [b.reason for b in buys]:
            for st in ("ULTRA", "STRONG", "MEDIUM", "WEAK", "FRAGILE", "NONE"):
                if f"seal={st}" in (r or ""):
                    seal = st; break
            if seal != "NONE": break
        seal_counts[seal] += 1
        if pnl > 0: seal_wins[seal] += 1
        seal_pnl[seal] += pnl

    print("\n  ─── 封板强度归因 ───")
    print(f"  {'tier':<10} {'n':>4} {'胜率':>7} {'平均盈亏%':>10}")
    for st in ("ULTRA","STRONG","MEDIUM","WEAK","FRAGILE","NONE"):
        n = seal_counts[st]
        if n == 0: continue
        wr = seal_wins[st]/n*100
        avg = seal_pnl[st]/n
        print(f"  {st:<10} {n:>4} {wr:>6.1f}% {avg:>+9.2f}%")

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
    out_path = Path(f"reports/youzi_backtest_2026ytd_DRAGONv2_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
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
