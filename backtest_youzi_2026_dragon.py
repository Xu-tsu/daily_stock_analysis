# -*- coding: utf-8 -*-
"""游资·龙头板策略回测（2026 YTD）·终版
===========================================
在 calibrated 版基础上，加入连板接力 & 连板天花板清仓逻辑。

【入场信号优先级（T 日收盘决策，T+1 开盘成交）】
  1. 一进二 (D1→D2)：T 日首板 (today_limit & consec_limit==1)
                     → T+1 开盘追（若高开 ≤ +5%），博 2 连板
  2. 二进三 (D2→D3)：T 日二板 (today_limit & consec_limit==2)
                     → T+1 开盘追（若高开 ≤ +5%），博 3 连板
  3. 预涨停候选：T 日涨幅 6.5~9.5% + vol_ratio 1.3~3 + 收盘位于日内高位
                 （close 距 high < 1% = 接近封板式强势）→ T+1 追
  4. 启动位 (ouyang 风格)：经 oracle 校准的 trend / momentum 票
                            （今日温和上涨、未涨停、未过热）→ T+1 买

【仓位分配（资源倾斜）】
  -  一进二 / 二进三：单票最多 50% 仓
  -  预涨停：单票 30% 仓
  -  启动位：单票 30% 仓

【出场规则】
  强制止盈（连板天花板）：
    板 ≥ 5            → 次日开盘全清
    板 == 4           → 当日 TP_HALF 50%
    板 == 3 且 RSI>78 → 当日 TP_HALF 50%
  炸板止损：
    持有期内若某日高开涨停后 close 回落 >4% （炸板）→ 次日开盘全清
  常规 SL/TRAIL/EXPIRE 继承 V2 REGIME_CONFIG

【REGIME】
  CRASH → 强制空仓（保护资金）
  BEAR  → 仅允许 ouyang 启动位 + 高质量 1→2
  SIDE  → 全量
  BULL  → 全量 + 板强化

【风格权重修正】
  ouyang 全 regime 加权（BULL 0.8 → 1.6，CRASH 0.5 → 1.3）
  chen_xiaoqun 仅在"首板/二板候选"时加权，其他时候降权避免全做"龙头打板"
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


# ── Dragon 模式的 regime 权重（ouyang 大幅加权） ──
REGIME_WEIGHTS_DRAGON = {
    "BULL":  {"chen_xiaoqun": 1.1, "zhao_laoge": 1.1, "zhang_jiahu": 1.0, "ouyang": 1.6},
    "SIDE":  {"chen_xiaoqun": 0.9, "zhao_laoge": 0.9, "zhang_jiahu": 0.7, "ouyang": 1.6},
    "BEAR":  {"chen_xiaoqun": 0.4, "zhao_laoge": 0.4, "zhang_jiahu": 0.3, "ouyang": 1.5},
    "CRASH": {"chen_xiaoqun": 0.0, "zhao_laoge": 0.0, "zhang_jiahu": 0.0, "ouyang": 0.0},
}

# ── 连板天花板规则 ──
CEILING_FULL_CLEAR = 5       # 5 板及以上全清
CEILING_HALF_CLEAR = 4       # 4 板减半
CEILING_HALF_RSI   = 78      # 3 板 + 高 RSI 也减半

# ── 炸板判定：涨停开盘后当日跌幅超过此值 ──
BOARD_BREAK_DROP = 4.0


# ── 候选特征与分类 ──

def _classify_entry(ind: Dict) -> Optional[str]:
    """把候选分到 4 类入场信号之一，否则返回 None 被丢弃。"""
    tc = ind["today_chg"]
    vr = ind["vol_ratio"]
    cl = int(ind.get("consec_limit", 0))
    tl = bool(ind["today_limit"])
    rsi = ind["rsi"]
    chg5 = ind["chg_5d"]

    # 共通强过滤（oracle 校准）
    if rsi > 75: return None
    if chg5 > 18: return None

    # 一进二：T 日首板（今天刚涨停，历史 0 板）
    if tl and cl == 1:
        return "1to2"

    # 二进三：T 日二板
    if tl and cl == 2:
        return "2to3"

    # 连板 ≥ 3：不买（已处高位）
    if tl and cl >= 3:
        return None

    # 预涨停：今日 6.5~9.5% + 温和放量 + 非已涨停 + 非已跌
    if 6.5 <= tc <= 9.5 and 1.3 <= vr <= 3.2 and not tl:
        return "pre_limit"

    # 启动位：温和上涨 + 未过热（ouyang 区间）
    if -2 <= tc <= 5 and 0.8 <= vr <= 2.5 and 3 <= ind["turnover"] <= 13:
        return "ignition"

    return None


def _score_by_entry(entry: str, ind: Dict) -> int:
    """按入场类型给基础分（后续叠加 youzi delta）。"""
    s = 50
    tc = ind["today_chg"]
    vr = ind["vol_ratio"]
    turn = ind["turnover"]
    cl = int(ind.get("consec_limit", 0))

    if entry == "1to2":
        s += 30                                  # 1→2 有天然溢价
        if 5 <= turn <= 15: s += 10              # 换手充分
        if 1.3 <= vr <= 2.5: s += 8              # 温和放量封板
        if ind["ma5_slope"] > 0: s += 5
    elif entry == "2to3":
        s += 20                                  # 加速板略保守
        if turn > 8: s += 5                      # 二板需换手
        if vr > 2.5: s -= 10                     # 二板爆量反而见顶信号
    elif entry == "pre_limit":
        s += 15
        if 1.5 <= vr <= 2.5: s += 10
        if 7.5 <= tc <= 9.0: s += 8              # 离涨停 1-2% 最佳
    elif entry == "ignition":
        s += 10
        if ind["chg_3d"] < 3: s += 8             # 更早期
        if 0.9 <= vr <= 1.8: s += 5              # 温和异动

    # 共通
    if ind["chg_5d"] > 12: s -= 15
    if ind["rsi"] > 70: s -= 8
    if cl >= 3: s -= 20  # 安全网

    return max(0, min(100, s))


def _build_candidate_dragon(code: str, name: str, df: pd.DataFrame, idx: int) -> Optional[Dict]:
    ind = calc_indicators(df, idx)
    if ind is None: return None
    row = df.iloc[idx]
    price = float(ind["price"])
    if price < 5 or price > 40: return None
    if ind["turnover"] < 2: return None

    entry = _classify_entry(ind)
    if entry is None: return None

    # 额外的"收盘封板强度"判定（预涨停要强势收）
    if entry == "pre_limit":
        high = float(row.get("high", price))
        if high > 0 and (high - price) / price > 0.012:   # 收盘距最高 >1.2% → 盘中拉高后回落
            return None

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
        "tech_score": _score_by_entry(entry, ind),
        "ma_trend": "bull" if ind["ma5_slope"] > 0 and ind["ma10_slope"] > 0
                    else ("bear" if ind["ma5_slope"] < 0 and ind["ma10_slope"] < 0 else "neutral"),
        "rsi": ind["rsi"],
        "_consec_limit": int(ind.get("consec_limit", 0)),
        "_today_limit": bool(ind["today_limit"]),
        "_chg_5d": ind["chg_5d"],
        "_entry_type": entry,
    }


class YouziBacktestDragon(YouziBacktest):
    """龙头板增强版。"""

    # ── 建仓时记录入场类型 & 连板 ──
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
            if sc > 0 and op >= sc * 1.098: continue  # 一字
            if sc > 0 and op > sc * 1.05: continue    # 高开 >5% 不追

            cost = op * (1 + SLIPPAGE_PCT)
            total_asset = self.cash + sum(h.market_value for h in self.holdings.values())
            mv_now = sum(h.market_value for h in self.holdings.values())

            # 按入场类型分配单票仓位
            entry = pb.get("_entry_type", "ignition")
            single_pct_map = {
                "1to2": 0.50, "2to3": 0.45,
                "pre_limit": 0.35, "ignition": 0.35,
            }
            single_pct = single_pct_map.get(entry, cfg["single_pct"])
            single_pct = min(single_pct, cfg["single_pct"] * 1.1)  # 不超 cfg 上限太多

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
            # 额外属性
            h.entry_type = entry
            h.entry_boards = pb.get("_consec_limit", 0)
            h.current_boards = pb.get("_consec_limit", 0)
            h.board_broken = False
            self.holdings[code] = h

            self.trades.append(Trade(
                date, code, pb.get("name", code), "buy", shares, cost, amount, comm,
                f"DRAGON|{entry}|b{h.entry_boards}|votes={'+'.join(pb.get('buy_votes',[]))}",
                self._current_regime,
            ))
            self._style_attribution[code] = {
                "entry_date": date, "cost": cost,
                "buy_votes": pb.get("buy_votes", []),
                "weighted_score": pb.get("weighted_score", 0),
                "per_style_verdicts": pb.get("per_style_verdicts", {}),
                "entry_type": entry,
            }
            for sn in pb.get("buy_votes", []):
                self._style_buy_count[sn] += 1

    # ── 收盘更新：追踪连板/炸板 ──
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
                        # 炸板判定：今天原本冲板但收盘跌幅 > BOARD_BREAK_DROP
                        opn = float(row.get("open", h.current_price))
                        high = float(row.get("high", h.current_price))
                        if high >= opn * 1.095 and (high - h.current_price) / high * 100 > BOARD_BREAK_DROP:
                            h.board_broken = True
                        h.current_boards = new_boards

    # ── 盘后出场复盘：增加连板天花板清仓 ──
    def _review_sells(self, date, day_idx, cfg):
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx == day_idx: continue
            pnl = h.pnl_pct
            hold = day_idx - h.buy_day_idx
            pk = (h.peak_price - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            dd = (h.peak_price - h.current_price) / h.peak_price * 100 if h.peak_price > 0 else 0
            boards = getattr(h, "current_boards", 0)
            broken = getattr(h, "board_broken", False)

            # 1. 连板天花板全清
            if boards >= CEILING_FULL_CLEAR:
                self._pending_sells.append((code, h.shares, f"CEIL{boards}板全清"))
                continue
            # 2. 炸板
            if broken:
                self._pending_sells.append((code, h.shares, f"BOARDBREAK@{boards}板"))
                continue
            # 3. 4 板减半
            if boards == CEILING_HALF_CLEAR and not h.partial_sold:
                half = max(100, (h.shares // 200) * 100)
                if half < h.shares:
                    self._pending_sells.append((code, half, f"CEIL{boards}板减半"))
                    h.partial_sold = True
                    continue
            # 4. 3 板高 RSI 减半
            if boards == 3 and not h.partial_sold:
                # 用最新日 rsi
                idx = self._get_idx(code, date)
                if idx >= 0:
                    ind = calc_indicators(self.all_data[code], idx) or {}
                    if ind.get("rsi", 0) > CEILING_HALF_RSI:
                        half = max(100, (h.shares // 200) * 100)
                        if half < h.shares:
                            self._pending_sells.append((code, half, f"CEIL3板RSI{ind.get('rsi',0):.0f}"))
                            h.partial_sold = True
                            continue

            # 原 V2 规则
            if pnl <= cfg["stop_loss"]:
                self._pending_sells.append((code, h.shares, f"SL{pnl:.1f}%")); continue
            if pk >= cfg["trail_trigger"] and dd >= cfg["trail_dd"]:
                self._pending_sells.append((code, h.shares, f"TRAIL pk{pk:.1f}%dd{dd:.1f}%")); continue
            if not h.partial_sold and pnl >= cfg["tp_half"]:
                half = max(100, (h.shares // 200) * 100)
                if half < h.shares:
                    self._pending_sells.append((code, half, f"TP_HALF{pnl:.1f}%"))
                    h.partial_sold = True; continue
            if pnl >= cfg["tp_full"]:
                self._pending_sells.append((code, h.shares, f"TP_FULL{pnl:.1f}%")); continue
            max_hold = cfg["hold_max"]
            if pnl > 2: max_hold += 2
            if hold >= max_hold and pnl < 1:
                self._pending_sells.append((code, h.shares, f"EXPIRE{hold}d"))

    # ── 选股逻辑：dragon 版 ──
    def _scan_buys_youzi(self, date, day_idx, cfg):
        if len(self.holdings) >= cfg["max_positions"]: return
        if self._pending_buys: return
        if self._current_regime == "CRASH":
            return  # 强制空仓

        weights = REGIME_WEIGHTS_DRAGON.get(self._current_regime, REGIME_WEIGHTS_DRAGON["SIDE"])

        pool = []
        for code, df in self.all_data.items():
            if code in self.holdings: continue
            idx = self._get_idx(code, date)
            if idx < 30: continue
            c = _build_candidate_dragon(code, self.name_map.get(code, code), df, idx)
            if c is None: continue
            pool.append((c, df, idx))

        # 入场类型优先级排序：1to2 > 2to3 > pre_limit > ignition
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

            # BEAR 限制：只收 ignition + 1to2
            if self._current_regime == "BEAR" and cand["_entry_type"] in ("2to3", "pre_limit"):
                continue

            scored.append({
                "code": cand["code"], "name": cand["name"],
                "signal_close": cand["price"],
                "final_score": final_score,
                "weighted_score": agg["weighted_score"],
                "buy_votes": agg["buy_votes"],
                "per_style_verdicts": {k: v.verdict for k, v in per.items()},
                "reason": f"DRAGON|{self._current_regime}|{cand['_entry_type']}|ws={agg['weighted_score']:.0f}",
                "strategy": self._current_regime,
                "_entry_type": cand["_entry_type"],
                "_consec_limit": cand["_consec_limit"],
                "_today_limit": cand["_today_limit"],
            })

        if not scored: return
        # 先按入场优先级，再按 final_score
        scored.sort(key=lambda x: (ENTRY_PRIORITY.get(x["_entry_type"], 9),
                                    -x["final_score"]))
        slots = cfg["max_positions"] - len(self.holdings)
        for c in scored[:max(1, slots)]:
            self._pending_buys.append(c)


def main(start: str = "2026-01-05", end: str = "2026-04-09"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")

    cache = Path("data/backtest_cache_2026ytd.pkl")
    all_data = pickle.loads(cache.read_bytes())
    logger.info(f"loaded {len(all_data)} stocks")

    engine = YouziBacktestDragon(all_data, top_universe=60)
    engine.run(start, end)

    # 入场类型统计
    entry_stats = defaultdict(lambda: {"n": 0, "wins": 0, "pnl_sum": 0.0})
    # 从 trades 配对算胜率（按入场类型）
    by_code = defaultdict(list)
    for t in engine.trades: by_code[t.code].append(t)
    for code, lst in by_code.items():
        buys = [t for t in lst if t.direction == "buy"]
        sells = [t for t in lst if t.direction == "sell"]
        if not buys or not sells: continue
        bp = sum(t.price*t.shares for t in buys) / max(1, sum(t.shares for t in buys))
        sp = sum(t.price*t.shares for t in sells) / max(1, sum(t.shares for t in sells))
        pnl = (sp-bp)/bp*100
        # entry type 在 buy reason: DRAGON|entry|bN
        etype = "unknown"
        for r_ in [b.reason for b in buys]:
            for et in ("1to2","2to3","pre_limit","ignition"):
                if et in (r_ or ""):
                    etype = et; break
            if etype != "unknown": break
        entry_stats[etype]["n"] += 1
        if pnl > 0: entry_stats[etype]["wins"] += 1
        entry_stats[etype]["pnl_sum"] += pnl

    print("\n  ─── 入场类型归因 ───")
    print(f"  {'类型':<12} {'n':>4} {'胜率':>7} {'平均盈亏%':>10}")
    for et in ("1to2","2to3","pre_limit","ignition","unknown"):
        s = entry_stats[et]
        if s["n"] == 0: continue
        wr = s["wins"]/s["n"]*100
        avg = s["pnl_sum"]/s["n"]
        print(f"  {et:<12} {s['n']:>4} {wr:>6.1f}% {avg:>+9.2f}%")
    print()

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
        "entry_attribution": {et: dict(v) for et, v in entry_stats.items()},
    }
    import json
    out_path = Path(f"reports/youzi_backtest_2026ytd_DRAGON_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
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
