# -*- coding: utf-8 -*-
"""游资风格回测 · 2026 YTD
=====================================================
严格无未来函数：T 日收盘后用 df.iloc[:idx+1] 决策，T+1 开盘执行。

Pipeline:
    每日收盘：
      1. 用 MarketRegimeDetector 判断 regime（只看 <= today 数据）
      2. 对每支股票 calc_indicators(df, idx) → 候选 dict
      3. build_timeframe_context(code, kline=df.iloc[:idx+1])
         → 仅含截至 today 的日/5日/周/月指标
      4. score_all_styles(candidate, tf, news=NewsContext(), regime)
         → 每家游资打分
      5. aggregate_verdict 取 buy 共识
      6. 入队 pending_buys（T+1 开盘成交）
    次日开盘：
      7. pending_sells 按开盘价卖（V2 止盈/止损规则）
      8. pending_buys 按开盘价买
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

# 复用 V2 引擎的指标 / regime 判别 / 参数
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


# ──────────────────────────────────────────────────────────
# 候选构造：calc_indicators → candidate dict
# ──────────────────────────────────────────────────────────

def _derive_signal(ind: Dict) -> str:
    """从 calc_indicators 输出推断 signal_type。"""
    if ind["today_limit"] and ind.get("consec_limit", 0) == 1:
        return "board_hit"
    if ind["today_limit"] and ind.get("consec_limit", 0) >= 2:
        return "board_hit"   # 二连板仍归打板
    if ind["yest_limit"] and ind["today_chg"] > 3:
        return "relay"
    if ind.get("vol_breakout"):
        return "breakout"
    if ind["chg_3d"] >= 10 and ind["today_chg"] > 2:
        return "momentum"
    return "trend"


def _base_tech_score(ind: Dict) -> int:
    """无 LLM 的纯量化基础分（0~100），后续由 youzi 调整。"""
    score = 40
    if ind["today_limit"]: score += 20
    if ind["vol_breakout"]: score += 15
    if ind["chg_3d"] > 5: score += 10
    if 5 <= ind["turnover"] <= 15: score += 10
    if ind["ma5_slope"] > 0 and ind["ma10_slope"] > 0: score += 10
    if ind["rsi"] > 70: score -= 10
    if ind["chg_5d"] > 50: score -= 30
    return max(0, min(100, score))


def _build_candidate(code: str, name: str, df: pd.DataFrame, idx: int) -> Optional[Dict]:
    ind = calc_indicators(df, idx)
    if ind is None:
        return None
    row = df.iloc[idx]
    price = float(ind["price"])
    if price < 3 or price > 50:  # 与 V2 price_range 对齐
        return None
    if ind["turnover"] < 1:
        return None
    # 市值（元）：close × outstanding_share
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
        "signal_type": _derive_signal(ind),
        "tech_score": _base_tech_score(ind),
        "ma_trend": "bull" if ind["ma5_slope"] > 0 and ind["ma10_slope"] > 0
                    else ("bear" if ind["ma5_slope"] < 0 and ind["ma10_slope"] < 0 else "neutral"),
        "rsi": ind["rsi"],
    }


# ──────────────────────────────────────────────────────────
# 无未来 TimeframeContext（只用 df.iloc[:idx+1]）
# ──────────────────────────────────────────────────────────

def _build_tf_no_lookahead(code: str, df: pd.DataFrame, idx: int) -> TimeframeContext:
    past = df.iloc[:idx + 1].copy()
    if "date" in past.columns:
        past["date"] = pd.to_datetime(past["date"])
        past = past.set_index("date")
    return build_timeframe_context(code, kline=past)


# ──────────────────────────────────────────────────────────
# 回测引擎
# ──────────────────────────────────────────────────────────

REGIME_WEIGHTS = {
    "BULL":  {"chen_xiaoqun": 1.3, "zhao_laoge": 1.2, "zhang_jiahu": 1.2, "ouyang": 0.8},
    "SIDE":  {"chen_xiaoqun": 1.0, "zhao_laoge": 1.0, "zhang_jiahu": 0.7, "ouyang": 1.3},
    "BEAR":  {"chen_xiaoqun": 0.5, "zhao_laoge": 0.5, "zhang_jiahu": 0.3, "ouyang": 1.2},
    "CRASH": {"chen_xiaoqun": 0.2, "zhao_laoge": 0.2, "zhang_jiahu": 0.0, "ouyang": 0.5},
}


class YouziBacktest:
    def __init__(self, all_data: Dict[str, pd.DataFrame], top_universe: int = 60):
        """
        top_universe: 每天按基础分保留 top_universe 只进入 youzi 评分（控制耗时）
        """
        self.all_data = all_data
        self.top_universe = top_universe
        self.cash = INITIAL_CAPITAL
        self.holdings: Dict[str, Holding] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[Snapshot] = []
        self.prev_total = INITIAL_CAPITAL
        self._pending_buys: List[dict] = []
        self._pending_sells: List[tuple] = []  # (code, shares, reason)
        self._regime_detector = MarketRegimeDetector()
        self._current_regime = "SIDE"
        self._regime_history = []

        self._win_streak = 0
        self._loss_streak = 0

        # 每笔买入的游资风格归因
        self._style_attribution: Dict[str, Dict] = {}  # code -> {votes, weighted_score, per_style_verdicts, entry_date}
        # 每个风格的累计 buy 数
        self._style_buy_count: Dict[str, int] = defaultdict(int)
        self._style_win_count: Dict[str, int] = defaultdict(int)
        self._style_pnl_sum: Dict[str, float] = defaultdict(float)

        dates_set = set()
        for df in all_data.values():
            dates_set.update(df["date"].tolist())
        # 统一 date 为 str
        self.trading_days = sorted(str(d)[:10] for d in dates_set)
        self.day_idx_map = {d: i for i, d in enumerate(self.trading_days)}

        # 代码 → 名称（如果可用）
        self.name_map: Dict[str, str] = {}
        for c, df in all_data.items():
            if "stock_name" in df.columns and len(df) > 0:
                self.name_map[c] = str(df["stock_name"].iloc[0])
            else:
                self.name_map[c] = c

    def _position_multiplier(self) -> float:
        if self._win_streak >= 3: return 1.3
        elif self._win_streak >= 2: return 1.15
        elif self._loss_streak >= 3: return 0.6
        elif self._loss_streak >= 2: return 0.8
        return 1.0

    def run(self, start_date: str, end_date: str):
        days = [d for d in self.trading_days if start_date <= d <= end_date]
        logger.info(f"[Youzi Backtest] {start_date} ~ {end_date}  ({len(days)}d)")
        logger.info(f"[Youzi Backtest] initial capital: Y{INITIAL_CAPITAL:,.0f}  "
                    f"universe pool={len(self.all_data)}  top_universe={self.top_universe}")

        t0 = time.time()
        for day_num, date in enumerate(days):
            day_idx = self.day_idx_map.get(date, day_num)

            # 1) 开盘执行上一日 T+1 队列
            self._exec_pending_sells(date, day_idx)
            self._exec_pending_buys(date, day_idx)

            # 2) 盘中极端止损
            self._emergency_stop(date, day_idx)

            # 3) 收盘更新持仓估值
            self._update_close(date)

            # 4) 盘后 regime 判断
            regime_info = self._regime_detector.detect(self.all_data, date)
            self._current_regime = regime_info["regime"]
            self._regime_history.append({"date": date, **regime_info})

            # 5) 盘后复盘卖出
            cfg = REGIME_CONFIG_V2[self._current_regime]
            self._review_sells(date, day_idx, cfg)

            # 6) 盘后游资选股（无未来）
            self._scan_buys_youzi(date, day_idx, cfg)

            # 7) 快照
            mv = sum(h.market_value for h in self.holdings.values())
            total = self.cash + mv
            ret = (total - self.prev_total) / self.prev_total * 100 if self.prev_total > 0 else 0
            self.prev_total = total
            self.snapshots.append(Snapshot(date, total, self.cash, mv,
                                            len(self.holdings), ret, self._current_regime))

            if (day_num + 1) % 10 == 0:
                r = (total - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                logger.info(f"  [{date}] Day{day_num+1:3d} | Y{total:>10,.0f} | {r:+6.2f}% | "
                            f"{self._current_regime:5s} | Hold={len(self.holdings)}  "
                            f"WS={self._win_streak} LS={self._loss_streak}")

        self._print_report(time.time() - t0)

    # ── pending queue execution ──
    def _get_row(self, code, date):
        df = self.all_data.get(code)
        if df is None: return None
        rows = df[df["date"].astype(str).str[:10] == date]
        return rows.iloc[0] if not rows.empty else None

    def _get_idx(self, code, date) -> int:
        df = self.all_data.get(code)
        if df is None: return -1
        m = df["date"].astype(str).str[:10] == date
        idxs = df.index[m]
        return int(idxs[0]) if len(idxs) else -1

    def _exec_pending_sells(self, date, day_idx):
        sells = self._pending_sells[:]; self._pending_sells = []
        for code, shares, reason in sells:
            h = self.holdings.get(code)
            if not h: continue
            if h.buy_day_idx >= day_idx:
                self._pending_sells.append((code, shares, reason + "(T+1)"))
                continue
            row = self._get_row(code, date)
            if row is None: continue
            op = float(row["open"])
            if op <= 0: continue
            # 开盘跌停
            if h.current_price > 0 and op <= h.current_price * 0.902:
                self._pending_sells.append((code, shares, reason + "(LD)"))
                continue
            self._sell(code, min(shares, h.shares), op, date, reason)

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
            if sc > 0 and op >= sc * 1.098: continue    # 一字
            if sc > 0 and op > sc * 1.05: continue       # 高开 >5% 不追

            cost = op * (1 + SLIPPAGE_PCT)
            total_asset = self.cash + sum(h.market_value for h in self.holdings.values())
            mv_now = sum(h.market_value for h in self.holdings.values())
            max_mv = total_asset * cfg["total_pct"] * pm
            remaining = max_mv - mv_now
            if remaining <= 0: continue

            buy_amount = min(self.cash * 0.95, total_asset * cfg["single_pct"] * pm, remaining)
            shares = int(buy_amount / cost / 100) * 100
            if shares < 100: continue
            amount = cost * shares; comm = max(amount * COMMISSION_RATE, 5)
            if amount + comm > self.cash: continue

            self.cash -= (amount + comm)
            self.holdings[code] = Holding(
                code=code, name=pb.get("name", code), shares=shares,
                cost_price=cost, buy_date=date, buy_day_idx=day_idx,
                strategy=pb.get("strategy", self._current_regime),
                current_price=op, peak_price=op,
            )
            self.trades.append(Trade(
                date, code, pb.get("name", code), "buy", shares, cost, amount, comm,
                f"YOUZI|{pb.get('reason','')}|votes={'+'.join(pb.get('buy_votes',[]))}",
                self._current_regime,
            ))
            # 归因
            self._style_attribution[code] = {
                "entry_date": date, "cost": cost,
                "buy_votes": pb.get("buy_votes", []),
                "weighted_score": pb.get("weighted_score", 0),
                "per_style_verdicts": pb.get("per_style_verdicts", {}),
            }
            for sn in pb.get("buy_votes", []):
                self._style_buy_count[sn] += 1

    def _emergency_stop(self, date, day_idx):
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx >= day_idx: continue
            row = self._get_row(code, date)
            if row is None: continue
            low = float(row["low"])
            pnl = (low - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            if pnl <= -7.0:
                trigger = h.cost_price * 0.93
                self._sell(code, h.shares, trigger, date, f"E-STOP{pnl:.1f}%")

    def _update_close(self, date):
        for code, h in self.holdings.items():
            row = self._get_row(code, date)
            if row is None: continue
            h.current_price = float(row["close"])
            h.peak_price = max(h.peak_price, float(row["high"]))

    def _review_sells(self, date, day_idx, cfg):
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx == day_idx: continue
            pnl = h.pnl_pct
            hold = day_idx - h.buy_day_idx
            pk = (h.peak_price - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            dd = (h.peak_price - h.current_price) / h.peak_price * 100 if h.peak_price > 0 else 0

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

    # ── Youzi 选股 ──
    def _scan_buys_youzi(self, date, day_idx, cfg):
        if len(self.holdings) >= cfg["max_positions"]: return
        if self._pending_buys: return

        weights = REGIME_WEIGHTS.get(self._current_regime, REGIME_WEIGHTS["SIDE"])

        # 1) 第一轮：calc_indicators → 候选（控制进入 youzi 的规模）
        pool = []
        for code, df in self.all_data.items():
            if code in self.holdings: continue
            idx = self._get_idx(code, date)
            if idx < 30: continue
            c = _build_candidate(code, self.name_map.get(code, code), df, idx)
            if c is None: continue
            pool.append((c, df, idx))

        # 按 base tech_score 降序取 top_universe 进入游资打分
        pool.sort(key=lambda t: t[0]["tech_score"], reverse=True)
        pool = pool[:self.top_universe]

        # 2) 第二轮：youzi 打分
        scored = []
        for cand, df, idx in pool:
            try:
                tf = _build_tf_no_lookahead(cand["code"], df, idx)
            except Exception:
                continue
            news = NewsContext()  # 回测没有历史舆情
            per = score_all_styles(cand, tf, news,
                                    regime=self._current_regime, active=list_styles())
            agg = aggregate_verdict(per, weights=weights)
            # 合并 delta 到 tech_score（与实盘一致）
            total_delta, ws = 0.0, 0.0
            for sn, r in per.items():
                if r.verdict == "veto": continue
                w = weights.get(sn, 1.0)
                total_delta += r.score_delta * w; ws += w
            avg_delta = total_delta / ws if ws > 0 else 0
            final_score = cand["tech_score"] + avg_delta

            # 共识门槛：至少 1 家 buy 或 weighted_score>=80，且无全 veto
            vetoed_all = bool(agg["vetoed_by"]) and not agg["buy_votes"]
            if vetoed_all:
                continue
            if not agg["buy_votes"] and agg["weighted_score"] < 80:
                continue

            scored.append({
                "code": cand["code"], "name": cand["name"],
                "signal_close": cand["price"],
                "final_score": final_score,
                "weighted_score": agg["weighted_score"],
                "buy_votes": agg["buy_votes"],
                "per_style_verdicts": {k: v.verdict for k, v in per.items()},
                "reason": f"youzi|{self._current_regime}|ws={agg['weighted_score']:.0f}",
                "strategy": self._current_regime,
            })

        if not scored:
            return

        # 按综合 final_score 排序
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        slots = cfg["max_positions"] - len(self.holdings)
        for c in scored[:max(1, slots)]:
            self._pending_buys.append(c)

    def _sell(self, code, shares, price, date, reason):
        h = self.holdings.get(code)
        if not h: return
        sp = price * (1 - SLIPPAGE_PCT); amt = sp * shares
        comm = max(amt * COMMISSION_RATE, 5); tax = amt * STAMP_TAX_RATE
        self.cash += (amt - comm - tax)
        self.trades.append(Trade(date, code, h.name, "sell", shares, sp, amt,
                                  comm + tax, reason, self._current_regime))
        pnl = (sp - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
        if shares >= h.shares:
            # 归因：每家 buy 票的胜负
            attr = self._style_attribution.get(code, {})
            for sn in attr.get("buy_votes", []):
                if pnl > 0:
                    self._style_win_count[sn] += 1
                self._style_pnl_sum[sn] += pnl
            if pnl > 0:
                self._win_streak += 1; self._loss_streak = 0
            else:
                self._loss_streak += 1; self._win_streak = 0
            del self.holdings[code]
            self._style_attribution.pop(code, None)
        else:
            h.shares -= shares

    # ── 报告 ──
    def _print_report(self, elapsed: float):
        if not self.snapshots: return

        final = self.snapshots[-1].total_asset
        ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        peak, max_dd = INITIAL_CAPITAL, 0.0
        for s in self.snapshots:
            peak = max(peak, s.total_asset)
            dd = (peak - s.total_asset) / peak * 100
            max_dd = max(max_dd, dd)

        # 胜率
        completed = defaultdict(list)
        for t in self.trades: completed[t.code].append(t)
        wins, losses = 0, 0; wl, ll = [], []
        for code, tl in completed.items():
            bs = [t for t in tl if t.direction == "buy"]
            ss = [t for t in tl if t.direction == "sell"]
            if bs and ss:
                ab = sum(t.price*t.shares for t in bs) / max(1, sum(t.shares for t in bs))
                asl = sum(t.price*t.shares for t in ss) / max(1, sum(t.shares for t in ss))
                p = (asl-ab)/ab*100
                if p > 0: wins += 1; wl.append(p)
                else: losses += 1; ll.append(p)
        tc = wins + losses
        wr = wins/tc*100 if tc>0 else 0
        aw = np.mean(wl) if wl else 0
        al = np.mean(ll) if ll else 0
        pf = abs(sum(wl)/sum(ll)) if ll and sum(ll)!=0 else 0

        days = len(self.snapshots)
        ann = ret * (252/days) if days>0 else 0
        dr = [s.daily_return_pct for s in self.snapshots]
        sharpe = np.mean(dr)/np.std(dr)*np.sqrt(252) if len(dr)>1 and np.std(dr)>0 else 0

        regime_counts = defaultdict(int)
        for s in self.snapshots: regime_counts[s.regime] += 1
        monthly = defaultdict(list)
        for s in self.snapshots: monthly[s.date[:7]].append(s.daily_return_pct)

        print("\n" + "="*72)
        print("  游资风格回测 · 2026 YTD")
        print("="*72)
        print(f"  覆盖交易日: {days}  耗时: {elapsed:.1f}s")
        print(f"  初始资金:  Y{INITIAL_CAPITAL:>12,.0f}")
        print(f"  最终资产:  Y{final:>12,.0f}")
        print(f"  累计收益:  {ret:>+.2f}%    年化: {ann:>+.2f}%")
        print(f"  最大回撤:  {max_dd:>.2f}%")
        print(f"  夏普比率:  {sharpe:>.2f}")
        print(f"  交易对数:  {tc}   胜率: {wr:.1f}%  PF={pf:.2f}")
        print(f"  平均盈利:  +{aw:.2f}%   平均亏损: {al:.2f}%")
        print()

        # 每日收益表
        print("  ─── 日度收益（部分）───")
        print(f"  {'日期':<12} {'总资产':>12} {'日内%':>7} {'持仓':>4} {'Regime':<6} {'累计%':>7}")
        for s in self.snapshots:
            cum = (s.total_asset - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            print(f"  {s.date:<12} {s.total_asset:>12,.0f} {s.daily_return_pct:>+7.2f} "
                  f"{s.holdings_count:>4d} {s.regime:<6} {cum:>+7.2f}")
        print()

        # 月度
        if monthly:
            print("  ─── 月度 ───")
            for m in sorted(monthly):
                rets = monthly[m]
                mret = (np.prod([1 + r/100 for r in rets]) - 1) * 100
                print(f"  {m}:  {mret:+.2f}%  ({len(rets)}d)")
            print()

        # Regime 分布
        print("  ─── Regime 分布 ───")
        for r in ("BULL","SIDE","BEAR","CRASH"):
            c = regime_counts.get(r, 0)
            if c:
                print(f"  {r:6s}: {c}d ({c/days*100:.1f}%)")
        print()

        # 各游资风格命中率归因
        print("  ─── 游资风格归因 ───")
        print(f"  {'风格':<15} {'Buy次数':>8} {'胜率':>7} {'平均盈亏%':>10}")
        for sn in list_styles():
            bc = self._style_buy_count.get(sn, 0)
            wc = self._style_win_count.get(sn, 0)
            pnl = self._style_pnl_sum.get(sn, 0)
            wr_s = wc/bc*100 if bc else 0
            avg = pnl/bc if bc else 0
            print(f"  {sn:<15} {bc:>8d} {wr_s:>6.1f}% {avg:>+9.2f}%")
        print()

        print("="*72)


# ──────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────

def main(start: str = "2026-01-05", end: str = "2026-04-09"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")

    cache = Path("data/backtest_cache_2026ytd.pkl")
    if not cache.exists():
        logger.info(f"缓存不存在，开始拉取 2025-10 ~ 2026-04-09 数据...")
        from backtest_2025 import fetch_all_a_share_daily
        all_data = fetch_all_a_share_daily("20251001", "20260410",
                                            cache_name="backtest_cache_2026ytd.pkl")
    else:
        logger.info(f"从缓存加载 2026 YTD 数据...")
        all_data = pickle.loads(cache.read_bytes())

    logger.info(f"加载完成 {len(all_data)} 只股票")

    engine = YouziBacktest(all_data, top_universe=60)
    engine.run(start, end)

    # 落盘结果供进一步分析
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
    out_path = Path(f"reports/youzi_backtest_2026ytd_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    logger.info(f"结果已保存: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-01-05")
    p.add_argument("--end", default="2026-04-09")
    args = p.parse_args()
    main(args.start, args.end)
