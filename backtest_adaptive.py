"""
华尔街之狼 自适应策略回测
====================================
无未来函数 | T日盘后决策 → T+1开盘执行

核心: 每日收盘后判断市场环境，动态切换策略+仓位

市场状态:
  BULL   (牛市) → 趋势跟随，仓位80%，最多3只
  SIDE   (震荡) → 网格低吸，仓位50%，最多3只
  BEAR   (熊市) → 空仓等待 / 极少量超跌反弹，仓位20%，最多1只
  CRASH  (暴跌) → 全面空仓，0%仓位

市场判断依据（仅用已知数据）:
  1. 市场宽度: 站上MA20的股票比例
  2. 指数动量: 大盘近5日/20日涨跌
  3. 跌停家数: 恐慌度量
  4. 连续状态: 防止频繁切换（需连续2天确认）
"""
import json, logging, os, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 200_000
COMMISSION_RATE = 0.00025
STAMP_TAX_RATE = 0.0005
SLIPPAGE_PCT = 0.002


@dataclass
class Holding:
    code: str; name: str; shares: int; cost_price: float
    buy_date: str; buy_day_idx: int; strategy: str
    current_price: float = 0.0; peak_price: float = 0.0
    @property
    def market_value(self): return self.shares * self.current_price
    @property
    def pnl_pct(self):
        return (self.current_price - self.cost_price) / self.cost_price * 100 if self.cost_price > 0 else 0


@dataclass
class Trade:
    date: str; code: str; name: str; direction: str; shares: int
    price: float; amount: float; commission: float; reason: str; regime: str


@dataclass
class Snapshot:
    date: str; total_asset: float; cash: float; market_value: float
    holdings_count: int; daily_return_pct: float; regime: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 技术指标
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_indicators(df, idx):
    if idx < 25: return None
    w = df.iloc[max(0, idx-60):idx+1]
    c = w["close"].values; h = w["high"].values; l = w["low"].values; v = w["volume"].values
    if len(c) < 20: return None
    p = c[-1]
    if p <= 0: return None

    ma5 = np.mean(c[-5:]); ma10 = np.mean(c[-10:]); ma20 = np.mean(c[-20:])
    ma5_slope = (ma5 - np.mean(c[-6:-1])) / np.mean(c[-6:-1]) * 100 if len(c) >= 6 and np.mean(c[-6:-1]) > 0 else 0

    # RSI
    if len(c) >= 15:
        d = np.diff(c[-15:])
        g = np.mean(d[d > 0]) if np.any(d > 0) else 0
        ls = -np.mean(d[d < 0]) if np.any(d < 0) else 0.001
        rsi = 100 - 100 / (1 + g / ls)
    else: rsi = 50

    bias5 = (p - ma5) / ma5 * 100 if ma5 > 0 else 0
    bias20 = (p - ma20) / ma20 * 100 if ma20 > 0 else 0
    h20 = np.max(h[-20:]); l20 = np.min(l[-20:])
    pp = (p - l20) / (h20 - l20) if (h20 - l20) > 0 else 0.5

    vm20 = np.mean(v[-20:]) if len(v) >= 20 else np.mean(v[-5:])
    vr = v[-1] / vm20 if vm20 > 0 else 1
    vs = v[-1] / np.max(v[-10:]) if np.max(v[-10:]) > 0 else 1

    tc = (c[-1]/c[-2]-1)*100 if len(c) >= 2 else 0
    c3 = (p/c[-4]-1)*100 if len(c) >= 4 else 0
    c5 = (p/c[-6]-1)*100 if len(c) >= 6 else 0

    ema12 = pd.Series(c).ewm(span=12).mean().values
    ema26 = pd.Series(c).ewm(span=26).mean().values
    dif = ema12[-1] - ema26[-1]
    dea_arr = pd.Series(ema12-ema26).ewm(span=9).mean().values
    dea = dea_arr[-1]
    mc = dif > dea and (len(dea_arr) >= 2 and (ema12[-2]-ema26[-2]) <= dea_arr[-2])

    # 连板/近期涨停
    cl = 0
    for j in range(-1, max(-8, -len(c)), -1):
        dd = (c[j]/c[j-1]-1)*100 if j-1 >= -len(c) else 0
        if dd > 9.5: cl += 1
        else: break
    hrl = any((c[j]/c[j-1]-1)*100 > 9.5 for j in range(-2, max(-8,-len(c)),-1) if j-1 >= -len(c))

    cu = 0
    for j in range(-1, max(-8, -len(c)), -1):
        if c[j] > c[j-1]: cu += 1
        else: break

    to = float(df.iloc[idx].get("turnover_rate", 0))
    if pd.isna(to): to = 0

    return dict(price=p, ma5=ma5, ma10=ma10, ma20=ma20, ma5_slope=ma5_slope,
                rsi=rsi, bias5=bias5, bias20=bias20, price_pos=pp,
                vol_ratio=vr, vol_shrink=vs, today_chg=tc, chg_3d=c3, chg_5d=c5,
                dif=dif, dea=dea, macd_cross=mc, consec_limit=cl,
                has_recent_limit=hrl, consec_up=cu, turnover=to, h20=h20, l20=l20)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 市场环境判断
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MarketRegimeDetector:
    """收盘后用已知数据判断市场状态"""

    def __init__(self):
        self._prev_regime = "SIDE"
        self._regime_days = 0        # 当前状态持续天数
        self._pending_regime = None  # 待确认的新状态
        self._pending_count = 0

    def detect(self, all_data: dict, date: str) -> dict:
        """返回 {regime, score, breadth, detail}"""
        score = 0
        details = []

        # 1. 市场宽度（站上MA20的比例）
        above, below, limit_down_count = 0, 0, 0
        total = 0
        for code, df in all_data.items():
            idx_arr = df.index[df["date"] == date]
            if len(idx_arr) == 0: continue
            i = idx_arr[0]
            total += 1
            cl = float(df["close"].iloc[i])
            if i >= 20:
                m20 = df["close"].iloc[i-19:i+1].mean()
                if cl > m20: above += 1
                else: below += 1
            # 跌停计数
            if i >= 1:
                chg = (cl / float(df["close"].iloc[i-1]) - 1) * 100
                if chg < -9.5: limit_down_count += 1

        breadth = above / total if total > 0 else 0.5
        limit_down_pct = limit_down_count / total * 100 if total > 0 else 0

        # 宽度评分
        if breadth > 0.65:
            score += 40; details.append(f"宽度{breadth:.0%}强(+40)")
        elif breadth > 0.55:
            score += 25; details.append(f"宽度{breadth:.0%}偏强(+25)")
        elif breadth > 0.45:
            score += 10; details.append(f"宽度{breadth:.0%}中性(+10)")
        elif breadth > 0.35:
            score -= 15; details.append(f"宽度{breadth:.0%}偏弱(-15)")
        else:
            score -= 35; details.append(f"宽度{breadth:.0%}极弱(-35)")

        # 2. 跌停家数（恐慌指标）
        if limit_down_pct > 5:
            score -= 30; details.append(f"跌停{limit_down_count}家({limit_down_pct:.1f}%)恐慌(-30)")
        elif limit_down_pct > 2:
            score -= 15; details.append(f"跌停{limit_down_count}家(-15)")

        # 3. 近期动量（用市场宽度的变化代替指数）
        # 这里简化：用当日涨跌家数比
        up_count = sum(1 for code, df in all_data.items()
                       if not df[df["date"]==date].empty and
                       float(df[df["date"]==date].iloc[0].get("change_pct", 0) or 0) > 0)
        up_ratio = up_count / total if total > 0 else 0.5
        if up_ratio > 0.6:
            score += 15; details.append(f"涨跌比{up_ratio:.0%}(+15)")
        elif up_ratio < 0.35:
            score -= 15; details.append(f"涨跌比{up_ratio:.0%}(-15)")

        # 判定
        if score >= 40:
            raw = "BULL"
        elif score >= 10:
            raw = "SIDE"
        elif score >= -20:
            raw = "BEAR"
        else:
            raw = "CRASH"

        # 防频繁切换: 新状态需连续2天确认
        if raw != self._prev_regime:
            if raw == self._pending_regime:
                self._pending_count += 1
            else:
                self._pending_regime = raw
                self._pending_count = 1

            if self._pending_count >= 2:
                # 确认切换
                self._prev_regime = raw
                self._regime_days = 1
                self._pending_regime = None
                self._pending_count = 0
            # 未确认，保持原状态
        else:
            self._regime_days += 1
            self._pending_regime = None
            self._pending_count = 0

        return {
            "regime": self._prev_regime,
            "raw": raw,
            "score": score,
            "breadth": breadth,
            "limit_down": limit_down_count,
            "detail": ", ".join(details),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略评分
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_bull(ind):
    """牛市: 趋势跟随+龙头预埋"""
    s = 0
    if ind["ma5"] > ind["ma10"] > ind["ma20"]: s += 15
    elif ind["ma5"] > ind["ma10"]: s += 10
    else: return 0

    if ind["macd_cross"]: s += 15
    elif ind["dif"] > ind["dea"]: s += 8

    if 35 < ind["rsi"] < 55: s += 10
    elif ind["rsi"] > 70: s -= 15

    if 1.0 < ind["vol_ratio"] < 2.0: s += 10
    elif ind["vol_ratio"] > 3: s -= 10

    if 0.3 < ind["price_pos"] < 0.65: s += 10
    elif ind["price_pos"] > 0.85: s -= 10

    if 2 <= ind["consec_up"] <= 5 and ind["today_chg"] < 5: s += 8

    # 龙头预埋加分
    if ind["has_recent_limit"] and ind["price_pos"] < 0.5:
        s += 15

    if ind["today_chg"] > 5: s -= 15
    elif ind["today_chg"] > 3: s -= 5
    return max(0, s)


def score_side(ind):
    """震荡: 网格低吸（验证过的最优策略）"""
    s = 0
    if ind["price_pos"] < 0.1: s += 30
    elif ind["price_pos"] < 0.2: s += 25
    elif ind["price_pos"] < 0.3: s += 18
    elif ind["price_pos"] < 0.4: s += 10
    elif ind["price_pos"] > 0.6: return 0

    if ind["bias20"] < -8: s += 20
    elif ind["bias20"] < -5: s += 15
    elif ind["bias20"] < -3: s += 10
    elif ind["bias20"] < 0: s += 5

    if ind["vol_ratio"] < 0.6: s += 15
    elif ind["vol_ratio"] < 0.8: s += 10
    elif ind["vol_ratio"] < 1.0: s += 5
    elif ind["vol_ratio"] > 2.0: s -= 10

    if abs(ind["ma5_slope"]) < 1: s += 5
    if 0 < ind["today_chg"] < 2: s += 5
    if ind["today_chg"] < -5: s -= 10

    # MACD底部收敛加分
    if ind["macd_cross"]: s += 8
    elif ind["dif"] < 0 and ind["dif"] > ind["dea"]: s += 5

    return max(0, s)


def score_bear(ind):
    """熊市: 极度保守，仅做最确定的超跌反弹"""
    s = 0
    # RSI极端超卖
    if ind["rsi"] < 20: s += 30
    elif ind["rsi"] < 25: s += 20
    elif ind["rsi"] < 30: s += 10
    else: return 0  # 不超卖不做

    if ind["bias5"] < -8: s += 20
    elif ind["bias5"] < -5: s += 15
    elif ind["bias5"] < -3: s += 8

    if ind["price_pos"] < 0.1: s += 15
    elif ind["price_pos"] < 0.2: s += 10

    # 缩量企稳
    if ind["vol_shrink"] < 0.3: s += 10
    elif ind["vol_shrink"] < 0.5: s += 5

    # MACD金叉
    if ind["macd_cross"]: s += 10

    if ind["today_chg"] < -8: s -= 20
    return max(0, s)


# 策略参数表
REGIME_CONFIG = {
    "BULL": {
        "score_fn": score_bull, "min_score": 45,
        "max_positions": 3, "single_pct": 0.30, "total_pct": 0.80,
        "stop_loss": -4.0, "take_profit": 8.0, "trail_trigger": 4.0, "trail_dd": 2.0,
        "hold_max": 7, "price_range": (3, 50),
    },
    "SIDE": {
        "score_fn": score_side, "min_score": 45,
        "max_positions": 3, "single_pct": 0.20, "total_pct": 0.50,
        "stop_loss": -3.0, "take_profit": 5.0, "trail_trigger": 3.0, "trail_dd": 1.5,
        "hold_max": 5, "price_range": (3, 30),
    },
    "BEAR": {
        "score_fn": score_bear, "min_score": 55,
        "max_positions": 1, "single_pct": 0.15, "total_pct": 0.20,
        "stop_loss": -3.0, "take_profit": 4.0, "trail_trigger": 2.0, "trail_dd": 1.5,
        "hold_max": 3, "price_range": (3, 20),
    },
    "CRASH": {
        "score_fn": None, "min_score": 999,  # 不买
        "max_positions": 0, "single_pct": 0, "total_pct": 0,
        "stop_loss": -2.0, "take_profit": 3.0, "trail_trigger": 2.0, "trail_dd": 1.0,
        "hold_max": 1, "price_range": (0, 0),
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 自适应回测引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveBacktest:
    def __init__(self, all_data):
        self.all_data = all_data
        self.cash = INITIAL_CAPITAL
        self.holdings: Dict[str, Holding] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[Snapshot] = []
        self.prev_total = INITIAL_CAPITAL
        self._pending_buys: List[dict] = []
        self._pending_sells: List[tuple] = []
        self._regime_detector = MarketRegimeDetector()
        self._current_regime = "SIDE"
        self._regime_history = []

        dates_set = set()
        for df in all_data.values():
            dates_set.update(df["date"].tolist())
        self.trading_days = sorted(dates_set)
        self.day_idx_map = {d: i for i, d in enumerate(self.trading_days)}

    def run(self, start_date, end_date):
        days = [d for d in self.trading_days if start_date <= d <= end_date]
        logger.info(f"自适应回测: {start_date} ~ {end_date} ({len(days)}天)")
        logger.info(f"初始资金: Y{INITIAL_CAPITAL:,.0f}")

        for day_num, date in enumerate(days):
            day_idx = self.day_idx_map.get(date, day_num)

            # 1. 开盘: 执行昨晚挂单
            self._exec_pending_sells(date, day_idx)
            self._exec_pending_buys(date, day_idx)

            # 2. 盘中: 极端止损
            self._emergency_stop(date, day_idx)

            # 3. 收盘: 更新价格
            self._update_close(date)

            # 4. 盘后: 判断市场环境
            regime_info = self._regime_detector.detect(self.all_data, date)
            self._current_regime = regime_info["regime"]
            self._regime_history.append({
                "date": date, **regime_info,
            })

            # 5. 盘后: CRASH时清仓所有持仓
            if self._current_regime == "CRASH":
                for code, h in list(self.holdings.items()):
                    if h.buy_day_idx < day_idx:  # T+1检查
                        self._pending_sells.append((code, h.shares, "CRASH清仓"))

            # 6. 盘后: 复盘持仓+选股
            cfg = REGIME_CONFIG[self._current_regime]
            self._review_sells(date, day_idx, cfg)
            if cfg["score_fn"] is not None:
                self._scan_buys(date, day_idx, cfg)

            # 快照
            mv = sum(h.market_value for h in self.holdings.values())
            total = self.cash + mv
            ret = (total - self.prev_total) / self.prev_total * 100 if self.prev_total > 0 else 0
            self.prev_total = total
            self.snapshots.append(Snapshot(date, total, self.cash, mv,
                                           len(self.holdings), ret, self._current_regime))

            if (day_num + 1) % 20 == 0:
                r = (total - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                logger.info(f"  [{date}] Day{day_num+1} | Y{total:,.0f} | {r:+.2f}% | "
                            f"{self._current_regime} | Hold={len(self.holdings)}")

        self._print_report()

    def _get_row(self, code, date):
        df = self.all_data.get(code)
        if df is None: return None
        rows = df[df["date"] == date]
        return rows.iloc[0] if not rows.empty else None

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
            if h.current_price > 0 and op <= h.current_price * 0.902:
                self._pending_sells.append((code, shares, reason + "(跌停)"))
                continue
            self._sell(code, min(shares, h.shares), op, date, reason)

    def _exec_pending_buys(self, date, day_idx):
        buys = self._pending_buys[:]; self._pending_buys = []
        cfg = REGIME_CONFIG[self._current_regime]
        for pb in buys:
            code = pb["code"]
            if code in self.holdings or len(self.holdings) >= cfg["max_positions"]: continue
            row = self._get_row(code, date)
            if row is None: continue
            op = float(row["open"])
            if op <= 0: continue
            sc = pb.get("signal_close", 0)
            if sc > 0 and op >= sc * 1.098: continue  # 一字板
            if sc > 0 and op > sc * 1.04: continue     # 高开4%+不追

            # 仓位控制
            cost = op * (1 + SLIPPAGE_PCT)
            total_asset = self.cash + sum(h.market_value for h in self.holdings.values())
            mv_now = sum(h.market_value for h in self.holdings.values())
            max_mv = total_asset * cfg["total_pct"]
            remaining = max_mv - mv_now
            if remaining <= 0: continue

            buy_amount = min(self.cash * 0.95, total_asset * cfg["single_pct"], remaining)
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
                f"T+1|{pb.get('reason', '')}",
                self._current_regime,
            ))

    def _emergency_stop(self, date, day_idx):
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx >= day_idx: continue
            row = self._get_row(code, date)
            if row is None: continue
            low = float(row["low"])
            pnl = (low - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            if pnl <= -6.0:
                trigger = h.cost_price * 0.94
                self._sell(code, h.shares, trigger, date, f"盘中极端止损{pnl:.1f}%")

    def _update_close(self, date):
        for code, h in self.holdings.items():
            row = self._get_row(code, date)
            if row is None: continue
            h.current_price = float(row["close"])
            h.peak_price = max(h.peak_price, float(row["high"]))

    def _review_sells(self, date, day_idx, cfg):
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx == day_idx: continue
            pnl = h.pnl_pct; hold = day_idx - h.buy_day_idx
            pk = (h.peak_price - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            dd = (h.peak_price - h.current_price) / h.peak_price * 100 if h.peak_price > 0 else 0

            if pnl >= cfg["take_profit"]:
                self._pending_sells.append((code, h.shares, f"止盈{pnl:.1f}%"))
            elif pnl <= cfg["stop_loss"]:
                self._pending_sells.append((code, h.shares, f"止损{pnl:.1f}%"))
            elif pk >= cfg["trail_trigger"] and dd >= cfg["trail_dd"]:
                self._pending_sells.append((code, h.shares, f"移动止盈pk{pk:.1f}%dd{dd:.1f}%"))
            elif hold >= cfg["hold_max"] and pnl < 1:
                self._pending_sells.append((code, h.shares, f"超期{hold}d"))

    def _scan_buys(self, date, day_idx, cfg):
        if len(self.holdings) >= cfg["max_positions"]: return
        if self._pending_buys: return

        fn = cfg["score_fn"]; ms = cfg["min_score"]
        pmin, pmax = cfg["price_range"]
        cands = []
        for code, df in self.all_data.items():
            if code in self.holdings: continue
            ri = df.index[df["date"] == date]
            if len(ri) == 0: continue
            idx = ri[0]; row = df.iloc[idx]
            price = float(row["close"])
            if price < pmin or price > pmax: continue
            ind = calc_indicators(df, idx)
            if ind is None: continue
            s = fn(ind)
            if s < ms: continue
            cands.append({"code": code, "name": str(row.get("stock_name", code)),
                          "signal_close": price, "score": s,
                          "reason": f"{self._current_regime}|s={s}",
                          "strategy": self._current_regime})

        cands.sort(key=lambda x: x["score"], reverse=True)
        slots = cfg["max_positions"] - len(self.holdings)
        for c in cands[:max(1, slots)]:
            self._pending_buys.append(c)

    def _sell(self, code, shares, price, date, reason):
        h = self.holdings.get(code)
        if not h: return
        sp = price * (1 - SLIPPAGE_PCT); amt = sp * shares
        comm = max(amt * COMMISSION_RATE, 5); tax = amt * STAMP_TAX_RATE
        self.cash += (amt - comm - tax)
        self.trades.append(Trade(date, code, h.name, "sell", shares, sp, amt,
                                  comm + tax, reason, self._current_regime))
        if shares >= h.shares: del self.holdings[code]
        else: h.shares -= shares

    def _print_report(self):
        if not self.snapshots: return

        final = self.snapshots[-1].total_asset
        ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        peak, max_dd = INITIAL_CAPITAL, 0
        for s in self.snapshots:
            peak = max(peak, s.total_asset)
            dd = (peak - s.total_asset) / peak * 100
            max_dd = max(max_dd, dd)

        # 胜率
        completed = defaultdict(list)
        for t in self.trades: completed[t.code].append(t)
        wins, losses, wl, ll = 0, 0, [], []
        for code, tl in completed.items():
            bs = [t for t in tl if t.direction == "buy"]
            ss = [t for t in tl if t.direction == "sell"]
            if bs and ss:
                ab = sum(t.price*t.shares for t in bs) / sum(t.shares for t in bs)
                asl = sum(t.price*t.shares for t in ss) / sum(t.shares for t in ss)
                p = (asl-ab)/ab*100
                if p > 0: wins += 1; wl.append(p)
                else: losses += 1; ll.append(p)

        tc = wins + losses
        wr = wins / tc * 100 if tc > 0 else 0
        aw = np.mean(wl) if wl else 0; al = np.mean(ll) if ll else 0
        pf = abs(aw / al) if al != 0 else 0
        days = len(self.snapshots)
        ann = ret * (252 / days) if days > 0 else 0
        dr = [s.daily_return_pct for s in self.snapshots]
        sharpe = np.mean(dr) / np.std(dr) * np.sqrt(252) if len(dr) > 1 and np.std(dr) > 0 else 0

        # 状态分布
        regime_counts = defaultdict(int)
        for s in self.snapshots: regime_counts[s.regime] += 1

        # 月度
        monthly = defaultdict(list)
        for s in self.snapshots: monthly[s.date[:7]].append(s.daily_return_pct)

        # 按策略统计
        regime_trades = defaultdict(lambda: {"buys": 0, "sells": 0, "pnl": []})
        for t in self.trades:
            if t.direction == "buy": regime_trades[t.regime]["buys"] += 1
            else: regime_trades[t.regime]["sells"] += 1

        print("\n" + "=" * 70)
        print("  Adaptive Strategy Backtest Report 2026 YTD")
        print("=" * 70)
        print(f"\n  Initial:    Y{INITIAL_CAPITAL:>12,}")
        print(f"  Final:      Y{final:>12,.0f}")
        print(f"  Return:     {ret:>+11.2f}%")
        print(f"  Annualized: {ann:>+11.2f}%")
        print(f"  Max DD:     {max_dd:>11.2f}%")
        print(f"  Sharpe:     {sharpe:>11.2f}")
        print(f"\n  Trades: {len(self.trades)}  Win: {wins}  Loss: {losses}  "
              f"WinRate: {wr:.1f}%  AvgW: {aw:+.2f}%  AvgL: {al:+.2f}%  PF: {pf:.2f}")

        print(f"\n  [Market Regime Distribution]")
        for r in ["BULL", "SIDE", "BEAR", "CRASH"]:
            d = regime_counts.get(r, 0)
            pct = d / days * 100 if days > 0 else 0
            buys = regime_trades[r]["buys"]
            sells = regime_trades[r]["sells"]
            print(f"    {r:<6} {d:>3}d ({pct:>4.1f}%) | Buys={buys} Sells={sells}")

        print(f"\n  [Monthly Returns]")
        cum = 0
        for m in sorted(monthly.keys()):
            mr = sum(monthly[m]); cum += mr
            # 当月状态
            month_regimes = [s.regime for s in self.snapshots if s.date.startswith(m)]
            dominant = max(set(month_regimes), key=month_regimes.count) if month_regimes else "?"
            mark = "+" if mr >= 0 else "-"
            print(f"    {m}  {mr:>+7.2f}%  cum{cum:>+7.2f}%  [{mark}]  regime={dominant}")

        print("=" * 70)

        # 保存
        out_dir = Path("data/backtest_adaptive")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "return": round(ret, 2), "annual": round(ann, 2),
            "max_dd": round(max_dd, 2), "sharpe": round(sharpe, 2),
            "win_rate": round(wr, 1), "profit_factor": round(pf, 2),
            "trades": len(self.trades), "final": round(final, 0),
            "regime_dist": dict(regime_counts),
            "monthly": {m: round(sum(r), 2) for m, r in sorted(monthly.items())},
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 交易记录
        trade_rows = [{"date": t.date, "code": t.code, "name": t.name,
                        "dir": t.direction, "shares": t.shares, "price": round(t.price, 3),
                        "amount": round(t.amount, 0), "reason": t.reason, "regime": t.regime}
                       for t in self.trades]
        pd.DataFrame(trade_rows).to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")

        # 日线
        snap_rows = [{"date": s.date, "total": round(s.total_asset, 0), "cash": round(s.cash, 0),
                       "mv": round(s.market_value, 0), "hold": s.holdings_count,
                       "ret": round(s.daily_return_pct, 4), "regime": s.regime}
                      for s in self.snapshots]
        pd.DataFrame(snap_rows).to_csv(out_dir / "daily.csv", index=False, encoding="utf-8-sig")

        # 状态变化日志
        regime_changes = []
        prev = None
        for rh in self._regime_history:
            if rh["regime"] != prev:
                regime_changes.append(rh)
                prev = rh["regime"]
        with open(out_dir / "regime_changes.json", "w", encoding="utf-8") as f:
            json.dump(regime_changes, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    from backtest_2025 import fetch_all_a_share_daily

    logger.info("Loading data...")
    all_data = fetch_all_a_share_daily("20251001", "20260410",
                                        cache_name="backtest_cache_2026ytd.pkl")
    logger.info(f"Data loaded: {len(all_data)} stocks")

    engine = AdaptiveBacktest(all_data)
    engine.run("2026-01-05", "2026-04-10")
