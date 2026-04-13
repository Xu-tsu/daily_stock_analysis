"""
华尔街之狼 自适应策略回测 V2 — 激进版
========================================
无未来函数 | T日盘后决策 → T+1开盘执行

V2核心改进:
  1. 牛市集中ALL-IN (1-2只), 止盈放宽到15%+移动止盈
  2. 新增打板/接力策略: 涨停第二天追, 3连板加速
  3. 震荡市用网格+事件驱动(放量突破)
  4. 空仓期做短线超跌反弹(2天快进快出)
  5. 更灵活的市场识别: 1天切入牛市, 2天确认熊市/暴跌
  6. 梯度止盈: 不是一刀切, 分批止盈锁利
  7. 动态仓位: 连赢加仓, 连亏减仓(凯利准则简化版)
"""
import json, logging, os, time
from collections import defaultdict
from dataclasses import dataclass, field
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
    partial_sold: bool = False  # 已部分止盈
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
# 技术指标（增强版）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_indicators(df, idx):
    if idx < 30: return None
    w = df.iloc[max(0, idx-60):idx+1]
    c = w["close"].values; h = w["high"].values; l = w["low"].values
    v = w["volume"].values; o = w["open"].values
    if len(c) < 20: return None
    p = c[-1]
    if p <= 0: return None

    ma5 = np.mean(c[-5:]); ma10 = np.mean(c[-10:]); ma20 = np.mean(c[-20:])
    ma5_prev = np.mean(c[-6:-1]); ma10_prev = np.mean(c[-11:-1])
    ma5_slope = (ma5 - ma5_prev) / ma5_prev * 100 if ma5_prev > 0 else 0
    ma10_slope = (ma10 - ma10_prev) / ma10_prev * 100 if ma10_prev > 0 else 0

    # RSI
    if len(c) >= 15:
        d = np.diff(c[-15:])
        g = np.mean(d[d > 0]) if np.any(d > 0) else 0
        ls = -np.mean(d[d < 0]) if np.any(d < 0) else 0.001
        rsi = 100 - 100 / (1 + g / ls)
    else: rsi = 50

    bias5 = (p - ma5) / ma5 * 100 if ma5 > 0 else 0
    bias10 = (p - ma10) / ma10 * 100 if ma10 > 0 else 0
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

    # 连板/涨停相关
    cl = 0
    for j in range(-1, max(-8, -len(c)), -1):
        dd = (c[j]/c[j-1]-1)*100 if j-1 >= -len(c) else 0
        if dd > 9.5: cl += 1
        else: break

    # 近5日涨停次数
    limit_count_5d = sum(1 for j in range(-1, max(-6,-len(c)),-1)
                         if j-1 >= -len(c) and (c[j]/c[j-1]-1)*100 > 9.5)

    hrl = any((c[j]/c[j-1]-1)*100 > 9.5 for j in range(-2, max(-8,-len(c)),-1) if j-1 >= -len(c))

    # 今日是否涨停
    today_limit = tc > 9.5

    # 昨日是否涨停
    yest_limit = (c[-2]/c[-3]-1)*100 > 9.5 if len(c) >= 3 else False

    cu = 0
    for j in range(-1, max(-8, -len(c)), -1):
        if c[j] > c[j-1]: cu += 1
        else: break

    to = float(df.iloc[idx].get("turnover_rate", 0))
    if pd.isna(to): to = 0

    # 实体/上影线比（判断抛压）
    body = abs(c[-1] - o[-1])
    upper_shadow = h[-1] - max(c[-1], o[-1])
    lower_shadow = min(c[-1], o[-1]) - l[-1]
    bar_range = h[-1] - l[-1]
    body_ratio = body / bar_range if bar_range > 0 else 0.5
    upper_ratio = upper_shadow / bar_range if bar_range > 0 else 0

    # 放量突破
    vol_breakout = vr > 2.0 and tc > 3 and p > h20 * 0.98

    # 缩量回踩MA
    vol_pullback = vr < 0.7 and abs(p - ma10) / ma10 < 0.01 if ma10 > 0 else False

    return dict(price=p, ma5=ma5, ma10=ma10, ma20=ma20,
                ma5_slope=ma5_slope, ma10_slope=ma10_slope,
                rsi=rsi, bias5=bias5, bias10=bias10, bias20=bias20, price_pos=pp,
                vol_ratio=vr, vol_shrink=vs, today_chg=tc, chg_3d=c3, chg_5d=c5,
                dif=dif, dea=dea, macd_cross=mc, consec_limit=cl,
                limit_count_5d=limit_count_5d,
                has_recent_limit=hrl, consec_up=cu, turnover=to, h20=h20, l20=l20,
                today_limit=today_limit, yest_limit=yest_limit,
                body_ratio=body_ratio, upper_ratio=upper_ratio,
                vol_breakout=vol_breakout, vol_pullback=vol_pullback)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 市场环境判断 V2（更灵活）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MarketRegimeDetector:
    """收盘后用已知数据判断市场状态 — 更灵活版"""

    def __init__(self):
        self._prev_regime = "SIDE"
        self._regime_days = 0
        self._pending_regime = None
        self._pending_count = 0
        self._breadth_history = []

    def detect(self, all_data: dict, date: str) -> dict:
        score = 0
        details = []

        above, below, limit_down_count, limit_up_count = 0, 0, 0, 0
        total = 0
        for code, df in all_data.items():
            idx_arr = df.index[df["date"] == date]
            if len(idx_arr) == 0: continue
            i = idx_arr[0]; total += 1
            cl = float(df["close"].iloc[i])
            if i >= 20:
                m20 = df["close"].iloc[i-19:i+1].mean()
                if cl > m20: above += 1
                else: below += 1
            if i >= 1:
                chg = (cl / float(df["close"].iloc[i-1]) - 1) * 100
                if chg < -9.5: limit_down_count += 1
                if chg > 9.5: limit_up_count += 1

        breadth = above / total if total > 0 else 0.5
        limit_down_pct = limit_down_count / total * 100 if total > 0 else 0
        limit_up_pct = limit_up_count / total * 100 if total > 0 else 0
        self._breadth_history.append(breadth)

        # 宽度评分
        if breadth > 0.65:
            score += 40; details.append(f"breadth={breadth:.0%}(+40)")
        elif breadth > 0.55:
            score += 25; details.append(f"breadth={breadth:.0%}(+25)")
        elif breadth > 0.45:
            score += 10; details.append(f"breadth={breadth:.0%}(+10)")
        elif breadth > 0.35:
            score -= 15; details.append(f"breadth={breadth:.0%}(-15)")
        else:
            score -= 35; details.append(f"breadth={breadth:.0%}(-35)")

        # 宽度趋势（近3日是否改善）
        if len(self._breadth_history) >= 3:
            b3 = self._breadth_history[-3:]
            if b3[-1] > b3[0] + 0.05:
                score += 10; details.append("breadth_up(+10)")
            elif b3[-1] < b3[0] - 0.05:
                score -= 10; details.append("breadth_dn(-10)")

        # 跌停恐慌
        if limit_down_pct > 5:
            score -= 30; details.append(f"LD={limit_down_count}(-30)")
        elif limit_down_pct > 2:
            score -= 15; details.append(f"LD={limit_down_count}(-15)")

        # 涨停活跃度（牛市信号）
        if limit_up_pct > 3:
            score += 15; details.append(f"LU={limit_up_count}(+15)")
        elif limit_up_pct > 1.5:
            score += 8; details.append(f"LU={limit_up_count}(+8)")

        # 涨跌比
        up_count = sum(1 for code, df in all_data.items()
                       if not df[df["date"]==date].empty and
                       float(df[df["date"]==date].iloc[0].get("change_pct", 0) or 0) > 0)
        up_ratio = up_count / total if total > 0 else 0.5
        if up_ratio > 0.6:
            score += 15; details.append(f"upR={up_ratio:.0%}(+15)")
        elif up_ratio < 0.35:
            score -= 15; details.append(f"upR={up_ratio:.0%}(-15)")

        # 判定
        if score >= 45:
            raw = "BULL"
        elif score >= 10:
            raw = "SIDE"
        elif score >= -15:
            raw = "BEAR"
        else:
            raw = "CRASH"

        # V2: 非对称确认 — 牛市1天确认(快进), 其余2天确认(慢出)
        confirm_needed = 1 if raw == "BULL" else 2

        if raw != self._prev_regime:
            if raw == self._pending_regime:
                self._pending_count += 1
            else:
                self._pending_regime = raw
                self._pending_count = 1

            if self._pending_count >= confirm_needed:
                self._prev_regime = raw
                self._regime_days = 1
                self._pending_regime = None
                self._pending_count = 0
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
            "limit_up": limit_up_count,
            "detail": ", ".join(details),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V2策略评分（更激进）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_bull_v2(ind):
    """牛市V2: 打板/接力 + 趋势强势股

    核心思路:
    1. 涨停次日低开/平开接力（龙头接力）
    2. 放量突破新高（突破买入）
    3. 缩量回踩MA10/MA20（趋势回踩）
    """
    s = 0

    # === A. 龙头接力（高优先级）===
    # 昨日涨停 + 今日未大跌(回踩不超过3%) + 换手适中
    if ind["yest_limit"]:
        s += 25
        if ind["today_chg"] > -3 and ind["today_chg"] < 3:
            s += 15  # 高位震荡消化，明天可能继续
        if 3 < ind["turnover"] < 15:
            s += 10  # 换手适中，不是出货
        if ind["consec_limit"] >= 1:
            s += 10  # 连板加速
        if ind["upper_ratio"] < 0.3:
            s += 5   # 没有大上影线

    # === B. 放量突破 ===
    if ind["vol_breakout"]:
        s += 20
        if ind["ma5"] > ind["ma10"] > ind["ma20"]: s += 15
        if ind["today_chg"] > 5: s += 10  # 强势突破
        if 3 < ind["turnover"] < 12: s += 5

    # === C. 趋势回踩 ===
    if not ind["yest_limit"] and not ind["vol_breakout"]:
        if ind["ma5"] > ind["ma10"] > ind["ma20"]: s += 10
        elif ind["ma5"] > ind["ma10"]: s += 5
        else: return 0  # 非多头排列不做

        if ind["vol_pullback"]: s += 15
        if ind["macd_cross"]: s += 12
        elif ind["dif"] > ind["dea"]: s += 5

        if 0.3 < ind["price_pos"] < 0.6: s += 8
        if 30 < ind["rsi"] < 50: s += 8
        if 2 <= ind["consec_up"] <= 4: s += 5

    # 通用扣分
    if ind["today_chg"] > 9.5: s -= 20  # 今天涨停了明天追风险大
    if ind["upper_ratio"] > 0.5: s -= 10  # 上影线太长
    if ind["rsi"] > 80: s -= 15  # 严重超买
    if ind["turnover"] > 25: s -= 15  # 换手率过高，可能出货

    return max(0, s)


def score_side_v2(ind):
    """震荡V2: 网格低吸 + 放量突破（双模式）"""
    s = 0

    # === A. 网格低吸（优先）===
    if ind["price_pos"] < 0.15:
        s += 30
    elif ind["price_pos"] < 0.25:
        s += 22
    elif ind["price_pos"] < 0.35:
        s += 15
    elif ind["price_pos"] < 0.45:
        s += 8

    if ind["bias20"] < -8: s += 18
    elif ind["bias20"] < -5: s += 12
    elif ind["bias20"] < -3: s += 8
    elif ind["bias20"] < 0: s += 4
    else: s -= 5

    if ind["vol_ratio"] < 0.6: s += 12
    elif ind["vol_ratio"] < 0.8: s += 8

    if ind["macd_cross"]: s += 10
    elif ind["dif"] < 0 and ind["dif"] > ind["dea"]: s += 5

    if 0 < ind["today_chg"] < 2: s += 5
    if ind["rsi"] < 30: s += 10
    elif ind["rsi"] < 40: s += 5

    # === B. 放量突破加分 ===
    if ind["vol_breakout"]:
        s += 15  # 震荡市突破也有机会

    if ind["today_chg"] < -5: s -= 10
    return max(0, s)


def score_bear_v2(ind):
    """熊市/空仓V2: 极端超跌反弹（快进快出）"""
    s = 0
    if ind["rsi"] < 18: s += 35
    elif ind["rsi"] < 22: s += 25
    elif ind["rsi"] < 28: s += 12
    else: return 0

    if ind["chg_3d"] < -12: s += 20
    elif ind["chg_3d"] < -8: s += 12
    elif ind["chg_3d"] < -5: s += 5

    if ind["price_pos"] < 0.08: s += 15
    elif ind["price_pos"] < 0.15: s += 10

    if ind["vol_shrink"] < 0.3: s += 10
    if ind["macd_cross"]: s += 10
    if ind["bias5"] < -7: s += 8

    if ind["today_chg"] < -8: s -= 15
    return max(0, s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V2 参数表（激进版）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGIME_CONFIG_V2 = {
    "BULL": {
        "score_fn": score_bull_v2, "min_score": 40,
        # V2: 集中！最多2只，单只50%，总仓90%
        "max_positions": 2, "single_pct": 0.50, "total_pct": 0.90,
        # V2: 梯度止盈 — 不用一刀切
        "stop_loss": -4.0,
        "tp_half": 6.0,      # 盈利6%先卖一半
        "tp_full": 15.0,     # 盈利15%清仓
        "trail_trigger": 8.0, "trail_dd": 3.0,
        "hold_max": 5, "price_range": (3, 50),
    },
    "SIDE": {
        "score_fn": score_side_v2, "min_score": 40,
        # V2: 稍集中
        "max_positions": 2, "single_pct": 0.35, "total_pct": 0.60,
        "stop_loss": -3.0,
        "tp_half": 4.0,
        "tp_full": 8.0,
        "trail_trigger": 4.0, "trail_dd": 2.0,
        "hold_max": 4, "price_range": (3, 30),
    },
    "BEAR": {
        "score_fn": score_bear_v2, "min_score": 50,
        # V2: 仍然做! 但轻仓快进快出
        "max_positions": 1, "single_pct": 0.25, "total_pct": 0.25,
        "stop_loss": -3.0,
        "tp_half": 3.0,
        "tp_full": 6.0,
        "trail_trigger": 3.0, "trail_dd": 1.5,
        "hold_max": 2, "price_range": (3, 20),
    },
    "CRASH": {
        # V2: CRASH也做极端超跌反弹!（极轻仓）
        "score_fn": score_bear_v2, "min_score": 60,  # 门槛更高
        "max_positions": 1, "single_pct": 0.15, "total_pct": 0.15,
        "stop_loss": -2.5,
        "tp_half": 2.0,
        "tp_full": 5.0,
        "trail_trigger": 2.0, "trail_dd": 1.5,
        "hold_max": 2, "price_range": (3, 15),
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V2 自适应回测引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveBacktestV2:
    def __init__(self, all_data):
        self.all_data = all_data
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

        # V2: 连赢/连亏追踪（用于动态仓位）
        self._recent_results = []  # 最近5笔交易的盈亏 True/False
        self._win_streak = 0
        self._loss_streak = 0

        dates_set = set()
        for df in all_data.values():
            dates_set.update(df["date"].tolist())
        self.trading_days = sorted(dates_set)
        self.day_idx_map = {d: i for i, d in enumerate(self.trading_days)}

    def _position_multiplier(self):
        """凯利准则简化: 连赢放大仓位, 连亏缩小"""
        if self._win_streak >= 3: return 1.3
        elif self._win_streak >= 2: return 1.15
        elif self._loss_streak >= 3: return 0.6
        elif self._loss_streak >= 2: return 0.8
        return 1.0

    def run(self, start_date, end_date):
        days = [d for d in self.trading_days if start_date <= d <= end_date]
        logger.info(f"V2 Adaptive Backtest: {start_date} ~ {end_date} ({len(days)}d)")
        logger.info(f"Initial: Y{INITIAL_CAPITAL:,.0f}")

        for day_num, date in enumerate(days):
            day_idx = self.day_idx_map.get(date, day_num)

            # 1. 开盘执行
            self._exec_pending_sells(date, day_idx)
            self._exec_pending_buys(date, day_idx)

            # 2. 盘中极端止损
            self._emergency_stop(date, day_idx)

            # 3. 收盘更新
            self._update_close(date)

            # 4. 盘后市场判断
            regime_info = self._regime_detector.detect(self.all_data, date)
            self._current_regime = regime_info["regime"]
            self._regime_history.append({"date": date, **regime_info})

            # 5. 盘后复盘持仓
            cfg = REGIME_CONFIG_V2[self._current_regime]
            self._review_sells_v2(date, day_idx, cfg)

            # 6. 盘后选股
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
                            f"{self._current_regime} | Hold={len(self.holdings)} | "
                            f"WinStreak={self._win_streak} LossStreak={self._loss_streak}")

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
            # 跌停检测
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
            if code in self.holdings or len(self.holdings) >= cfg["max_positions"]: continue
            row = self._get_row(code, date)
            if row is None: continue
            op = float(row["open"])
            if op <= 0: continue
            sc = pb.get("signal_close", 0)
            if sc > 0 and op >= sc * 1.098: continue   # 一字板
            if sc > 0 and op > sc * 1.05: continue      # 高开5%+不追

            cost = op * (1 + SLIPPAGE_PCT)
            total_asset = self.cash + sum(h.market_value for h in self.holdings.values())
            mv_now = sum(h.market_value for h in self.holdings.values())
            max_mv = total_asset * cfg["total_pct"] * pm  # V2: 乘以动态系数
            remaining = max_mv - mv_now
            if remaining <= 0: continue

            buy_amount = min(self.cash * 0.95,
                             total_asset * cfg["single_pct"] * pm,
                             remaining)
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
            if pnl <= -7.0:
                trigger = h.cost_price * 0.93
                self._sell(code, h.shares, trigger, date, f"E-STOP{pnl:.1f}%")

    def _update_close(self, date):
        for code, h in self.holdings.items():
            row = self._get_row(code, date)
            if row is None: continue
            h.current_price = float(row["close"])
            h.peak_price = max(h.peak_price, float(row["high"]))

    def _review_sells_v2(self, date, day_idx, cfg):
        """V2: 梯度止盈"""
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx == day_idx: continue
            pnl = h.pnl_pct
            hold = day_idx - h.buy_day_idx
            pk = (h.peak_price - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            dd = (h.peak_price - h.current_price) / h.peak_price * 100 if h.peak_price > 0 else 0

            # 止损
            if pnl <= cfg["stop_loss"]:
                self._pending_sells.append((code, h.shares, f"SL{pnl:.1f}%"))
                continue

            # 移动止盈
            if pk >= cfg["trail_trigger"] and dd >= cfg["trail_dd"]:
                self._pending_sells.append((code, h.shares, f"TRAIL pk{pk:.1f}%dd{dd:.1f}%"))
                continue

            # V2: 梯度止盈 — 先卖一半
            if not h.partial_sold and pnl >= cfg["tp_half"]:
                half = max(100, (h.shares // 200) * 100)
                if half < h.shares:
                    self._pending_sells.append((code, half, f"TP_HALF{pnl:.1f}%"))
                    h.partial_sold = True
                    continue

            # 全止盈
            if pnl >= cfg["tp_full"]:
                self._pending_sells.append((code, h.shares, f"TP_FULL{pnl:.1f}%"))
                continue

            # 超期（但如果盈利就放宽）
            max_hold = cfg["hold_max"]
            if pnl > 2: max_hold += 2  # 盈利多给2天
            if hold >= max_hold and pnl < 1:
                self._pending_sells.append((code, h.shares, f"EXPIRE{hold}d"))

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
            to = float(row.get("turnover_rate", 0) or 0)
            if to < 1: continue  # 换手率太低的不做
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

        # V2: 更新连赢/连亏
        pnl = (sp - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
        if shares >= h.shares:
            # 全部卖出 — 统计胜负
            if pnl > 0:
                self._win_streak += 1; self._loss_streak = 0
            else:
                self._loss_streak += 1; self._win_streak = 0
            self._recent_results.append(pnl > 0)
            if len(self._recent_results) > 10: self._recent_results.pop(0)
            del self.holdings[code]
        else:
            h.shares -= shares

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
        pf = abs(sum(wl) / sum(ll)) if ll and sum(ll) != 0 else 0
        days = len(self.snapshots)
        ann = ret * (252 / days) if days > 0 else 0
        dr = [s.daily_return_pct for s in self.snapshots]
        sharpe = np.mean(dr) / np.std(dr) * np.sqrt(252) if len(dr) > 1 and np.std(dr) > 0 else 0

        regime_counts = defaultdict(int)
        for s in self.snapshots: regime_counts[s.regime] += 1

        monthly = defaultdict(list)
        for s in self.snapshots: monthly[s.date[:7]].append(s.daily_return_pct)

        regime_trades = defaultdict(lambda: {"buys": 0, "sells": 0})
        for t in self.trades:
            if t.direction == "buy": regime_trades[t.regime]["buys"] += 1
            else: regime_trades[t.regime]["sells"] += 1

        print("\n" + "=" * 70)
        print("  V2 Adaptive Strategy Backtest Report 2026 YTD")
        print("=" * 70)
        print(f"\n  Initial:    Y{INITIAL_CAPITAL:>12,}")
        print(f"  Final:      Y{final:>12,.0f}")
        print(f"  Return:     {ret:>+11.2f}%")
        print(f"  Annualized: {ann:>+11.2f}%")
        print(f"  Max DD:     {max_dd:>11.2f}%")
        print(f"  Sharpe:     {sharpe:>11.2f}")
        print(f"\n  Trades: {len(self.trades)}  Win: {wins}  Loss: {losses}  "
              f"WinRate: {wr:.1f}%  AvgW: {aw:+.2f}%  AvgL: {al:+.2f}%  PF: {pf:.2f}")
        print(f"  Kelly mult: wins={self._win_streak} losses={self._loss_streak}")

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
            month_regimes = [s.regime for s in self.snapshots if s.date.startswith(m)]
            dominant = max(set(month_regimes), key=month_regimes.count) if month_regimes else "?"
            mark = "+" if mr >= 0 else "-"
            print(f"    {m}  {mr:>+7.2f}%  cum{cum:>+7.2f}%  [{mark}]  regime={dominant}")

        print("=" * 70)

        # 保存
        out_dir = Path("data/backtest_adaptive_v2")
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

        trade_rows = [{"date": t.date, "code": t.code, "name": t.name,
                        "dir": t.direction, "shares": t.shares, "price": round(t.price, 3),
                        "amount": round(t.amount, 0), "reason": t.reason, "regime": t.regime}
                       for t in self.trades]
        pd.DataFrame(trade_rows).to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")

        snap_rows = [{"date": s.date, "total": round(s.total_asset, 0), "cash": round(s.cash, 0),
                       "mv": round(s.market_value, 0), "hold": s.holdings_count,
                       "ret": round(s.daily_return_pct, 4), "regime": s.regime}
                      for s in self.snapshots]
        pd.DataFrame(snap_rows).to_csv(out_dir / "daily.csv", index=False, encoding="utf-8-sig")

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

    engine = AdaptiveBacktestV2(all_data)
    engine.run("2026-01-05", "2026-04-10")
