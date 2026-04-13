"""
华尔街之狼 多策略对比回测
====================================
无未来函数：T日盘后决策 → T+1开盘执行

策略:
  A. 超跌反弹 (oversold_bounce)  — RSI低+偏离大+近支撑 → 买入等反弹
  B. 龙头埋伏 (pre_dragon)       — 近期涨停后回调到位 → 埋伏等二波
  C. 网格低吸 (grid_support)     — 20日低位+缩量 → 低吸高抛
  D. 趋势跟随 (trend_follow)     — MA金叉+温和放量+低位 → 顺势做多
  E. 均值回归 (mean_revert)      — 综合偏离度最大的 → 回归均值

python backtest_strategies.py
"""
import json, logging, os, time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 公共参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INITIAL_CAPITAL = 200_000
COMMISSION_RATE = 0.00025
STAMP_TAX_RATE = 0.0005
SLIPPAGE_PCT = 0.002


@dataclass
class Holding:
    code: str
    name: str
    shares: int
    cost_price: float
    buy_date: str
    buy_day_idx: int
    current_price: float = 0.0
    peak_price: float = 0.0

    @property
    def market_value(self): return self.shares * self.current_price
    @property
    def pnl_pct(self):
        return (self.current_price - self.cost_price) / self.cost_price * 100 if self.cost_price > 0 else 0


@dataclass
class Trade:
    date: str; code: str; name: str; direction: str
    shares: int; price: float; amount: float; commission: float; reason: str


@dataclass
class Snapshot:
    date: str; total_asset: float; cash: float; market_value: float
    holdings_count: int; daily_return_pct: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 技术指标计算（公共）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_indicators(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """计算 ≤idx 的所有技术指标（无未来函数）"""
    if idx < 25:
        return None
    window = df.iloc[max(0, idx - 60):idx + 1]
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    vol = window["volume"].values

    if len(close) < 20:
        return None

    price = close[-1]
    if price <= 0:
        return None

    # MA
    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:])

    # MA5斜率
    if len(close) >= 6:
        prev_ma5 = np.mean(close[-6:-1])
        ma5_slope = (ma5 - prev_ma5) / prev_ma5 * 100 if prev_ma5 > 0 else 0
    else:
        ma5_slope = 0

    # RSI
    if len(close) >= 15:
        deltas = np.diff(close[-15:])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.001
        rsi = 100 - 100 / (1 + gain / loss)
    else:
        rsi = 50

    # Bias
    bias5 = (price - ma5) / ma5 * 100 if ma5 > 0 else 0
    bias20 = (price - ma20) / ma20 * 100 if ma20 > 0 else 0

    # 20日价格位置
    h20 = np.max(high[-20:])
    l20 = np.min(low[-20:])
    price_pos = (price - l20) / (h20 - l20) if (h20 - l20) > 0 else 0.5

    # 量比
    vol_ma20 = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol[-5:])
    vol_ratio = vol[-1] / vol_ma20 if vol_ma20 > 0 else 1

    # 涨跌幅
    today_chg = (close[-1] / close[-2] - 1) * 100 if len(close) >= 2 else 0
    chg_3d = (price / close[-4] - 1) * 100 if len(close) >= 4 else 0
    chg_5d = (price / close[-6] - 1) * 100 if len(close) >= 6 else 0

    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    dif = ema12[-1] - ema26[-1]
    dea_arr = pd.Series(ema12 - ema26).ewm(span=9).mean().values
    dea = dea_arr[-1]
    macd_bar = 2 * (dif - dea)
    macd_cross = dif > dea and (ema12[-2] - ema26[-2]) <= (dea_arr[-2] if len(dea_arr) >= 2 else dea)

    # 连板
    consec_limit = 0
    for j in range(-1, max(-8, -len(close)), -1):
        d = (close[j] / close[j-1] - 1) * 100 if j-1 >= -len(close) else 0
        if d > 9.5:
            consec_limit += 1
        else:
            break

    # 近7日是否有过涨停
    has_recent_limit = False
    for j in range(-2, max(-8, -len(close)), -1):
        d = (close[j] / close[j-1] - 1) * 100 if j-1 >= -len(close) else 0
        if d > 9.5:
            has_recent_limit = True
            break

    # 连阳
    consec_up = 0
    for j in range(-1, max(-8, -len(close)), -1):
        if close[j] > close[j-1]:
            consec_up += 1
        else:
            break

    # 缩量程度
    vol_shrink = vol[-1] / np.max(vol[-10:]) if np.max(vol[-10:]) > 0 else 1

    turnover = float(df.iloc[idx].get("turnover_rate", 0))
    if pd.isna(turnover):
        turnover = 0

    return {
        "price": price, "ma5": ma5, "ma10": ma10, "ma20": ma20,
        "ma5_slope": ma5_slope,
        "rsi": rsi, "bias5": bias5, "bias20": bias20,
        "price_pos": price_pos, "vol_ratio": vol_ratio, "vol_shrink": vol_shrink,
        "today_chg": today_chg, "chg_3d": chg_3d, "chg_5d": chg_5d,
        "dif": dif, "dea": dea, "macd_bar": macd_bar, "macd_cross": macd_cross,
        "consec_limit": consec_limit, "has_recent_limit": has_recent_limit,
        "consec_up": consec_up, "turnover": turnover,
        "h20": h20, "l20": l20,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略评分函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_oversold_bounce(ind: dict) -> int:
    """策略A: 超跌反弹 — RSI低+偏离大+近支撑位"""
    score = 0
    # RSI超卖是核心信号
    if ind["rsi"] < 25: score += 30
    elif ind["rsi"] < 30: score += 25
    elif ind["rsi"] < 35: score += 18
    elif ind["rsi"] < 40: score += 10
    elif ind["rsi"] > 60: return 0  # 不超卖不买

    # Bias5偏离（越负越好）
    if ind["bias5"] < -6: score += 25
    elif ind["bias5"] < -4: score += 20
    elif ind["bias5"] < -2: score += 12
    elif ind["bias5"] < 0: score += 5
    elif ind["bias5"] > 3: score -= 15

    # 20日价格位置（低位更好）
    if ind["price_pos"] < 0.15: score += 20
    elif ind["price_pos"] < 0.25: score += 15
    elif ind["price_pos"] < 0.35: score += 10
    elif ind["price_pos"] > 0.7: score -= 15

    # 缩量企稳（卖压释放）
    if ind["vol_shrink"] < 0.4: score += 10
    elif ind["vol_shrink"] < 0.6: score += 5

    # MACD底部收敛/金叉
    if ind["macd_cross"]: score += 10
    elif ind["macd_bar"] > 0 and ind["dif"] < 0: score += 5  # 底部放红

    # 不追跌停（跌太猛可能有问题）
    if ind["today_chg"] < -8: score -= 20
    # 已经开始反弹更好
    if 0 < ind["today_chg"] < 3: score += 5

    return max(0, score)


def score_pre_dragon(ind: dict) -> int:
    """策略B: 龙头埋伏 — 近期涨停后回调到位，等二波"""
    if not ind["has_recent_limit"]:
        return 0  # 没有近期涨停历史，不考虑

    score = 20  # 基础分：有过涨停

    # 回调幅度（从高点回落越多越好，但不能太多）
    if ind["price_pos"] < 0.3: score += 20   # 回调到低位
    elif ind["price_pos"] < 0.5: score += 15
    elif ind["price_pos"] < 0.7: score += 5
    elif ind["price_pos"] > 0.85: return 0   # 还在高位，不是回调

    # 缩量回调（洗盘完毕信号）
    if ind["vol_shrink"] < 0.3: score += 15
    elif ind["vol_shrink"] < 0.5: score += 10
    elif ind["vol_shrink"] < 0.7: score += 5

    # RSI不能太低（不是崩盘）也不能太高
    if 30 < ind["rsi"] < 50: score += 10
    elif 25 < ind["rsi"] < 30: score += 5

    # MA5开始走平或上翘（止跌信号）
    if -1 < ind["ma5_slope"] < 1: score += 10  # 走平
    elif ind["ma5_slope"] > 1: score += 5       # 上翘

    # 今天涨幅不大（还在底部，没被别人抢跑）
    if ind["today_chg"] > 5: return 0  # 已经启动了
    if 0 < ind["today_chg"] < 3: score += 5

    return max(0, score)


def score_grid_support(ind: dict) -> int:
    """策略C: 网格低吸 — 20日低位+缩量+支撑位"""
    score = 0

    # 20日价格位置是核心（越低越好）
    if ind["price_pos"] < 0.1: score += 30
    elif ind["price_pos"] < 0.2: score += 25
    elif ind["price_pos"] < 0.3: score += 18
    elif ind["price_pos"] < 0.4: score += 10
    elif ind["price_pos"] > 0.6: return 0  # 不在低位不买

    # 偏离MA20（负偏离=便宜）
    if ind["bias20"] < -8: score += 20
    elif ind["bias20"] < -5: score += 15
    elif ind["bias20"] < -3: score += 10
    elif ind["bias20"] < 0: score += 5

    # 缩量（卖盘枯竭）
    if ind["vol_ratio"] < 0.6: score += 15
    elif ind["vol_ratio"] < 0.8: score += 10
    elif ind["vol_ratio"] < 1.0: score += 5
    elif ind["vol_ratio"] > 2.0: score -= 10  # 放量杀跌不好

    # MA5斜率开始企稳
    if abs(ind["ma5_slope"]) < 1: score += 5

    # 今天不能大跌（抄底抄在半山腰）
    if ind["today_chg"] < -5: score -= 10
    if 0 < ind["today_chg"] < 2: score += 5  # 微涨企稳

    return max(0, score)


def score_trend_follow(ind: dict) -> int:
    """策略D: 趋势跟随 — MA金叉+温和放量+低位启动"""
    score = 0

    # MA排列
    if ind["ma5"] > ind["ma10"] > ind["ma20"]:
        score += 15  # 多头排列
    elif ind["ma5"] > ind["ma10"]:
        score += 10  # 短期多头
    elif ind["ma5"] < ind["ma10"] < ind["ma20"]:
        return 0     # 空头不做

    # MACD金叉
    if ind["macd_cross"]: score += 15
    elif ind["dif"] > ind["dea"]: score += 8

    # RSI中等偏低（有上行空间）
    if 35 < ind["rsi"] < 50: score += 10
    elif 50 <= ind["rsi"] < 60: score += 5
    elif ind["rsi"] > 70: score -= 15

    # 温和放量（不是爆量）
    if 1.0 < ind["vol_ratio"] < 1.8: score += 10
    elif 0.8 < ind["vol_ratio"] <= 1.0: score += 5
    elif ind["vol_ratio"] > 2.5: score -= 5

    # 价格位置适中（不在底部也不在顶部）
    if 0.3 < ind["price_pos"] < 0.6: score += 10
    elif 0.2 < ind["price_pos"] <= 0.3: score += 5
    elif ind["price_pos"] > 0.85: score -= 10

    # 连阳蓄势
    if 2 <= ind["consec_up"] <= 4 and ind["today_chg"] < 5:
        score += 8

    # 不追高
    if ind["today_chg"] > 5: score -= 15
    elif ind["today_chg"] > 3: score -= 5

    return max(0, score)


def score_mean_revert(ind: dict) -> int:
    """策略E: 均值回归 — 综合偏离度"""
    score = 0

    # Bias5（核心指标，与回报负相关 corr=-0.103）
    if ind["bias5"] < -6: score += 25
    elif ind["bias5"] < -4: score += 20
    elif ind["bias5"] < -2: score += 15
    elif ind["bias5"] < 0: score += 8
    elif ind["bias5"] > 5: score -= 15

    # MA趋势（反转加分：空头排列=回归空间大）
    if ind["ma5"] < ind["ma10"] < ind["ma20"]:
        score += 12  # 空头排列，回归空间最大
    elif ind["ma5"] < ind["ma10"]:
        score += 8   # 短期空头
    elif ind["ma5"] > ind["ma10"] > ind["ma20"]:
        score += 3   # 多头，回归空间小

    # RSI
    if ind["rsi"] < 30: score += 20
    elif ind["rsi"] < 40: score += 15
    elif ind["rsi"] < 50: score += 8
    elif ind["rsi"] > 65: score -= 10

    # MA5斜率（下跌越快→反弹越急）
    if ind["ma5_slope"] < -3: score += 12
    elif ind["ma5_slope"] < -1: score += 8
    elif ind["ma5_slope"] < 0: score += 4

    # 20日价格位置
    if ind["price_pos"] < 0.2: score += 15
    elif ind["price_pos"] < 0.35: score += 10
    elif ind["price_pos"] < 0.5: score += 5
    elif ind["price_pos"] > 0.8: score -= 10

    # 量比（缩量到位更好）
    if 0.5 < ind["vol_ratio"] < 1.0: score += 8
    elif ind["vol_ratio"] > 2.5: score -= 8

    # 不追高
    if ind["today_chg"] > 5: score -= 15
    # 深跌加分
    if ind["today_chg"] < -3: score += 5

    return max(0, score)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGIES = {
    "oversold_bounce": {
        "name": "超跌反弹", "score_fn": score_oversold_bounce,
        "min_score": 55, "max_positions": 3, "single_pct": 0.35,
        "stop_loss": -4.0, "take_profit": 6.0, "hold_max": 5,
        "price_range": (3, 30),
    },
    "pre_dragon": {
        "name": "龙头埋伏", "score_fn": score_pre_dragon,
        "min_score": 50, "max_positions": 2, "single_pct": 0.45,
        "stop_loss": -5.0, "take_profit": 10.0, "hold_max": 5,
        "price_range": (3, 50),
    },
    "grid_support": {
        "name": "网格低吸", "score_fn": score_grid_support,
        "min_score": 50, "max_positions": 4, "single_pct": 0.25,
        "stop_loss": -3.0, "take_profit": 5.0, "hold_max": 7,
        "price_range": (3, 30),
    },
    "trend_follow": {
        "name": "趋势跟随", "score_fn": score_trend_follow,
        "min_score": 45, "max_positions": 3, "single_pct": 0.35,
        "stop_loss": -4.0, "take_profit": 8.0, "hold_max": 7,
        "price_range": (3, 40),
    },
    "mean_revert": {
        "name": "均值回归", "score_fn": score_mean_revert,
        "min_score": 55, "max_positions": 3, "single_pct": 0.35,
        "stop_loss": -4.0, "take_profit": 6.0, "hold_max": 5,
        "price_range": (3, 30),
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 回测引擎（无未来函数 v3）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StrategyBacktest:
    def __init__(self, all_data, strategy_key: str):
        self.all_data = all_data
        self.cfg = STRATEGIES[strategy_key]
        self.strategy_key = strategy_key
        self.cash = INITIAL_CAPITAL
        self.holdings: Dict[str, Holding] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[Snapshot] = []
        self.prev_total = INITIAL_CAPITAL
        self._pending_buys: List[dict] = []
        self._pending_sells: List[tuple] = []

        dates_set = set()
        for df in all_data.values():
            dates_set.update(df["date"].tolist())
        self.trading_days = sorted(dates_set)
        self.day_idx_map = {d: i for i, d in enumerate(self.trading_days)}

    def run(self, start_date, end_date) -> dict:
        days = [d for d in self.trading_days if start_date <= d <= end_date]

        for day_num, date in enumerate(days):
            day_idx = self.day_idx_map.get(date, day_num)

            # 1. 开盘: 执行昨晚挂的卖出(T+1开盘价)
            self._exec_pending_sells(date, day_idx)
            # 2. 开盘: 执行昨晚挂的买入(T+1开盘价)
            self._exec_pending_buys(date, day_idx)
            # 3. 盘中: 极端止损(-5%条件单)
            self._emergency_stop(date, day_idx)
            # 4. 收盘: 更新价格
            self._update_close(date)
            # 5. 盘后复盘: 决定明天卖什么+买什么
            self._review_sells(date, day_idx)
            self._scan_buys(date, day_idx)

            # 快照
            mv = sum(h.market_value for h in self.holdings.values())
            total = self.cash + mv
            ret = (total - self.prev_total) / self.prev_total * 100 if self.prev_total > 0 else 0
            self.prev_total = total
            self.snapshots.append(Snapshot(date, total, self.cash, mv, len(self.holdings), ret))

        return self._summary()

    def _get_row(self, code, date):
        df = self.all_data.get(code)
        if df is None: return None
        rows = df[df["date"] == date]
        return rows.iloc[0] if not rows.empty else None

    def _exec_pending_sells(self, date, day_idx):
        sells = self._pending_sells[:]
        self._pending_sells = []
        for code, shares, reason in sells:
            h = self.holdings.get(code)
            if not h: continue
            if h.buy_day_idx >= day_idx:
                self._pending_sells.append((code, shares, reason + "(T+1延后)"))
                continue
            row = self._get_row(code, date)
            if row is None: continue
            open_p = float(row["open"])
            if open_p <= 0: continue
            # 跌停检测
            if h.current_price > 0 and open_p <= h.current_price * 0.902:
                self._pending_sells.append((code, shares, reason + "(跌停重挂)"))
                continue
            self._sell(code, min(shares, h.shares), open_p, date, f"盘后→开盘卖|{reason}")

    def _exec_pending_buys(self, date, day_idx):
        buys = self._pending_buys[:]
        self._pending_buys = []
        for pb in buys:
            code = pb["code"]
            if code in self.holdings or len(self.holdings) >= self.cfg["max_positions"]:
                continue
            row = self._get_row(code, date)
            if row is None: continue
            open_p = float(row["open"])
            if open_p <= 0: continue
            # 一字涨停买不进
            sig_close = pb.get("signal_close", 0)
            if sig_close > 0 and open_p >= sig_close * 1.098:
                continue
            # 开盘价比信号价涨太多（高开5%+不追）
            if sig_close > 0 and open_p > sig_close * 1.05:
                continue

            cost = open_p * (1 + SLIPPAGE_PCT)
            total_asset = self.cash + sum(h.market_value for h in self.holdings.values())
            buy_amount = min(self.cash * 0.95, total_asset * self.cfg["single_pct"])
            shares = int(buy_amount / cost / 100) * 100
            if shares < 100: continue
            amount = cost * shares
            comm = max(amount * COMMISSION_RATE, 5)
            if amount + comm > self.cash: continue

            self.cash -= (amount + comm)
            self.holdings[code] = Holding(
                code=code, name=pb.get("name", code), shares=shares,
                cost_price=cost, buy_date=date, buy_day_idx=day_idx,
                current_price=open_p, peak_price=open_p,
            )
            self.trades.append(Trade(
                date, code, pb.get("name", code), "buy", shares, cost, amount, comm,
                f"T+1开盘买|{pb.get('reason', '')}",
            ))

    def _emergency_stop(self, date, day_idx):
        to_sell = []
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx >= day_idx: continue
            row = self._get_row(code, date)
            if row is None: continue
            low = float(row["low"])
            pnl = (low - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            if pnl <= -5.0:
                trigger = h.cost_price * 0.95
                to_sell.append((code, h.shares, trigger, f"盘中极端止损{pnl:.1f}%"))
        for code, shares, price, reason in to_sell:
            self._sell(code, shares, price, date, reason)

    def _update_close(self, date):
        for code, h in self.holdings.items():
            row = self._get_row(code, date)
            if row is None: continue
            h.current_price = float(row["close"])
            h.peak_price = max(h.peak_price, float(row["high"]))

    def _review_sells(self, date, day_idx):
        """盘后复盘: 用今日收盘数据决定明天卖什么"""
        sl = self.cfg["stop_loss"]
        tp = self.cfg["take_profit"]
        hm = self.cfg["hold_max"]

        for code, h in list(self.holdings.items()):
            if h.buy_day_idx == day_idx: continue  # 今天买的，后天才能卖
            pnl = h.pnl_pct
            hold = day_idx - h.buy_day_idx
            peak_pnl = (h.peak_price - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            dd = (h.peak_price - h.current_price) / h.peak_price * 100 if h.peak_price > 0 else 0

            if pnl >= tp:
                self._pending_sells.append((code, h.shares, f"止盈{pnl:.1f}%"))
            elif pnl <= sl:
                self._pending_sells.append((code, h.shares, f"止损{pnl:.1f}%"))
            elif peak_pnl >= 3 and dd >= 2:
                self._pending_sells.append((code, h.shares, f"移动止盈 pk{peak_pnl:.1f}% dd{dd:.1f}%"))
            elif hold >= hm and pnl < 1:
                self._pending_sells.append((code, h.shares, f"超期{hold}d pnl{pnl:.1f}%"))

    def _scan_buys(self, date, day_idx):
        """盘后选股: 用≤今日数据评分，存入明天执行"""
        if len(self.holdings) >= self.cfg["max_positions"]:
            return
        if self._pending_buys:
            return

        score_fn = self.cfg["score_fn"]
        min_score = self.cfg["min_score"]
        pmin, pmax = self.cfg["price_range"]

        candidates = []
        for code, df in self.all_data.items():
            if code in self.holdings: continue
            row_idx = df.index[df["date"] == date]
            if len(row_idx) == 0: continue
            idx = row_idx[0]
            row = df.iloc[idx]
            price = float(row["close"])
            if price < pmin or price > pmax: continue

            ind = calc_indicators(df, idx)
            if ind is None: continue

            s = score_fn(ind)
            if s < min_score: continue

            candidates.append({
                "code": code, "name": str(row.get("股票简称", code)),
                "signal_close": price, "signal_date": date,
                "score": s, "reason": f"score={s}",
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 选前N只（填满仓位）
        slots = self.cfg["max_positions"] - len(self.holdings) - len(self._pending_sells)
        for c in candidates[:max(1, slots)]:
            self._pending_buys.append(c)

    def _sell(self, code, shares, price, date, reason):
        h = self.holdings.get(code)
        if not h: return
        sp = price * (1 - SLIPPAGE_PCT)
        amount = sp * shares
        comm = max(amount * COMMISSION_RATE, 5)
        tax = amount * STAMP_TAX_RATE
        self.cash += (amount - comm - tax)
        self.trades.append(Trade(date, code, h.name, "sell", shares, sp, amount, comm + tax, reason))
        if shares >= h.shares:
            del self.holdings[code]
        else:
            h.shares -= shares

    def _summary(self) -> dict:
        if not self.snapshots:
            return {"strategy": self.strategy_key, "total_return": 0}

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
        wins, losses = 0, 0
        win_pnls, loss_pnls = [], []
        for code, tl in completed.items():
            buys = [t for t in tl if t.direction == "buy"]
            sells = [t for t in tl if t.direction == "sell"]
            if buys and sells:
                avg_b = sum(t.price * t.shares for t in buys) / sum(t.shares for t in buys)
                avg_s = sum(t.price * t.shares for t in sells) / sum(t.shares for t in sells)
                pnl = (avg_s - avg_b) / avg_b * 100
                if pnl > 0:
                    wins += 1; win_pnls.append(pnl)
                else:
                    losses += 1; loss_pnls.append(pnl)

        total_trades = wins + losses
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        days = len(self.snapshots)
        annual = ret * (252 / days) if days > 0 else 0
        daily_rets = [s.daily_return_pct for s in self.snapshots]
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if len(daily_rets) > 1 and np.std(daily_rets) > 0 else 0

        # 月度
        monthly = defaultdict(list)
        for s in self.snapshots: monthly[s.date[:7]].append(s.daily_return_pct)

        return {
            "strategy": self.strategy_key,
            "name": self.cfg["name"],
            "total_return": round(ret, 2),
            "annual_return": round(annual, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "total_trades": len(self.trades),
            "completed": total_trades,
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(pf, 2),
            "final_asset": round(final, 0),
            "monthly": {m: round(sum(r), 2) for m, r in sorted(monthly.items())},
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 运行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_all_strategies(start="2026-01-05", end="2026-04-10"):
    from backtest_2025 import fetch_all_a_share_daily

    logger.info("加载数据...")
    all_data = fetch_all_a_share_daily("20251001", "20260410",
                                        cache_name="backtest_cache_2026ytd.pkl")
    logger.info(f"数据加载完成: {len(all_data)} 只股票")

    results = []
    for key in STRATEGIES:
        logger.info(f"\n{'='*50}")
        logger.info(f"  回测策略: {STRATEGIES[key]['name']} ({key})")
        logger.info(f"{'='*50}")
        engine = StrategyBacktest(all_data, key)
        r = engine.run(start, end)
        results.append(r)
        logger.info(f"  → 收益={r['total_return']:+.2f}% 胜率={r['win_rate']}% "
                     f"夏普={r['sharpe']} 最大回撤={r['max_drawdown']:.1f}%")

    # ── 打印对比表 ──
    print("\n" + "=" * 90)
    print("  多策略对比回测 2026 YTD（无未来函数）")
    print("=" * 90)
    print(f"  {'策略':<12} {'收益率':>8} {'年化':>8} {'胜率':>6} {'盈亏比':>6} "
          f"{'最大回撤':>8} {'夏普':>6} {'交易数':>6} {'最终资产':>12}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x["total_return"], reverse=True):
        print(f"  {r['name']:<10} {r['total_return']:>+7.2f}% {r['annual_return']:>+7.1f}% "
              f"{r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
              f"{r['max_drawdown']:>7.2f}% {r['sharpe']:>5.2f} "
              f"{r['total_trades']:>5} Y{r['final_asset']:>10,.0f}")
    print("-" * 90)

    # 月度对比
    print(f"\n  月度收益对比:")
    months = sorted(set(m for r in results for m in r.get("monthly", {})))
    header = f"  {'策略':<12}" + "".join(f" {m[5:]:>7}" for m in months) + f" {'累计':>8}"
    print(header)
    for r in sorted(results, key=lambda x: x["total_return"], reverse=True):
        row = f"  {r['name']:<10}"
        cum = 0
        for m in months:
            v = r.get("monthly", {}).get(m, 0)
            cum += v
            row += f" {v:>+6.1f}%"
        row += f" {cum:>+7.1f}%"
        print(row)

    print("=" * 90)

    # 保存
    out_dir = Path("data/backtest_strategies")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n结果已保存到 {out_dir}/comparison.json")

    return results


if __name__ == "__main__":
    run_all_strategies()
