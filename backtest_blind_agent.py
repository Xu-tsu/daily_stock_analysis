"""
华尔街之狼 盲态Agent回测 V4
====================================
无未来函数 | 无市场分类 | 纯自适应

核心原则:
  1. Agent对市场一无所知，没有BULL/BEAR/CRASH标签
  2. 选股用统一评分函数，好股票自然高分，烂市场自然没有高分股
  3. 仓位完全由近期交易结果决定:
     - 近5笔胜率高 → 仓位系数上升 (最高1.0 = 满仓)
     - 近5笔胜率低 → 仓位系数下降 (最低0.15 = 轻仓)
     - 连亏3笔 → 暂停1天不买入(冷静期)
  4. 买入门槛动态调整:
     - 近期赚钱 → 门槛降低(敢于出手)
     - 近期亏钱 → 门槛升高(严格筛选)
  5. 自然空仓:
     - 不是"因为判断熊市所以空仓"
     - 而是"没有达标的股票+仓位系数很低→自然空仓"
"""
import json, logging, os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
    buy_date: str; buy_day_idx: int
    current_price: float = 0.0; peak_price: float = 0.0
    partial_sold: bool = False
    @property
    def market_value(self): return self.shares * self.current_price
    @property
    def pnl_pct(self):
        return (self.current_price - self.cost_price) / self.cost_price * 100 if self.cost_price > 0 else 0


@dataclass
class Trade:
    date: str; code: str; name: str; direction: str; shares: int
    price: float; amount: float; commission: float; reason: str


@dataclass
class Snapshot:
    date: str; total_asset: float; cash: float; market_value: float
    holdings_count: int; daily_return_pct: float
    position_coeff: float  # 当时的仓位系数
    buy_threshold: int     # 当时的买入门槛


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 技术指标（纯个股层面，不含市场判断）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_indicators(df, idx):
    """纯个股技术面指标，不涉及任何市场层面信息"""
    if idx < 30: return None
    w = df.iloc[max(0, idx-60):idx+1]
    c = w["close"].values; h = w["high"].values; l = w["low"].values
    v = w["volume"].values; o = w["open"].values
    if len(c) < 20: return None
    p = c[-1]
    if p <= 0: return None

    ma5 = np.mean(c[-5:]); ma10 = np.mean(c[-10:]); ma20 = np.mean(c[-20:])
    ma5_prev = np.mean(c[-6:-1])
    ma5_slope = (ma5 - ma5_prev) / ma5_prev * 100 if ma5_prev > 0 else 0

    # RSI-14
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
    macd_cross = dif > dea and (len(dea_arr) >= 2 and (ema12[-2]-ema26[-2]) <= dea_arr[-2])

    # 涨停相关
    cl = 0
    for j in range(-1, max(-8, -len(c)), -1):
        dd = (c[j]/c[j-1]-1)*100 if j-1 >= -len(c) else 0
        if dd > 9.5: cl += 1
        else: break

    yest_limit = (c[-2]/c[-3]-1)*100 > 9.5 if len(c) >= 3 else False
    today_limit = tc > 9.5

    cu = 0
    for j in range(-1, max(-8, -len(c)), -1):
        if c[j] > c[j-1]: cu += 1
        else: break

    to = float(df.iloc[idx].get("turnover_rate", 0))
    if pd.isna(to): to = 0

    body = abs(c[-1] - o[-1])
    upper_shadow = h[-1] - max(c[-1], o[-1])
    bar_range = h[-1] - l[-1]
    upper_ratio = upper_shadow / bar_range if bar_range > 0 else 0

    vol_breakout = vr > 1.8 and tc > 3 and p > h20 * 0.97
    vol_pullback = vr < 0.7 and abs(p - ma10) / ma10 < 0.015 if ma10 > 0 else False

    return dict(
        price=p, ma5=ma5, ma10=ma10, ma20=ma20, ma5_slope=ma5_slope,
        rsi=rsi, bias5=bias5, bias20=bias20, price_pos=pp,
        vol_ratio=vr, vol_shrink=vs, today_chg=tc, chg_3d=c3, chg_5d=c5,
        dif=dif, dea=dea, macd_cross=macd_cross, consec_limit=cl,
        yest_limit=yest_limit, today_limit=today_limit,
        consec_up=cu, turnover=to, h20=h20, l20=l20,
        upper_ratio=upper_ratio,
        vol_breakout=vol_breakout, vol_pullback=vol_pullback,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 统一评分函数（不分市场状态）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def unified_score(ind):
    """
    统一选股评分: 综合多种买入模式
    好市场→更多股票能得高分→自然多买
    差市场→几乎没有股票能达标→自然空仓

    四种模式同时评估，取最高分:
    A. 低吸模式: 超跌+缩量+MACD底部 (适用于任何市场)
    B. 突破模式: 放量突破+均线多头 (适用于强势市场)
    C. 回踩模式: 缩量回踩均线+MACD金叉 (适用于趋势市)
    D. 涨停接力: 昨日涨停+今日稳住 (适用于活跃市)
    """
    scores = []

    # === A. 低吸模式 ===
    sa = 0
    if ind["price_pos"] < 0.12: sa += 25
    elif ind["price_pos"] < 0.20: sa += 20
    elif ind["price_pos"] < 0.30: sa += 15
    elif ind["price_pos"] < 0.40: sa += 8
    elif ind["price_pos"] > 0.60: sa -= 999

    if sa > -100:
        if ind["bias20"] < -8: sa += 18
        elif ind["bias20"] < -5: sa += 13
        elif ind["bias20"] < -3: sa += 8
        elif ind["bias20"] < 0: sa += 3

        if ind["vol_ratio"] < 0.6: sa += 12
        elif ind["vol_ratio"] < 0.8: sa += 8
        elif ind["vol_ratio"] > 2.0: sa -= 8

        if ind["macd_cross"]: sa += 10
        elif ind["dif"] < 0 and ind["dif"] > ind["dea"]: sa += 5

        if ind["rsi"] < 25: sa += 10
        elif ind["rsi"] < 35: sa += 5

        if 0 < ind["today_chg"] < 2: sa += 5
        if ind["today_chg"] < -5: sa -= 10
    scores.append(max(0, sa))

    # === B. 突破模式 ===
    sb = 0
    if ind["vol_breakout"]:
        sb += 20
        if ind["ma5"] > ind["ma10"] > ind["ma20"]: sb += 15
        elif ind["ma5"] > ind["ma10"]: sb += 8
        if 2 < ind["today_chg"] < 7: sb += 10
        if 5 < ind["turnover"] < 15: sb += 8
        if ind["upper_ratio"] < 0.3: sb += 5
        if ind["rsi"] > 75: sb -= 15
        if ind["chg_5d"] > 20: sb -= 15
    scores.append(max(0, sb))

    # === C. 回踩模式 ===
    sc = 0
    if ind["ma5"] > ind["ma10"]:
        if ind["vol_pullback"]: sc += 15
        if ind["macd_cross"]: sc += 12
        elif ind["dif"] > ind["dea"]: sc += 5
        if sc > 0:  # 至少有一个触发
            if ind["ma5"] > ind["ma10"] > ind["ma20"]: sc += 10
            if 0.25 < ind["price_pos"] < 0.55: sc += 8
            if 30 < ind["rsi"] < 50: sc += 8
            if 2 <= ind["consec_up"] <= 4: sc += 5
            if ind["upper_ratio"] < 0.25: sc += 5
    scores.append(max(0, sc))

    # === D. 涨停接力模式 ===
    sd = 0
    if ind["yest_limit"] and ind["consec_limit"] <= 2:
        sd += 20
        if -2 < ind["today_chg"] < 3: sd += 15
        if 5 < ind["turnover"] < 15: sd += 10
        if ind["upper_ratio"] < 0.25: sd += 8
        if ind["ma5"] > ind["ma20"]: sd += 5
        if ind["price_pos"] > 0.8: sd -= 20
        if ind["turnover"] > 20: sd -= 15
    scores.append(max(0, sd))

    # 通用扣分
    best = max(scores)
    if ind["today_limit"]: best -= 20  # 今天已涨停→明天追风险大
    if ind["chg_5d"] > 25: best -= 10  # 短期过热

    return max(0, best)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 自适应状态管理器（不预设市场分类）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveState:
    """
    Agent的"记忆" — 只基于自身交易经验，不对市场做分类

    核心输出:
    1. position_coeff: 0.0~1.0 控制总仓位上限
    2. buy_threshold: 动态买入门槛
    3. cooldown: 连亏后的冷静期天数
    """

    def __init__(self):
        self._trade_results: List[float] = []  # 完成交易的盈亏%
        self._cooldown_days = 0
        self._cooldown_consumed = False
        self._daily_equity: List[float] = []  # 每日净值
        self._max_equity = INITIAL_CAPITAL

    def record_trade(self, pnl_pct: float):
        """记录一笔完成的交易"""
        self._trade_results.append(pnl_pct)
        if len(self._trade_results) > 20:
            self._trade_results.pop(0)
        # 新交易会更新亏损序列，允许后续在新的连亏阶段再次触发冷静期。
        if self.consecutive_losses < 4:
            self._cooldown_consumed = False

    def update_equity(self, equity: float):
        """每日更新净值"""
        self._daily_equity.append(equity)
        self._max_equity = max(self._max_equity, equity)

    @property
    def recent_win_rate(self) -> float:
        """最近N笔交易的胜率"""
        recent = self._trade_results[-7:]  # 最近7笔
        if len(recent) < 2: return 0.5  # 默认中性
        return sum(1 for r in recent if r > 0) / len(recent)

    @property
    def recent_avg_pnl(self) -> float:
        """最近N笔的平均盈亏"""
        recent = self._trade_results[-7:]
        if not recent: return 0
        return np.mean(recent)

    @property
    def consecutive_losses(self) -> int:
        """当前连亏次数"""
        count = 0
        for r in reversed(self._trade_results):
            if r < 0: count += 1
            else: break
        return count

    @property
    def drawdown_pct(self) -> float:
        """当前从最高点回撤百分比"""
        if not self._daily_equity: return 0
        current = self._daily_equity[-1]
        return (self._max_equity - current) / self._max_equity * 100

    @property
    def position_coeff(self) -> float:
        """
        仓位系数: 0.0 ~ 1.0
        基于近期胜率 + 连亏惩罚 + 回撤惩罚
        """
        # 基础: 由近期胜率决定
        wr = self.recent_win_rate
        if wr >= 0.7: base = 0.80
        elif wr >= 0.5: base = 0.60
        elif wr >= 0.3: base = 0.40
        else: base = 0.20

        # 连亏惩罚
        cl = self.consecutive_losses
        if cl >= 4: base *= 0.3
        elif cl >= 3: base *= 0.5
        elif cl >= 2: base *= 0.7

        # 回撤惩罚: 回撤超过5%开始减仓
        dd = self.drawdown_pct
        if dd > 10: base *= 0.3
        elif dd > 7: base *= 0.5
        elif dd > 5: base *= 0.7

        # 近期平均盈利加成
        avg = self.recent_avg_pnl
        if avg > 3: base = min(base * 1.2, 1.0)
        elif avg > 1: base = min(base * 1.1, 1.0)

        return max(0.10, min(1.0, base))  # 最低10%，不完全空仓

    @property
    def buy_threshold(self) -> int:
        """
        动态买入门槛: 分数需 >= 此值才买入
        赚钱时降低门槛(更敢买), 亏钱时提高门槛(更谨慎)
        """
        base = 40

        # 连亏提高门槛
        cl = self.consecutive_losses
        if cl >= 3: base += 15
        elif cl >= 2: base += 8
        elif cl >= 1: base += 4

        # 回撤提高门槛
        dd = self.drawdown_pct
        if dd > 8: base += 12
        elif dd > 5: base += 6

        # 近期赚钱降低门槛
        if self.recent_avg_pnl > 2: base -= 5
        elif self.recent_avg_pnl > 0: base -= 2

        return max(30, min(65, base))

    @property
    def in_cooldown(self) -> bool:
        """连亏4+笔后进入1天冷静期"""
        return self._cooldown_days > 0

    def tick_cooldown(self):
        """每天减少冷静期"""
        if self._cooldown_days > 0:
            self._cooldown_days -= 1

    def trigger_cooldown(self):
        """触发冷静期"""
        self._cooldown_days = 1
        self._cooldown_consumed = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 止盈止损参数（固定, 不依赖市场判断）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK_PARAMS = {
    "stop_loss": -3.0,        # 止损
    "tp_half": 5.0,           # 半仓止盈
    "tp_full": 12.0,          # 全仓止盈
    "trail_trigger": 5.5,     # 移动止盈触发
    "trail_dd": 1.8,          # 移动止盈回撤幅度
    "hold_max": 5,            # 最大持有天数
    "emergency_stop": -5.0,   # 盘中极端止损
    "max_positions": 3,       # 最大持仓数
    "single_pct": 0.35,       # 单只最大仓位
    "price_range": (3, 50),   # 价格范围
    "min_turnover": 3.4,      # 最低换手率
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 盲态Agent回测引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BlindAgentBacktest:
    def __init__(self, all_data):
        self.all_data = all_data
        self.cash = INITIAL_CAPITAL
        self.holdings: Dict[str, Holding] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[Snapshot] = []
        self.prev_total = INITIAL_CAPITAL
        self._pending_buys: List[dict] = []
        self._pending_sells: List[tuple] = []
        self._state = AdaptiveState()

        dates_set = set()
        for df in all_data.values():
            dates_set.update(df["date"].tolist())
        self.trading_days = sorted(dates_set)
        self.day_idx_map = {d: i for i, d in enumerate(self.trading_days)}

    def run(self, start_date, end_date):
        days = [d for d in self.trading_days if start_date <= d <= end_date]
        logger.info(f"Blind Agent: {start_date} ~ {end_date} ({len(days)}d)")
        logger.info(f"Initial: Y{INITIAL_CAPITAL:,.0f}")

        for day_num, date in enumerate(days):
            day_idx = self.day_idx_map.get(date, day_num)

            # 冷静期倒计时
            self._state.tick_cooldown()

            # 1. 开盘: 执行昨晚挂单
            self._exec_pending_sells(date, day_idx)
            self._exec_pending_buys(date, day_idx)

            # 2. 盘中: 极端止损
            self._emergency_stop(date, day_idx)

            # 3. 收盘: 更新价格
            self._update_close(date)

            # 4. 盘后: 复盘持仓 → 生成卖出信号
            self._review_sells(date, day_idx)

            # 5. 盘后: 选股 → 生成买入信号
            self._scan_buys(date, day_idx)

            # 快照
            mv = sum(h.market_value for h in self.holdings.values())
            total = self.cash + mv
            ret = (total - self.prev_total) / self.prev_total * 100 if self.prev_total > 0 else 0
            self.prev_total = total
            self._state.update_equity(total)

            self.snapshots.append(Snapshot(
                date, total, self.cash, mv, len(self.holdings), ret,
                self._state.position_coeff, self._state.buy_threshold,
            ))

            if (day_num + 1) % 10 == 0:
                r = (total - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                pc = self._state.position_coeff
                bt = self._state.buy_threshold
                wr = self._state.recent_win_rate
                cl = self._state.consecutive_losses
                logger.info(
                    f"  [{date}] Day{day_num+1:>2} | Y{total:>9,.0f} | {r:>+6.2f}% | "
                    f"pos={pc:.0%} thr={bt} wr={wr:.0%} cl={cl} hold={len(self.holdings)}"
                )

        self._print_report()

    def _get_row(self, code, date):
        df = self.all_data.get(code)
        if df is None: return None
        rows = df[df["date"] == date]
        return rows.iloc[0] if not rows.empty else None

    # ── 执行层 ──

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
                self._pending_sells.append((code, shares, reason + "(LD)"))
                continue
            self._sell(code, min(shares, h.shares), op, date, reason)

    def _exec_pending_buys(self, date, day_idx):
        buys = self._pending_buys[:]; self._pending_buys = []
        pc = self._state.position_coeff

        for pb in buys:
            code = pb["code"]
            if code in self.holdings: continue
            if len(self.holdings) >= RISK_PARAMS["max_positions"]: continue
            row = self._get_row(code, date)
            if row is None: continue
            op = float(row["open"])
            if op <= 0: continue
            sc = pb.get("signal_close", 0)
            if sc > 0 and op >= sc * 1.098: continue  # 一字板
            if sc > 0 and op > sc * 1.04: continue      # 高开4%不追

            cost = op * (1 + SLIPPAGE_PCT)
            total_asset = self.cash + sum(h.market_value for h in self.holdings.values())
            mv_now = sum(h.market_value for h in self.holdings.values())
            # 总仓位上限 = 仓位系数 * 总资产
            max_mv = total_asset * pc
            remaining = max_mv - mv_now
            if remaining <= 0: continue

            buy_amount = min(
                self.cash * 0.95,
                total_asset * RISK_PARAMS["single_pct"],
                remaining,
            )
            shares = int(buy_amount / cost / 100) * 100
            if shares < 100: continue
            amount = cost * shares
            comm = max(amount * COMMISSION_RATE, 5)
            if amount + comm > self.cash: continue

            self.cash -= (amount + comm)
            self.holdings[code] = Holding(
                code=code, name=pb.get("name", code), shares=shares,
                cost_price=cost, buy_date=date, buy_day_idx=day_idx,
                current_price=op, peak_price=op,
            )
            self.trades.append(Trade(
                date, code, pb.get("name", code), "buy", shares, cost, amount, comm,
                f"s={pb.get('score',0)}|pc={pc:.0%}",
            ))

    def _emergency_stop(self, date, day_idx):
        threshold = RISK_PARAMS["emergency_stop"]
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx >= day_idx: continue
            row = self._get_row(code, date)
            if row is None: continue
            low = float(row["low"])
            pnl = (low - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            if pnl <= threshold:
                trigger = h.cost_price * (1 + threshold / 100)
                self._sell(code, h.shares, trigger, date, f"ESTOP{pnl:.1f}%")

    def _update_close(self, date):
        for code, h in self.holdings.items():
            row = self._get_row(code, date)
            if row is None: continue
            h.current_price = float(row["close"])
            h.peak_price = max(h.peak_price, float(row["high"]))

    # ── 决策层 ──

    def _review_sells(self, date, day_idx):
        rp = RISK_PARAMS
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx == day_idx: continue  # 今天买的不能卖
            pnl = h.pnl_pct
            hold = day_idx - h.buy_day_idx
            pk = (h.peak_price - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            dd = (h.peak_price - h.current_price) / h.peak_price * 100 if h.peak_price > 0 else 0

            # 止损
            if pnl <= rp["stop_loss"]:
                self._pending_sells.append((code, h.shares, f"SL{pnl:.1f}%"))
                continue

            # 移动止盈
            if pk >= rp["trail_trigger"] and dd >= rp["trail_dd"]:
                self._pending_sells.append((code, h.shares, f"TRAIL pk{pk:.1f}% dd{dd:.1f}%"))
                continue

            # 半仓止盈
            if not h.partial_sold and pnl >= rp["tp_half"]:
                half = max(100, (h.shares // 200) * 100)
                if half < h.shares:
                    self._pending_sells.append((code, half, f"TP_HALF{pnl:.1f}%"))
                    h.partial_sold = True
                    continue

            # 全止盈
            if pnl >= rp["tp_full"]:
                self._pending_sells.append((code, h.shares, f"TP_FULL{pnl:.1f}%"))
                continue

            # 超期
            max_hold = rp["hold_max"]
            if pnl > 2: max_hold += 2  # 盈利多给2天
            if hold >= max_hold and pnl < 1:
                self._pending_sells.append((code, h.shares, f"EXPIRE{hold}d"))

    def _scan_buys(self, date, day_idx):
        if len(self.holdings) >= RISK_PARAMS["max_positions"]: return
        if self._pending_buys: return

        # 冷静期不买
        if self._state.in_cooldown:
            return

        # 连亏4+触发冷静
        if self._state.consecutive_losses >= 4 and not self._state._cooldown_consumed:
            self._state.trigger_cooldown()
            return

        threshold = self._state.buy_threshold
        pmin, pmax = RISK_PARAMS["price_range"]
        min_to = RISK_PARAMS["min_turnover"]

        cands = []
        for code, df in self.all_data.items():
            if code in self.holdings: continue
            ri = df.index[df["date"] == date]
            if len(ri) == 0: continue
            idx = ri[0]; row = df.iloc[idx]
            price = float(row["close"])
            if price < pmin or price > pmax: continue
            to = float(row.get("turnover_rate", 0) or 0)
            if to < min_to: continue
            ind = calc_indicators(df, idx)
            if ind is None: continue
            s = unified_score(ind)
            if s < threshold: continue
            cands.append({
                "code": code,
                "name": str(row.get("stock_name", code)),
                "signal_close": price,
                "score": s,
            })

        cands.sort(key=lambda x: x["score"], reverse=True)
        slots = RISK_PARAMS["max_positions"] - len(self.holdings)
        for c in cands[:max(1, slots)]:
            self._pending_buys.append(c)

    # ── 交易执行 ──

    def _sell(self, code, shares, price, date, reason):
        h = self.holdings.get(code)
        if not h: return
        sp = price * (1 - SLIPPAGE_PCT)
        amt = sp * shares
        comm = max(amt * COMMISSION_RATE, 5)
        tax = amt * STAMP_TAX_RATE
        self.cash += (amt - comm - tax)
        self.trades.append(Trade(
            date, code, h.name, "sell", shares, sp, amt, comm + tax, reason,
        ))

        pnl = (sp - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
        if shares >= h.shares:
            self._state.record_trade(pnl)
            del self.holdings[code]
        else:
            h.shares -= shares

    # ── 报告 ──

    def _print_report(self):
        if not self.snapshots: return

        final = self.snapshots[-1].total_asset
        ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        peak, max_dd = INITIAL_CAPITAL, 0
        for s in self.snapshots:
            peak = max(peak, s.total_asset)
            dd = (peak - s.total_asset) / peak * 100
            max_dd = max(max_dd, dd)

        # 胜率统计
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

        monthly = defaultdict(list)
        for s in self.snapshots: monthly[s.date[:7]].append(s.daily_return_pct)

        # 仓位系数变化
        pc_series = [s.position_coeff for s in self.snapshots]
        bt_series = [s.buy_threshold for s in self.snapshots]

        print("\n" + "=" * 70)
        print("  Blind Agent Backtest Report 2026 YTD")
        print("  (No regime labels - pure self-adaptive)")
        print("=" * 70)
        print(f"\n  Initial:    Y{INITIAL_CAPITAL:>12,}")
        print(f"  Final:      Y{final:>12,.0f}")
        print(f"  Return:     {ret:>+11.2f}%")
        print(f"  Annualized: {ann:>+11.2f}%")
        print(f"  Max DD:     {max_dd:>11.2f}%")
        print(f"  Sharpe:     {sharpe:>11.2f}")
        print(f"\n  Trades: {len(self.trades)}  Win: {wins}  Loss: {losses}  "
              f"WinRate: {wr:.1f}%  AvgW: {aw:+.2f}%  AvgL: {al:+.2f}%  PF: {pf:.2f}")

        print(f"\n  [Agent Self-Adaptation Trace]")
        print(f"    Position coeff: {pc_series[0]:.0%} -> {pc_series[-1]:.0%}  "
              f"(min={min(pc_series):.0%} max={max(pc_series):.0%})")
        print(f"    Buy threshold:  {bt_series[0]} -> {bt_series[-1]}  "
              f"(min={min(bt_series)} max={max(bt_series)})")
        print(f"    Final win_rate: {self._state.recent_win_rate:.0%}")
        print(f"    Final consec_losses: {self._state.consecutive_losses}")
        print(f"    Final drawdown: {self._state.drawdown_pct:.2f}%")

        print(f"\n  [Monthly Returns]")
        cum = 0
        for m in sorted(monthly.keys()):
            mr = sum(monthly[m]); cum += mr
            # 月均仓位系数
            month_pcs = [s.position_coeff for s in self.snapshots if s.date.startswith(m)]
            avg_pc = np.mean(month_pcs) if month_pcs else 0
            mark = "+" if mr >= 0 else "-"
            print(f"    {m}  {mr:>+7.2f}%  cum{cum:>+7.2f}%  [{mark}]  avg_pos={avg_pc:.0%}")

        print("=" * 70)

        # 保存
        out_dir = Path("data/backtest_blind_agent")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "return": round(ret, 2), "annual": round(ann, 2),
            "max_dd": round(max_dd, 2), "sharpe": round(sharpe, 2),
            "win_rate": round(wr, 1), "profit_factor": round(pf, 2),
            "trades": len(self.trades), "final": round(final, 0),
            "monthly": {m: round(sum(r), 2) for m, r in sorted(monthly.items())},
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        trade_rows = [{"date": t.date, "code": t.code, "name": t.name,
                        "dir": t.direction, "shares": t.shares, "price": round(t.price, 3),
                        "amount": round(t.amount, 0), "reason": t.reason}
                       for t in self.trades]
        pd.DataFrame(trade_rows).to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")

        snap_rows = [{"date": s.date, "total": round(s.total_asset, 0), "cash": round(s.cash, 0),
                       "mv": round(s.market_value, 0), "hold": s.holdings_count,
                       "ret": round(s.daily_return_pct, 4),
                       "pos_coeff": round(s.position_coeff, 3),
                       "buy_thr": s.buy_threshold}
                      for s in self.snapshots]
        pd.DataFrame(snap_rows).to_csv(out_dir / "daily.csv", index=False, encoding="utf-8-sig")

        logger.info(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    from backtest_2025 import fetch_all_a_share_daily

    logger.info("Loading data...")
    all_data = fetch_all_a_share_daily("20251001", "20260410",
                                        cache_name="backtest_cache_2026ytd.pkl")
    logger.info(f"Data loaded: {len(all_data)} stocks")

    engine = BlindAgentBacktest(all_data)
    engine.run("2026-01-05", "2026-04-10")
