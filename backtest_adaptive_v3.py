"""
华尔街之狼 自适应策略回测 V3 — 极致优化版
==========================================
无未来函数 | T日盘后决策 → T+1开盘执行

V3核心改进（基于V1/V2教训）:
  1. 结合V1的保守（低回撤）和V2的激进（高收益）
  2. 更严格的入场: 分数+量价确认双重过滤
  3. 更快的止损: -3%就跑, 不等到-7%才emergency
  4. BULL期间ALL-IN单只（集中火力）
  5. CRASH期间真正空仓（V2在CRASH还亏钱）
  6. 加入"强势板块轮动": 不是随便选股, 而是从近5日涨幅最大的板块里选
  7. 涨停接力优化: 只做首板次日+2连板加速, 不做高位板
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
    partial_sold: bool = False
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

    cl = 0
    for j in range(-1, max(-8, -len(c)), -1):
        dd = (c[j]/c[j-1]-1)*100 if j-1 >= -len(c) else 0
        if dd > 9.5: cl += 1
        else: break

    hrl = any((c[j]/c[j-1]-1)*100 > 9.5 for j in range(-2, max(-8,-len(c)),-1) if j-1 >= -len(c))
    today_limit = tc > 9.5
    yest_limit = (c[-2]/c[-3]-1)*100 > 9.5 if len(c) >= 3 else False

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

    # V3新增: 近5日涨幅排名辅助
    chg_5d_rank_score = 0  # 占位, 由外部计算

    return dict(price=p, ma5=ma5, ma10=ma10, ma20=ma20, ma5_slope=ma5_slope,
                rsi=rsi, bias5=bias5, bias20=bias20, price_pos=pp,
                vol_ratio=vr, vol_shrink=vs, today_chg=tc, chg_3d=c3, chg_5d=c5,
                dif=dif, dea=dea, macd_cross=mc, consec_limit=cl,
                has_recent_limit=hrl, consec_up=cu, turnover=to, h20=h20, l20=l20,
                today_limit=today_limit, yest_limit=yest_limit,
                upper_ratio=upper_ratio,
                vol_breakout=vol_breakout, vol_pullback=vol_pullback)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 市场环境判断
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MarketRegimeDetector:
    def __init__(self):
        self._prev_regime = "SIDE"
        self._pending_regime = None
        self._pending_count = 0
        self._breadth_history = []

    def detect(self, all_data: dict, date: str) -> dict:
        score = 0
        above, total, limit_down_count, limit_up_count = 0, 0, 0, 0

        for code, df in all_data.items():
            idx_arr = df.index[df["date"] == date]
            if len(idx_arr) == 0: continue
            i = idx_arr[0]; total += 1
            cl = float(df["close"].iloc[i])
            if i >= 20:
                m20 = df["close"].iloc[i-19:i+1].mean()
                if cl > m20: above += 1
            if i >= 1:
                chg = (cl / float(df["close"].iloc[i-1]) - 1) * 100
                if chg < -9.5: limit_down_count += 1
                if chg > 9.5: limit_up_count += 1

        breadth = above / total if total > 0 else 0.5
        self._breadth_history.append(breadth)

        if breadth > 0.65: score += 40
        elif breadth > 0.55: score += 25
        elif breadth > 0.45: score += 10
        elif breadth > 0.35: score -= 15
        else: score -= 35

        if len(self._breadth_history) >= 3:
            b3 = self._breadth_history[-3:]
            if b3[-1] > b3[0] + 0.05: score += 10
            elif b3[-1] < b3[0] - 0.05: score -= 10

        ld_pct = limit_down_count / total * 100 if total > 0 else 0
        if ld_pct > 5: score -= 30
        elif ld_pct > 2: score -= 15

        lu_pct = limit_up_count / total * 100 if total > 0 else 0
        if lu_pct > 3: score += 15
        elif lu_pct > 1.5: score += 8

        up_count = sum(1 for code, df in all_data.items()
                       if not df[df["date"]==date].empty and
                       float(df[df["date"]==date].iloc[0].get("change_pct", 0) or 0) > 0)
        up_ratio = up_count / total if total > 0 else 0.5
        if up_ratio > 0.6: score += 15
        elif up_ratio < 0.35: score -= 15

        if score >= 45: raw = "BULL"
        elif score >= 10: raw = "SIDE"
        elif score >= -15: raw = "BEAR"
        else: raw = "CRASH"

        # 牛市1天确认, 其余2天
        confirm_needed = 1 if raw == "BULL" else 2
        if raw != self._prev_regime:
            if raw == self._pending_regime:
                self._pending_count += 1
            else:
                self._pending_regime = raw
                self._pending_count = 1
            if self._pending_count >= confirm_needed:
                self._prev_regime = raw
                self._pending_regime = None
                self._pending_count = 0
        else:
            self._pending_regime = None
            self._pending_count = 0

        return {"regime": self._prev_regime, "score": score, "breadth": breadth,
                "limit_up": limit_up_count, "limit_down": limit_down_count}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V3 策略评分
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_bull_v3(ind):
    """牛市V3: 更严格筛选, 只做最确定的机会

    三种模式:
    A. 首板次日接力 (昨日首次涨停, 今日稳住)
    B. 放量突破 (突破20日新高, 放量)
    C. MACD金叉+回踩 (安全但稳健)
    """
    s = 0

    # A. 首板次日接力
    if ind["yest_limit"] and ind["consec_limit"] <= 2:
        s += 20
        if -2 < ind["today_chg"] < 3: s += 15  # 平稳消化
        if 5 < ind["turnover"] < 15: s += 10    # 换手适中
        if ind["upper_ratio"] < 0.25: s += 8    # 无上影
        if ind["ma5"] > ind["ma20"]: s += 5     # 趋势对
        # 扣分: 高位涨停(price_pos>0.8)不追
        if ind["price_pos"] > 0.8: s -= 20
        if ind["turnover"] > 20: s -= 15  # 换手太高=出货

    # B. 放量突破
    elif ind["vol_breakout"]:
        s += 15
        if ind["ma5"] > ind["ma10"] > ind["ma20"]: s += 15
        if 2 < ind["today_chg"] < 7: s += 10
        if 5 < ind["turnover"] < 15: s += 8
        if ind["upper_ratio"] < 0.3: s += 5

    # C. MACD金叉回踩
    elif ind["macd_cross"] or ind["vol_pullback"]:
        if ind["ma5"] > ind["ma10"]: s += 8
        else: return 0
        if ind["macd_cross"]: s += 12
        if ind["vol_pullback"]: s += 12
        if 0.25 < ind["price_pos"] < 0.55: s += 8
        if 30 < ind["rsi"] < 50: s += 8
    else:
        return 0

    # 通用扣分
    if ind["today_limit"]: s -= 25  # 今天已涨停, 明天追风险大
    if ind["rsi"] > 75: s -= 15
    if ind["chg_5d"] > 20: s -= 15  # 5日涨20%+, 短期过热
    return max(0, s)


def score_side_v3(ind):
    """震荡V3: 网格低吸（V1验证过的最优策略, 参数微调）"""
    s = 0
    if ind["price_pos"] < 0.12: s += 28
    elif ind["price_pos"] < 0.22: s += 22
    elif ind["price_pos"] < 0.32: s += 15
    elif ind["price_pos"] < 0.42: s += 8
    elif ind["price_pos"] > 0.55: return 0

    if ind["bias20"] < -8: s += 18
    elif ind["bias20"] < -5: s += 13
    elif ind["bias20"] < -3: s += 8
    elif ind["bias20"] < 0: s += 3

    if ind["vol_ratio"] < 0.6: s += 12
    elif ind["vol_ratio"] < 0.8: s += 8
    elif ind["vol_ratio"] > 2.0: s -= 8

    if ind["macd_cross"]: s += 10
    elif ind["dif"] < 0 and ind["dif"] > ind["dea"]: s += 5

    if 0 < ind["today_chg"] < 2: s += 5
    if ind["rsi"] < 28: s += 10
    elif ind["rsi"] < 35: s += 5

    if ind["today_chg"] < -5: s -= 10
    return max(0, s)


# V3参数表
REGIME_CONFIG_V3 = {
    "BULL": {
        "score_fn": score_bull_v3, "min_score": 40,
        "max_positions": 1, "single_pct": 0.90, "total_pct": 0.90,  # V3: ALL-IN 1只
        "stop_loss": -3.0,          # V3: 更快止损
        "tp_half": 5.0,
        "tp_full": 12.0,
        "trail_trigger": 6.0, "trail_dd": 2.5,
        "hold_max": 5, "price_range": (3, 50),
    },
    "SIDE": {
        "score_fn": score_side_v3, "min_score": 42,
        "max_positions": 3, "single_pct": 0.25, "total_pct": 0.55,
        "stop_loss": -3.0,
        "tp_half": 4.0,
        "tp_full": 8.0,
        "trail_trigger": 3.5, "trail_dd": 1.5,
        "hold_max": 5, "price_range": (3, 30),
    },
    "BEAR": {
        "score_fn": None, "min_score": 999,  # V3: BEAR不做, 空仓等待
        "max_positions": 0, "single_pct": 0, "total_pct": 0,
        "stop_loss": -3.0, "tp_half": 3.0, "tp_full": 5.0,
        "trail_trigger": 2.0, "trail_dd": 1.5,
        "hold_max": 2, "price_range": (0, 0),
    },
    "CRASH": {
        "score_fn": None, "min_score": 999,  # V3: CRASH不做
        "max_positions": 0, "single_pct": 0, "total_pct": 0,
        "stop_loss": -2.0, "tp_half": 2.0, "tp_full": 3.0,
        "trail_trigger": 1.5, "trail_dd": 1.0,
        "hold_max": 1, "price_range": (0, 0),
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V3 回测引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveBacktestV3:
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
        self._recent_pnl = []  # 最近交易盈亏率

        dates_set = set()
        for df in all_data.values():
            dates_set.update(df["date"].tolist())
        self.trading_days = sorted(dates_set)
        self.day_idx_map = {d: i for i, d in enumerate(self.trading_days)}

    def _confidence_mult(self):
        """根据最近5笔交易表现动态调仓"""
        if len(self._recent_pnl) < 3: return 1.0
        recent = self._recent_pnl[-5:]
        wins = sum(1 for p in recent if p > 0)
        if wins >= 4: return 1.2    # 连赢, 加仓
        if wins <= 1: return 0.7    # 连亏, 减仓
        return 1.0

    def run(self, start_date, end_date):
        days = [d for d in self.trading_days if start_date <= d <= end_date]
        logger.info(f"V3 Adaptive: {start_date} ~ {end_date} ({len(days)}d)")
        logger.info(f"Initial: Y{INITIAL_CAPITAL:,.0f}")

        for day_num, date in enumerate(days):
            day_idx = self.day_idx_map.get(date, day_num)

            self._exec_pending_sells(date, day_idx)
            self._exec_pending_buys(date, day_idx)
            self._emergency_stop(date, day_idx)
            self._update_close(date)

            regime_info = self._regime_detector.detect(self.all_data, date)
            self._current_regime = regime_info["regime"]
            self._regime_history.append({"date": date, **regime_info})

            cfg = REGIME_CONFIG_V3[self._current_regime]

            # BEAR/CRASH: 清仓
            if self._current_regime in ("BEAR", "CRASH"):
                for code, h in list(self.holdings.items()):
                    if h.buy_day_idx < day_idx:
                        self._pending_sells.append((code, h.shares, f"{self._current_regime}_CLEAR"))

            self._review_sells_v3(date, day_idx, cfg)
            if cfg["score_fn"] is not None:
                self._scan_buys(date, day_idx, cfg)

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
                self._pending_sells.append((code, shares, reason + "(LD)"))
                continue
            self._sell(code, min(shares, h.shares), op, date, reason)

    def _exec_pending_buys(self, date, day_idx):
        buys = self._pending_buys[:]; self._pending_buys = []
        cfg = REGIME_CONFIG_V3[self._current_regime]
        cm = self._confidence_mult()

        for pb in buys:
            code = pb["code"]
            if code in self.holdings or len(self.holdings) >= cfg["max_positions"]: continue
            row = self._get_row(code, date)
            if row is None: continue
            op = float(row["open"])
            if op <= 0: continue
            sc = pb.get("signal_close", 0)
            if sc > 0 and op >= sc * 1.098: continue
            if sc > 0 and op > sc * 1.04: continue

            cost = op * (1 + SLIPPAGE_PCT)
            total_asset = self.cash + sum(h.market_value for h in self.holdings.values())
            mv_now = sum(h.market_value for h in self.holdings.values())
            max_mv = total_asset * cfg["total_pct"] * cm
            remaining = max_mv - mv_now
            if remaining <= 0: continue

            buy_amount = min(self.cash * 0.95,
                             total_asset * cfg["single_pct"] * cm,
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
            if pnl <= -5.0:
                trigger = h.cost_price * 0.95
                self._sell(code, h.shares, trigger, date, f"E-STOP{pnl:.1f}%")

    def _update_close(self, date):
        for code, h in self.holdings.items():
            row = self._get_row(code, date)
            if row is None: continue
            h.current_price = float(row["close"])
            h.peak_price = max(h.peak_price, float(row["high"]))

    def _review_sells_v3(self, date, day_idx, cfg):
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx == day_idx: continue
            pnl = h.pnl_pct
            hold = day_idx - h.buy_day_idx
            pk = (h.peak_price - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            dd = (h.peak_price - h.current_price) / h.peak_price * 100 if h.peak_price > 0 else 0

            if pnl <= cfg["stop_loss"]:
                self._pending_sells.append((code, h.shares, f"SL{pnl:.1f}%"))
                continue

            if pk >= cfg["trail_trigger"] and dd >= cfg["trail_dd"]:
                self._pending_sells.append((code, h.shares, f"TRAIL pk{pk:.1f}%dd{dd:.1f}%"))
                continue

            if not h.partial_sold and pnl >= cfg["tp_half"]:
                half = max(100, (h.shares // 200) * 100)
                if half < h.shares:
                    self._pending_sells.append((code, half, f"TP_HALF{pnl:.1f}%"))
                    h.partial_sold = True
                    continue

            if pnl >= cfg["tp_full"]:
                self._pending_sells.append((code, h.shares, f"TP_FULL{pnl:.1f}%"))
                continue

            max_hold = cfg["hold_max"]
            if pnl > 2: max_hold += 2
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
            if to < 2: continue
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

        pnl = (sp - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
        if shares >= h.shares:
            self._recent_pnl.append(pnl)
            if len(self._recent_pnl) > 10: self._recent_pnl.pop(0)
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
        print("  V3 Adaptive Strategy Backtest Report 2026 YTD")
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
            month_regimes = [s.regime for s in self.snapshots if s.date.startswith(m)]
            dominant = max(set(month_regimes), key=month_regimes.count) if month_regimes else "?"
            mark = "+" if mr >= 0 else "-"
            print(f"    {m}  {mr:>+7.2f}%  cum{cum:>+7.2f}%  [{mark}]  regime={dominant}")

        print("=" * 70)

        out_dir = Path("data/backtest_adaptive_v3")
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

    engine = AdaptiveBacktestV3(all_data)
    engine.run("2026-01-05", "2026-04-10")
