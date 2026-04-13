"""
华尔街之狼 事件驱动+盲态Agent回测 V5b
==========================================
无未来函数 | 无预设标签 | 事件冲击检测 + 板块冲击检测 + 自适应仓位

V5b增强:
  在V5基础上新增板块级冲击检测:
    1. 加载stock_industry_map.json获取真实行业分类
    2. 每日计算每个板块的平均涨跌幅、跌>3%家数占比
    3. 当板块平均跌幅>3%且跌>3%占比>50%时，判定为板块冲击
    4. 板块冲击后1-3天(RECOVERY)，优先从该板块超跌个股中寻找反弹机会
    5. 板块冲击有3天冷却期，避免无限延长RECOVERY窗口

信号层次:
  L1. 常规信号: 个股技术面评分(低吸/突破/回踩/接力)
  L2. 事件信号: 检测到市场冲击 → 冲击后超跌个股加分
  L2b.板块事件: 检测到板块冲击 → 板块内超跌个股额外加分
  L3. 恐慌信号: 全市场恐慌(跌停家数暴增) → 全面防守

仓位管理: 完全由近期交易结果+当前回撤决定(盲态自适应)
"""
import json, logging, os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 200_000
COMMISSION_RATE = 0.00025
STAMP_TAX_RATE = 0.0005
SLIPPAGE_PCT = 0.002

# 板块冲击参数
SECTOR_SHOCK_AVG_THRESHOLD = -3.0    # 板块平均跌幅阈值
SECTOR_SHOCK_DOWN_RATIO = 0.50       # 板块内跌>3%占比阈值
SECTOR_RECOVERY_DAYS = 3             # 板块冲击后恢复窗口(天)
SECTOR_MIN_STOCKS = 5                # 板块最少股票数(不足的合并)


@dataclass
class Holding:
    code: str; name: str; shares: int; cost_price: float
    buy_date: str; buy_day_idx: int; signal_type: str
    sector: str = ""
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
    sector: str = ""


@dataclass
class Snapshot:
    date: str; total_asset: float; cash: float; market_value: float
    holdings_count: int; daily_return_pct: float
    position_coeff: float; event_level: str


@dataclass
class SectorShock:
    """记录一次板块冲击"""
    sector: str
    date: str
    day_idx: int
    avg_change: float
    down_count: int
    total_count: int
    hit_codes: List[str]  # 该板块中跌幅>3%的股票


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 事件冲击检测器（从价格推断新闻事件 + 板块冲击）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EventDetector:
    """
    从价格数据中推断"是否发生了重大事件"

    检测维度:
    1. 市场恐慌度: 跌停家数、跌幅>5%家数占比
    2. 同步暴跌: 是否有大量股票同一天跌>3%
    3. 异常放量: 全市场成交量是否异常放大
    4. 事件后恢复: 暴跌后是否缩量企稳(反弹信号)
    5. [V5b] 板块冲击: 同行业股票集体暴跌
    """

    def __init__(self):
        self._history: List[dict] = []
        self._recent_shocks: List[dict] = []  # 近期市场级冲击记录
        self._sector_shocks: List[SectorShock] = []  # 近期板块冲击记录

        # 加载行业映射
        map_path = Path(__file__).parent / "data" / "stock_industry_map.json"
        with open(map_path, "r", encoding="utf-8") as f:
            self._raw_industry_map: Dict[str, str] = json.load(f)

        # 构建: code -> sector, sector -> [codes]
        self._industry_map: Dict[str, str] = {}
        self._sector_codes: Dict[str, List[str]] = defaultdict(list)
        self._build_sector_groups()

        logger.info(f"[EventDetector V5b] Loaded {len(self._raw_industry_map)} stock-industry mappings, "
                     f"grouped into {len(self._sector_codes)} sectors")

    def _build_sector_groups(self):
        """按行业分组，小行业(<5只股票)合并到更宽泛的类别"""
        raw_groups = defaultdict(list)
        for code, industry in self._raw_industry_map.items():
            raw_groups[industry].append(code)

        # 合并策略: 含Ⅱ后缀的细分行业合并到去掉Ⅱ的大类
        # 不足5只股票的行业归入"其他"
        merged = defaultdict(list)
        for industry, codes in raw_groups.items():
            if len(codes) >= SECTOR_MIN_STOCKS:
                merged[industry].extend(codes)
            else:
                # 尝试去掉Ⅱ后缀合并
                broader = industry.replace("Ⅱ", "").strip()
                if broader != industry and broader in raw_groups:
                    merged[broader].extend(codes)
                else:
                    merged["其他"].extend(codes)

        # 二次检查: 合并后仍不足5只的归入"其他"
        final = defaultdict(list)
        for sector, codes in merged.items():
            if sector == "其他" or len(codes) >= SECTOR_MIN_STOCKS:
                final[sector].extend(codes)
            else:
                final["其他"].extend(codes)

        for sector, codes in final.items():
            for code in codes:
                self._industry_map[code] = sector
            self._sector_codes[sector] = codes

    def get_sector(self, code: str) -> str:
        """获取股票所属板块"""
        return self._industry_map.get(code, "未知")

    def analyze(self, all_data: dict, date: str, day_idx: int = 0) -> dict:
        """分析当日市场状态，返回事件信号"""
        total = 0
        down_3pct = 0      # 跌幅>3%家数
        down_5pct = 0      # 跌幅>5%家数
        limit_down = 0     # 跌停家数
        limit_up = 0       # 涨停家数
        up_count = 0       # 上涨家数
        total_vol_ratio = 0  # 全市场量比加总
        vol_counted = 0

        # 统计每只股票的跌幅 + 按板块聚合
        stock_changes = {}
        sector_changes = defaultdict(list)  # sector -> [change_pct, ...]

        for code, df in all_data.items():
            idx_arr = df.index[df["date"] == date]
            if len(idx_arr) == 0: continue
            i = idx_arr[0]; total += 1

            chg = float(df.iloc[i].get("change_pct", 0) or 0)
            stock_changes[code] = chg

            # 按板块聚合
            sector = self.get_sector(code)
            if sector != "未知":
                sector_changes[sector].append((code, chg))

            if chg < -3: down_3pct += 1
            if chg < -5: down_5pct += 1
            if chg < -9.5: limit_down += 1
            if chg > 9.5: limit_up += 1
            if chg > 0: up_count += 1

            # 量比
            if i >= 20:
                v20 = df["volume"].iloc[i-19:i+1].mean()
                if v20 > 0:
                    vr = float(df["volume"].iloc[i]) / v20
                    total_vol_ratio += vr
                    vol_counted += 1

        if total == 0:
            return {"level": "NORMAL", "panic": 0, "shock": False,
                    "sector_shocks": [], "recovering_sectors": []}

        down_3_pct = down_3pct / total * 100
        down_5_pct = down_5pct / total * 100
        limit_down_pct = limit_down / total * 100
        up_ratio = up_count / total
        avg_vol_ratio = total_vol_ratio / vol_counted if vol_counted > 0 else 1.0

        # === 恐慌度评分 ===
        panic = 0
        if limit_down_pct > 5: panic += 40
        elif limit_down_pct > 2: panic += 20
        elif limit_down_pct > 1: panic += 10

        if down_5_pct > 15: panic += 30
        elif down_5_pct > 8: panic += 15

        if down_3_pct > 30: panic += 20
        elif down_3_pct > 20: panic += 10

        if up_ratio < 0.2: panic += 15
        elif up_ratio < 0.3: panic += 8

        # === 事件冲击判定 ===
        shock = False
        shock_detail = ""
        if panic >= 50:
            shock = True
            shock_detail = f"HEAVY panic={panic} LD={limit_down} D5%={down_5pct}"
        elif panic >= 25 and avg_vol_ratio > 1.3:
            shock = True
            shock_detail = f"VOL_SHOCK panic={panic} avgVR={avg_vol_ratio:.1f}"

        if shock:
            self._recent_shocks.append({
                "date": date, "panic": panic,
                "down_codes": [c for c, chg in stock_changes.items() if chg < -5],
                "limit_down_codes": [c for c, chg in stock_changes.items() if chg < -9.5],
            })
            if len(self._recent_shocks) > 5:
                self._recent_shocks.pop(0)

        # === [V5b] 板块冲击检测 ===
        today_sector_shocks = []
        for sector, code_chgs in sector_changes.items():
            if sector == "其他": continue
            n = len(code_chgs)
            if n < 3: continue  # 当日参与交易的太少不算

            avg_chg = np.mean([chg for _, chg in code_chgs])
            down_cnt = sum(1 for _, chg in code_chgs if chg < -3)
            down_ratio = down_cnt / n

            if avg_chg < SECTOR_SHOCK_AVG_THRESHOLD and down_ratio > SECTOR_SHOCK_DOWN_RATIO:
                hit_codes = [c for c, chg in code_chgs if chg < -3]
                ss = SectorShock(
                    sector=sector, date=date, day_idx=day_idx,
                    avg_change=round(avg_chg, 2),
                    down_count=down_cnt, total_count=n,
                    hit_codes=hit_codes,
                )
                today_sector_shocks.append(ss)
                self._sector_shocks.append(ss)

        # 清理过期板块冲击(只保留最近10条)
        if len(self._sector_shocks) > 10:
            self._sector_shocks = self._sector_shocks[-10:]

        # === 恢复信号: 冲击后1-3天是否企稳 ===
        recovering_from = None
        if self._recent_shocks and not shock:
            last_shock = self._recent_shocks[-1]
            shock_date = last_shock["date"]
            if panic < 15 and up_ratio > 0.4:
                recovering_from = last_shock

        # === [V5b] 板块恢复信号: 板块冲击后1-3天 ===
        recovering_sectors: List[SectorShock] = []
        if not shock:
            for ss in self._sector_shocks:
                days_since = day_idx - ss.day_idx
                if 1 <= days_since <= SECTOR_RECOVERY_DAYS:
                    # 检查该板块今天是否已企稳(不再继续暴跌)
                    sector_today = sector_changes.get(ss.sector, [])
                    if sector_today:
                        sector_avg_today = np.mean([chg for _, chg in sector_today])
                        # 只要今天板块不再暴跌(>-2%)就算恢复中
                        if sector_avg_today > -2.0:
                            recovering_sectors.append(ss)

        # === 事件级别 ===
        if panic >= 50:
            level = "PANIC"       # 极端恐慌，不买
        elif shock:
            level = "SHOCK"       # 冲击发生中，观望
        elif recovering_from or recovering_sectors:
            level = "RECOVERY"    # 冲击后恢复，抄底机会！
        elif panic >= 15:
            level = "CAUTION"     # 略有不安，谨慎
        else:
            level = "NORMAL"      # 正常

        result = {
            "level": level,
            "panic": panic,
            "shock": shock,
            "shock_detail": shock_detail,
            "up_ratio": up_ratio,
            "down_3pct_ratio": down_3_pct,
            "limit_down": limit_down,
            "limit_up": limit_up,
            "avg_vol_ratio": avg_vol_ratio,
            "recovering_from": recovering_from,
            "stock_changes": stock_changes,
            # V5b新增
            "sector_shocks": today_sector_shocks,
            "recovering_sectors": recovering_sectors,
        }
        self._history.append({"date": date, "level": level, "panic": panic})
        return result


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
    macd_cross = dif > dea and (len(dea_arr) >= 2 and (ema12[-2]-ema26[-2]) <= dea_arr[-2])

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
# 统一评分 + 事件加分
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def unified_score(ind, event_bonus=0):
    """
    统一评分 + 事件冲击后的超跌加分

    event_bonus: 事件驱动额外加分(冲击后超跌的股票获得加分)
    """
    scores = []

    # A. 低吸模式
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

    # B. 突破模式
    sb = 0
    if ind["vol_breakout"]:
        sb += 20
        if ind["ma5"] > ind["ma10"] > ind["ma20"]: sb += 15
        elif ind["ma5"] > ind["ma10"]: sb += 8
        if 2 < ind["today_chg"] < 7: sb += 10
        if 5 < ind["turnover"] < 15: sb += 8
        if ind["upper_ratio"] < 0.3: sb += 5
        if ind["rsi"] > 75: sb -= 15
    scores.append(max(0, sb))

    # C. 回踩模式
    sc = 0
    if ind["ma5"] > ind["ma10"]:
        if ind["vol_pullback"]: sc += 15
        if ind["macd_cross"]: sc += 12
        elif ind["dif"] > ind["dea"]: sc += 5
        if sc > 0:
            if ind["ma5"] > ind["ma10"] > ind["ma20"]: sc += 10
            if 0.25 < ind["price_pos"] < 0.55: sc += 8
            if 30 < ind["rsi"] < 50: sc += 8
    scores.append(max(0, sc))

    # D. 涨停接力
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

    best = max(scores)

    # 事件加分（冲击后超跌的股票额外加分）
    best += event_bonus

    if ind["today_limit"]: best -= 20
    if ind["chg_5d"] > 25: best -= 10

    return max(0, best)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 自适应状态管理器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveState:
    def __init__(self):
        self._trade_results: List[float] = []
        self._daily_equity: List[float] = []
        self._max_equity = INITIAL_CAPITAL
        self._cooldown = 0

    def record_trade(self, pnl_pct: float):
        self._trade_results.append(pnl_pct)
        if len(self._trade_results) > 20:
            self._trade_results.pop(0)

    def update_equity(self, equity: float):
        self._daily_equity.append(equity)
        self._max_equity = max(self._max_equity, equity)

    @property
    def recent_win_rate(self) -> float:
        recent = self._trade_results[-7:]
        if len(recent) < 2: return 0.5
        return sum(1 for r in recent if r > 0) / len(recent)

    @property
    def consecutive_losses(self) -> int:
        count = 0
        for r in reversed(self._trade_results):
            if r < 0: count += 1
            else: break
        return count

    @property
    def drawdown_pct(self) -> float:
        if not self._daily_equity: return 0
        return (self._max_equity - self._daily_equity[-1]) / self._max_equity * 100

    @property
    def position_coeff(self) -> float:
        wr = self.recent_win_rate
        if wr >= 0.7: base = 0.80
        elif wr >= 0.5: base = 0.60
        elif wr >= 0.3: base = 0.40
        else: base = 0.20

        cl = self.consecutive_losses
        if cl >= 4: base *= 0.3
        elif cl >= 3: base *= 0.5
        elif cl >= 2: base *= 0.7

        dd = self.drawdown_pct
        if dd > 10: base *= 0.3
        elif dd > 7: base *= 0.5
        elif dd > 5: base *= 0.7

        avg = np.mean(self._trade_results[-7:]) if self._trade_results else 0
        if avg > 3: base = min(base * 1.2, 1.0)

        return max(0.10, min(1.0, base))

    @property
    def buy_threshold(self) -> int:
        base = 40
        cl = self.consecutive_losses
        if cl >= 3: base += 15
        elif cl >= 2: base += 8
        elif cl >= 1: base += 4
        dd = self.drawdown_pct
        if dd > 8: base += 12
        elif dd > 5: base += 6
        avg = np.mean(self._trade_results[-7:]) if self._trade_results else 0
        if avg > 2: base -= 5
        return max(30, min(65, base))

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown > 0

    def tick(self):
        if self._cooldown > 0: self._cooldown -= 1

    def trigger_cooldown(self):
        self._cooldown = 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 回测引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK_PARAMS = {
    "stop_loss": -3.0,
    "tp_half": 5.0,
    "tp_full": 12.0,
    "trail_trigger": 6.0,
    "trail_dd": 2.5,
    "hold_max": 5,
    "emergency_stop": -5.0,
    "max_positions": 3,
    "single_pct": 0.35,
    "price_range": (3, 50),
    "min_turnover": 2.0,
}


class EventAgentBacktest:
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
        self._event_detector = EventDetector()
        self._event_log: List[dict] = []
        self._sector_shock_log: List[dict] = []  # V5b: 板块冲击日志

        dates_set = set()
        for df in all_data.values():
            dates_set.update(df["date"].tolist())
        self.trading_days = sorted(dates_set)
        self.day_idx_map = {d: i for i, d in enumerate(self.trading_days)}

    def run(self, start_date, end_date):
        days = [d for d in self.trading_days if start_date <= d <= end_date]
        logger.info(f"Event Agent V5b: {start_date} ~ {end_date} ({len(days)}d)")
        logger.info(f"Initial: Y{INITIAL_CAPITAL:,.0f}")

        for day_num, date in enumerate(days):
            day_idx = self.day_idx_map.get(date, day_num)
            self._state.tick()

            # 1. 开盘执行
            self._exec_pending_sells(date, day_idx)
            self._exec_pending_buys(date, day_idx)

            # 2. 盘中极端止损
            self._emergency_stop(date, day_idx)

            # 3. 收盘更新
            self._update_close(date)

            # 4. 盘后: 事件分析 (V5b: 传入day_idx用于板块冲击冷却)
            event = self._event_detector.analyze(self.all_data, date, day_idx)
            event_level = event["level"]

            # 5. 盘后: 决策
            self._review_sells(date, day_idx)

            if event_level == "PANIC":
                # 极端恐慌: 清仓保命
                for code, h in list(self.holdings.items()):
                    if h.buy_day_idx < day_idx:
                        self._pending_sells.append((code, h.shares, "PANIC_EXIT"))
            elif event_level == "SHOCK":
                # 冲击中: 不新买, 持有观望
                pass
            elif event_level == "RECOVERY":
                # 恢复期: 事件驱动抄底!
                self._scan_event_recovery(date, day_idx, event)
            else:
                # 正常/谨慎: 常规选股
                if event_level != "CAUTION" or self._state.position_coeff > 0.3:
                    self._scan_buys(date, day_idx, event)

            # 记录板块冲击
            for ss in event.get("sector_shocks", []):
                self._sector_shock_log.append({
                    "date": date, "sector": ss.sector,
                    "avg_change": ss.avg_change,
                    "down_ratio": f"{ss.down_count}/{ss.total_count}",
                    "hit_count": len(ss.hit_codes),
                })

            # 快照
            mv = sum(h.market_value for h in self.holdings.values())
            total = self.cash + mv
            ret = (total - self.prev_total) / self.prev_total * 100 if self.prev_total > 0 else 0
            self.prev_total = total
            self._state.update_equity(total)

            self.snapshots.append(Snapshot(
                date, total, self.cash, mv, len(self.holdings), ret,
                self._state.position_coeff, event_level,
            ))

            if event_level in ("PANIC", "SHOCK", "RECOVERY"):
                self._event_log.append({
                    "date": date, "level": event_level,
                    "panic": event["panic"],
                    "LD": event["limit_down"], "LU": event["limit_up"],
                    "sector_shocks": [ss.sector for ss in event.get("sector_shocks", [])],
                    "recovering_sectors": [ss.sector for ss in event.get("recovering_sectors", [])],
                })

            if (day_num + 1) % 10 == 0:
                r = (total - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                pc = self._state.position_coeff
                logger.info(
                    f"  [{date}] Day{day_num+1:>2} | Y{total:>9,.0f} | {r:>+6.2f}% | "
                    f"pos={pc:.0%} evt={event_level:<8} hold={len(self.holdings)}"
                )

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
        pc = self._state.position_coeff

        for pb in buys:
            code = pb["code"]
            if code in self.holdings or len(self.holdings) >= RISK_PARAMS["max_positions"]: continue
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
            max_mv = total_asset * pc
            remaining = max_mv - mv_now
            if remaining <= 0: continue

            buy_amount = min(self.cash * 0.95, total_asset * RISK_PARAMS["single_pct"], remaining)
            shares = int(buy_amount / cost / 100) * 100
            if shares < 100: continue
            amount = cost * shares; comm = max(amount * COMMISSION_RATE, 5)
            if amount + comm > self.cash: continue

            sector = self._event_detector.get_sector(code)
            self.cash -= (amount + comm)
            self.holdings[code] = Holding(
                code=code, name=pb.get("name", code), shares=shares,
                cost_price=cost, buy_date=date, buy_day_idx=day_idx,
                signal_type=pb.get("signal_type", "regular"),
                sector=sector,
                current_price=op, peak_price=op,
            )
            self.trades.append(Trade(
                date, code, pb.get("name", code), "buy", shares, cost, amount, comm,
                f"{pb.get('signal_type','reg')}|s={pb.get('score',0)}|pc={pc:.0%}",
                sector=sector,
            ))

    def _emergency_stop(self, date, day_idx):
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx >= day_idx: continue
            row = self._get_row(code, date)
            if row is None: continue
            low = float(row["low"])
            pnl = (low - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            if pnl <= RISK_PARAMS["emergency_stop"]:
                trigger = h.cost_price * (1 + RISK_PARAMS["emergency_stop"] / 100)
                self._sell(code, h.shares, trigger, date, f"ESTOP{pnl:.1f}%")

    def _update_close(self, date):
        for code, h in self.holdings.items():
            row = self._get_row(code, date)
            if row is None: continue
            h.current_price = float(row["close"])
            h.peak_price = max(h.peak_price, float(row["high"]))

    def _review_sells(self, date, day_idx):
        rp = RISK_PARAMS
        for code, h in list(self.holdings.items()):
            if h.buy_day_idx == day_idx: continue
            pnl = h.pnl_pct
            hold = day_idx - h.buy_day_idx
            pk = (h.peak_price - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
            dd = (h.peak_price - h.current_price) / h.peak_price * 100 if h.peak_price > 0 else 0

            # 事件驱动买入的快进快出: 2天没赚就跑
            if h.signal_type in ("event_recovery", "sector_recovery") and hold >= 2 and pnl < 1:
                self._pending_sells.append((code, h.shares, f"EVT_EXPIRE{hold}d"))
                continue

            if pnl <= rp["stop_loss"]:
                self._pending_sells.append((code, h.shares, f"SL{pnl:.1f}%"))
                continue

            if pk >= rp["trail_trigger"] and dd >= rp["trail_dd"]:
                self._pending_sells.append((code, h.shares, f"TRAIL pk{pk:.1f}%"))
                continue

            if not h.partial_sold and pnl >= rp["tp_half"]:
                half = max(100, (h.shares // 200) * 100)
                if half < h.shares:
                    self._pending_sells.append((code, half, f"TP_HALF{pnl:.1f}%"))
                    h.partial_sold = True
                    continue

            if pnl >= rp["tp_full"]:
                self._pending_sells.append((code, h.shares, f"TP_FULL{pnl:.1f}%"))
                continue

            max_hold = rp["hold_max"]
            if pnl > 2: max_hold += 2
            if hold >= max_hold and pnl < 1:
                self._pending_sells.append((code, h.shares, f"EXPIRE{hold}d"))

    def _scan_event_recovery(self, date, day_idx, event):
        """
        事件冲击后恢复期: 从冲击中跌幅最大的股票中找超跌反弹机会
        V5b增强: 优先从受冲击板块中选股，给板块内股票额外加分
        """
        if len(self.holdings) >= RISK_PARAMS["max_positions"]: return
        if self._pending_buys: return
        if self._state.in_cooldown: return

        recovering_from = event.get("recovering_from")
        recovering_sectors: List[SectorShock] = event.get("recovering_sectors", [])

        # 收集受冲击的板块代码集合
        sector_shocked_codes: Dict[str, Set[str]] = {}  # sector -> set of codes
        for ss in recovering_sectors:
            sector_shocked_codes[ss.sector] = set(ss.hit_codes)

        # 市场级冲击的受影响代码
        market_shocked_codes: Set[str] = set()
        if recovering_from:
            market_shocked_codes = set(recovering_from.get("down_codes", []))

        # 合并候选池: 板块冲击codes + 市场冲击codes
        all_candidate_codes = set()
        for codes in sector_shocked_codes.values():
            all_candidate_codes |= codes
        all_candidate_codes |= market_shocked_codes

        if not all_candidate_codes:
            return

        pmin, pmax = RISK_PARAMS["price_range"]
        threshold = max(30, self._state.buy_threshold - 10)  # 事件恢复降低门槛

        cands = []
        for code in all_candidate_codes:
            if code in self.holdings: continue
            df = self.all_data.get(code)
            if df is None: continue
            ri = df.index[df["date"] == date]
            if len(ri) == 0: continue
            idx = ri[0]; row = df.iloc[idx]
            price = float(row["close"])
            if price < pmin or price > pmax: continue
            to = float(row.get("turnover_rate", 0) or 0)
            if to < 1: continue

            ind = calc_indicators(df, idx)
            if ind is None: continue

            # 事件加分: 冲击后超跌+缩量企稳 = 反弹概率高
            event_bonus = 0
            if ind["rsi"] < 30: event_bonus += 15
            if ind["chg_3d"] < -8: event_bonus += 10
            if ind["vol_ratio"] < 0.8: event_bonus += 8  # 缩量=恐慌减退
            if 0 < ind["today_chg"] < 3: event_bonus += 5  # 今天已开始反弹

            # V5b: 板块冲击额外加分 -- 同板块集体超跌更容易集体反弹
            code_sector = self._event_detector.get_sector(code)
            is_sector_recovery = False
            if code_sector in sector_shocked_codes:
                event_bonus += 8   # 板块冲击加分
                is_sector_recovery = True
                # 如果该板块冲击幅度特别大，额外加分
                for ss in recovering_sectors:
                    if ss.sector == code_sector and ss.avg_change < -5:
                        event_bonus += 5
                        break

            s = unified_score(ind, event_bonus=event_bonus)
            if s < threshold: continue

            signal_type = "sector_recovery" if is_sector_recovery else "event_recovery"
            cands.append({
                "code": code, "name": str(row.get("stock_name", code)),
                "signal_close": price, "score": s,
                "signal_type": signal_type,
                "sector": code_sector,
            })

        cands.sort(key=lambda x: x["score"], reverse=True)
        slots = RISK_PARAMS["max_positions"] - len(self.holdings)
        for c in cands[:max(1, slots)]:
            self._pending_buys.append(c)

    def _scan_buys(self, date, day_idx, event):
        """常规选股"""
        if len(self.holdings) >= RISK_PARAMS["max_positions"]: return
        if self._pending_buys: return
        if self._state.in_cooldown: return
        if self._state.consecutive_losses >= 4:
            self._state.trigger_cooldown()
            return

        threshold = self._state.buy_threshold
        pmin, pmax = RISK_PARAMS["price_range"]

        cands = []
        for code, df in self.all_data.items():
            if code in self.holdings: continue
            ri = df.index[df["date"] == date]
            if len(ri) == 0: continue
            idx = ri[0]; row = df.iloc[idx]
            price = float(row["close"])
            if price < pmin or price > pmax: continue
            to = float(row.get("turnover_rate", 0) or 0)
            if to < RISK_PARAMS["min_turnover"]: continue
            ind = calc_indicators(df, idx)
            if ind is None: continue
            s = unified_score(ind)
            if s < threshold: continue
            cands.append({
                "code": code, "name": str(row.get("stock_name", code)),
                "signal_close": price, "score": s,
                "signal_type": "regular",
            })

        cands.sort(key=lambda x: x["score"], reverse=True)
        slots = RISK_PARAMS["max_positions"] - len(self.holdings)
        for c in cands[:max(1, slots)]:
            self._pending_buys.append(c)

    def _sell(self, code, shares, price, date, reason):
        h = self.holdings.get(code)
        if not h: return
        sp = price * (1 - SLIPPAGE_PCT); amt = sp * shares
        comm = max(amt * COMMISSION_RATE, 5); tax = amt * STAMP_TAX_RATE
        self.cash += (amt - comm - tax)
        self.trades.append(Trade(
            date, code, h.name, "sell", shares, sp, amt, comm + tax, reason,
            sector=h.sector,
        ))

        pnl = (sp - h.cost_price) / h.cost_price * 100 if h.cost_price > 0 else 0
        if shares >= h.shares:
            self._state.record_trade(pnl)
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
        trade_by_sector = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": []})

        for code, tl in completed.items():
            bs = [t for t in tl if t.direction == "buy"]
            ss = [t for t in tl if t.direction == "sell"]
            if bs and ss:
                ab = sum(t.price*t.shares for t in bs) / sum(t.shares for t in bs)
                asl = sum(t.price*t.shares for t in ss) / sum(t.shares for t in ss)
                p = (asl-ab)/ab*100
                sector = bs[0].sector or "未知"
                signal_type = "regular"
                for t in bs:
                    if "event_recovery" in t.reason or "sector_recovery" in t.reason:
                        signal_type = "event"
                        break

                if p > 0:
                    wins += 1; wl.append(p)
                    trade_by_sector[sector]["wins"] += 1
                else:
                    losses += 1; ll.append(p)
                    trade_by_sector[sector]["losses"] += 1
                trade_by_sector[sector]["pnl"].append(p)

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

        # 事件统计
        event_counts = defaultdict(int)
        for s in self.snapshots: event_counts[s.event_level] += 1
        event_trades = sum(1 for t in self.trades
                           if "event_recovery" in t.reason or "sector_recovery" in t.reason
                           or "EVT_" in t.reason)
        sector_recovery_trades = sum(1 for t in self.trades if "sector_recovery" in t.reason)

        print("\n" + "=" * 70)
        print("  Event-Driven Blind Agent V5b Backtest Report 2026 YTD")
        print("  (Sector shock detection + self-adaptive)")
        print("=" * 70)
        print(f"\n  Initial:    Y{INITIAL_CAPITAL:>12,}")
        print(f"  Final:      Y{final:>12,.0f}")
        print(f"  Return:     {ret:>+11.2f}%")
        print(f"  Annualized: {ann:>+11.2f}%")
        print(f"  Max DD:     {max_dd:>11.2f}%")
        print(f"  Sharpe:     {sharpe:>11.2f}")
        print(f"\n  Trades: {len(self.trades)}  Win: {wins}  Loss: {losses}  "
              f"WinRate: {wr:.1f}%  AvgW: {aw:+.2f}%  AvgL: {al:+.2f}%  PF: {pf:.2f}")
        print(f"  Event trades: {event_trades}  (sector_recovery: {sector_recovery_trades})")

        print(f"\n  [Event Distribution]")
        for lvl in ["NORMAL", "CAUTION", "SHOCK", "RECOVERY", "PANIC"]:
            d = event_counts.get(lvl, 0)
            pct = d / days * 100 if days > 0 else 0
            print(f"    {lvl:<10} {d:>3}d ({pct:>4.1f}%)")

        print(f"\n  [Detected Market Events]")
        for e in self._event_log:
            sectors_str = ""
            if e.get("sector_shocks"):
                sectors_str = f" sectors={','.join(e['sector_shocks'])}"
            if e.get("recovering_sectors"):
                sectors_str += f" recovering={','.join(e['recovering_sectors'])}"
            print(f"    {e['date']} {e['level']:<10} panic={e['panic']:>2} "
                  f"LD={e['LD']} LU={e['LU']}{sectors_str}")

        # V5b: 板块冲击日志
        if self._sector_shock_log:
            print(f"\n  [Sector Shocks Detected] ({len(self._sector_shock_log)} events)")
            for ss in self._sector_shock_log:
                print(f"    {ss['date']} {ss['sector']:<12} avg={ss['avg_change']:>+5.1f}% "
                      f"down={ss['down_ratio']} hit={ss['hit_count']}")

        # V5b: 按板块统计交易盈亏
        if trade_by_sector:
            print(f"\n  [Per-Sector Trade P&L]")
            sorted_sectors = sorted(trade_by_sector.items(),
                                     key=lambda x: sum(x[1]["pnl"]), reverse=True)
            for sector, stats in sorted_sectors:
                total_pnl = sum(stats["pnl"])
                n = len(stats["pnl"])
                w = stats["wins"]; l = stats["losses"]
                wr_s = w / n * 100 if n > 0 else 0
                print(f"    {sector:<14} trades={n:>2} W={w} L={l} "
                      f"WR={wr_s:>4.0f}% totalP&L={total_pnl:>+6.2f}%")

        print(f"\n  [Agent Adaptation]")
        pc_series = [s.position_coeff for s in self.snapshots]
        print(f"    Position: {pc_series[0]:.0%} -> {pc_series[-1]:.0%} "
              f"(min={min(pc_series):.0%} max={max(pc_series):.0%})")
        print(f"    Final: wr={self._state.recent_win_rate:.0%} cl={self._state.consecutive_losses} "
              f"dd={self._state.drawdown_pct:.2f}%")

        print(f"\n  [Monthly Returns]")
        cum = 0
        for m in sorted(monthly.keys()):
            mr = sum(monthly[m]); cum += mr
            month_pcs = [s.position_coeff for s in self.snapshots if s.date.startswith(m)]
            avg_pc = np.mean(month_pcs) if month_pcs else 0
            mark = "+" if mr >= 0 else "-"
            print(f"    {m}  {mr:>+7.2f}%  cum{cum:>+7.2f}%  [{mark}]  avg_pos={avg_pc:.0%}")
        print("=" * 70)

        # 保存
        out_dir = Path("data/backtest_event_agent_v5b")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "return": round(ret, 2), "annual": round(ann, 2),
            "max_dd": round(max_dd, 2), "sharpe": round(sharpe, 2),
            "win_rate": round(wr, 1), "profit_factor": round(pf, 2),
            "trades": len(self.trades), "event_trades": event_trades,
            "sector_recovery_trades": sector_recovery_trades,
            "final": round(final, 0),
            "monthly": {m: round(sum(r), 2) for m, r in sorted(monthly.items())},
            "event_log": self._event_log,
            "sector_shock_log": self._sector_shock_log,
            "sector_trade_pnl": {
                sector: {
                    "trades": len(stats["pnl"]),
                    "wins": stats["wins"],
                    "losses": stats["losses"],
                    "total_pnl": round(sum(stats["pnl"]), 2),
                }
                for sector, stats in trade_by_sector.items()
            },
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        trade_rows = [{"date": t.date, "code": t.code, "name": t.name,
                        "dir": t.direction, "shares": t.shares, "price": round(t.price, 3),
                        "amount": round(t.amount, 0), "reason": t.reason,
                        "sector": t.sector}
                       for t in self.trades]
        pd.DataFrame(trade_rows).to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")

        snap_rows = [{"date": s.date, "total": round(s.total_asset, 0),
                       "cash": round(s.cash, 0), "mv": round(s.market_value, 0),
                       "hold": s.holdings_count, "ret": round(s.daily_return_pct, 4),
                       "pos_coeff": round(s.position_coeff, 3), "event": s.event_level}
                      for s in self.snapshots]
        pd.DataFrame(snap_rows).to_csv(out_dir / "daily.csv", index=False, encoding="utf-8-sig")

        logger.info(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    from backtest_2025 import fetch_all_a_share_daily

    logger.info("Loading data...")
    all_data = fetch_all_a_share_daily("20251001", "20260410",
                                        cache_name="backtest_cache_2026ytd.pkl")
    logger.info(f"Data loaded: {len(all_data)} stocks")

    engine = EventAgentBacktest(all_data)
    engine.run("2026-01-05", "2026-04-10")
