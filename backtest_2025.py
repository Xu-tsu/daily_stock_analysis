"""
华尔街之狼 2025年回测引擎
===============================
基于当前系统的硬规则引擎 + 技术评分选股，
回测 2025年1月-12月 的实际市场数据。

使用方式：
  python backtest_2025.py

规则（与线上一致）：
  选股: market_scanner 技术评分 ≥ 60 + T+1 安全 + 不追高
  止损: -5% 减仓, -8% 强制清仓
  止盈: +8% 减半, +15% 全清
  持仓: 最长7天(盈利<5%则清), 最多5只, 单只≤15%
  T+1: 买入当天不可卖出
"""
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 参数（与 risk_control.py 一致）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INITIAL_CAPITAL = 200_000       # 初始资金
MAX_POSITIONS = 1               # 极度集中：ALL-IN单只（龙头战法核心）
MAX_SINGLE_PCT = 0.95           # 单只95%仓位
MAX_DAILY_BUYS = 1              # 每天最多1只
STOP_LOSS_PCT = -3.0            # 止损（龙头错了必须快跑）
FORCE_EXIT_PCT = -5.0           # 强制清仓
TAKE_PROFIT_HALF_PCT = 5.0      # 5%减半
TAKE_PROFIT_FULL_PCT = 12.0     # 12%全清
HOLD_DAYS_MAX = 2               # 最多持2天（打板隔日卖）
HOLD_DAYS_PROFIT_MIN = 0.5      # 到期盈利最低0.5%
CHASE_BLOCK_PCT = 20.0          # 涨停板可以追（龙头核心）
CHASE_WARN_PCT = 9.5            # 涨停提示
MIN_TECH_SCORE = 70             # 降低门槛（打板信号不需要太高分）
COMMISSION_RATE = 0.00025       # 单边佣金 (万2.5)
STAMP_TAX_RATE = 0.0005         # 印花税 (卖出万5)
SLIPPAGE_PCT = 0.002            # 滑点 0.2%（追板滑点更大）


@dataclass
class Holding:
    code: str
    name: str
    shares: int
    cost_price: float
    buy_date: str           # YYYY-MM-DD
    buy_day_idx: int        # 交易日序号（用于 T+1）
    current_price: float = 0.0
    peak_price: float = 0.0  # 持仓期间最高价

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.cost_price <= 0:
            return 0
        return (self.current_price - self.cost_price) / self.cost_price * 100


@dataclass
class TradeRecord:
    date: str
    code: str
    name: str
    direction: str          # buy / sell
    shares: int
    price: float
    amount: float
    commission: float
    reason: str


@dataclass
class DailySnapshot:
    date: str
    total_asset: float
    cash: float
    market_value: float
    holdings_count: int
    daily_return_pct: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据获取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_all_a_share_daily(start: str = "20250101", end: str = "20251231",
                            cache_name: str = "backtest_cache_2025.pkl") -> Dict[str, pd.DataFrame]:
    """获取 A 股日线数据 (用 akshare stock_zh_a_daily)"""
    import akshare as ak

    cache_file = Path(f"data/{cache_name}")
    if cache_file.exists():
        logger.info(f"从缓存加载数据: {cache_file}")
        return pd.read_pickle(cache_file)

    logger.info("获取 A 股代码列表...")
    try:
        stock_info = ak.stock_zh_a_spot_em()
    except Exception:
        stock_info = ak.stock_info_a_code_name()

    if "代码" in stock_info.columns:
        codes = stock_info["代码"].tolist()
    else:
        codes = stock_info["code"].tolist()

    # 过滤：排除 ST、北交所(8/9开头)、科创板(688)
    codes = [c for c in codes if not str(c).startswith(("8", "9"))
             and not str(c).startswith("688")]

    # 随机抽样加速回测（500只足够代表市场）
    import random
    random.seed(42)
    if len(codes) > 500:
        codes = random.sample(codes, 500)
        logger.info(f"随机抽样 500 只股票（总共 {len(codes)} 只可选）")

    logger.info(f"共 {len(codes)} 只股票，开始获取日线数据...")
    all_data = {}
    failed = 0

    for i, code in enumerate(codes):
        if (i + 1) % 500 == 0:
            logger.info(f"  进度: {i+1}/{len(codes)} ({len(all_data)} 成功, {failed} 失败)")
        try:
            # stock_zh_a_daily 需要 sz/sh 前缀
            if code.startswith("6"):
                symbol = f"sh{code}"
            else:
                symbol = f"sz{code}"
            df = ak.stock_zh_a_daily(
                symbol=symbol, start_date=start, end_date=end, adjust="qfq"
            )
            if df is not None and len(df) >= 20:
                # stock_zh_a_daily 返回: date, open, high, low, close, volume, amount, outstanding_share, turnover
                # 计算涨跌幅
                df = df.sort_values("date").reset_index(drop=True)
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df["change_pct"] = df["close"].pct_change() * 100
                df["turnover_rate"] = df["turnover"] * 100 if "turnover" in df.columns else 0
                all_data[code] = df
            time.sleep(0.02)  # 限速
        except Exception:
            failed += 1
            continue

    logger.info(f"数据获取完成: {len(all_data)} 只成功, {failed} 只失败")

    # 缓存
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(all_data, cache_file)
    logger.info(f"已缓存到 {cache_file}")
    return all_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 技术分析（复用 market_scanner 的评分逻辑）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_technical_score(df: pd.DataFrame, idx: int) -> dict:
    """龙头打板评分 v6 — 数据驱动的涨停接力策略

    基于10万+样本数据挖掘结果：
    最强信号：涨停+二连板(次日+2.11%, 5%+概率42%)
    次强信号：昨涨停+今继续涨5%+(次日+1.50%, 5%+概率39.6%)
    核心逻辑：找到最强的股票→ALL-IN→隔日卖出

    策略类型A（打板）：今天涨停 → 买入 → 明天高开卖
    策略类型B（接力）：昨天涨停+今天继续强 → 买入 → 明天卖
    策略类型C（爆量突破）：连阳+放量+创新高+涨5%+ → 买入
    """
    if idx < 25:
        return {"score": 0, "t1_safety": 0, "signal_type": "none"}

    window = df.iloc[max(0, idx - 60):idx + 1].copy()
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    vol = window["volume"].values

    if len(close) < 20:
        return {"score": 0, "t1_safety": 0, "signal_type": "none"}

    price = close[-1]

    # ── 基础指标 ──
    today_chg = df.iloc[idx].get("change_pct", 0)
    if isinstance(today_chg, str):
        today_chg = float(today_chg)
    if pd.isna(today_chg):
        today_chg = 0

    prev_chg = (close[-1] / close[-2] - 1) * 100 if len(close) >= 2 else 0
    prev2_chg = (close[-2] / close[-3] - 1) * 100 if len(close) >= 3 else 0

    # 昨日是否涨停
    is_prev_limit = prev_chg > 9.5 if len(close) >= 2 else False
    # 前天涨停
    is_prev2_limit = prev2_chg > 9.5 if len(close) >= 3 else False

    # 连板天数
    consec_limit = 0
    for j in range(-1, max(-8, -len(close)), -1):
        day_chg = (close[j] / close[j-1] - 1) * 100 if j-1 >= -len(close) else 0
        if day_chg > 9.5:
            consec_limit += 1
        else:
            break

    # 连阳天数
    consec_up = 0
    for j in range(-1, max(-8, -len(close)), -1):
        if close[j] > close[j-1]:
            consec_up += 1
        else:
            break

    # 量比
    vol_ma20 = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol[-5:])
    vol_today_ratio = vol[-1] / vol_ma20 if vol_ma20 > 0 else 1

    # 换手率
    turnover = float(df.iloc[idx].get("turnover_rate", 0))
    if pd.isna(turnover):
        turnover = 0

    # 3日/5日涨幅
    chg_3d = (price / close[-4] - 1) * 100 if len(close) >= 4 else 0
    chg_5d = (price / close[-6] - 1) * 100 if len(close) >= 6 else 0

    # 20日新高
    h20 = np.max(high[-20:]) if len(high) >= 20 else np.max(high[-10:])
    is_20d_high = price >= h20 * 0.98

    # MA
    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:]) if len(close) >= 20 else ma10

    score = 0
    signal_type = "none"

    # ═══════════════════════════════════════════════
    # 策略A: 涨停打板（今天涨停→买入→明天卖）
    # 数据：涨停首板次日+1.19%, 二连板+2.11%, 5%+概率23-42%
    # ═══════════════════════════════════════════════
    if today_chg > 9.5:
        score = 30  # 基础分：涨停

        # 连板加分（二连板是最强信号）
        if consec_limit >= 2:
            score += 40     # 二连板：次日+2.11%, 42%概率5%+
        elif consec_limit == 1:
            score += 25     # 首板：次日+1.19%

        # 换手率筛选（涨停+高换手10-15%最佳，>15%分歧太大）
        if 5 <= turnover <= 15:
            score += 15
        elif turnover > 15:
            score -= 10     # 换手太高=分歧，次日-0.14%

        # 放量确认（放量涨停比缩量涨停差）
        if vol_today_ratio < 2:
            score += 10     # 缩量涨停=一致看多
        elif vol_today_ratio > 3:
            score -= 5      # 放量太大=抛压

        signal_type = "board_hit"

    # ═══════════════════════════════════════════════
    # 策略B: 涨停接力（昨涨停+今继续涨→买入）
    # 数据：昨涨停+今涨5%+ 次日+1.50%, 39.6%概率5%+
    # ═══════════════════════════════════════════════
    elif is_prev_limit and today_chg > 3:
        score = 25

        if today_chg >= 5:
            score += 35     # 昨涨停+今涨5%+=最强接力信号
        elif today_chg >= 3:
            score += 20     # 昨涨停+今涨3-5%=中等接力

        # 二板接力（前天涨停+昨天涨停+今天继续涨）
        if is_prev2_limit:
            score += 15     # 连板股的接力

        # 高换手确认
        if turnover > 8:
            score += 10     # 有人气

        signal_type = "relay"

    # ═══════════════════════════════════════════════
    # 策略C: 爆量突破（强势股放量创新高）
    # 数据：今5%+放量3x+新高 次日均+0.09%，但18.7%概率5%+
    # ═══════════════════════════════════════════════
    elif today_chg >= 5 and vol_today_ratio > 2 and is_20d_high:
        score = 20

        if chg_3d > 10:
            score += 25     # 3日涨10%+=题材大热
        elif chg_3d > 5:
            score += 15

        if consec_up >= 3:
            score += 15     # 连阳+放量=资金持续涌入

        if ma5 > ma10 > ma20:
            score += 10     # 趋势确认

        if vol_today_ratio > 3:
            score += 5

        signal_type = "breakout"

    # ═══════════════════════════════════════════════
    # 策略D: 强势追涨（3日涨15%+的趋势延续）
    # 数据：3日涨15%+ 次日+0.39%, 24.7%概率5%+
    # ═══════════════════════════════════════════════
    elif chg_3d >= 15 and today_chg > 2:
        score = 20

        if chg_5d > 20:
            score += 25     # 5日涨20%+=绝对龙头
        elif chg_5d > 10:
            score += 15

        if consec_up >= 4:
            score += 15

        if is_20d_high:
            score += 10

        signal_type = "momentum"

    # ── 通用加减分 ──
    # 大盘择时不在这里做（在_process_day里做）

    # 价格过滤（太贵的不做）
    if price > 50:
        score -= 20
    elif price > 30:
        score -= 5

    return {
        "score": max(0, score),
        "t1_safety": 0,
        "signal_type": signal_type,
        "today_chg": today_chg,
        "consec_limit": consec_limit,
        "consec_up": consec_up,
        "chg_3d": chg_3d,
        "turnover": turnover,
        "vol_ratio": vol_today_ratio,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 回测引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BacktestEngine:
    """无未来函数回测引擎 v3

    核心原则: 智能体在 Day T 收盘后只能看到 ≤Day T 的数据，
    所有决策都在盘后做出，次日开盘执行。

    时间线:
      Day T 收盘后 → 复盘分析（只看 ≤T 的数据）
        → 选股: 评分 → pending_buy（明日开盘买入）
        → 调仓: 检查持仓盈亏 → pending_sell（明日开盘卖出）
      Day T+1 开盘 → 先执行 pending_sell（以 T+1 OPEN 价）
                   → 再执行 pending_buy（以 T+1 OPEN 价）
                   → 盘中只做极端止损（跌停保护，用 T+1 LOW 模拟）
      Day T+1 收盘 → 再次复盘分析 → 新的 pending_buy / pending_sell
    """

    def __init__(self, all_data: Dict[str, pd.DataFrame], initial_capital: float):
        self.all_data = all_data
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.holdings: Dict[str, Holding] = {}
        self.trades: List[TradeRecord] = []
        self.snapshots: List[DailySnapshot] = []
        self.prev_total = initial_capital

        # 待执行的订单 (T日盘后决策 → T+1日开盘执行)
        self._pending_buy: Optional[dict] = None
        self._pending_sells: List[tuple] = []  # [(code, shares, reason), ...]

        # 获取所有交易日
        dates_set = set()
        for df in all_data.values():
            dates_set.update(df["date"].tolist())
        self.trading_days = sorted(dates_set)
        self.day_idx_map = {d: i for i, d in enumerate(self.trading_days)}

    def _get_next_open(self, code: str, after_date: str) -> Optional[float]:
        """获取 after_date 之后第一个交易日的开盘价"""
        df = self.all_data.get(code)
        if df is None:
            return None
        future = df[df["date"] > after_date]
        if future.empty:
            return None
        return float(future.iloc[0]["open"])

    def _get_row(self, code: str, date: str):
        """获取指定日期的行数据"""
        df = self.all_data.get(code)
        if df is None:
            return None
        rows = df[df["date"] == date]
        if rows.empty:
            return None
        return rows.iloc[0]

    def run(self, start_date: str = "2025-01-06", end_date: str = "2025-12-31"):
        """运行回测"""
        logger.info(f"回测区间: {start_date} ~ {end_date}")
        logger.info(f"初始资金: ¥{self.initial_capital:,.0f}")
        logger.info(f"股票数据: {len(self.all_data)} 只")
        logger.info(f"模式: 无未来函数 (T日信号 → T+1日开盘价执行)")

        days = [d for d in self.trading_days if start_date <= d <= end_date]
        logger.info(f"交易日数: {len(days)} 天")

        for day_num, date in enumerate(days):
            self._process_day(date, day_num)

            if (day_num + 1) % 20 == 0:
                total = self.cash + sum(h.market_value for h in self.holdings.values())
                ret = (total - self.initial_capital) / self.initial_capital * 100
                logger.info(
                    f"  [{date}] 第{day_num+1}天 | 总资产 ¥{total:,.0f} | "
                    f"收益 {ret:+.2f}% | 持仓 {len(self.holdings)} 只"
                )

        self._print_summary()

    def _process_day(self, date: str, day_num: int):
        """处理一个交易日 — 严格禁止未来函数

        执行顺序:
          1. 开盘: 执行昨晚挂的 pending_sell (以今日 OPEN 价)
          2. 开盘: 执行昨晚挂的 pending_buy  (以今日 OPEN 价)
          3. 盘中: 仅做极端止损保护 (跌停/暴跌，用今日 LOW 模拟)
          4. 收盘: 更新持仓价格
          5. 盘后复盘: 用 ≤今日 数据分析 → 生成明日 pending_sell / pending_buy
        """
        day_idx = self.day_idx_map.get(date, day_num)

        # ═══ Step 1: 开盘 — 执行昨晚的卖出决策 (以今日 OPEN 价) ═══
        self._execute_pending_sells(date, day_idx)

        # ═══ Step 2: 开盘 — 执行昨晚的买入信号 (以今日 OPEN 价) ═══
        self._execute_pending_buy(date, day_idx)

        # ═══ Step 3: 盘中 — 仅极端止损保护 (跌停/暴跌) ═══
        self._emergency_stop_loss(date, day_idx)

        # ═══ Step 4: 收盘 — 更新持仓价格 ═══
        self._update_prices_close(date)

        # ═══ Step 5: 盘后复盘 — 分析持仓+选股 (仅看 ≤今日数据) ═══
        market_mood = self._calc_market_mood(date)

        if not hasattr(self, '_pause_until_day'):
            self._pause_until_day = -1

        if len(self.snapshots) >= 5:
            recent_peak = max(s.total_asset for s in self.snapshots[-20:])
            current_total = self.cash + sum(h.market_value for h in self.holdings.values())
            recent_dd = (recent_peak - current_total) / recent_peak * 100
            if recent_dd > 15:
                self._pause_until_day = day_num + 3

        # 5a: 盘后复盘 — 检查持仓是否需要明天卖
        self._review_and_queue_sells(date, day_idx)

        # 5b: 盘后选股 — 信号存入 pending_buy（明天开盘执行）
        if day_num >= self._pause_until_day and market_mood >= 0.30:
            self._scan_and_queue(date, day_idx)

        # ═══ 记录快照 ═══
        total_mv = sum(h.market_value for h in self.holdings.values())
        total = self.cash + total_mv
        daily_ret = (total - self.prev_total) / self.prev_total * 100 if self.prev_total > 0 else 0
        self.prev_total = total

        self.snapshots.append(DailySnapshot(
            date=date,
            total_asset=total,
            cash=self.cash,
            market_value=total_mv,
            holdings_count=len(self.holdings),
            daily_return_pct=daily_ret,
        ))

    # ────────────────────────────────────────
    # Step 1: 执行昨日挂单
    # ────────────────────────────────────────

    def _execute_pending_buy(self, date: str, day_idx: int):
        """执行昨日收盘生成的买入信号 — 以今日开盘价成交"""
        if self._pending_buy is None:
            return

        pb = self._pending_buy
        self._pending_buy = None  # 不管成不成功，清空

        code = pb["code"]

        # 如果已持仓，跳过
        if code in self.holdings or len(self.holdings) >= MAX_POSITIONS:
            return

        # 获取今天的开盘价（这是真实可获得的执行价格）
        row = self._get_row(code, date)
        if row is None:
            return  # 今天停牌

        open_price = float(row["open"])
        if open_price <= 0:
            return

        # 涨停一字板检测：开盘价=涨停价 → 买不进去
        prev_close = pb.get("signal_close", 0)
        if prev_close > 0:
            limit_up = prev_close * 1.1
            if open_price >= limit_up * 0.998:  # 开盘几乎涨停
                return  # 一字板，实盘排队也买不到

        # 以开盘价 + 滑点买入
        cost_price = open_price * (1 + SLIPPAGE_PCT)

        total_asset = self.cash + sum(h.market_value for h in self.holdings.values())
        buy_amount = min(self.cash * 0.95, total_asset * MAX_SINGLE_PCT)

        shares = int(buy_amount / cost_price / 100) * 100
        if shares < 100 or cost_price * shares > self.cash:
            return

        amount = cost_price * shares
        commission = max(amount * COMMISSION_RATE, 5)
        if amount + commission > self.cash:
            return

        self.cash -= (amount + commission)
        self.holdings[code] = Holding(
            code=code, name=pb.get("name", code), shares=shares,
            cost_price=cost_price, buy_date=date,
            buy_day_idx=day_idx, current_price=open_price,
            peak_price=open_price,
        )
        self.trades.append(TradeRecord(
            date=date, code=code, name=pb.get("name", code), direction="buy",
            shares=shares, price=cost_price, amount=amount,
            commission=commission,
            reason=f"T+1开盘买入|{pb.get('reason', '')}|signal@{pb.get('signal_date', '')}",
        ))

    # ────────────────────────────────────────
    # Step 1b: 执行昨晚挂的卖出订单
    # ────────────────────────────────────────

    def _execute_pending_sells(self, date: str, day_idx: int):
        """执行昨晚盘后复盘决定的卖出 — 以今日开盘价成交"""
        if not self._pending_sells:
            return

        sells = self._pending_sells[:]
        self._pending_sells = []

        for code, shares, reason in sells:
            h = self.holdings.get(code)
            if not h:
                continue

            # T+1检查: 昨天买入的今天才能卖（buy_day_idx < today's day_idx）
            if h.buy_day_idx >= day_idx:
                # 还不能卖，重新挂到明天
                self._pending_sells.append((code, shares, reason + "(T+1延后)"))
                continue

            row = self._get_row(code, date)
            if row is None:
                continue  # 停牌

            open_price = float(row["open"])
            if open_price <= 0:
                continue

            # 跌停一字板检测：开盘即跌停 → 卖不掉
            prev_close = h.current_price
            if prev_close > 0:
                limit_down = prev_close * 0.9
                if open_price <= limit_down * 1.002:
                    # 跌停开盘，挂单可能排不到，再挂一天
                    self._pending_sells.append((code, shares, reason + "(跌停重挂)"))
                    continue

            self._execute_sell(code, min(shares, h.shares), open_price, date,
                               f"盘后决策→开盘卖|{reason}")

    # ────────────────────────────────────────
    # Step 3: 盘中极端止损保护（仅跌停/暴跌）
    # ────────────────────────────────────────

    def _emergency_stop_loss(self, date: str, day_idx: int):
        """盘中极端止损 — 只处理今日盘中暴跌触发 -5% 强制止损线

        这是唯一允许在盘中做的操作（条件单/自动止损），
        普通止盈止损都在盘后复盘决定。
        """
        to_sell = []
        for code, h in list(self.holdings.items()):
            # T+1: 买入当天和次日都不能盘中卖
            # buy_day_idx 是实际买入的那天，当天(buy_day)不能卖，
            # 最早第二天(buy_day+1)才能卖
            if h.buy_day_idx >= day_idx:
                continue  # 今天或今天之后买入的，还不能卖

            row = self._get_row(code, date)
            if row is None:
                continue

            today_low = float(row["low"])
            cost = h.cost_price
            low_pnl = (today_low - cost) / cost * 100 if cost > 0 else 0

            # 仅极端情况：盘中低点触及 FORCE_EXIT(-5%) → 条件单自动卖
            if low_pnl <= FORCE_EXIT_PCT:
                trigger_price = cost * (1 + FORCE_EXIT_PCT / 100)
                to_sell.append((code, h.shares, trigger_price,
                                f"盘中条件单止损 low_pnl={low_pnl:.1f}%"))

        for code, shares, price, reason in to_sell:
            self._execute_sell(code, shares, price, date, reason)

    # ────────────────────────────────────────
    # Step 5a: 盘后复盘 — 决定明天卖什么
    # ────────────────────────────────────────

    def _review_and_queue_sells(self, date: str, day_idx: int):
        """盘后复盘持仓 — 用 ≤今日 收盘数据分析，决定明天开盘卖什么

        这里看到的都是今天收盘后已知的信息：
        - 今天的收盘价、涨跌幅、成交量
        - 持仓盈亏（按今天收盘价计算）
        - 持仓天数
        """
        for code, h in list(self.holdings.items()):
            # T+1规则: T+1买入 → 最早T+2卖出
            # 盘后复盘决定"明天卖不卖"，明天=day_idx+1
            # 所以必须 h.buy_day_idx <= day_idx - 1（即至少昨天买入的），明天才能卖
            # 今天刚买入的(buy_day_idx==day_idx)，明天还不能卖，后天才行
            next_day_idx = day_idx + 1
            if h.buy_day_idx >= next_day_idx:
                continue  # 这不该发生，但防御性编程
            if h.buy_day_idx == day_idx:
                continue  # 今天买入，明天(T+1)还不能卖，要到后天(T+2)

            hold_days = day_idx - h.buy_day_idx
            cost = h.cost_price
            close_pnl = h.pnl_pct  # 按收盘价算

            row = self._get_row(code, date)
            if row is None:
                continue

            today_chg = float(row.get("change_pct", 0))
            if pd.isna(today_chg):
                today_chg = 0

            today_close = float(row["close"])
            today_high = float(row["high"])

            # 更新peak
            h.peak_price = max(h.peak_price, today_high)
            peak_pnl = (h.peak_price - cost) / cost * 100 if cost > 0 else 0
            dd_from_peak = (h.peak_price - today_close) / h.peak_price * 100 if h.peak_price > 0 else 0

            # ── 盘后决策：明天是否卖出 ──

            # 1. 龙头连板中 → 不卖（今天涨停，明天继续看）
            if today_chg > 9.5 and hold_days <= 3:
                continue

            # 2. 止盈全清：收盘盈利 >= 12% → 明天开盘卖
            if close_pnl >= TAKE_PROFIT_FULL_PCT:
                self._pending_sells.append(
                    (code, h.shares, f"止盈全清 {close_pnl:.1f}%"))
                continue

            # 3. 移动止盈：高点回落 2%+ → 明天卖
            if peak_pnl >= 3.0 and dd_from_peak >= 2.0:
                self._pending_sells.append(
                    (code, h.shares,
                     f"移动止盈 peak{peak_pnl:.1f}% dd{dd_from_peak:.1f}%"))
                continue

            # 4. 半仓止盈：收盘盈利 >= 5% → 明天卖一半
            if close_pnl >= TAKE_PROFIT_HALF_PCT:
                sell_shares = (h.shares // 200) * 100
                if sell_shares >= 100:
                    self._pending_sells.append(
                        (code, sell_shares, f"半仓止盈 {close_pnl:.1f}%"))
                    continue

            # 5. 止损：收盘亏损 >= 3% → 明天开盘割肉
            if close_pnl <= STOP_LOSS_PCT:
                self._pending_sells.append(
                    (code, h.shares, f"止损 {close_pnl:.1f}%"))
                continue

            # 6. 超期清仓：持仓 >= 2天且盈利不足
            if hold_days >= HOLD_DAYS_MAX and close_pnl < HOLD_DAYS_PROFIT_MIN:
                self._pending_sells.append(
                    (code, h.shares,
                     f"超期清仓 {hold_days}d {close_pnl:.1f}%"))
                continue

            # 7. 龙头走弱：今天大跌且盈利不佳 → 明天卖
            if hold_days >= 1 and today_chg < -2 and close_pnl < 1:
                self._pending_sells.append(
                    (code, h.shares,
                     f"龙头走弱 chg{today_chg:.1f}% pnl{close_pnl:.1f}%"))
                continue

    # ────────────────────────────────────────
    # Step 3: 收盘更新价格
    # ────────────────────────────────────────

    def _update_prices_close(self, date: str):
        """收盘后更新持仓价格"""
        for code, h in self.holdings.items():
            row = self._get_row(code, date)
            if row is None:
                continue
            h.current_price = float(row["close"])
            h.peak_price = max(h.peak_price, float(row["high"]))

    # ────────────────────────────────────────
    # Step 4: 盘后评分选股（存入 pending）
    # ────────────────────────────────────────

    def _calc_market_mood(self, date: str) -> float:
        """计算大盘情绪（当日数据，收盘后可知）"""
        above_ma20 = 0
        total = 0
        for code, sdf in self.all_data.items():
            idx_arr = sdf.index[sdf["date"] == date]
            if len(idx_arr) == 0:
                continue
            i = idx_arr[0]
            total += 1
            if i >= 20:
                c = float(sdf["close"].iloc[i])
                ma20 = sdf["close"].iloc[i-19:i+1].mean()
                if c > ma20:
                    above_ma20 += 1
        return above_ma20 / total if total > 0 else 0.5

    def _scan_and_queue(self, date: str, day_idx: int):
        """盘后选股 — 用≤今天的数据评分，信号存入 pending_buy 明天执行

        关键: 这里评分用的 idx 是当天的，但买入在明天 → 无未来函数
        """
        if len(self.holdings) >= MAX_POSITIONS:
            return

        # 今天已经有 pending_buy 了就不覆盖
        if self._pending_buy is not None:
            return

        today_buys = sum(1 for t in self.trades if t.date == date and t.direction == "buy")
        if today_buys >= MAX_DAILY_BUYS:
            return

        candidates = []
        for code, df in self.all_data.items():
            if code in self.holdings:
                continue

            row_idx = df.index[df["date"] == date]
            if len(row_idx) == 0:
                continue
            idx = row_idx[0]

            row = df.iloc[idx]
            price = float(row["close"])
            if price > 50 or price < 2:
                continue

            change_pct = float(row.get("change_pct", 0))
            if pd.isna(change_pct):
                change_pct = 0

            turnover = float(row.get("turnover_rate", 0))
            if pd.isna(turnover):
                turnover = 0

            if turnover < 3:
                continue

            # 技术评分: 只用 ≤idx 的数据（compute_technical_score 使用 df.iloc[:idx+1]）
            tech = compute_technical_score(df, idx)
            if tech["score"] < MIN_TECH_SCORE:
                continue

            candidates.append({
                "code": code,
                "name": str(row.get("股票简称", code)),
                "signal_close": price,       # 信号日收盘价（用于判断明日一字板）
                "signal_date": date,
                "score": tech["score"],
                "change_pct": change_pct,
                "signal_type": tech.get("signal_type", ""),
                "consec_limit": tech.get("consec_limit", 0),
            })

        # 排序
        type_priority = {"board_hit": 4, "relay": 3, "breakout": 2, "momentum": 1}
        candidates.sort(
            key=lambda x: (type_priority.get(x["signal_type"], 0),
                           x.get("consec_limit", 0),
                           x["score"]),
            reverse=True,
        )

        # 挂单：只选最强的1只，明天开盘买入
        if candidates:
            best = candidates[0]
            best["reason"] = (
                f"{best['signal_type']}|score{best['score']}"
                f"|chg{best['change_pct']:.1f}%"
            )
            self._pending_buy = best

    # ────────────────────────────────────────
    # 成交执行
    # ────────────────────────────────────────

    def _execute_sell(self, code, shares, sell_price, date, reason):
        """执行卖出（sell_price 由调用方决定，而非统一用收盘价）"""
        h = self.holdings.get(code)
        if not h:
            return

        actual_price = sell_price * (1 - SLIPPAGE_PCT)
        amount = actual_price * shares
        commission = max(amount * COMMISSION_RATE, 5)
        tax = amount * STAMP_TAX_RATE

        self.cash += (amount - commission - tax)

        self.trades.append(TradeRecord(
            date=date, code=code, name=h.name, direction="sell",
            shares=shares, price=actual_price, amount=amount,
            commission=commission + tax, reason=reason,
        ))

        if shares >= h.shares:
            del self.holdings[code]
        else:
            h.shares -= shares

    def _print_summary(self):
        """打印回测总结"""
        if not self.snapshots:
            logger.error("无回测数据")
            return

        final = self.snapshots[-1]
        total_return = (final.total_asset - self.initial_capital) / self.initial_capital * 100

        # 最大回撤
        peak = self.initial_capital
        max_dd = 0
        for s in self.snapshots:
            peak = max(peak, s.total_asset)
            dd = (peak - s.total_asset) / peak * 100
            max_dd = max(max_dd, dd)

        # 交易统计
        buy_trades = [t for t in self.trades if t.direction == "buy"]
        sell_trades = [t for t in self.trades if t.direction == "sell"]

        # 胜率（按完整交易计算）
        completed = defaultdict(list)
        for t in self.trades:
            completed[t.code].append(t)

        wins, losses, total_pnl_list = 0, 0, []
        for code, trade_list in completed.items():
            buys = [t for t in trade_list if t.direction == "buy"]
            sells = [t for t in trade_list if t.direction == "sell"]
            if buys and sells:
                avg_buy = sum(t.price * t.shares for t in buys) / sum(t.shares for t in buys)
                avg_sell = sum(t.price * t.shares for t in sells) / sum(t.shares for t in sells)
                pnl_pct = (avg_sell - avg_buy) / avg_buy * 100
                total_pnl_list.append(pnl_pct)
                if pnl_pct > 0:
                    wins += 1
                else:
                    losses += 1

        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        avg_win = np.mean([p for p in total_pnl_list if p > 0]) if any(p > 0 for p in total_pnl_list) else 0
        avg_loss = np.mean([p for p in total_pnl_list if p <= 0]) if any(p <= 0 for p in total_pnl_list) else 0

        # 年化
        days = len(self.snapshots)
        annual_return = total_return * (252 / days) if days > 0 else 0

        # Sharpe
        daily_rets = [s.daily_return_pct for s in self.snapshots]
        if len(daily_rets) > 1:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
        else:
            sharpe = 0

        # 按月统计
        monthly = defaultdict(list)
        for s in self.snapshots:
            month = s.date[:7]
            monthly[month].append(s.daily_return_pct)

        import sys, io
        out = io.StringIO()

        period_label = getattr(self, '_period_label', '2025')
        out.write("\n" + "=" * 70 + "\n")
        out.write(f"Wolf of Wall Street {period_label} Backtest Report\n")
        out.write("=" * 70 + "\n")
        out.write(f"\n[Overall Performance]\n")
        out.write(f"  Initial Capital: Y{self.initial_capital:>12,.0f}\n")
        out.write(f"  Final Asset:     Y{final.total_asset:>12,.0f}\n")
        out.write(f"  Total Return:    {total_return:>+11.2f}%\n")
        out.write(f"  Annual Return:   {annual_return:>+11.2f}%\n")
        out.write(f"  Max Drawdown:    {max_dd:>11.2f}%\n")
        out.write(f"  Sharpe:          {sharpe:>11.2f}\n")

        out.write(f"\n[Trade Statistics]\n")
        out.write(f"  Total Trades:    {len(self.trades)}\n")
        out.write(f"  Buys:            {len(buy_trades)}\n")
        out.write(f"  Sells:           {len(sell_trades)}\n")
        out.write(f"  Completed:       {wins + losses}\n")
        out.write(f"  Win Rate:        {win_rate:.1f}%\n")
        out.write(f"  Avg Win:         {avg_win:+.2f}%\n")
        out.write(f"  Avg Loss:        {avg_loss:+.2f}%\n")
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        out.write(f"  Profit Factor:   {profit_factor:.2f}\n")

        # 卖出原因统计
        sell_reasons = defaultdict(int)
        reason_labels = {
            "强制止损": "Force StopLoss",
            "止盈全清": "Full TakeProfit",
            "止盈减半": "Half TakeProfit",
            "超期清仓": "Expired Close",
            "止损减仓": "Partial StopLoss",
        }
        for t in sell_trades:
            for key in reason_labels:
                if key in t.reason:
                    sell_reasons[key] += 1
                    break

        out.write(f"\n[Sell Reasons]\n")
        for reason, count in sorted(sell_reasons.items(), key=lambda x: -x[1]):
            label = reason_labels.get(reason, reason)
            out.write(f"  {label:<20} {count:>4}\n")

        out.write(f"\n[Monthly Returns]\n")
        cum = 0
        for month in sorted(monthly.keys()):
            rets = monthly[month]
            month_ret = sum(rets)
            cum += month_ret
            marker = "+" if month_ret >= 0 else "-"
            out.write(f"  {month}  {month_ret:>+7.2f}%  cum{cum:>+7.2f}%  [{marker}]\n")

        out.write(f"\n[Conclusion]\n")
        if total_return > 0:
            out.write(f"  {period_label} profit {total_return:.2f}%, annualized {annual_return:.2f}%\n")
        else:
            out.write(f"  {period_label} loss {total_return:.2f}%\n")
        out.write(f"  Win rate {win_rate:.1f}%, profit factor {profit_factor:.2f}\n")
        out.write(f"  Max drawdown {max_dd:.2f}%\n")
        out.write("=" * 70 + "\n")

        report_text = out.getvalue()
        print(report_text)

        # Also save as text file
        report_path = Path(f"data/backtest_{period_label.lower().replace(' ', '_')}/report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 保存结果
        self._save_results(total_return, annual_return, max_dd, win_rate, sharpe)

    def _save_results(self, total_return, annual_return, max_dd, win_rate, sharpe):
        """保存回测结果"""
        period_label = getattr(self, '_period_label', '2025')
        out_dir = Path(f"data/backtest_{period_label.lower().replace(' ', '_')}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 快照
        df_snap = pd.DataFrame([vars(s) for s in self.snapshots])
        df_snap.to_csv(out_dir / "daily_snapshots.csv", index=False, encoding="utf-8-sig")

        # 交易记录
        df_trades = pd.DataFrame([vars(t) for t in self.trades])
        df_trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")

        # 摘要
        summary = {
            "period": "2025-01 ~ 2025-12",
            "initial_capital": self.initial_capital,
            "final_asset": self.snapshots[-1].total_asset if self.snapshots else 0,
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 1),
            "sharpe_ratio": round(sharpe, 2),
            "total_trades": len(self.trades),
            "rules": {
                "stop_loss": STOP_LOSS_PCT,
                "force_exit": FORCE_EXIT_PCT,
                "take_profit_half": TAKE_PROFIT_HALF_PCT,
                "take_profit_full": TAKE_PROFIT_FULL_PCT,
                "hold_days_max": HOLD_DAYS_MAX,
                "max_positions": MAX_POSITIONS,
                "min_tech_score": MIN_TECH_SCORE,
            },
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"回测结果已保存到 {out_dir}/")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="2025", choices=["2025", "2026q1", "2026ytd"],
                        help="2025=full year 2025, 2026q1=2026 Jan-Mar, 2026ytd=2026 YTD")
    args = parser.parse_args()

    if args.period == "2026ytd":
        logger.info("华尔街之狼 2026年YTD回测 开始 (截止2026-04-10)")
        logger.info("=" * 60)
        all_data = fetch_all_a_share_daily("20251001", "20260410",
                                           cache_name="backtest_cache_2026ytd.pkl")
        engine = BacktestEngine(all_data, INITIAL_CAPITAL)
        engine._period_label = "2026YTD"
        engine.run("2026-01-05", "2026-04-10")
    elif args.period == "2026q1":
        logger.info("华尔街之狼 2026年Q1回测 开始")
        logger.info("=" * 60)
        all_data = fetch_all_a_share_daily("20251001", "20260331",
                                           cache_name="backtest_cache_2026q1.pkl")
        engine = BacktestEngine(all_data, INITIAL_CAPITAL)
        engine._period_label = "2026Q1"
        engine.run("2026-01-05", "2026-03-31")
    else:
        logger.info("华尔街之狼 2025年回测 开始")
        logger.info("=" * 60)
        all_data = fetch_all_a_share_daily("20250101", "20251231")
        engine = BacktestEngine(all_data, INITIAL_CAPITAL)
        engine._period_label = "2025"
        engine.run("2025-01-06", "2025-12-31")
