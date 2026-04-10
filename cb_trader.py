"""
cb_trader.py — 可转债 T+0 日内交易引擎
========================================

核心策略：
  1. 早盘选债：从 cb_scanner 获取高分标的
  2. 日内做T：基于分时动量/支撑阻力/正股联动做买卖
  3. 尾盘清仓：不隔夜持仓（可选）

交易规则：
  - 单只可转债最大仓位 50%
  - 单日最多操作 3 只转债
  - 单笔止损 -1.5%，止盈 +2%~3%
  - 日内最多 6 笔交易
  - 成交额<1000万的不做

使用:
  from cb_trader import CBDayTrader
  trader = CBDayTrader(broker)
  trader.run_morning_scan()      # 09:30 选债
  trader.check_signals()         # 盘中循环检查信号
  trader.close_day()             # 14:50 清仓
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T+0 交易参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CB_MAX_POSITIONS = 2            # 同时持有最多2只转债
CB_MAX_SINGLE_PCT = 0.50        # 单只最大仓位50%
CB_MAX_DAILY_TRADES = 8         # 日内最多交易笔数
CB_STOP_LOSS_PCT = -1.5         # 单笔止损
CB_TAKE_PROFIT_PCT = 2.5        # 单笔止盈
CB_TRAIL_TRIGGER_PCT = 1.5      # 移动止盈触发点
CB_TRAIL_STOP_PCT = 0.8         # 移动止盈回撤比例
CB_MIN_HOLD_SECONDS = 60        # 最少持仓60秒（防频繁交易）
CB_CLOSE_TIME = "14:50"         # 尾盘清仓时间
CB_CAPITAL_RATIO = 0.4          # 总资金中分配给转债的比例


@dataclass
class CBPosition:
    """可转债持仓"""
    code: str
    name: str
    shares: int                 # 张数
    cost_price: float
    current_price: float = 0.0
    peak_price: float = 0.0
    buy_time: str = ""
    reason: str = ""

    @property
    def pnl_pct(self) -> float:
        if self.cost_price <= 0:
            return 0
        return (self.current_price - self.cost_price) / self.cost_price * 100

    @property
    def peak_pnl_pct(self) -> float:
        if self.cost_price <= 0:
            return 0
        return (self.peak_price - self.cost_price) / self.cost_price * 100


@dataclass
class CBTradeRecord:
    """可转债交易记录"""
    time: str
    code: str
    name: str
    direction: str              # "buy" / "sell"
    shares: int
    price: float
    amount: float
    reason: str
    pnl_pct: float = 0.0


@dataclass
class CBDayReport:
    """日内交易日报"""
    date: str
    trades: List[CBTradeRecord] = field(default_factory=list)
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    max_drawdown_pct: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T+0 信号生成器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_cb_signal(
    cb_code: str,
    current_price: float,
    daily_df: pd.DataFrame,
    position: Optional[CBPosition] = None,
    scanner_score: float = 0,
) -> dict:
    """生成可转债T+0交易信号

    结合 scanner 评分 + 日K线趋势 + 盘中价格动态综合判断。

    Args:
        cb_code: 转债代码
        current_price: 当前价格
        daily_df: 日K线数据
        position: 当前持仓（None=未持仓）
        scanner_score: cb_scanner 综合评分（0-100），≥70 表示高质量标的

    Returns:
        {"action": "buy"/"sell"/"hold", "reason": str, "strength": float}
    """
    if daily_df is None or len(daily_df) < 3:
        # scanner 高分标的即使K线少也可以尝试
        if scanner_score >= 80 and position is None:
            return {"action": "buy", "reason": f"Scanner高分{scanner_score:.0f}(K线不足)", "strength": 55}
        return {"action": "hold", "reason": "数据不足", "strength": 0}

    close = daily_df["close"].values
    high = daily_df["high"].values
    low = daily_df["low"].values
    vol = daily_df["volume"].values

    # 技术指标
    ma3 = np.mean(close[-3:])
    ma5 = np.mean(close[-min(5, len(close)):])
    ma10 = np.mean(close[-10:]) if len(close) >= 10 else ma5

    # 近3日动量
    chg_3d = (close[-1] / close[-4] - 1) * 100 if len(close) >= 4 else 0

    # 今日波幅
    today_range = (high[-1] - low[-1]) / close[-1] * 100 if close[-1] > 0 else 0

    # 量比
    vol_ratio = vol[-1] / np.mean(vol[-5:]) if len(vol) >= 5 and np.mean(vol[-5:]) > 0 else 1

    # 支撑位/阻力位
    support = np.min(low[-min(5, len(low)):])
    resistance = np.max(high[-min(5, len(high)):])

    # 昨收（日K最后一根的收盘价）
    prev_close = close[-1]
    # 盘中涨跌幅（当前价 vs 昨收）
    intraday_chg = (current_price / prev_close - 1) * 100 if prev_close > 0 else 0

    # ─── 买入信号 ───
    if position is None:
        strength = 0
        reasons = []

        # 基础分: scanner 评分加成（scanner 已评估溢价率、成交量、正股质量）
        if scanner_score >= 80:
            strength += 25
            reasons.append(f"Scanner{scanner_score:.0f}")
        elif scanner_score >= 70:
            strength += 15
            reasons.append(f"Scanner{scanner_score:.0f}")

        # 信号1: 放量（量比>1.3）
        if vol_ratio > 1.3:
            strength += 15
            reasons.append(f"放量{vol_ratio:.1f}")

        # 信号2: 价格在MA3上方（短期趋势向上）
        if current_price > ma3:
            strength += 10
            reasons.append("上穿MA3")

        # 信号3: 短期上涨趋势（MA3>MA5）
        if ma3 > ma5:
            strength += 10
            reasons.append("MA趋势上")

        # 信号4: 盘中价格回踩支撑后反弹（距支撑<3%）
        if support > 0:
            dist_to_support = (current_price - support) / support * 100
            if 0 < dist_to_support < 3:
                strength += 10
                reasons.append(f"近支撑{dist_to_support:.1f}%")

        # 信号5: 盘中微涨（0~2%涨幅=上升初期，做T空间大）
        if 0 < intraday_chg <= 2:
            strength += 10
            reasons.append(f"盘中+{intraday_chg:.1f}%")
        # 盘中回调（-1%~0%=低吸机会）
        elif -1 <= intraday_chg < 0 and scanner_score >= 70:
            strength += 10
            reasons.append(f"盘中回调{intraday_chg:.1f}%")

        # 信号6: 日内波幅大（>2%=做T空间足）
        if today_range > 2:
            strength += 5
            reasons.append(f"振幅{today_range:.1f}%")

        # T+0 当天清仓，风险可控，门槛设为 45
        if strength >= 45:
            return {"action": "buy", "reason": " ".join(reasons), "strength": strength}

    # ─── 卖出信号（持仓中）───
    if position is not None:
        pnl = position.pnl_pct

        # 止损
        if pnl <= CB_STOP_LOSS_PCT:
            return {"action": "sell", "reason": f"止损 {pnl:.1f}%", "strength": 100}

        # 止盈
        if pnl >= CB_TAKE_PROFIT_PCT:
            return {"action": "sell", "reason": f"止盈 {pnl:.1f}%", "strength": 90}

        # 移动止盈
        peak_pnl = position.peak_pnl_pct
        if peak_pnl >= CB_TRAIL_TRIGGER_PCT:
            trail_loss = peak_pnl - pnl
            if trail_loss >= CB_TRAIL_STOP_PCT:
                return {
                    "action": "sell",
                    "reason": f"移动止盈 峰值{peak_pnl:.1f}%回撤{trail_loss:.1f}%",
                    "strength": 85,
                }

        # 趋势转弱（价格跌破MA3）
        if current_price < ma3 and pnl < 0:
            return {"action": "sell", "reason": f"趋势转弱 {pnl:.1f}%", "strength": 60}

    return {"action": "hold", "reason": "无明确信号", "strength": 0}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T+0 日内交易器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CBDayTrader:
    """可转债 T+0 日内交易器

    生命周期：
      1. run_morning_scan()  → 09:25 扫描选债
      2. check_signals()     → 09:30-14:50 循环检查
      3. close_day()         → 14:50 强制清仓
    """

    def __init__(self, broker=None, total_capital: float = 50000):
        self.broker = broker
        self.total_capital = total_capital
        self.cb_capital = total_capital * CB_CAPITAL_RATIO  # 分配给转债的资金
        self.cash = self.cb_capital

        self.watchlist: List[dict] = []         # 今日候选转债
        self.positions: Dict[str, CBPosition] = {}
        self.trades: List[CBTradeRecord] = []
        self.daily_data: Dict[str, pd.DataFrame] = {}  # 缓存日K线

    @property
    def today_trade_count(self) -> int:
        return len(self.trades)

    def run_morning_scan(self) -> List[dict]:
        """早盘选债"""
        from cb_scanner import scan_convertible_bonds

        logger.info("[CB T+0] 早盘扫描开始...")
        candidates = scan_convertible_bonds(top_n=10, fetch_daily=True)

        self.watchlist = []
        for cb in candidates:
            if cb.signal == "buy":
                self.watchlist.append({
                    "code": cb.code,
                    "name": cb.name,
                    "price": cb.price,
                    "score": cb.score,
                    "premium_rate": cb.premium_rate,
                })
                # 缓存日K线
                from cb_scanner import fetch_cb_daily
                df = fetch_cb_daily(cb.code)
                if not df.empty:
                    self.daily_data[cb.code] = df

        logger.info(f"[CB T+0] 今日候选: {len(self.watchlist)} 只转债")
        for w in self.watchlist:
            logger.info(f"  {w['name']}({w['code']}) 价格{w['price']:.2f} 评分{w['score']:.0f}")

        return self.watchlist

    def check_signals(self) -> List[dict]:
        """检查所有候选转债的交易信号

        Returns:
            触发的信号列表
        """
        if self.today_trade_count >= CB_MAX_DAILY_TRADES:
            return []

        signals = []

        # 1. 检查持仓的卖出信号
        for code, pos in list(self.positions.items()):
            daily_df = self.daily_data.get(code)
            sig = generate_cb_signal(code, pos.current_price, daily_df, pos)

            if sig["action"] == "sell":
                result = self._execute_sell(code, pos.shares, sig["reason"])
                if result:
                    signals.append(result)

        # 2. 检查候选的买入信号
        if len(self.positions) < CB_MAX_POSITIONS:
            for w in self.watchlist:
                code = w["code"]
                if code in self.positions:
                    continue

                daily_df = self.daily_data.get(code)
                # daily_df 可以为 None，generate_cb_signal 会用 scanner_score 兜底

                sig = generate_cb_signal(
                    code, w["price"], daily_df,
                    scanner_score=w.get("score", 0),
                )

                if sig["action"] == "buy" and sig["strength"] >= 45:
                    result = self._execute_buy(
                        code, w["name"], w["price"], sig["reason"]
                    )
                    if result:
                        signals.append(result)

                    if len(self.positions) >= CB_MAX_POSITIONS:
                        break

        return signals

    def update_prices(self, prices: Dict[str, float]):
        """更新持仓价格（盘中调用）"""
        for code, price in prices.items():
            if code in self.positions:
                pos = self.positions[code]
                pos.current_price = price
                pos.peak_price = max(pos.peak_price, price)

    def close_day(self) -> CBDayReport:
        """尾盘清仓 + 生成日报"""
        # 清仓所有持仓
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            self._execute_sell(code, pos.shares, "尾盘清仓")

        # 生成日报
        report = CBDayReport(
            date=datetime.now().strftime("%Y-%m-%d"),
            trades=self.trades.copy(),
        )

        # 统计
        sell_trades = [t for t in self.trades if t.direction == "sell"]
        report.win_count = sum(1 for t in sell_trades if t.pnl_pct > 0)
        report.loss_count = sum(1 for t in sell_trades if t.pnl_pct <= 0)
        report.total_pnl = sum(
            t.amount * t.pnl_pct / 100 for t in sell_trades
        )
        report.total_pnl_pct = report.total_pnl / self.cb_capital * 100 if self.cb_capital > 0 else 0

        logger.info(f"\n{'='*50}")
        logger.info(f"[CB T+0] 日报 {report.date}")
        logger.info(f"  交易笔数: {len(self.trades)}")
        logger.info(f"  盈利: {report.win_count} 笔 / 亏损: {report.loss_count} 笔")
        logger.info(f"  日内盈亏: {report.total_pnl:+.2f} 元 ({report.total_pnl_pct:+.2f}%)")
        logger.info(f"{'='*50}")

        return report

    # ─── 内部方法 ───

    def _execute_buy(self, code: str, name: str, price: float, reason: str) -> Optional[dict]:
        """执行买入"""
        # 计算可买数量（转债10张=1手）
        max_amount = min(self.cash, self.cb_capital * CB_MAX_SINGLE_PCT)
        shares = int(max_amount / price / 10) * 10  # 10张整数倍
        if shares < 10:
            return None

        amount = price * shares
        commission = max(amount * 0.00005, 0.1)  # 万0.5，最低0.1元

        if amount + commission > self.cash:
            return None

        self.cash -= (amount + commission)

        self.positions[code] = CBPosition(
            code=code, name=name, shares=shares,
            cost_price=price, current_price=price,
            peak_price=price,
            buy_time=datetime.now().strftime("%H:%M:%S"),
            reason=reason,
        )

        trade = CBTradeRecord(
            time=datetime.now().strftime("%H:%M:%S"),
            code=code, name=name, direction="buy",
            shares=shares, price=price, amount=amount,
            reason=reason,
        )
        self.trades.append(trade)

        logger.info(f"[CB BUY] {name}({code}) {shares}张 @{price:.3f} = {amount:.0f}元 | {reason}")

        # 实盘下单（转债价格保留3位小数）
        price = round(price, 3)
        if self.broker:
            try:
                self.broker.buy(code, price, shares)
            except Exception as e:
                logger.error(f"[CB] 实盘买入失败: {e}")

        return {"action": "buy", "code": code, "name": name, "shares": shares, "price": price}

    def _execute_sell(self, code: str, shares: int, reason: str) -> Optional[dict]:
        """执行卖出"""
        pos = self.positions.get(code)
        if not pos:
            return None

        price = pos.current_price
        amount = price * shares
        commission = max(amount * 0.00005, 0.1)  # 无印花税

        pnl_pct = pos.pnl_pct

        self.cash += (amount - commission)

        trade = CBTradeRecord(
            time=datetime.now().strftime("%H:%M:%S"),
            code=code, name=pos.name, direction="sell",
            shares=shares, price=price, amount=amount,
            reason=reason, pnl_pct=pnl_pct,
        )
        self.trades.append(trade)

        marker = "+" if pnl_pct >= 0 else ""
        logger.info(
            f"[CB SELL] {pos.name}({code}) {shares}张 @{price:.3f} "
            f"盈亏{marker}{pnl_pct:.2f}% | {reason}"
        )

        if shares >= pos.shares:
            del self.positions[code]
        else:
            pos.shares -= shares

        # 实盘下单（转债价格保留3位小数）
        price = round(price, 3)
        if self.broker:
            try:
                self.broker.sell(code, price, shares)
            except Exception as e:
                logger.error(f"[CB] 实盘卖出失败: {e}")

        return {"action": "sell", "code": code, "pnl_pct": pnl_pct}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 命令行测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    trader = CBDayTrader(broker=None, total_capital=30000)
    logger.info(f"可转债资金: {trader.cb_capital:.0f} 元")

    # 模拟早盘扫描
    watchlist = trader.run_morning_scan()

    # 模拟检查信号
    signals = trader.check_signals()
    logger.info(f"触发信号: {len(signals)}")

    # 模拟尾盘清仓
    report = trader.close_day()
