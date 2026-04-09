"""
cb_backtest.py — 可转债 T+0 回测引擎
=====================================

用2025年可转债日K线数据，模拟每日T+0操作：
  - 每天选出高分转债
  - 基于日K线模拟日内买卖（用开盘/最高/最低/收盘近似分时）
  - 统计胜率、盈亏比、累计收益

使用:
  python cb_backtest.py                # 默认2025全年
  python cb_backtest.py --period 2026q1  # 2026年Q1
"""

import argparse
import logging
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 回测参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INITIAL_CAPITAL = 20000         # 初始资金（模拟盘）
CB_CAPITAL_RATIO = 0.4          # 分配给转债的资金比例
MAX_POSITIONS = 2               # 同时持有最多2只
MAX_SINGLE_PCT = 0.50           # 单只最大仓位
STOP_LOSS_PCT = -1.5            # 止损
TAKE_PROFIT_PCT = 2.5           # 止盈
TRAIL_TRIGGER = 1.5             # 移动止盈触发
TRAIL_STOP = 0.8                # 移动止盈回撤
COMMISSION_RATE = 0.00005       # 手续费万0.5

# 选债条件
CB_PRICE_MIN = 100
CB_PRICE_MAX = 150
CB_VOLUME_MIN = 3000            # 最低成交额(万)
CB_PREMIUM_MAX = 30.0           # 最高溢价率

# 回测使用的转债数量（全量太慢，随机抽样）
CB_SAMPLE_SIZE = 50

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据获取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_cb_list() -> pd.DataFrame:
    """获取可转债列表（含基础信息）"""
    import akshare as ak
    try:
        jsl = ak.bond_cb_jsl()
        jsl = jsl.rename(columns={
            "代码": "code", "转债名称": "name", "现价": "price",
            "转股溢价率": "premium_rate", "债券评级": "rating",
            "剩余年限": "remain_year", "成交额": "volume_wan",
        })
        return jsl
    except Exception as e:
        logger.error(f"获取转债列表失败: {e}")
        return pd.DataFrame()


def fetch_cb_daily_data(code: str, start: str = "20250101", end: str = "20261231") -> pd.DataFrame:
    """获取单只可转债日K线"""
    import akshare as ak

    if code.startswith("11"):
        symbol = f"sh{code}"
    else:
        symbol = f"sz{code}"

    try:
        df = ak.bond_zh_hs_cov_daily(symbol=symbol)
        if df is not None and len(df) > 0:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:]}"
            end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:]}"
            df = df[(df["date"] >= start_fmt) & (df["date"] <= end_fmt)]
            df = df.sort_values("date").reset_index(drop=True)
            # 确保数值类型
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
    except Exception as e:
        logger.debug(f"获取 {code} 日K失败: {e}")
    return pd.DataFrame()


def load_or_fetch_cb_data(period: str = "2025", cache_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """加载或下载可转债数据，带缓存"""
    cache_file = os.path.join(cache_dir, f"cb_cache_{period}.pkl")

    if os.path.exists(cache_file):
        logger.info(f"从缓存加载: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logger.info("获取可转债列表...")
    cb_list = fetch_cb_list()
    if cb_list.empty:
        return {}

    # 过滤掉明显不适合的转债
    cb_list["price"] = pd.to_numeric(cb_list.get("price", 0), errors="coerce")
    cb_list["volume_wan"] = pd.to_numeric(cb_list.get("volume_wan", 0), errors="coerce")

    # 放宽过滤条件获取更多数据
    valid = cb_list[
        (cb_list["price"] >= 90) & (cb_list["price"] <= 200) &
        (cb_list["volume_wan"] >= 1000)
    ]

    codes = valid["code"].tolist()
    names = valid["name"].tolist()
    logger.info(f"符合条件的转债: {len(codes)} 只")

    # 抽样（太多太慢）
    if len(codes) > CB_SAMPLE_SIZE:
        random.seed(42)
        indices = random.sample(range(len(codes)), CB_SAMPLE_SIZE)
        codes = [codes[i] for i in indices]
        names = [names[i] for i in indices]
        logger.info(f"随机抽样 {CB_SAMPLE_SIZE} 只")

    # 设置日期范围
    if period == "2025":
        start, end = "20250101", "20251231"
    elif period == "2026q1":
        start, end = "20251001", "20260331"
    else:
        start, end = "20250101", "20261231"

    # 下载数据
    all_data = {}
    for code, name in tqdm(zip(codes, names), total=len(codes)):
        df = fetch_cb_daily_data(code, start, end)
        if df is not None and len(df) >= 20:
            all_data[code] = df
            all_data[code].attrs["name"] = name
        time.sleep(0.1)

    logger.info(f"共获取 {len(all_data)} 只转债数据")

    # 缓存
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(all_data, f)
    logger.info(f"数据已缓存: {cache_file}")

    return all_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T+0 日内模拟（基于日K线近似）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_intraday(
    open_p: float, high_p: float, low_p: float, close_p: float,
    prev_close: float, volume: float,
    ma5: float, ma10: float, ma20: float,
    vol_ma5: float,
    atr_pct: float,
) -> Tuple[Optional[float], Optional[float], float]:
    """用日K线的OHLC模拟日内T+0操作

    T+0策略核心思路：
      - 日内趋势策略：开盘后价格向上突破 → 买入，触及止盈/止损 → 卖出
      - 日内反转策略：急跌到低点附近 → 买入，反弹到一定幅度 → 卖出

    Returns:
        (buy_price, sell_price, pnl_pct)
        如果没交易则返回 (None, None, 0)
    """
    if prev_close <= 0 or open_p <= 0:
        return None, None, 0

    day_range_pct = (high_p - low_p) / prev_close * 100
    open_gap_pct = (open_p - prev_close) / prev_close * 100

    # ═══ 策略1: 日内趋势跟踪 ═══
    # 条件: 开盘高于MA5, 量比>1, 日内波幅>1.5%
    vol_ratio = volume / vol_ma5 if vol_ma5 > 0 else 1
    trend_buy = False
    reversal_buy = False

    if open_p > ma5 and vol_ratio > 1.0 and day_range_pct > 1.5:
        trend_buy = True

    # ═══ 策略2: 日内反转（急跌买入）═══
    # 条件: 开盘跌>1%, 但最终收阳 or 下影线长
    lower_shadow = min(open_p, close_p) - low_p
    upper_shadow = high_p - max(open_p, close_p)
    body = abs(close_p - open_p)

    if (open_gap_pct < -1 and close_p > open_p and
            lower_shadow > body * 0.5 and day_range_pct > 1):
        reversal_buy = True

    if not trend_buy and not reversal_buy:
        return None, None, 0

    # ─── 模拟日内执行价格 ───
    if trend_buy:
        # 趋势买入: 在open和low之间买入（近似开盘后回踩买入）
        buy_price = open_p + (low_p - open_p) * 0.3  # 买在回踩30%位置
        buy_price = max(buy_price, low_p)

        # 卖出价格: 模拟日内走势
        # 如果收盘>开盘（阳线），大概率能吃到部分涨幅
        if close_p > buy_price:
            # 止盈位
            tp_price = buy_price * (1 + TAKE_PROFIT_PCT / 100)
            if high_p >= tp_price:
                sell_price = tp_price  # 触及止盈
            else:
                # 没到止盈，尾盘卖出
                sell_price = close_p * 0.998  # 尾盘略低于收盘价
        else:
            # 收阴线，止损or尾盘出
            sl_price = buy_price * (1 + STOP_LOSS_PCT / 100)
            if low_p <= sl_price:
                sell_price = sl_price  # 触及止损
            else:
                sell_price = close_p * 0.998

    else:  # reversal_buy
        # 反转买入: 在低点附近买入
        buy_price = low_p + (high_p - low_p) * 0.15  # 买在低点上方15%位置
        # 反弹卖出
        sell_price = low_p + (high_p - low_p) * 0.65  # 卖在反弹65%位置
        sell_price = min(sell_price, close_p * 1.001)  # 不超过收盘价太多

    sell_price = max(sell_price, low_p)  # 保底不低于最低价
    pnl_pct = (sell_price - buy_price) / buy_price * 100 - COMMISSION_RATE * 2 * 100

    return buy_price, sell_price, pnl_pct


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 可转债评分（回测用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_cb_for_backtest(df: pd.DataFrame, idx: int) -> float:
    """对可转债在特定日期评分

    Args:
        df: 日K线数据
        idx: 当前行索引（用 idx-1 及之前的数据评分，idx 当天交易）

    Returns:
        评分 (0-100)
    """
    if idx < 20:
        return 0

    close = df["close"].values[:idx]
    high = df["high"].values[:idx]
    low = df["low"].values[:idx]
    volume = df["volume"].values[:idx]

    price = close[-1]
    score = 0

    # 1. 价格区间 (10分)
    if CB_PRICE_MIN <= price <= 115:
        score += 10
    elif 115 < price <= 130:
        score += 7
    elif 130 < price <= CB_PRICE_MAX:
        score += 3
    elif price > CB_PRICE_MAX or price < CB_PRICE_MIN:
        return 0  # 不参与

    # 2. 波动率 (25分) — T+0核心：日内波幅越大越好
    ranges_5d = (high[-5:] - low[-5:]) / close[-5:] * 100
    avg_range = np.mean(ranges_5d)
    if avg_range > 5:
        score += 25
    elif avg_range > 3:
        score += 20
    elif avg_range > 2:
        score += 15
    elif avg_range > 1.5:
        score += 10
    else:
        score += 3

    # 3. 成交量活跃度 (20分)
    vol_ma5 = np.mean(volume[-5:])
    vol_ma20 = np.mean(volume[-20:])
    if vol_ma5 > vol_ma20 * 1.5:
        score += 20  # 近期放量
    elif vol_ma5 > vol_ma20:
        score += 15
    elif vol_ma5 > vol_ma20 * 0.7:
        score += 10
    else:
        score += 3

    # 4. 短期动量 (20分)
    chg_3d = (close[-1] / close[-4] - 1) * 100 if len(close) >= 4 else 0
    chg_1d = (close[-1] / close[-2] - 1) * 100 if len(close) >= 2 else 0

    if 1 <= chg_3d <= 8:
        score += 15
    elif -2 <= chg_3d < 1:
        score += 10  # 横盘微调=好时机
    elif chg_3d > 8:
        score += 5   # 涨太多
    else:
        score += 3

    if -1 <= chg_1d <= 3:
        score += 5  # 今天没暴涨暴跌

    # 5. MA趋势 (15分)
    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:])

    if ma5 > ma10 > ma20:
        score += 15  # 多头排列=强势
    elif ma5 > ma10:
        score += 10
    elif ma5 > ma20:
        score += 5
    else:
        score += 3   # 弱势也给基础分（反转机会）

    # 6. 日内模式 (10分) — 近5日是否频繁出现大波幅
    big_range_days = np.sum(ranges_5d > 2)
    if big_range_days >= 4:
        score += 10
    elif big_range_days >= 3:
        score += 7
    elif big_range_days >= 2:
        score += 4
    else:
        score += 1

    return score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 回测引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CBBacktestResult:
    """回测结果"""
    period: str
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_pnl_per_trade: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trade_log: List[dict] = field(default_factory=list)


def run_cb_backtest(
    all_data: Dict[str, pd.DataFrame],
    period: str = "2025",
) -> CBBacktestResult:
    """运行可转债T+0回测

    每个交易日：
      1. 对所有转债评分
      2. 选出Top N高分标的
      3. 用日K线OHLC模拟日内T+0交易
      4. 统计盈亏
    """
    result = CBBacktestResult(period=period)
    capital = INITIAL_CAPITAL * CB_CAPITAL_RATIO
    equity = capital
    peak_equity = capital
    result.equity_curve.append(equity)

    # 确定回测日期范围
    if period == "2025":
        date_start, date_end = "2025-01-06", "2025-12-31"
    elif period == "2026q1":
        date_start, date_end = "2026-01-02", "2026-03-31"
    else:
        date_start, date_end = "2025-01-06", "2026-03-31"

    # 获取所有交易日
    all_dates = set()
    for code, df in all_data.items():
        dates = df["date"].tolist()
        all_dates.update(dates)
    all_dates = sorted([d for d in all_dates if date_start <= d <= date_end])

    logger.info(f"回测期间: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 个交易日")
    logger.info(f"可转债数量: {len(all_data)} 只")
    logger.info(f"初始资金: {capital:.0f} 元")

    total_gross_profit = 0
    total_gross_loss = 0

    for date in all_dates:
        day_pnl = 0
        day_trades = 0

        # 对每只转债评分并选择
        scored = []
        for code, df in all_data.items():
            # 找到这一天的数据
            date_idx = df.index[df["date"] == date].tolist()
            if not date_idx:
                continue
            idx = date_idx[0]
            if idx < 20:
                continue

            cb_score = score_cb_for_backtest(df, idx)
            if cb_score >= 60:
                scored.append((code, cb_score, idx))

        # 按评分排序，取Top N
        scored.sort(key=lambda x: x[1], reverse=True)
        top_cbs = scored[:MAX_POSITIONS]

        # 对每只入选转债模拟T+0
        for code, cb_score, idx in top_cbs:
            df = all_data[code]
            row = df.iloc[idx]

            open_p = float(row["open"])
            high_p = float(row["high"])
            low_p = float(row["low"])
            close_p = float(row["close"])
            volume = float(row["volume"])

            prev_close = float(df.iloc[idx - 1]["close"])

            # 计算技术指标
            close_arr = df["close"].values[:idx + 1]
            vol_arr = df["volume"].values[:idx + 1]
            high_arr = df["high"].values[:idx + 1]
            low_arr = df["low"].values[:idx + 1]

            ma5 = np.mean(close_arr[-5:])
            ma10 = np.mean(close_arr[-10:])
            ma20 = np.mean(close_arr[-20:]) if len(close_arr) >= 20 else ma10
            vol_ma5 = np.mean(vol_arr[-5:])

            # ATR
            ranges = high_arr[-10:] - low_arr[-10:]
            atr_pct = np.mean(ranges / close_arr[-10:]) * 100

            # 模拟日内交易
            buy_p, sell_p, pnl_pct = simulate_intraday(
                open_p, high_p, low_p, close_p,
                prev_close, volume,
                ma5, ma10, ma20, vol_ma5, atr_pct,
            )

            if buy_p is not None and sell_p is not None:
                # 计算实际盈亏
                trade_capital = min(equity * MAX_SINGLE_PCT, equity)
                shares = int(trade_capital / buy_p / 10) * 10
                if shares < 10:
                    continue

                trade_amount = buy_p * shares
                pnl_amount = trade_amount * pnl_pct / 100

                equity += pnl_amount
                day_pnl += pnl_amount
                day_trades += 1

                if pnl_pct > 0:
                    result.win_trades += 1
                    total_gross_profit += pnl_amount
                else:
                    result.loss_trades += 1
                    total_gross_loss += abs(pnl_amount)

                result.total_trades += 1

                cb_name = all_data[code].attrs.get("name", code)
                result.trade_log.append({
                    "date": date,
                    "code": code,
                    "name": cb_name,
                    "buy": buy_p,
                    "sell": sell_p,
                    "pnl_pct": pnl_pct,
                    "pnl_amount": pnl_amount,
                    "score": cb_score,
                })

        # 日结算
        result.daily_returns.append(day_pnl / capital * 100 if capital > 0 else 0)
        result.equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity * 100
        result.max_drawdown_pct = max(result.max_drawdown_pct, drawdown)

    # 汇总
    result.total_pnl_pct = (equity - capital) / capital * 100
    result.avg_pnl_per_trade = result.total_pnl_pct / result.total_trades if result.total_trades > 0 else 0
    result.win_rate = result.win_trades / result.total_trades * 100 if result.total_trades > 0 else 0
    result.profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float("inf")

    return result


def print_backtest_result(result: CBBacktestResult):
    """打印回测结果"""
    print("\n" + "=" * 60)
    print(f"  CB T+0 Backtest Result [{result.period}]")
    print("=" * 60)
    print(f"  Total Trades:      {result.total_trades}")
    print(f"  Win / Loss:        {result.win_trades} / {result.loss_trades}")
    print(f"  Win Rate:          {result.win_rate:.1f}%")
    print(f"  Total Return:      {result.total_pnl_pct:+.2f}%")
    print(f"  Avg PnL/Trade:     {result.avg_pnl_per_trade:+.3f}%")
    print(f"  Profit Factor:     {result.profit_factor:.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown_pct:.2f}%")

    if result.daily_returns:
        daily = np.array(result.daily_returns)
        pos_days = np.sum(daily > 0)
        neg_days = np.sum(daily < 0)
        zero_days = np.sum(daily == 0)
        print(f"  Trading Days:      {len(daily)} ({pos_days} up / {neg_days} down / {zero_days} flat)")
        if len(daily) > 0:
            sharpe = np.mean(daily) / np.std(daily) * np.sqrt(252) if np.std(daily) > 0 else 0
            print(f"  Sharpe Ratio:      {sharpe:.2f}")

    print("=" * 60)

    # 最近20笔交易明细
    if result.trade_log:
        print(f"\n  Recent Trades (last 20):")
        print(f"  {'Date':<12} {'Name':<10} {'Buy':>8} {'Sell':>8} {'PnL%':>8} {'Score':>5}")
        print(f"  {'-'*55}")
        for t in result.trade_log[-20:]:
            marker = "+" if t["pnl_pct"] > 0 else " "
            print(
                f"  {t['date']:<12} {t['name'][:8]:<10} "
                f"{t['buy']:>8.2f} {t['sell']:>8.2f} "
                f"{marker}{t['pnl_pct']:>7.2f}% {t['score']:>5.0f}"
            )

    # 月度收益
    if result.trade_log:
        print(f"\n  Monthly Summary:")
        trades_df = pd.DataFrame(result.trade_log)
        trades_df["month"] = trades_df["date"].str[:7]
        monthly = trades_df.groupby("month").agg(
            trades=("pnl_pct", "count"),
            total_pnl=("pnl_amount", "sum"),
            avg_pnl=("pnl_pct", "mean"),
            win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100),
        )
        print(f"  {'Month':<10} {'Trades':>7} {'Total PnL':>12} {'Avg PnL%':>10} {'WinRate':>8}")
        print(f"  {'-'*50}")
        for month, row in monthly.iterrows():
            print(
                f"  {month:<10} {int(row['trades']):>7} "
                f"{row['total_pnl']:>+12.1f} "
                f"{row['avg_pnl']:>+10.3f}% "
                f"{row['win_rate']:>7.1f}%"
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 命令行入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="2025", choices=["2025", "2026q1", "all"])
    args = parser.parse_args()

    # 加载数据
    all_data = load_or_fetch_cb_data(period=args.period)

    if not all_data:
        print("No data available!")
    else:
        result = run_cb_backtest(all_data, period=args.period)
        print_backtest_result(result)
