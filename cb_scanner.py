"""
cb_scanner.py — 可转债扫描器 + T+0 选债评分系统
===============================================

大A小资金T+0核心模块：
  - 实时扫描全市场可转债
  - 基于溢价率/成交额/正股联动/动量的评分体系
  - 输出当日T+0操作标的列表

数据来源：akshare (集思录 + 东方财富)

使用:
  from cb_scanner import scan_convertible_bonds
  candidates = scan_convertible_bonds()
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 可转债筛选参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CB_PRICE_MIN = 100              # 最低价格（排除低于面值的垃圾债）
CB_PRICE_MAX = 150              # 最高价格（太高的溢价空间小）
CB_VOLUME_MIN = 5000            # 最低成交额(万元)，流动性保证
CB_PREMIUM_MAX = 30.0           # 最高转股溢价率%（越低=跟正股越紧）
CB_RATING_MIN = "A+"            # 最低评级

RATING_ORDER = {"AAA": 6, "AA+": 5, "AA": 4, "AA-": 3, "A+": 2, "A": 1}


@dataclass
class CBCandidate:
    """可转债候选标的"""
    code: str                       # 转债代码
    name: str                       # 转债名称
    price: float = 0.0              # 当前价
    change_pct: float = 0.0         # 涨跌幅%
    volume_wan: float = 0.0         # 成交额(万)
    premium_rate: float = 0.0       # 转股溢价率%
    stock_code: str = ""            # 正股代码
    stock_name: str = ""            # 正股名称
    stock_change_pct: float = 0.0   # 正股涨跌幅%
    rating: str = ""                # 评级
    remain_year: float = 0.0        # 剩余年限
    score: float = 0.0              # 综合评分
    signal: str = ""                # 信号: "buy"/"watch"/"avoid"
    reason: str = ""                # 理由


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据获取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_cb_realtime() -> pd.DataFrame:
    """获取可转债实时行情 + 基础数据"""
    import akshare as ak

    # 1. 集思录数据（溢价率、评级、剩余年限）
    try:
        jsl = ak.bond_cb_jsl()
        jsl = jsl.rename(columns={
            "代码": "code", "转债名称": "name", "现价": "price",
            "涨跌幅": "change_pct", "正股代码": "stock_code",
            "正股名称": "stock_name", "正股涨跌": "stock_change_pct",
            "转股溢价率": "premium_rate", "债券评级": "rating",
            "剩余年限": "remain_year", "成交额": "volume_wan",
            "换手率": "turnover",
        })
        logger.info(f"集思录: {len(jsl)} 只可转债")
    except Exception as e:
        logger.warning(f"集思录数据获取失败: {e}")
        jsl = pd.DataFrame()

    # 2. 东方财富实时行情（更全更及时）
    try:
        spot = ak.bond_zh_hs_cov_spot()
        spot = spot.rename(columns={
            "code": "code", "name": "name", "trade": "price_rt",
            "changepercent": "change_pct_rt", "volume": "volume_rt",
            "amount": "amount_rt", "high": "high", "low": "low",
            "open": "open",
        })
        logger.info(f"东方财富行情: {len(spot)} 只可转债")
    except Exception as e:
        logger.warning(f"东方财富行情获取失败: {e}")
        spot = pd.DataFrame()

    # 优先用集思录（有溢价率、评级），补充东方财富的实时价格
    if not jsl.empty:
        return jsl
    return spot


def fetch_cb_daily(code: str, start: str = "20250101", end: str = "20261231") -> pd.DataFrame:
    """获取单只可转债日K线"""
    import akshare as ak

    # code格式: 113050 → sh113050 或 sz128xxx
    if code.startswith("11"):
        symbol = f"sh{code}"
    else:
        symbol = f"sz{code}"

    try:
        df = ak.bond_zh_hs_cov_daily(symbol=symbol)
        if df is not None and len(df) > 0:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df[(df["date"] >= start[:4] + "-" + start[4:6] + "-" + start[6:]) &
                    (df["date"] <= end[:4] + "-" + end[4:6] + "-" + end[6:])]
            df = df.sort_values("date").reset_index(drop=True)
            # 计算涨跌幅
            df["change_pct"] = df["close"].pct_change() * 100
            return df
    except Exception as e:
        logger.debug(f"获取 {code} 日K失败: {e}")
    return pd.DataFrame()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 可转债评分系统
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_cb(row: dict, daily_df: Optional[pd.DataFrame] = None) -> CBCandidate:
    """对单只可转债评分

    评分维度（总分100）：
      1. 溢价率 (25分)  — 越低越好，跟正股联动强
      2. 成交额 (20分)  — 流动性保证，做T必须有量
      3. 短期动量 (20分) — 近3日涨幅，题材热度
      4. 正股联动 (15分) — 正股在涨，转债跟涨
      5. 波动率 (10分)   — 日内波幅越大，T+0空间越大
      6. 安全边际 (10分) — 价格适中，离面值不远
    """
    cb = CBCandidate(
        code=str(row.get("code", "")),
        name=str(row.get("name", "")),
        price=float(row.get("price", 0) or 0),
        change_pct=float(row.get("change_pct", 0) or 0),
        volume_wan=float(row.get("volume_wan", 0) or 0),
        premium_rate=float(row.get("premium_rate", 0) or 0),
        stock_code=str(row.get("stock_code", "")),
        stock_name=str(row.get("stock_name", "")),
        stock_change_pct=float(row.get("stock_change_pct", 0) or 0),
        rating=str(row.get("rating", "")),
        remain_year=float(row.get("remain_year", 0) or 0),
    )

    score = 0

    # ── 1. 溢价率评分 (25分) ──
    pr = cb.premium_rate
    if pr < 5:
        score += 25         # 极低溢价=几乎等于正股
    elif pr < 10:
        score += 20
    elif pr < 15:
        score += 15
    elif pr < 20:
        score += 10
    elif pr < 30:
        score += 5
    else:
        score -= 5          # 高溢价=与正股脱节

    # ── 2. 成交额评分 (20分) ──
    vol = cb.volume_wan
    if vol > 50000:
        score += 20         # 5亿+超级活跃
    elif vol > 20000:
        score += 18
    elif vol > 10000:
        score += 15         # 1亿+活跃
    elif vol > 5000:
        score += 10
    elif vol > 1000:
        score += 5
    else:
        score -= 10         # 成交太少=无法做T

    # ── 3. 短期动量 (20分) ──
    chg = cb.change_pct
    if 2 <= chg <= 8:
        score += 20         # 今天在涨2-8%=热度正好
    elif 0.5 <= chg < 2:
        score += 12         # 小涨
    elif chg > 8:
        score += 5          # 涨太多=追高风险
    elif -2 <= chg < 0.5:
        score += 8          # 微跌=可能是低吸机会
    else:
        score -= 5          # 暴跌=回避

    # 使用日K线计算近3日动量
    if daily_df is not None and len(daily_df) >= 4:
        close = daily_df["close"].values
        chg_3d = (close[-1] / close[-4] - 1) * 100
        if 3 <= chg_3d <= 15:
            score += 10     # 3日连涨=题材持续
        elif 0 < chg_3d < 3:
            score += 5

    # ── 4. 正股联动 (15分) ──
    stock_chg = cb.stock_change_pct
    if stock_chg > 3:
        score += 15         # 正股大涨=转债跟涨
    elif stock_chg > 1:
        score += 10
    elif stock_chg > 0:
        score += 5
    elif stock_chg < -3:
        score -= 5          # 正股暴跌=回避

    # ── 5. 波动率 (10分) ──
    if daily_df is not None and len(daily_df) >= 5:
        high_arr = daily_df["high"].values[-5:]
        low_arr = daily_df["low"].values[-5:]
        close_arr = daily_df["close"].values[-5:]
        # 平均日内波幅
        avg_range = np.mean((high_arr - low_arr) / close_arr * 100)
        if avg_range > 5:
            score += 10     # 日内波幅>5%=T+0空间大
        elif avg_range > 3:
            score += 7
        elif avg_range > 1.5:
            score += 4
        cb._avg_range = avg_range
    else:
        score += 3          # 无数据给基础分

    # ── 6. 安全边际 (10分) ──
    p = cb.price
    if 100 <= p <= 115:
        score += 10         # 价格适中，下跌空间有限
    elif 115 < p <= 130:
        score += 7
    elif 130 < p <= 150:
        score += 3
    elif p > 150:
        score -= 3          # 高价转债风险大

    # ── 信号判断 ──
    cb.score = max(0, score)

    if score >= 70:
        cb.signal = "buy"
        cb.reason = "高分标的，适合T+0操作"
    elif score >= 50:
        cb.signal = "watch"
        cb.reason = "关注中，等待更好时机"
    else:
        cb.signal = "avoid"
        cb.reason = "评分不足，暂不操作"

    return cb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主扫描入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scan_convertible_bonds(
    top_n: int = 10,
    fetch_daily: bool = True,
) -> List[CBCandidate]:
    """扫描全市场可转债，返回评分排名前N的标的

    Args:
        top_n: 返回前N只
        fetch_daily: 是否获取日K线（更精确但更慢）

    Returns:
        评分从高到低排列的 CBCandidate 列表
    """
    logger.info("=" * 50)
    logger.info("可转债扫描开始")

    # 1. 获取实时数据
    df = fetch_cb_realtime()
    if df.empty:
        logger.error("无法获取可转债数据")
        return []

    # 2. 基础过滤
    filtered = df.copy()
    if "price" in filtered.columns:
        filtered["price"] = pd.to_numeric(filtered["price"], errors="coerce")
        filtered = filtered[
            (filtered["price"] >= CB_PRICE_MIN) &
            (filtered["price"] <= CB_PRICE_MAX)
        ]

    if "volume_wan" in filtered.columns:
        filtered["volume_wan"] = pd.to_numeric(filtered["volume_wan"], errors="coerce")
        filtered = filtered[filtered["volume_wan"] >= CB_VOLUME_MIN]

    if "premium_rate" in filtered.columns:
        filtered["premium_rate"] = pd.to_numeric(filtered["premium_rate"], errors="coerce")
        filtered = filtered[filtered["premium_rate"] <= CB_PREMIUM_MAX]

    logger.info(f"基础过滤后: {len(filtered)} 只 (原{len(df)}只)")

    # 3. 逐只评分
    candidates = []
    for _, row in filtered.iterrows():
        daily_df = None
        if fetch_daily:
            code = str(row.get("code", ""))
            if code:
                daily_df = fetch_cb_daily(code)
                time.sleep(0.05)  # 限速

        cb = score_cb(row.to_dict(), daily_df)
        candidates.append(cb)

    # 4. 按评分排序
    candidates.sort(key=lambda x: x.score, reverse=True)

    # 5. 输出结果
    logger.info(f"\n{'='*60}")
    logger.info(f"可转债扫描结果 Top {top_n}")
    logger.info(f"{'='*60}")
    for i, cb in enumerate(candidates[:top_n]):
        marker = {"buy": "[BUY]", "watch": "[---]", "avoid": "[XXX]"}.get(cb.signal, "")
        logger.info(
            f"  #{i+1} {marker} {cb.name}({cb.code}) "
            f"价格{cb.price:.2f} 涨跌{cb.change_pct:+.2f}% "
            f"溢价{cb.premium_rate:.1f}% 成交{cb.volume_wan:.0f}万 "
            f"评分{cb.score:.0f}"
        )

    buy_list = [c for c in candidates[:top_n] if c.signal == "buy"]
    logger.info(f"\n可买入标的: {len(buy_list)} 只")

    return candidates[:top_n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 命令行入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    results = scan_convertible_bonds(top_n=15, fetch_daily=False)
    print(f"\n共 {len(results)} 只候选")
