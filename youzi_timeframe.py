# -*- coding: utf-8 -*-
"""多周期上下文聚合器（日 / 5日 / 周 / 月）。

游资评分器需要同时看短（5日线）/ 中（周线）/ 长（月线）三层视角，
本模块把 market_scanner._fetch_kline 拉到的日线 K 线在本地做 resample，
避免重复请求外部接口。

Usage:
    from youzi_timeframe import build_timeframe_context
    ctx = build_timeframe_context("002219")
    # ctx.daily_ma5, ctx.weekly_trend, ctx.monthly_high, ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TimeframeContext:
    """日 / 5日 / 周 / 月 四层指标上下文。"""
    code: str
    ok: bool = False
    error: str = ""

    # ── 日线 ──
    last_close: float = 0.0
    daily_ma5: float = 0.0
    daily_ma10: float = 0.0
    daily_ma20: float = 0.0
    daily_ma60: float = 0.0
    daily_above_ma5: bool = False
    daily_above_ma10: bool = False
    daily_above_ma20: bool = False
    daily_above_ma60: bool = False
    daily_long_arrangement: bool = False   # MA5>MA10>MA20 多头排列
    daily_short_arrangement: bool = False  # MA5<MA10<MA20 空头
    daily_bias5: float = 0.0               # 偏离 MA5 百分比

    # ── 5日线（短期动量）──
    chg_3d: float = 0.0
    chg_5d: float = 0.0
    chg_10d: float = 0.0
    consec_up: int = 0       # 连阳
    consec_limit: int = 0    # 连板

    # ── 周线 ──
    weekly_ma4: float = 0.0
    weekly_ma10: float = 0.0
    weekly_above_ma4: bool = False
    weekly_trend: str = "unknown"  # up / down / flat
    weekly_rsi: float = 50.0
    weekly_chg: float = 0.0        # 本周涨幅

    # ── 月线 ──
    monthly_ma3: float = 0.0
    monthly_ma6: float = 0.0
    monthly_trend: str = "unknown"
    monthly_chg: float = 0.0
    is_52w_high: bool = False
    is_52w_low: bool = False
    pct_from_52w_high: float = 0.0  # -10 表示跌 10%

    # ── 成交量 ──
    vol_ratio_today: float = 0.0
    vol_ratio_5d: float = 0.0     # 5日均量 / 20日均量
    shrinking: bool = False
    exploding: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """把日线 df(index=date) 降采样为周/月 K。freq: 'W-FRI' / 'ME'."""
    if df is None or df.empty:
        return pd.DataFrame()
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    cols = [c for c in agg if c in df.columns]
    use = {c: agg[c] for c in cols}
    r = df[cols].resample(freq).agg(use).dropna(how="all")
    return r


def _slope(arr: np.ndarray) -> float:
    """最小二乘拟合斜率（百分比/单位时间）。"""
    if len(arr) < 3:
        return 0.0
    x = np.arange(len(arr))
    try:
        k, _ = np.polyfit(x, arr, 1)
        base = float(arr[0]) if arr[0] else 1.0
        return float(k) / base * 100.0
    except Exception:
        return 0.0


def _trend_label(slope_pct: float) -> str:
    if slope_pct > 0.6:
        return "up"
    if slope_pct < -0.6:
        return "down"
    return "flat"


def _rsi(close: np.ndarray, n: int = 14) -> float:
    if len(close) < n + 1:
        return 50.0
    d = np.diff(close[-(n + 1):])
    gain = np.mean(d[d > 0]) if np.any(d > 0) else 0.0
    loss = -np.mean(d[d < 0]) if np.any(d < 0) else 1e-6
    return float(100 - 100 / (1 + gain / max(loss, 1e-6)))


# ──────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────

def build_timeframe_context(
    code: str,
    kline: Optional[pd.DataFrame] = None,
) -> TimeframeContext:
    """构造 code 的多周期上下文。

    参数:
        code:   '002219' 等 6 位代码（或带 sh/sz 前缀）
        kline:  可选，外部已拉好的日线 df（index 必须是 DatetimeIndex）。
                不传则调用 market_scanner._fetch_kline 拉 120 天。
    """
    ctx = TimeframeContext(code=str(code).replace("sh", "").replace("sz", "")[-6:])

    df = kline
    if df is None or df.empty:
        try:
            from market_scanner import _fetch_kline
            df = _fetch_kline(ctx.code, days=180)
        except Exception as e:
            ctx.error = f"fetch_kline_failed: {e}"
            return ctx

    if df is None or len(df) < 25:
        ctx.error = "insufficient_data"
        return ctx

    # 保证索引是时间
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("date", "Date", "trade_date"):
            if col in df.columns:
                df = df.copy()
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        ctx.error = "no_datetime_index"
        return ctx
    df = df.sort_index()

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)

    close_arr = close.values
    latest = float(close_arr[-1])
    ctx.last_close = latest

    # ── 日线 MA ──
    ctx.daily_ma5  = float(close.rolling(5).mean().iloc[-1])  if len(close) >= 5  else latest
    ctx.daily_ma10 = float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else latest
    ctx.daily_ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else latest
    ctx.daily_ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else latest

    ctx.daily_above_ma5  = bool(latest > ctx.daily_ma5)
    ctx.daily_above_ma10 = bool(latest > ctx.daily_ma10)
    ctx.daily_above_ma20 = bool(latest > ctx.daily_ma20)
    ctx.daily_above_ma60 = bool(latest > ctx.daily_ma60)
    ctx.daily_long_arrangement  = bool(ctx.daily_ma5 > ctx.daily_ma10 > ctx.daily_ma20)
    ctx.daily_short_arrangement = bool(ctx.daily_ma5 < ctx.daily_ma10 < ctx.daily_ma20)
    ctx.daily_bias5 = round((latest - ctx.daily_ma5) / ctx.daily_ma5 * 100, 2) if ctx.daily_ma5 else 0.0

    # ── 短期动量 ──
    def _chg(offset: int) -> float:
        if len(close_arr) > offset:
            return round((latest / close_arr[-offset - 1] - 1) * 100, 2)
        return 0.0

    ctx.chg_3d  = _chg(3)
    ctx.chg_5d  = _chg(5)
    ctx.chg_10d = _chg(10)

    # 连阳 / 连板
    up, board = 0, 0
    for j in range(-1, max(-8, -len(close_arr)), -1):
        if j - 1 < -len(close_arr):
            break
        d = (close_arr[j] / close_arr[j - 1] - 1) * 100
        if d > 9.5 and board == up:  # 必须连续且首日未断
            board += 1
        if close_arr[j] > close_arr[j - 1]:
            up += 1
        else:
            break
    ctx.consec_up = up
    # 连板重算：从最近往前数涨停日
    board = 0
    for j in range(-1, max(-8, -len(close_arr)), -1):
        if j - 1 < -len(close_arr):
            break
        d = (close_arr[j] / close_arr[j - 1] - 1) * 100
        if d > 9.5:
            board += 1
        else:
            break
    ctx.consec_limit = board

    # ── 周线 ──
    try:
        wk = _resample(df, "W-FRI")
        if len(wk) >= 5:
            wc = wk["close"].astype(float)
            wc_arr = wc.values
            ctx.weekly_ma4  = float(wc.rolling(4).mean().iloc[-1])  if len(wc) >= 4  else float(wc_arr[-1])
            ctx.weekly_ma10 = float(wc.rolling(10).mean().iloc[-1]) if len(wc) >= 10 else float(wc_arr[-1])
            ctx.weekly_above_ma4 = bool(float(wc_arr[-1]) > ctx.weekly_ma4)
            last_n = wc_arr[-6:] if len(wc_arr) >= 6 else wc_arr
            ctx.weekly_trend = _trend_label(_slope(last_n))
            ctx.weekly_rsi = round(_rsi(wc_arr, n=9), 1)
            if len(wc_arr) >= 2:
                ctx.weekly_chg = round((wc_arr[-1] / wc_arr[-2] - 1) * 100, 2)
    except Exception as e:
        logger.debug(f"[youzi] weekly resample failed for {ctx.code}: {e}")

    # ── 月线 ──
    try:
        mo = _resample(df, "ME")
        if len(mo) >= 3:
            mc = mo["close"].astype(float)
            mc_arr = mc.values
            ctx.monthly_ma3 = float(mc.rolling(3).mean().iloc[-1]) if len(mc) >= 3 else float(mc_arr[-1])
            ctx.monthly_ma6 = float(mc.rolling(6).mean().iloc[-1]) if len(mc) >= 6 else float(mc_arr[-1])
            last_n = mc_arr[-5:] if len(mc_arr) >= 5 else mc_arr
            ctx.monthly_trend = _trend_label(_slope(last_n))
            if len(mc_arr) >= 2:
                ctx.monthly_chg = round((mc_arr[-1] / mc_arr[-2] - 1) * 100, 2)
    except Exception as e:
        logger.debug(f"[youzi] monthly resample failed for {ctx.code}: {e}")

    # ── 52周高低 ──
    if len(close_arr) >= 10:
        window = close_arr[-min(252, len(close_arr)):]
        w_high = float(np.max(window))
        w_low = float(np.min(window))
        ctx.is_52w_high = bool(latest >= w_high * 0.98)
        ctx.is_52w_low = bool(latest <= w_low * 1.02)
        ctx.pct_from_52w_high = round((latest / w_high - 1) * 100, 2) if w_high else 0.0

    # ── 成交量 ──
    vol_arr = vol.values.astype(float)
    if len(vol_arr) >= 20:
        vma20 = float(np.mean(vol_arr[-20:]))
        vma5 = float(np.mean(vol_arr[-5:]))
        ctx.vol_ratio_today = round(vol_arr[-1] / vma20, 2) if vma20 else 0.0
        ctx.vol_ratio_5d = round(vma5 / vma20, 2) if vma20 else 0.0
        ctx.shrinking = bool(ctx.vol_ratio_today < 0.7)
        ctx.exploding = bool(ctx.vol_ratio_today > 2.0)

    ctx.ok = True
    return ctx
