# -*- coding: utf-8 -*-
"""封板压单强度评估
=====================================================
用于预判"连板接力"的次日延续概率。

两种来源：
  1. 回测代理（仅日线数据）：
     根据 close==high、turnover、相对成交量、当日振幅、涨停时机
     综合推断封单强度 0~100。
  2. 实盘真实封单（akshare / ths）：
     直接取涨停池的"封板资金""封单占流通"。

经验映射（基于游资圈共识）：
  score ≥ 85  → ULTRA (一字板/铁封) · 次日高开概率 > 75%
  score 70~85 → STRONG  · 次日接力概率 ~55%
  score 55~70 → MEDIUM  · 次日 open 不强反包可能
  score 40~55 → WEAK    · 次日很可能低开/炸板
  score < 40  → FRAGILE · 次日炸板概率高，不追
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SealStrength:
    """封板强度评估结果。"""
    score: int = 0                  # 0~100
    tier: str = "NONE"              # ULTRA / STRONG / MEDIUM / WEAK / FRAGILE / NONE
    seal_amount_est: float = 0.0    # 估算的封单金额（万元，实盘直接拿真实值）
    reasons: list = None            # 推导依据
    is_limit_up: bool = False       # 今日是否真的涨停
    is_yizi: bool = False           # 是否一字板

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


def _tier_from_score(score: int) -> str:
    if score >= 85: return "ULTRA"
    if score >= 70: return "STRONG"
    if score >= 55: return "MEDIUM"
    if score >= 40: return "WEAK"
    if score > 0:   return "FRAGILE"
    return "NONE"


def estimate_seal_from_daily(df: pd.DataFrame, idx: int) -> SealStrength:
    """仅用日线 OHLCV 代理估算封板强度。

    df  : 单只股票的日线 DataFrame (date/open/high/low/close/volume/amount/turnover)
    idx : 当前待评估的行索引（T 日）

    规则（经验法则）：
      1. 必要条件：close == high 且 change_pct > 9.5%（确认封单收盘）
      2. 换手率越低 → 封单越强（相对少的筹码出来）
      3. 当日 vol / 20 日均量 越低 → 封单越紧（一字/铁封）
      4. 盘中振幅越小 → 封单越稳（没有剧烈打开又封回）
      5. 前 5 日是否已有连板 → 首板评估更保守
    """
    s = SealStrength()
    if idx < 5 or idx >= len(df):
        s.reasons.append("数据不足")
        return s

    row = df.iloc[idx]
    prev_close = float(df.iloc[idx - 1].get("close", 0))
    if prev_close <= 0:
        return s

    close = float(row.get("close", 0))
    high = float(row.get("high", 0))
    low = float(row.get("low", 0))
    opn = float(row.get("open", 0))
    vol = float(row.get("volume", 0))
    # turnover_rate 列已经是百分数（3.99 表示 3.99%）；
    # turnover 列是分数形式（0.0399 表示 3.99%）。
    # 之前的 `if turnover < 1: *= 100` 会把 0.5% 换手的一字板错误放大为 50% → 铁封被误判为脆封。
    if "turnover_rate" in row.index if hasattr(row, "index") else "turnover_rate" in row:
        turnover = float(row.get("turnover_rate", 0) or 0)
    elif "turnover" in (row.index if hasattr(row, "index") else row):
        turnover = float(row.get("turnover", 0) or 0) * 100  # 分数 → 百分数
    else:
        turnover = 0.0

    chg = (close - prev_close) / prev_close * 100

    # 1) 封板确认
    limit_price = round(prev_close * 1.10, 2)  # 简化为 10% 板
    is_limit = chg >= 9.5 and abs(close - high) < 0.02
    s.is_limit_up = is_limit
    if not is_limit:
        s.reasons.append(f"未涨停 chg={chg:.2f}% close/high gap={abs(close-high):.3f}")
        return s

    # 2) 基础分 65（强封基准）
    score = 65

    # 3) 一字板识别
    is_yizi = abs(opn - high) < 0.02 and abs(low - high) < 0.02 and turnover < 3
    s.is_yizi = is_yizi
    if is_yizi:
        score = 95
        s.reasons.append("一字板（铁封）")

    # 4) 换手率档位
    if not is_yizi:
        if turnover < 1.5:
            score += 20; s.reasons.append(f"换手仅 {turnover:.1f}% → 铁封")
        elif turnover < 3:
            score += 12; s.reasons.append(f"换手 {turnover:.1f}% → 强封")
        elif turnover < 6:
            score += 5;  s.reasons.append(f"换手 {turnover:.1f}% → 中强封")
        elif turnover < 10:
            score += 0;  s.reasons.append(f"换手 {turnover:.1f}% → 中封")
        elif turnover < 15:
            score -= 10; s.reasons.append(f"换手 {turnover:.1f}% → 弱封")
        else:
            score -= 25; s.reasons.append(f"换手 {turnover:.1f}% 过大 → 脆封")

    # 5) 相对成交量（vs 20 日均量）
    vol20 = float(df.iloc[max(0, idx - 20):idx]["volume"].mean()) if idx >= 5 else vol
    vol_ratio = vol / vol20 if vol20 > 0 else 1
    if not is_yizi:
        if vol_ratio < 0.4:
            score += 10; s.reasons.append(f"缩量封板 vol={vol_ratio:.2f}x")
        elif vol_ratio < 0.8:
            score += 5;  s.reasons.append(f"温和量 {vol_ratio:.2f}x")
        elif vol_ratio < 1.5:
            score += 0
        elif vol_ratio < 2.5:
            score -= 8;  s.reasons.append(f"放量 {vol_ratio:.2f}x → 承接减弱")
        else:
            score -= 18; s.reasons.append(f"巨量 {vol_ratio:.2f}x → 高位博弈")

    # 6) 盘中振幅（开盘后打开过几次）
    if opn > 0 and high > 0:
        intraday_range = (high - low) / opn * 100
        open_to_close = (close - opn) / opn * 100
        if intraday_range < 2 and open_to_close > 6:
            score += 8; s.reasons.append("盘中振幅小 → 稳封")
        elif intraday_range > 8:
            score -= 10; s.reasons.append("盘中振幅大 → 多次炸开")

    # 7) 连板高度调整：高位连板封强度会被市场预期压榨
    consec = 0
    for j in range(1, 8):
        if idx - j < 0: break
        rp = df.iloc[idx - j - 1]["close"] if idx - j - 1 >= 0 else None
        rc = df.iloc[idx - j]["close"]
        if rp and rc / rp > 1.095:
            consec += 1
        else:
            break
    if consec >= 4:
        score -= 15; s.reasons.append(f"已 {consec+1} 连板 → 高位 hope 压缩")
    elif consec == 3:
        score -= 8;  s.reasons.append("4 板高位 → 减分")
    elif consec == 2:
        score -= 3;  s.reasons.append("3 板 → 中性")

    score = max(0, min(100, score))
    s.score = int(score)
    s.tier = _tier_from_score(s.score)

    # 粗略封单金额估算：(封板价 × 流通股 × 1/换手率修正系数)
    # 仅当有 outstanding_share 列时估算
    if "outstanding_share" in df.columns:
        outstanding = float(row.get("outstanding_share", 0) or 0)
        if outstanding > 0:
            # 近似：强封时平均 2% 流通股挂单; 弱封 0.3%
            pct_queued = {"ULTRA": 0.04, "STRONG": 0.02, "MEDIUM": 0.01,
                          "WEAK": 0.005, "FRAGILE": 0.002}.get(s.tier, 0)
            s.seal_amount_est = outstanding * close * pct_queued / 10000.0  # 万元

    return s


def estimate_next_day_continuation_prob(seal: SealStrength, consec_limit: int) -> float:
    """给定今日封板强度 + 当前连板数，估算明日继续涨停概率。"""
    if not seal.is_limit_up:
        return 0.0
    base = {
        "ULTRA":    0.70,
        "STRONG":   0.50,
        "MEDIUM":   0.35,
        "WEAK":     0.20,
        "FRAGILE":  0.10,
        "NONE":     0.0,
    }[seal.tier]
    # 高位连板衰减
    decay = {0: 1.0, 1: 1.0, 2: 0.92, 3: 0.78, 4: 0.60, 5: 0.40}.get(
        min(consec_limit, 5), 0.30
    )
    return round(base * decay, 3)


# ──────────────────────────────────────────────────────────
# 实盘接口：从 akshare 拉真实封单数据
# ──────────────────────────────────────────────────────────

def fetch_live_seal_data(date: Optional[str] = None) -> Dict[str, Dict]:
    """获取当日涨停池的真实封单金额。

    返回: {code: {seal_amount: 万元, first_limit_time: HH:MM,
                  open_count: int, limit_price: float,
                  turnover_amount_wan: float}}
    """
    import datetime as dt
    d = date or dt.datetime.now().strftime("%Y%m%d")
    out: Dict[str, Dict] = {}
    try:
        import akshare as ak
        # 涨停池
        df = ak.stock_zt_pool_em(date=d)
        if df is not None and not df.empty:
            # 东方财富的列名
            cmap = {
                "代码": "code",
                "名称": "name",
                "封板资金": "seal_amount",
                "首次封板时间": "first_limit_time",
                "最后封板时间": "last_limit_time",
                "炸板次数": "open_count",
                "涨停统计": "limit_stat",
                "连板数": "consec",
                "所属行业": "industry",
            }
            for _, row in df.iterrows():
                code = str(row.get("代码", "")).zfill(6)
                out[code] = {
                    "seal_amount_wan": float(row.get("封板资金", 0)) / 10000.0,  # 元 → 万
                    "first_limit_time": str(row.get("首次封板时间", "")),
                    "last_limit_time": str(row.get("最后封板时间", "")),
                    "open_count": int(row.get("炸板次数", 0) or 0),
                    "consec": int(row.get("连板数", 1) or 1),
                    "name": str(row.get("名称", "")),
                    "industry": str(row.get("所属行业", "")),
                }
    except Exception as e:
        logger.warning(f"[seal_strength] 拉取 live 封单失败: {e}")
    return out


def classify_live_seal(info: Dict) -> SealStrength:
    """把实盘获取到的单票封单信息映射为 SealStrength。"""
    s = SealStrength()
    s.is_limit_up = True
    amt_wan = info.get("seal_amount_wan", 0)
    open_cnt = info.get("open_count", 0)
    first_t = info.get("first_limit_time", "")

    # 封板金额档（万元）—— 游资圈常用阈值
    score = 50
    if amt_wan >= 50000:       # 5 亿+
        score = 95; s.reasons.append(f"封单 {amt_wan/10000:.1f}亿 → 铁封")
    elif amt_wan >= 20000:     # 2 亿
        score = 85; s.reasons.append(f"封单 {amt_wan/10000:.2f}亿")
    elif amt_wan >= 10000:
        score = 75; s.reasons.append(f"封单 1+亿")
    elif amt_wan >= 5000:
        score = 65; s.reasons.append(f"封单 {amt_wan/1000:.1f}k万")
    elif amt_wan >= 2000:
        score = 55
    elif amt_wan >= 1000:
        score = 48
    else:
        score = 35; s.reasons.append(f"封单弱 {amt_wan:.0f}万")

    # 首次封板时间：越早越强
    try:
        hh, mm = first_t.split(":")[:2]
        minutes = int(hh) * 60 + int(mm)
        if minutes < 9 * 60 + 35:          # 9:35 前秒板
            score += 12; s.reasons.append("秒板")
        elif minutes < 10 * 60:
            score += 6; s.reasons.append("早盘封板")
        elif minutes < 13 * 60:
            score += 0
        elif minutes < 14 * 60:
            score -= 5; s.reasons.append("午后封板")
        else:
            score -= 10; s.reasons.append("尾盘偷袭板")
    except Exception:
        pass

    # 炸板次数
    if open_cnt == 0:
        score += 5; s.reasons.append("未炸板")
    elif open_cnt == 1:
        score -= 3
    elif open_cnt >= 2:
        score -= 15; s.reasons.append(f"炸板 {open_cnt} 次")

    score = max(0, min(100, score))
    s.score = int(score)
    s.tier = _tier_from_score(score)
    s.seal_amount_est = amt_wan
    return s
