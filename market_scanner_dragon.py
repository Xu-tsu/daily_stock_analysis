# -*- coding: utf-8 -*-
"""实盘 Dragon v2 扫描器
=====================================================
包装 market_scanner.scan_market()，在其结果上应用回测验证过的
Dragon v2 规则：
  1. Oracle 校准硬过滤（today_chg / chg_5d / RSI / vol_ratio / turnover / price）
  2. Signal 重分类（1to2 / 2to3 / pre_limit / ignition）
  3. 实盘封板强度（akshare 涨停池）→ SealStrength
  4. Regime 分流：CRASH 直接空扫描；BEAR 限制类型
  5. 仓位标记（ULTRA+1to2 → ALL-IN 提示）

接入方式（dual_trader.py）：
    from market_scanner_dragon import scan_market_dragon_v2
    results = scan_market_dragon_v2(top_n=10, regime=current_regime)

返回字段在原 scan_market 基础上追加：
    entry_type, seal_tier, seal_score, seal_amount_wan,
    next_prob, allin_suggest, dragon_tech_score
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from market_scanner import scan_market, _fetch_kline
from seal_strength import (
    SealStrength,
    estimate_seal_from_daily,
    estimate_next_day_continuation_prob,
    fetch_live_seal_data,
    classify_live_seal,
)

logger = logging.getLogger(__name__)


# ── Oracle 校准阈值（与回测一致） ──
CAL_TODAY_CHG_MAX = 7.0
CAL_TODAY_CHG_MIN = -3.5
CAL_CHG5D_MAX     = 15.0
CAL_RSI_MAX       = 72.0
CAL_VOL_MAX       = 3.0
CAL_VOL_MIN       = 0.6
CAL_TURN_MIN      = 3.0
CAL_TURN_MAX      = 15.0
CAL_PRICE_MIN     = 5.0
CAL_PRICE_MAX     = 40.0


def _classify_entry_from_result(s: Dict) -> Optional[str]:
    """从 market_scanner 返回的候选信息推断入场类型。

    market_scanner 返回字段：
      change_pct, turnover_rate, vol_ratio, rsi, chg_5d,
      signal_type (可能有)  ← 来自 analyze_kline
    """
    tc = s.get("change_pct", 0)
    vr = s.get("vol_ratio", 0)
    chg5 = s.get("chg_5d", 0)
    rsi = s.get("rsi", 50)
    turn = s.get("turnover_rate", 0)

    # 硬过滤
    if rsi > CAL_RSI_MAX: return None
    if chg5 > CAL_CHG5D_MAX: return None
    if tc > CAL_TODAY_CHG_MAX or tc < CAL_TODAY_CHG_MIN: return None
    if vr > CAL_VOL_MAX or vr < CAL_VOL_MIN: return None
    if turn < CAL_TURN_MIN or turn > CAL_TURN_MAX: return None

    is_limit = tc >= 9.5                # 今日涨停近似
    consec = int(s.get("consec_limit", 0)) if "consec_limit" in s else 0
    # 从 signal_type 推连板
    sig = (s.get("signal_type", "") or "").lower()

    # 一进二
    if is_limit and (consec == 1 or sig == "board_hit"):
        return "1to2"
    # 二进三
    if is_limit and consec == 2:
        return "2to3"
    # 连板 ≥ 3 不追
    if is_limit and consec >= 3:
        return None
    # 预涨停
    if 6.5 <= tc <= 9.5 and 1.3 <= vr <= 3.2 and not is_limit:
        return "pre_limit"
    # 启动位
    if -2 <= tc <= 5 and 0.8 <= vr <= 2.5 and CAL_TURN_MIN <= turn <= 13:
        return "ignition"
    return None


def _score_dragon(s: Dict, entry: str, seal: Optional[SealStrength]) -> int:
    score = s.get("tech_score", 0)
    tc = s.get("change_pct", 0)
    turn = s.get("turnover_rate", 0)
    vr = s.get("vol_ratio", 0)
    chg5 = s.get("chg_5d", 0)
    rsi = s.get("rsi", 50)

    # 入场类型分
    if entry == "1to2":
        score += 30
        if 5 <= turn <= 15: score += 8
        if 1.3 <= vr <= 2.5: score += 6
    elif entry == "2to3":
        score += 20
        if vr > 2.5: score -= 10
    elif entry == "pre_limit":
        score += 15
        if 7.5 <= tc <= 9.0: score += 8
    elif entry == "ignition":
        score += 10

    # 封板强度
    if seal is not None:
        bonus = {"ULTRA": 30, "STRONG": 18, "MEDIUM": 8, "WEAK": -5, "FRAGILE": -15}
        score += bonus.get(seal.tier, 0)

    # 共通惩罚
    if chg5 > 12: score -= 15
    if rsi > 70: score -= 8

    return max(0, min(150, score))


def scan_market_dragon_v2(
    regime: str = "BULL",
    top_n: int = 10,
    use_live_seal: bool = True,
    **kwargs,
) -> List[Dict]:
    """Dragon v2 扫描器（实盘版）。

    Args:
        regime: BULL / SIDE / BEAR / CRASH（来自 detect_market_regime）
        top_n:  最终返回条数
        use_live_seal: 是否调用 akshare 涨停池获取真实封单金额
        kwargs: 透传给 scan_market()

    Returns:
        List[Dict] 按 dragon_tech_score 降序排列
    """
    if regime == "CRASH":
        logger.info("[dragon_v2] regime=CRASH → 强制空扫描（保护资金）")
        return []

    # 先用 market_scanner 拉全量候选（用更宽的参数，后面自己再卡 oracle）
    raw = scan_market(
        max_price=kwargs.get("max_price", 40.0),
        min_turnover=kwargs.get("min_turnover", 3.0),
        max_market_cap=kwargs.get("max_market_cap", 300.0),
        min_change=-5.0,
        max_change=20.0,
        top_n=kwargs.get("prescan_top", 50),
        mode="dragon",
    )
    logger.info(f"[dragon_v2] market_scanner 原始候选 {len(raw)} 只")

    # 拉取真实封单池
    live_seal: Dict[str, Dict] = {}
    if use_live_seal:
        try:
            live_seal = fetch_live_seal_data()
            logger.info(f"[dragon_v2] 当日涨停池真实数据 {len(live_seal)} 只")
        except Exception as e:
            logger.warning(f"[dragon_v2] 拉取实盘封单失败: {e}")

    # Oracle 过滤 + 入场分类 + 封板强度
    out = []
    for s in raw:
        # 价格再卡一次（与回测一致）
        price = s.get("price", 0)
        if price < CAL_PRICE_MIN or price > CAL_PRICE_MAX:
            continue

        entry = _classify_entry_from_result(s)
        if entry is None:
            continue

        # BEAR 限制
        if regime == "BEAR" and entry in ("2to3", "pre_limit"):
            continue

        # 封板强度评估
        seal: Optional[SealStrength] = None
        seal_amt_wan = 0.0
        code = s["code"]
        if entry in ("1to2", "2to3"):
            # 优先用实盘真实封单
            if code in live_seal:
                seal = classify_live_seal(live_seal[code])
                seal_amt_wan = live_seal[code].get("seal_amount_wan", 0)
            else:
                # fallback: 日线估算
                try:
                    df = _fetch_kline(code, days=60)
                    if df is not None and len(df) >= 25:
                        seal = estimate_seal_from_daily(df, len(df) - 1)
                except Exception:
                    pass

            # WEAK/FRAGILE 过滤
            if seal and seal.tier in ("WEAK", "FRAGILE"):
                logger.info(f"  ✗ {code} seal={seal.tier} → 跳过（炸板概率高）")
                continue

        # 次日接力概率
        next_prob = 0.0
        if seal is not None:
            consec = int(s.get("consec_limit", 0)) if "consec_limit" in s else (1 if entry == "1to2" else 2 if entry == "2to3" else 0)
            next_prob = estimate_next_day_continuation_prob(seal, consec)

        dragon_score = _score_dragon(s, entry, seal)

        # ALL-IN 建议
        allin = (regime == "BULL" and seal is not None and seal.tier == "ULTRA" and entry == "1to2")

        enriched = {
            **s,
            "entry_type": entry,
            "seal_tier": seal.tier if seal else "NONE",
            "seal_score": seal.score if seal else 0,
            "seal_amount_wan": round(seal_amt_wan, 1),
            "next_prob": next_prob,
            "allin_suggest": allin,
            "dragon_tech_score": dragon_score,
        }
        out.append(enriched)

    # 排序：1→2 ULTRA 优先 > STRONG > others；同类按 score
    entry_rank = {"1to2": 0, "2to3": 1, "pre_limit": 2, "ignition": 3}
    seal_rank  = {"ULTRA": 0, "STRONG": 1, "MEDIUM": 2, "NONE": 3, "WEAK": 4, "FRAGILE": 5}
    out.sort(key=lambda x: (
        entry_rank.get(x["entry_type"], 9),
        seal_rank.get(x["seal_tier"], 9),
        -x["dragon_tech_score"],
    ))

    logger.info(f"[dragon_v2] 过滤后候选 {len(out)} 只，取 top {top_n}")
    for i, s in enumerate(out[:top_n], 1):
        logger.info(
            f"  #{i} {s['code']} {s.get('name','')[:6]:<6} "
            f"价{s['price']:.2f} 涨{s['change_pct']:+.1f}% "
            f"{s['entry_type']:<9} seal={s['seal_tier']:<8} "
            f"封{s['seal_amount_wan']:.0f}万 续板{s['next_prob']:.2f} "
            f"score={s['dragon_tech_score']} "
            f"{'★ALL-IN' if s['allin_suggest'] else ''}"
        )
    return out[:top_n]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default="BULL")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--no-live-seal", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    results = scan_market_dragon_v2(
        regime=args.regime,
        top_n=args.top,
        use_live_seal=not args.no_live_seal,
    )

    print(f"\n{'='*90}")
    print(f"  Dragon v2 扫描 · regime={args.regime} · Top {len(results)}")
    print(f"{'='*90}")
    print(f"{'代码':>8} {'名称':<8} {'价格':>6} {'涨%':>5} {'换%':>5} "
          f"{'入场':<9} {'封板':<8} {'封单(万)':>9} {'续板':>5} {'分数':>5}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['code']:>8} {r.get('name','')[:6]:<8} "
            f"{r['price']:>6.2f} {r['change_pct']:>+5.2f} "
            f"{r['turnover_rate']:>5.2f} "
            f"{r['entry_type']:<9} {r['seal_tier']:<8} "
            f"{r['seal_amount_wan']:>9.0f} {r['next_prob']:>5.2f} "
            f"{r['dragon_tech_score']:>5}"
            f"{'  ★ALL-IN' if r['allin_suggest'] else ''}"
        )
