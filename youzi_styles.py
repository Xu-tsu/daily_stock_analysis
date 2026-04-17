# -*- coding: utf-8 -*-
"""游资风格评分器（蒸馏自 A 股知名游资打法）。

每一个 Style 是一个可插拔的量化评分器：
    result = style.score(candidate, tf_ctx, news_ctx, regime)
返回 YouziStyleResult，包含：
    score_delta : 对基础 tech_score 的增减
    final_score : candidate 原始 tech_score + delta
    reasons     : 触发的加分理由列表（给报告用）
    vetoes      : 硬否决原因（若非空视为被该游资 pass）
    verdict     : buy / watch / skip / veto

风格列表：
    - ChenXiaoqunStyle  陈小群：龙头打板 / 题材首板 / 连板接力
    - ZhaoLaogeStyle    赵老哥：一进二 / 中军大阳线 / 高低切
    - ZhangJiahuStyle   章盟主：大阳追高 / 月线新高 / 主升段
    - OuyangStyle       欧阳：低吸反包 / 龙回头 / 退潮反击
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from youzi_timeframe import TimeframeContext

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# 数据契约
# ──────────────────────────────────────────────────────────

@dataclass
class NewsContext:
    """candidate 的消息面视图。"""
    in_hot_concept: bool = False
    hot_concept_names: List[str] = field(default_factory=list)
    hot_concept_rank: int = 0          # 该概念在当日热度榜的排名，1=最热，0=未上榜
    event_boost: bool = False
    event_bonus: int = 0
    recent_loser: bool = False
    news_score: int = 0                # 综合消息面分（0~100）

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class YouziStyleResult:
    style_name: str
    display_name: str
    base_score: int = 0
    score_delta: int = 0
    final_score: int = 0
    reasons: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    verdict: str = "watch"   # buy / watch / skip / veto
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────
# 基类
# ──────────────────────────────────────────────────────────

class YouziStyle:
    name: str = "base"
    display_name: str = "Base"
    description: str = ""

    # 由子类在 score() 中调用 _add(...) / _veto(...) 来累加
    def score(
        self,
        candidate: Dict,
        tf: TimeframeContext,
        news: NewsContext,
        regime: str = "SIDE",
    ) -> YouziStyleResult:
        raise NotImplementedError

    # ── helpers ──
    @staticmethod
    def _market_cap_yi(c: Dict) -> float:
        """返回市值（亿）。market_scanner 输出 market_cap 单位是"元"，除以 1e8。
        部分来源已经是亿，做自动识别：>10000 视为元。"""
        mc = float(c.get("market_cap", 0) or 0)
        if mc > 10000:
            return mc / 1e8
        return mc

    @staticmethod
    def _signal(c: Dict) -> str:
        return str(c.get("signal_type", c.get("ma_trend", "")) or "")

    def _finalize(
        self,
        candidate: Dict,
        delta: int,
        reasons: List[str],
        vetoes: List[str],
        buy_threshold: int = 90,
        skip_threshold: int = 50,
    ) -> YouziStyleResult:
        base = int(candidate.get("tech_score", 0) or 0)
        final = base + delta
        if vetoes:
            verdict = "veto"
        elif final >= buy_threshold:
            verdict = "buy"
        elif final >= skip_threshold:
            verdict = "watch"
        else:
            verdict = "skip"
        return YouziStyleResult(
            style_name=self.name,
            display_name=self.display_name,
            base_score=base,
            score_delta=delta,
            final_score=final,
            reasons=reasons,
            vetoes=vetoes,
            verdict=verdict,
        )


# ──────────────────────────────────────────────────────────
# 风格 1：陈小群 —— 龙头打板 / 题材首板 / 低吸龙头回马枪
# ──────────────────────────────────────────────────────────

class ChenXiaoqunStyle(YouziStyle):
    name = "chen_xiaoqun"
    display_name = "陈小群·龙头打板"
    description = "首板排兵布阵，龙头才是核心：小盘+题材+涨停+换手适中"

    def score(self, candidate, tf, news, regime="SIDE"):
        delta = 0
        reasons: List[str] = []
        vetoes: List[str] = []

        sig = self._signal(candidate)
        chg = float(candidate.get("change_pct", 0) or 0)
        tor = float(candidate.get("turnover_rate", 0) or 0)
        price = float(candidate.get("price", 0) or 0)
        mc = self._market_cap_yi(candidate)

        # ══ 信号强度 ══
        if sig == "board_hit":
            delta += 30; reasons.append("+30 涨停首板（陈小群最爱的起点）")
        elif sig == "relay":
            delta += 20; reasons.append("+20 涨停接力（二进三卡位）")
        elif sig == "floor_to_sky" or sig == "sky_floor_sky":
            delta += 35; reasons.append("+35 地天板反包，龙头回马枪")
        elif sig == "breakout":
            delta += 12; reasons.append("+12 放量突破但非涨停")
        elif sig == "momentum":
            delta += 5; reasons.append("+5 强势动量")
        elif sig == "trend":
            delta -= 15; reasons.append("-15 仅趋势股（非龙头打法）")

        # ══ 题材龙头加分 ══
        if news.in_hot_concept:
            if news.hot_concept_rank == 1:
                delta += 40; reasons.append(f"+40 站在当日 TOP1 热点『{','.join(news.hot_concept_names[:2])}』")
            elif news.hot_concept_rank <= 3:
                delta += 25; reasons.append(f"+25 位列热点 TOP3")
            else:
                delta += 10; reasons.append("+10 在热点概念池")

        # ══ 市值甜蜜区 30-200 亿 ══
        if 30 <= mc <= 200:
            delta += 20; reasons.append(f"+20 市值{mc:.0f}亿（游资主战场）")
        elif mc > 500:
            delta -= 15; reasons.append(f"-15 市值{mc:.0f}亿过大，涨停成本高")
        elif mc < 15 and mc > 0:
            delta -= 10; reasons.append(f"-10 市值{mc:.0f}亿过小，易控盘但难持续")

        # ══ 价格区间 5-50 ══
        if 5 <= price <= 50:
            delta += 8; reasons.append(f"+8 价{price:.2f}（打板甜蜜区）")
        elif price > 80:
            delta -= 10; reasons.append(f"-10 价{price:.2f}过高")

        # ══ 连板结构 ══
        if tf.consec_limit == 1:
            delta += 15; reasons.append("+15 首板（最佳介入点）")
        elif tf.consec_limit == 2:
            delta += 25; reasons.append("+25 二连板（陈小群经典卡位）")
        elif tf.consec_limit == 3:
            delta += 10; reasons.append("+10 三连板（仍可冲关）")
        elif tf.consec_limit >= 5:
            delta -= 40; reasons.append(f"-40 {tf.consec_limit}连板（接力风险过大）")
            vetoes.append(f"{tf.consec_limit}连板高位")

        # ══ 换手率（小换手=惜售，过大=出货）══
        if 5 <= tor <= 15:
            delta += 10; reasons.append(f"+10 换手{tor:.1f}%健康")
        elif tor > 25:
            delta -= 15; reasons.append(f"-15 换手{tor:.1f}%过度活跃（易炸板）")

        # ══ 多周期过滤 ══
        if tf.weekly_trend == "down":
            delta -= 10; reasons.append("-10 周线向下（陈小群不愿逆周线打板）")
        elif tf.weekly_trend == "up":
            delta += 8; reasons.append("+8 周线向上共振")

        if tf.monthly_trend == "up" and tf.is_52w_high:
            delta += 10; reasons.append("+10 月线向上+年线新高")

        if tf.daily_short_arrangement:
            vetoes.append("日线空头排列（MA5<MA10<MA20）")
            reasons.append("VETO 空头排列，不逆势打板")

        # ══ 短期已经起飞的反而要警惕 ══
        if tf.chg_3d > 40:
            delta -= 30; reasons.append(f"-30 3日涨{tf.chg_3d}%（高位接力风险）")
        if tf.chg_5d > 60:
            vetoes.append(f"5日涨{tf.chg_5d}% 翻倍，禁区")

        # ══ V2 regime 协同 ══
        if regime == "BULL" and sig in ("board_hit", "relay"):
            delta += 10; reasons.append("+10 牛市打板加码")
        elif regime in ("BEAR", "CRASH"):
            delta -= 20; reasons.append(f"-20 {regime} 环境下打板降权")

        # ══ 近期亏损惩罚 ══
        if news.recent_loser:
            delta -= 25; reasons.append("-25 近期该票刚亏过，规避二次踩雷")

        return self._finalize(candidate, delta, reasons, vetoes,
                              buy_threshold=100, skip_threshold=55)


# ──────────────────────────────────────────────────────────
# 风格 2：赵老哥 —— 一进二 / 中军 / 高低切
# ──────────────────────────────────────────────────────────

class ZhaoLaogeStyle(YouziStyle):
    name = "zhao_laoge"
    display_name = "赵老哥·一进二中军"
    description = "一进二是真金白银兑现，盘龙的不要，换手充分的中军最优"

    def score(self, candidate, tf, news, regime="SIDE"):
        delta = 0
        reasons: List[str] = []
        vetoes: List[str] = []

        sig = self._signal(candidate)
        chg = float(candidate.get("change_pct", 0) or 0)
        tor = float(candidate.get("turnover_rate", 0) or 0)
        price = float(candidate.get("price", 0) or 0)
        mc = self._market_cap_yi(candidate)
        vr = tf.vol_ratio_today or float(candidate.get("vol_ratio", 0) or 0)

        # ══ 一进二核心信号 ══
        if sig == "relay" and tf.consec_limit == 1 and chg > 3:
            # 前一天首板，今日跟进 = 一进二场景
            delta += 45; reasons.append("+45 一进二卡位（赵老哥招牌）")
        elif sig == "board_hit" and tf.consec_limit == 2:
            delta += 35; reasons.append("+35 二板封板（龙头中军）")
        elif sig == "board_hit":
            delta += 15; reasons.append("+15 涨停（非首板场景）")
        elif sig == "relay":
            delta += 20; reasons.append("+20 接力涨停板")
        elif sig == "breakout" and tf.consec_up >= 3:
            delta += 12; reasons.append("+12 连续阳线突破")
        elif sig == "trend":
            delta -= 12; reasons.append("-12 纯趋势票非赵老哥风格")

        # ══ 成交量：必须是放量 ══
        if vr > 3:
            delta += 20; reasons.append(f"+20 爆量{vr:.1f}倍（资金兑现）")
        elif vr > 1.5:
            delta += 10; reasons.append(f"+10 放量{vr:.1f}倍")
        elif vr < 0.7 and sig in ("board_hit", "relay"):
            delta -= 15; reasons.append(f"-15 缩量涨停（一二板通常要放量）")

        # ══ 换手率：中军级要 8-20% ══
        if 8 <= tor <= 20:
            delta += 15; reasons.append(f"+15 换手{tor:.1f}%充分交换")
        elif tor < 3:
            delta -= 15; reasons.append(f"-15 换手{tor:.1f}%太低（中军要筹码活跃）")

        # ══ 中军：市值 50-300 亿 ══
        if 50 <= mc <= 300:
            delta += 15; reasons.append(f"+15 市值{mc:.0f}亿（中军体量）")
        elif mc < 30:
            delta -= 10; reasons.append(f"-10 市值{mc:.0f}亿太小，不够中军")
        elif mc > 500:
            delta -= 15; reasons.append(f"-15 市值{mc:.0f}亿过重，非赵老哥偏好")

        # ══ 题材热度 ══
        if news.in_hot_concept:
            if news.hot_concept_rank <= 2:
                delta += 20; reasons.append(f"+20 热点 TOP2 内的中军")
            else:
                delta += 8; reasons.append("+8 在热点池")
        else:
            delta -= 8; reasons.append("-8 无题材加持，赵老哥一般不参与")

        # ══ 周线月线 ══
        if tf.weekly_trend in ("up", "flat"):
            delta += 8; reasons.append(f"+8 周线{tf.weekly_trend}")
        else:
            delta -= 10; reasons.append("-10 周线向下（一进二胜率低）")

        if tf.monthly_trend == "up":
            delta += 5; reasons.append("+5 月线上行共振")

        # ══ 价格 ══
        if 5 <= price <= 60:
            delta += 5; reasons.append(f"+5 价{price:.2f}合适")
        elif price > 100:
            delta -= 10; reasons.append(f"-10 价{price:.2f}过高")

        # ══ 否决条件 ══
        if tf.consec_limit >= 4:
            vetoes.append(f"{tf.consec_limit}连板，接力风险过大")
        if tf.chg_5d > 50:
            vetoes.append(f"5日涨{tf.chg_5d}%，高位不接")
        if tf.daily_short_arrangement and sig != "floor_to_sky":
            vetoes.append("空头排列且非反包")

        # ══ regime 协同 ══
        if regime == "BULL":
            delta += 8; reasons.append("+8 牛市一进二胜率高")
        elif regime == "CRASH":
            vetoes.append("CRASH 不参与一进二")

        if news.recent_loser:
            delta -= 20; reasons.append("-20 近期亏损票")

        return self._finalize(candidate, delta, reasons, vetoes,
                              buy_threshold=95, skip_threshold=55)


# ──────────────────────────────────────────────────────────
# 风格 3：章盟主 —— 大阳追高 / 月线新高 / 主升段
# ──────────────────────────────────────────────────────────

class ZhangJiahuStyle(YouziStyle):
    name = "zhang_jiahu"
    display_name = "章盟主·大阳追高"
    description = "月线突破、主升段起点、大阳线追高，不做横盘，只做趋势"

    def score(self, candidate, tf, news, regime="SIDE"):
        delta = 0
        reasons: List[str] = []
        vetoes: List[str] = []

        sig = self._signal(candidate)
        chg = float(candidate.get("change_pct", 0) or 0)
        tor = float(candidate.get("turnover_rate", 0) or 0)
        price = float(candidate.get("price", 0) or 0)
        mc = self._market_cap_yi(candidate)
        vr = tf.vol_ratio_today or float(candidate.get("vol_ratio", 0) or 0)

        # ══ 趋势结构：多头排列+月线向上 ══
        if tf.daily_long_arrangement:
            delta += 15; reasons.append("+15 日线多头排列")
        if tf.weekly_trend == "up":
            delta += 15; reasons.append("+15 周线主升")
        if tf.monthly_trend == "up":
            delta += 20; reasons.append("+20 月线上行（章盟主最核心）")

        # ══ 52 周新高 ══
        if tf.is_52w_high:
            delta += 25; reasons.append("+25 年线新高（主升起点）")
        elif tf.pct_from_52w_high > -5:
            delta += 10; reasons.append(f"+10 距年线高点{tf.pct_from_52w_high}%")

        # ══ 今日信号 ══
        if chg >= 7 and vr >= 2:
            delta += 30; reasons.append(f"+30 大阳{chg:.1f}%+爆量{vr:.1f}倍")
        elif chg >= 5:
            delta += 15; reasons.append(f"+15 阳线{chg:.1f}%")
        elif chg < 0:
            delta -= 15; reasons.append("-15 今日收阴，不做")

        if sig == "breakout":
            delta += 20; reasons.append("+20 突破形态")
        elif sig in ("board_hit", "momentum"):
            delta += 10; reasons.append(f"+10 {sig}")
        elif sig == "trend":
            # 章盟主其实喜欢趋势但要强趋势
            if tf.daily_long_arrangement and tf.weekly_trend == "up":
                delta += 10; reasons.append("+10 强趋势股")
            else:
                delta -= 5; reasons.append("-5 弱趋势不参与")
        elif sig == "relay":
            delta += 5; reasons.append("+5 涨停接力（可叠加趋势）")

        # ══ 市值：章盟主偏大中盘 ══
        if 80 <= mc <= 400:
            delta += 15; reasons.append(f"+15 市值{mc:.0f}亿（章盟主偏好）")
        elif mc < 30:
            delta -= 20; reasons.append(f"-20 市值{mc:.0f}亿过小")

        # ══ 题材：可选但不是必须 ══
        if news.in_hot_concept:
            delta += 10; reasons.append("+10 有题材加持")

        # ══ 否决条件 ══
        if tf.daily_short_arrangement:
            vetoes.append("空头排列")
        if tf.monthly_trend == "down":
            vetoes.append("月线下行，章盟主不逆势")
        if tf.consec_limit >= 5:
            vetoes.append("高位连板，非追高结构")

        # ══ regime ══
        if regime == "BULL":
            delta += 10; reasons.append("+10 牛市追高最佳")
        elif regime in ("BEAR", "CRASH"):
            delta -= 30; reasons.append(f"-30 {regime}不做趋势追高")

        if news.recent_loser:
            delta -= 15; reasons.append("-15 近期亏损")

        return self._finalize(candidate, delta, reasons, vetoes,
                              buy_threshold=110, skip_threshold=60)


# ──────────────────────────────────────────────────────────
# 风格 4：欧阳 —— 低吸反包 / 龙回头 / 退潮反击
# ──────────────────────────────────────────────────────────

class OuyangStyle(YouziStyle):
    name = "ouyang"
    display_name = "欧阳·低吸反包"
    description = "不追高，专做龙头回马枪和反包反弹，调整到位即出手"

    def score(self, candidate, tf, news, regime="SIDE"):
        delta = 0
        reasons: List[str] = []
        vetoes: List[str] = []

        sig = self._signal(candidate)
        chg = float(candidate.get("change_pct", 0) or 0)
        tor = float(candidate.get("turnover_rate", 0) or 0)
        price = float(candidate.get("price", 0) or 0)
        mc = self._market_cap_yi(candidate)
        vr = tf.vol_ratio_today or float(candidate.get("vol_ratio", 0) or 0)

        # ══ 反包信号 ══
        if sig == "floor_to_sky":
            delta += 45; reasons.append("+45 地天板反包（欧阳招牌）")
        elif sig == "sky_floor_sky":
            delta += 55; reasons.append("+55 天地天板，超强反转")
        elif sig == "relay" and chg > 3:
            delta += 15; reasons.append("+15 温和反包")

        # ══ 调整到位（龙回头）══
        # 关键指标：从高点回撤 15-30%、跌近 MA20 / MA60、RSI 低位
        if tf.pct_from_52w_high < -15 and tf.pct_from_52w_high > -40:
            delta += 20; reasons.append(f"+20 从高点回撤{abs(tf.pct_from_52w_high):.0f}%（龙回头位置）")
        elif tf.pct_from_52w_high < -40:
            delta -= 15; reasons.append(f"-15 回撤{abs(tf.pct_from_52w_high):.0f}%过深（主跌未止）")

        # 贴近 MA20/MA60
        if tf.daily_ma20 and abs((tf.last_close - tf.daily_ma20) / tf.daily_ma20 * 100) < 2:
            delta += 15; reasons.append("+15 贴 MA20 支撑")
        if tf.daily_ma60 and 0 > (tf.last_close - tf.daily_ma60) / tf.daily_ma60 * 100 > -3:
            delta += 10; reasons.append("+10 刚下穿 MA60（超跌接力）")

        # 周线 RSI 低位
        if tf.weekly_rsi < 40:
            delta += 15; reasons.append(f"+15 周线 RSI {tf.weekly_rsi}（超卖）")
        elif tf.weekly_rsi > 70:
            delta -= 20; reasons.append(f"-20 周线 RSI {tf.weekly_rsi}（超买）")

        # ══ 量能：反包要缩量调整 + 反弹日放量 ══
        if tf.vol_ratio_5d < 0.8:
            delta += 10; reasons.append(f"+10 近5日均量{tf.vol_ratio_5d}（缩量蓄势）")
        if vr > 1.5 and chg > 2:
            delta += 15; reasons.append(f"+15 反弹日放量{vr:.1f}倍")

        # ══ 市值 / 价格 偏好 ══
        if 20 <= mc <= 150:
            delta += 10; reasons.append(f"+10 市值{mc:.0f}亿（欧阳甜蜜区）")
        if 3 <= price <= 30:
            delta += 5; reasons.append(f"+5 价{price:.2f}适合低吸")

        # ══ 题材回流 ══
        if news.in_hot_concept:
            delta += 12; reasons.append("+12 题材回流低吸")

        # ══ 否决 ══
        if tf.consec_limit >= 2:
            vetoes.append(f"{tf.consec_limit}连板，非低吸结构")
        if sig == "trend" and tf.daily_long_arrangement:
            vetoes.append("已经多头排列，错过低吸窗口")
        if tf.is_52w_high:
            vetoes.append("52周新高，不是低吸位置")
        if regime == "CRASH":
            vetoes.append("CRASH 继续下跌，不接飞刀")

        # ══ regime ══
        if regime == "SIDE":
            delta += 8; reasons.append("+8 震荡市适合低吸")
        elif regime == "BULL":
            delta += 5; reasons.append("+5 牛市回踩低吸也有效")

        if news.recent_loser:
            delta -= 30; reasons.append("-30 近期亏损（低吸最忌连吃面）")

        return self._finalize(candidate, delta, reasons, vetoes,
                              buy_threshold=85, skip_threshold=45)


# ──────────────────────────────────────────────────────────
# 注册表 + 聚合器
# ──────────────────────────────────────────────────────────

STYLE_REGISTRY: Dict[str, YouziStyle] = {
    ChenXiaoqunStyle.name: ChenXiaoqunStyle(),
    ZhaoLaogeStyle.name:   ZhaoLaogeStyle(),
    ZhangJiahuStyle.name:  ZhangJiahuStyle(),
    OuyangStyle.name:      OuyangStyle(),
}


def list_styles() -> List[str]:
    return list(STYLE_REGISTRY.keys())


def get_style(name: str) -> Optional[YouziStyle]:
    return STYLE_REGISTRY.get(name)


def score_all_styles(
    candidate: Dict,
    tf: TimeframeContext,
    news: NewsContext,
    regime: str = "SIDE",
    active: Optional[List[str]] = None,
) -> Dict[str, YouziStyleResult]:
    """对单个 candidate 跑所有（或激活）游资风格，返回按 style_name 索引的结果。"""
    active = active or list_styles()
    out: Dict[str, YouziStyleResult] = {}
    for name in active:
        style = STYLE_REGISTRY.get(name)
        if not style:
            continue
        try:
            out[name] = style.score(candidate, tf, news, regime)
        except Exception as e:
            logger.warning(f"[youzi] {name} score failed for {candidate.get('code')}: {e}")
            out[name] = YouziStyleResult(
                style_name=name,
                display_name=STYLE_REGISTRY[name].display_name,
                base_score=int(candidate.get("tech_score", 0) or 0),
                reasons=[], vetoes=[f"exception:{e}"],
                verdict="veto",
            )
    return out


def aggregate_verdict(
    per_style: Dict[str, YouziStyleResult],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """跨风格聚合：投票 + 加权平均分。

    - buy_votes:     verdict==buy 的风格数
    - weighted_score: sum(final_score * weight) / sum(weight)（忽略 veto）
    - consensus:     >=2 家 buy 且至少 1 家 final>=90 → consensus=True
    - vetoed_by:     哪些风格 veto 了
    """
    weights = weights or {k: 1.0 for k in per_style}
    buy, watch, skip, veto = [], [], [], []
    weighted_sum, weight_total = 0.0, 0.0
    for name, r in per_style.items():
        if r.verdict == "buy":
            buy.append(name)
        elif r.verdict == "watch":
            watch.append(name)
        elif r.verdict == "skip":
            skip.append(name)
        else:
            veto.append(name)
        if r.verdict != "veto":
            w = float(weights.get(name, 1.0))
            weighted_sum += r.final_score * w
            weight_total += w

    avg = weighted_sum / weight_total if weight_total > 0 else 0
    consensus = len(buy) >= 2 and any(per_style[b].final_score >= 90 for b in buy)
    best_style = max(per_style.items(), key=lambda kv: kv[1].final_score)[0] if per_style else ""

    return {
        "buy_votes": buy,
        "watch_votes": watch,
        "skip_votes": skip,
        "vetoed_by": veto,
        "weighted_score": round(avg, 1),
        "consensus": consensus,
        "best_style": best_style,
    }
