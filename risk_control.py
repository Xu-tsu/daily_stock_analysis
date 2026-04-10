# -*- coding: utf-8 -*-
"""
risk_control.py - 交易风控模块

基于用户近几个月真实交割单的特征，这个模块不再把
“亏损 5% 必须无条件清仓”当作唯一答案，而是把它降级为
高优先级风险复核线：

1. A 股 T+1 追高依旧严格限制
2. 亏损 5% 进入强复核，不再自动一刀切
3. 只有市场、板块、资金和仓位都支持时，才允许保留底仓做 T
4. 深度亏损、弱势板块、宏观突发、量化砸盘环境下仍以退出为先
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.trading_calendar import count_stock_trading_days

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 核心参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 止损 / 止盈（可通过 .env 同名变量覆盖）
import os as _os
HARD_STOP_LOSS_PCT = float(_os.getenv("HARD_STOP_LOSS_PCT", "-3"))
FORCE_EXIT_LOSS_PCT = float(_os.getenv("FORCE_EXIT_LOSS_PCT", "-5"))
TRAILING_STOP_TRIGGER = float(_os.getenv("TRAILING_STOP_TRIGGER", "3"))
TRAILING_STOP_PCT = float(_os.getenv("TRAILING_STOP_PCT", "2"))
TAKE_PROFIT_HALF_PCT = float(_os.getenv("TAKE_PROFIT_HALF_PCT", "5"))
TAKE_PROFIT_FULL_PCT = float(_os.getenv("TAKE_PROFIT_FULL_PCT", "12"))

# 持仓天数（龙头打板：快进快出）
HOLD_DAYS_WARNING = 1              # 超过 1 天开始降级看待
HOLD_DAYS_FORCE_REVIEW = 2         # 超过 2 天必须复盘
HOLD_DAYS_MAX = 2                  # 超过 2 天且盈利不足，优先退出

# 仓位管理（龙头打板：集中火力）
MAX_SINGLE_POSITION_PCT = 95.0     # 单只ALL-IN
MAX_POSITIONS = 1                  # 同时只持1只（龙头战法核心）
MAX_DAILY_NEW_BUYS = 1             # 每日最多1只

# 补仓约束
FORBID_AVERAGING_DOWN = True       # 默认禁止对亏损股补仓
MAX_SAME_STOCK_TRADES = 3          # 同一只股票 30 天内最大交易次数

# 交易频率
MAX_DAILY_TRADES = 8

# T+1 追高风控
CHASE_HIGH_WARN_PCT = 9.5
CHASE_HIGH_BLOCK_PCT = 20.0          # 龙头打板：涨停板可以追
CHASE_HIGH_PENALTY = -20
LATE_SESSION_CHASE_BLOCK = True


class RiskAlert:
    """单条风控警报。"""

    def __init__(self, level: str, code: str, name: str, message: str, action: str):
        self.level = level  # critical / warning / info
        self.code = code
        self.name = name
        self.message = message
        self.action = action  # force_sell / reduce_half / review / hold

    def __repr__(self) -> str:
        return f"[{self.level}] {self.name}({self.code}): {self.message} -> {self.action}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "name": self.name,
            "message": self.message,
            "action": self.action,
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_label(value: Any, default: str = "neutral") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    return text or default


def _get_hold_days(holding: Dict[str, Any], today: datetime) -> Optional[int]:
    buy_date_str = holding.get("buy_date", "")
    if not buy_date_str:
        return None
    return count_stock_trading_days(
        str(holding.get("code", "") or ""),
        buy_date_str,
        today,
        default_market="cn",
    )


def _extract_sector_confirmation(
    holding: Dict[str, Any],
    market_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    sector_name = str(holding.get("sector", "") or "").strip()
    default = {
        "sector": sector_name,
        "confirmed": False,
        "strength": "weak",
        "rank": None,
        "change_pct": 0.0,
        "main_net": 0.0,
    }
    if not market_context:
        return default

    sector_confirmation = market_context.get("sector_confirmation")
    if isinstance(sector_confirmation, dict):
        confirmed_sector = str(sector_confirmation.get("sector", "") or "").strip()
        if sector_name and confirmed_sector == sector_name:
            return {
                "sector": confirmed_sector,
                "confirmed": bool(sector_confirmation.get("confirmed")),
                "strength": _normalize_label(sector_confirmation.get("strength"), "weak"),
                "rank": sector_confirmation.get("rank"),
                "change_pct": _safe_float(sector_confirmation.get("change_pct")),
                "main_net": _safe_float(sector_confirmation.get("main_net")),
            }

    for rank, sector in enumerate(market_context.get("top_sectors") or [], start=1):
        name = str(sector.get("name", "") or "").strip()
        if not sector_name or name != sector_name:
            continue
        change_pct = _safe_float(sector.get("change_pct"))
        main_net = _safe_float(sector.get("main_net"))
        confirmed = main_net > 0 and change_pct >= 0.8
        strength = "strong" if confirmed and rank <= 3 and main_net >= 5000 else "medium" if confirmed else "weak"
        return {
            "sector": name,
            "confirmed": confirmed,
            "strength": strength,
            "rank": rank,
            "change_pct": change_pct,
            "main_net": main_net,
        }

    return default


def _market_supports_base_position(
    holding: Dict[str, Any],
    market_context: Optional[Dict[str, Any]],
    hold_days: Optional[int],
) -> tuple[bool, List[str], Dict[str, Any]]:
    blockers: List[str] = []
    sector_confirmation = _extract_sector_confirmation(holding, market_context)

    if not market_context:
        blockers.append("缺少市场总览确认")
        return False, blockers, sector_confirmation

    market_bias = _normalize_label(market_context.get("bias"), "neutral")
    market_score = _safe_float(market_context.get("market_score"))
    macro_risk = _normalize_label(market_context.get("macro_risk_level"), "medium")
    quant_pressure = _normalize_label(
        (market_context.get("quant_pressure") or {}).get("signal"),
        "medium",
    )
    hot_money_signal = _normalize_label(
        (market_context.get("hot_money_probe") or {}).get("signal"),
        "quiet",
    )
    auction_direction = _normalize_label(
        (market_context.get("opening_auction") or {}).get("direction"),
        "unavailable",
    )

    if market_bias != "positive":
        blockers.append(f"市场偏{market_bias}")
    if market_score < 0.4:
        blockers.append(f"市场分值不足({market_score:.2f})")
    if macro_risk != "low":
        blockers.append(f"宏观/消息风险偏{macro_risk}")
    if quant_pressure == "high":
        blockers.append("量化砸盘压力偏高")
    if not sector_confirmation.get("confirmed"):
        blockers.append("所属板块缺少强确认")
    if hot_money_signal not in {"active", "constructive"}:
        blockers.append("题材试板资金不够明确")
    if auction_direction == "weak":
        blockers.append("集合竞价偏弱")
    if hold_days is not None and hold_days >= HOLD_DAYS_WARNING:
        blockers.append(f"持仓{hold_days}个交易日，已超短线优势窗口")

    return len(blockers) == 0, blockers, sector_confirmation


def _build_loss_alert(
    holding: Dict[str, Any],
    pnl_pct: float,
    hold_days: Optional[int],
    market_context: Optional[Dict[str, Any]],
) -> Optional[RiskAlert]:
    if pnl_pct > HARD_STOP_LOSS_PCT:
        return None

    code = holding.get("code", "")
    name = holding.get("name", "")

    if pnl_pct <= FORCE_EXIT_LOSS_PCT:
        return RiskAlert(
            "critical",
            code,
            name,
            f"亏损{pnl_pct:.1f}%，已超过强制退出线({FORCE_EXIT_LOSS_PCT}%)，不再保留底仓博弈。",
            "force_sell",
        )

    allow_base, blockers, sector_confirmation = _market_supports_base_position(
        holding=holding,
        market_context=market_context,
        hold_days=hold_days,
    )

    if allow_base:
        sector_name = sector_confirmation.get("sector") or "所属板块"
        return RiskAlert(
            "warning",
            code,
            name,
            f"亏损{pnl_pct:.1f}%，到达风险复核线({HARD_STOP_LOSS_PCT}%)，"
            f"但市场偏强且{sector_name}仍有确认，可减仓后保留底仓做T，禁止继续放大风险。",
            "review",
        )

    market_bias = _normalize_label((market_context or {}).get("bias"), "neutral")
    macro_risk = _normalize_label((market_context or {}).get("macro_risk_level"), "medium")
    quant_pressure = _normalize_label(
        ((market_context or {}).get("quant_pressure") or {}).get("signal"),
        "medium",
    )
    level = (
        "critical"
        if pnl_pct <= -6.5 or market_bias == "negative" or macro_risk == "high" or quant_pressure == "high"
        else "warning"
    )
    reason_text = "；".join(blockers[:3]) if blockers else "市场与板块未形成强确认"
    return RiskAlert(
        level,
        code,
        name,
        f"亏损{pnl_pct:.1f}%，且{reason_text}，优先退出或至少大幅降仓，不保留幻想仓位。",
        "force_sell",
    )


def check_stop_loss(
    holdings: List[Dict[str, Any]],
    market_context: Optional[Dict[str, Any]] = None,
) -> List[RiskAlert]:
    """检查所有持仓的止损 / 止盈 / 持仓天数。"""

    alerts: List[RiskAlert] = []
    today = datetime.now()

    for holding in holdings:
        code = holding.get("code", "")
        name = holding.get("name", "")
        cost = _safe_float(holding.get("cost_price"))
        current = _safe_float(holding.get("current_price"))
        pnl_pct = _safe_float(holding.get("pnl_pct"))

        if cost > 0 and current > 0:
            pnl_pct = (current - cost) / cost * 100

        hold_days = _get_hold_days(holding, today)

        loss_alert = _build_loss_alert(
            holding=holding,
            pnl_pct=pnl_pct,
            hold_days=hold_days,
            market_context=market_context,
        )
        if loss_alert is not None:
            alerts.append(loss_alert)
            continue

        if pnl_pct >= TAKE_PROFIT_FULL_PCT:
            alerts.append(
                RiskAlert(
                    "warning",
                    code,
                    name,
                    f"盈利{pnl_pct:.1f}%，建议分批兑现，避免高位利润回吐。",
                    "force_sell",
                )
            )
        elif pnl_pct >= TAKE_PROFIT_HALF_PCT:
            alerts.append(
                RiskAlert(
                    "warning",
                    code,
                    name,
                    f"盈利{pnl_pct:.1f}%，建议至少减仓一半锁定利润。",
                    "reduce_half",
                )
            )
        elif pnl_pct >= TRAILING_STOP_TRIGGER:
            alerts.append(
                RiskAlert(
                    "info",
                    code,
                    name,
                    f"盈利{pnl_pct:.1f}%，已进入保本保护区，回落到成本附近要果断收手。",
                    "hold",
                )
            )

        if hold_days is None:
            continue
        if hold_days >= HOLD_DAYS_MAX and pnl_pct < 5:
            alerts.append(
                RiskAlert(
                    "critical",
                    code,
                    name,
                    f"持仓{hold_days}个交易日且盈利不足5%，已超短线效率边界，优先退出。",
                    "force_sell",
                )
            )
        elif hold_days >= HOLD_DAYS_FORCE_REVIEW:
            alerts.append(
                RiskAlert(
                    "warning",
                    code,
                    name,
                    f"持仓{hold_days}个交易日，必须复盘是否继续持有。",
                    "review",
                )
            )
        elif hold_days >= HOLD_DAYS_WARNING:
            alerts.append(
                RiskAlert(
                    "info",
                    code,
                    name,
                    f"持仓{hold_days}个交易日，短线胜率开始衰减，注意别把短线做成长线。",
                    "review",
                )
            )

    return alerts


def check_buy_permission(
    code: str,
    name: str,
    holdings: List[Dict[str, Any]],
    total_asset: float = 0,
    buy_amount: float = 0,
    current_change_pct: float = 0,
    allow_averaging_down: bool = False,
) -> Dict[str, Any]:
    """检查是否允许买入。"""

    reasons: List[str] = []
    warnings: List[str] = []

    if current_change_pct >= CHASE_HIGH_BLOCK_PCT:
        reasons.append(
            f"T+1追高禁止: {name}今日已涨{current_change_pct:.1f}%（超过{CHASE_HIGH_BLOCK_PCT}%），"
            "A股今天买入明天才能卖，追高后次日被砸很难处理。"
        )
    elif current_change_pct >= CHASE_HIGH_WARN_PCT:
        now = datetime.now()
        if LATE_SESSION_CHASE_BLOCK and now.hour >= 14 and now.minute >= 30:
            reasons.append(
                f"尾盘追高禁止: {name}已涨{current_change_pct:.1f}%，"
                "14:30 后再追容易承受隔夜低开的被动风险。"
            )
        else:
            warnings.append(
                f"追高警告: {name}今日已涨{current_change_pct:.1f}%，"
                "更适合等回踩而不是在冲高途中接力。"
            )

    if FORBID_AVERAGING_DOWN and not allow_averaging_down:
        for holding in holdings:
            if holding.get("code") != code:
                continue
            pnl_pct = _safe_float(holding.get("pnl_pct"))
            cost = _safe_float(holding.get("cost_price"))
            current = _safe_float(holding.get("current_price"))
            if cost > 0 and current > 0:
                pnl_pct = (current - cost) / cost * 100
            if pnl_pct < 0:
                reasons.append(
                    f"默认不补仓亏损股: {name}({code})当前亏损{pnl_pct:.1f}%，"
                    "只有板块回流型 / 龙头错杀型等更严格条件满足时才允许例外。"
                )

    if total_asset > 0 and buy_amount > 0:
        existing_value = sum(
            _safe_float(holding.get("market_value"))
            for holding in holdings
            if holding.get("code") == code
        )
        new_pct = (existing_value + buy_amount) / total_asset * 100
        if new_pct > MAX_SINGLE_POSITION_PCT:
            reasons.append(
                f"仓位超限: 买入后 {name} 占比 {new_pct:.1f}% ，"
                f"超过单只上限 {MAX_SINGLE_POSITION_PCT}% 。"
            )

    held_codes = {
        holding.get("code")
        for holding in holdings
        if _safe_float(holding.get("shares")) > 0
    }
    if code not in held_codes and len(held_codes) >= MAX_POSITIONS:
        reasons.append(
            f"持仓数量超限: 当前已持有 {len(held_codes)} 只，"
            f"达到上限 {MAX_POSITIONS} 只。"
        )

    try:
        from trade_journal import _conn

        conn = _conn()
        recent = conn.execute(
            """
            SELECT COUNT(*) as cnt FROM trade_log
            WHERE code = ? AND trade_date >= date('now', '-30 days')
            """,
            (code,),
        ).fetchone()
        conn.close()
        if recent and recent["cnt"] >= MAX_SAME_STOCK_TRADES * 2:
            warnings.append(
                f"频繁交易提醒: {name} 近30天已交易 {recent['cnt']} 次，注意减少摩擦成本。"
            )
    except Exception:
        pass

    try:
        from trade_journal import _conn

        conn = _conn()
        today = datetime.now().strftime("%Y-%m-%d")
        today_buys = conn.execute(
            """
            SELECT COUNT(DISTINCT code) as cnt FROM trade_log
            WHERE trade_type = 'buy' AND trade_date = ?
            """,
            (today,),
        ).fetchone()
        conn.close()
        if today_buys and today_buys["cnt"] >= MAX_DAILY_NEW_BUYS:
            warnings.append(
                f"今日已新开 {today_buys['cnt']} 只标的，建议先停止扩散持仓。"
            )
    except Exception:
        pass

    return {
        "allowed": len(reasons) == 0,
        "reasons": reasons,
        "warnings": warnings,
    }


def get_position_sizing(
    total_asset: float,
    current_positions: int,
    signal_strength: str = "medium",
) -> float:
    """根据信号强度和当前持仓数计算建议买入金额。"""

    single_max = total_asset * MAX_SINGLE_POSITION_PCT / 100

    if signal_strength == "strong":
        base = single_max * 0.8
    elif signal_strength == "medium":
        base = single_max * 0.5
    else:
        base = single_max * 0.3

    if current_positions >= 4:
        base *= 0.5
    elif current_positions >= 3:
        base *= 0.7

    return round(min(base, single_max), 0)


def format_risk_alerts(alerts: List[RiskAlert]) -> str:
    """格式化风控警报。"""

    if not alerts:
        return "当前持仓风控状态正常"

    critical = [alert for alert in alerts if alert.level == "critical"]
    warning = [alert for alert in alerts if alert.level == "warning"]
    info = [alert for alert in alerts if alert.level == "info"]

    lines: List[str] = []
    if critical:
        lines.append("** 紧急（需立刻处理）**")
        for alert in critical:
            lines.append(f"  {alert.name}({alert.code}): {alert.message}")
            lines.append(f"    操作: {alert.action}")
    if warning:
        lines.append("\n** 警告 **")
        for alert in warning:
            lines.append(f"  {alert.name}({alert.code}): {alert.message}")
    if info:
        lines.append("\n** 提示 **")
        for alert in info:
            lines.append(f"  {alert.name}({alert.code}): {alert.message}")

    return "\n".join(lines)


def build_adaptive_trading_rules(
    market_context: Optional[Dict[str, Any]] = None,
    current_holding: Optional[Dict[str, Any]] = None,
) -> str:
    """构造供 LLM / Agent 使用的自适应高胜率规则。"""

    lines = [
        "## 基于真实交割单与盘中环境的高胜率规则",
        "",
        "### 数据基线",
        "- 已分析用户 2025.12-2026.03 的真实交易样本：胜率不低，但主要问题是赚得少、亏得深。",
        "- 优势在短线执行：0-1 天表现最好，超过 3 天后胜率和赔率都明显下降。",
        "",
        "### 规则优先级（从高到低）",
        "1. A股 T+1 追高禁令优先级最高：当日涨幅过大、尤其尾盘冲高，默认不追。",
        "2. 亏损 5% 是风险复核线，不再机械一刀切。",
        "   - 若市场偏弱、板块不确认、特朗普/关税等突发升级、量化砸盘压力偏高：优先清仓或大幅减仓。",
        "   - 只有市场偏强、主线板块资金仍在、题材试板资金活跃、且仓位不重时，才允许减仓后保留底仓做T。",
        "3. 亏损股默认不补仓。",
        "   - 仅在“板块回流型补仓”或“龙头错杀型补仓”条件同时满足时，才允许小额试探。",
        "4. 调仓先处理弱势和高风险仓位，再考虑弱转强切换。",
        "5. 看市场不能只看指数，必须联合判断：",
        "   - 新闻情绪与宏观突发（尤其特朗普、关税、贸易冲突）",
        "   - 09:15-09:25 集合竞价方向与开盘承接",
        "   - 热门题材滚动资金是否持续回流",
        "   - 游资是否只是试探拉板，还是有板块跟随",
        "   - 是否出现量化砸盘式的指数弱、承接差、分时共振下杀",
        "6. 盈利 8% 以上优先考虑锁利，不把短线利润无谓回吐。",
        "7. 持仓超过 3 天后，若没有继续走强证据，要主动降低预期。",
    ]

    lines.extend(
        [
            "",
            "### 逆向执行偏好",
            "8. 更高胜率的买点通常来自大盘下跌、分歧加大、情绪降温时，而不是全场一致兴奋时追进去。",
            "   - 前提是主线板块仍有资金确认、个股只是回踩支撑，而不是趋势已经走坏。",
            "9. 红盘冲高、题材一致高潮、讨论度极高时，优先考虑分批兑现或弱转强切换。",
            "   - 短线卖点更强调卖在人声鼎沸，而不是等热度退潮后被动回吐利润。",
        ]
    )

    if market_context:
        top_sectors = market_context.get("top_sectors") or []
        top_sector_text = " / ".join(
            f"{sector.get('name', '')} { _safe_float(sector.get('change_pct')):+.1f}%"
            for sector in top_sectors[:3]
            if sector.get("name")
        ) or "暂无"
        opening_auction = market_context.get("opening_auction") or {}
        hot_money = market_context.get("hot_money_probe") or {}
        quant_pressure = market_context.get("quant_pressure") or {}
        lines.extend(
            [
                "",
                "### 当前市场约束",
                f"- 市场偏向: {market_context.get('bias_label') or market_context.get('bias') or '未知'}",
                f"- 市场分值: {_safe_float(market_context.get('market_score')):.2f}",
                f"- 宏观/消息风险: {market_context.get('macro_risk_level', '未知')}",
                f"- 热门板块: {top_sector_text}",
                f"- 集合竞价: {opening_auction.get('direction', 'unavailable')}",
                f"- 游资/题材试板: {hot_money.get('signal', 'quiet')}",
                f"- 量化砸盘压力: {quant_pressure.get('signal', 'medium')}",
            ]
        )

    if current_holding:
        sector_name = str(current_holding.get("sector", "") or "").strip()
        pnl_pct = _safe_float(current_holding.get("pnl_pct"))
        lines.extend(
            [
                "",
                "### 当前标的约束",
                f"- 当前持仓盈亏: {pnl_pct:+.2f}%",
                f"- 所属板块: {sector_name or '未知'}",
                "- 如果该标的不在主线强确认里，就算跌到低位，也不能把“想做T”当成继续死扛的理由。",
            ]
        )

    return "\n".join(lines)


TRADING_RULES_FOR_LLM = build_adaptive_trading_rules()
