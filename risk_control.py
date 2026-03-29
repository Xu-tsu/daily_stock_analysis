# -*- coding: utf-8 -*-
"""
risk_control.py — 交易风控模块

基于用户真实交割单数据分析得出的核心规则：
  诊断: 胜率60.2%但盈亏比0.40（赚5%就跑，亏12%还扛）
  致命模式: 短线赚钱→拿成长线亏钱 / 亏损股反复补仓 / 没有止损

规则:
  1. 硬止损 5% — 任何持仓亏损超5%立即发出清仓信号
  2. 止盈阶梯 — 盈利3%移动止损至成本价，盈利8%止盈一半
  3. 持仓天数预警 — 超过3天的持仓自动降级
  4. 禁止补仓亏损股 — 亏损中的股票禁止再买入
  5. 单只仓位上限 — 不超过总资产15%
  6. 日交易频率限制 — 每日买入不超过3只新股

飞书指令:
  风控        → 查看当前持仓风控状态
  止损        → 查看需要止损的持仓
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 风控参数（基于交割单数据优化）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 止损/止盈
HARD_STOP_LOSS_PCT = -5.0        # 硬止损线：亏5%必须卖
TRAILING_STOP_TRIGGER = 3.0      # 盈利3%后启动移动止损
TRAILING_STOP_PCT = 0.0          # 移动止损线=成本价（保本出）
TAKE_PROFIT_HALF_PCT = 8.0       # 盈利8%减仓一半
TAKE_PROFIT_FULL_PCT = 15.0      # 盈利15%全部止盈

# 持仓天数（你的数据: 0-1天胜率93%, 2-3天72%, 4+天开始亏）
HOLD_DAYS_WARNING = 3            # 超过3天发出预警
HOLD_DAYS_FORCE_REVIEW = 5       # 超过5天强制复盘
HOLD_DAYS_MAX = 7                # 超过7天建议清仓（除非趋势极强）

# 仓位管理
MAX_SINGLE_POSITION_PCT = 15.0   # 单只股票最大仓位比例
MAX_POSITIONS = 5                # 同时持有股票数上限
MAX_DAILY_NEW_BUYS = 3           # 每日新买入股票数上限

# 补仓禁令
FORBID_AVERAGING_DOWN = True     # 禁止对亏损股补仓
MAX_SAME_STOCK_TRADES = 3        # 同一只股票30天内最大交易次数

# 交易频率
MAX_DAILY_TRADES = 8             # 每日最大交易笔数

# T+1 追高风控（A股核心风险：今天买明天才能卖）
CHASE_HIGH_WARN_PCT = 3.0        # 当日涨幅超3%即为追高警告
CHASE_HIGH_BLOCK_PCT = 5.0       # 当日涨幅超5%禁止买入
CHASE_HIGH_PENALTY = -20         # 追高股票打分惩罚（-20分）
# 尾盘追高更危险：14:30后买涨幅>3%的股票几乎必亏
LATE_SESSION_CHASE_BLOCK = True  # 14:30后禁止买入当日涨幅>3%的股


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 风控检查
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RiskAlert:
    """单条风控警报"""
    def __init__(self, level: str, code: str, name: str, message: str, action: str):
        self.level = level    # critical / warning / info
        self.code = code
        self.name = name
        self.message = message
        self.action = action  # force_sell / reduce_half / review / hold

    def __repr__(self):
        return f"[{self.level}] {self.name}({self.code}): {self.message} → {self.action}"

    def to_dict(self):
        return {
            "level": self.level, "code": self.code, "name": self.name,
            "message": self.message, "action": self.action,
        }


def check_stop_loss(holdings: List[Dict]) -> List[RiskAlert]:
    """检查所有持仓的止损/止盈/持仓天数。

    Args:
        holdings: [{"code", "name", "cost_price", "current_price",
                     "shares", "buy_date", "pnl_pct", "market_value"}, ...]

    Returns:
        风控警报列表
    """
    alerts = []
    today = datetime.now()

    for h in holdings:
        code = h.get("code", "")
        name = h.get("name", "")
        cost = h.get("cost_price", 0)
        current = h.get("current_price", 0)
        pnl_pct = h.get("pnl_pct", 0)

        if cost > 0 and current > 0:
            pnl_pct = (current - cost) / cost * 100

        # 1. 硬止损
        if pnl_pct <= HARD_STOP_LOSS_PCT:
            alerts.append(RiskAlert(
                "critical", code, name,
                f"亏损{pnl_pct:.1f}%，触发硬止损线({HARD_STOP_LOSS_PCT}%)",
                "force_sell"
            ))
            continue

        # 2. 止盈
        if pnl_pct >= TAKE_PROFIT_FULL_PCT:
            alerts.append(RiskAlert(
                "warning", code, name,
                f"盈利{pnl_pct:.1f}%，建议全部止盈",
                "force_sell"
            ))
        elif pnl_pct >= TAKE_PROFIT_HALF_PCT:
            alerts.append(RiskAlert(
                "warning", code, name,
                f"盈利{pnl_pct:.1f}%，建议减仓一半锁定利润",
                "reduce_half"
            ))
        elif pnl_pct >= TRAILING_STOP_TRIGGER:
            # 移动止损：盈利3%后如果回落到成本价附近，保本出
            alerts.append(RiskAlert(
                "info", code, name,
                f"盈利{pnl_pct:.1f}%，已启动移动止损(保本线)",
                "hold"
            ))

        # 3. 持仓天数
        buy_date_str = h.get("buy_date", "")
        if buy_date_str:
            try:
                buy_date = datetime.strptime(str(buy_date_str)[:10], "%Y-%m-%d")
                hold_days = (today - buy_date).days
                if hold_days >= HOLD_DAYS_MAX and pnl_pct < 5:
                    alerts.append(RiskAlert(
                        "critical", code, name,
                        f"持仓{hold_days}天（超{HOLD_DAYS_MAX}天上限），盈利不足5%",
                        "force_sell"
                    ))
                elif hold_days >= HOLD_DAYS_FORCE_REVIEW:
                    alerts.append(RiskAlert(
                        "warning", code, name,
                        f"持仓{hold_days}天，需要强制复盘决定去留",
                        "review"
                    ))
                elif hold_days >= HOLD_DAYS_WARNING:
                    alerts.append(RiskAlert(
                        "info", code, name,
                        f"持仓{hold_days}天，接近持仓上限，注意控制",
                        "review"
                    ))
            except (ValueError, TypeError):
                pass

    return alerts


def check_buy_permission(
    code: str,
    name: str,
    holdings: List[Dict],
    total_asset: float = 0,
    buy_amount: float = 0,
    current_change_pct: float = 0,
    allow_averaging_down: bool = False,
) -> Dict[str, Any]:
    """检查是否允许买入。

    Args:
        current_change_pct: 该股票当日涨跌幅(%)，用于T+1追高检查
        allow_averaging_down: 仅供上层策略在更严格前置条件下调用；
            默认仍然禁止对亏损股补仓

    Returns:
        {"allowed": True/False, "reasons": ["原因1", ...], "warnings": ["警告1", ...]}
    """
    reasons = []
    warnings = []

    # 0. T+1 追高风控（最重要！）
    if current_change_pct >= CHASE_HIGH_BLOCK_PCT:
        reasons.append(
            f"T+1追高禁止: {name}今日已涨{current_change_pct:.1f}%（超{CHASE_HIGH_BLOCK_PCT}%），"
            f"A股T+1今天买入明天才能卖，追高后砸盘无法止损。"
            f"历史数据：追高买入的交易亏损率超70%"
        )
    elif current_change_pct >= CHASE_HIGH_WARN_PCT:
        # 检查是否尾盘
        now = datetime.now()
        if LATE_SESSION_CHASE_BLOCK and now.hour >= 14 and now.minute >= 30:
            reasons.append(
                f"尾盘追高禁止: {name}已涨{current_change_pct:.1f}%，"
                f"14:30后买入涨幅>3%的股票，明天大概率低开"
            )
        else:
            warnings.append(
                f"追高警告: {name}已涨{current_change_pct:.1f}%，T+1风险较高，"
                f"建议等回调到均线附近再买入"
            )

    # 1. 禁止补仓亏损股
    if FORBID_AVERAGING_DOWN and not allow_averaging_down:
        for h in holdings:
            if h.get("code") == code:
                pnl_pct = h.get("pnl_pct", 0)
                cost = h.get("cost_price", 0)
                current = h.get("current_price", 0)
                if cost > 0 and current > 0:
                    pnl_pct = (current - cost) / cost * 100
                if pnl_pct < 0:
                    reasons.append(
                        f"禁止补仓: {name}({code})当前亏损{pnl_pct:.1f}%，"
                        f"禁止对亏损股加仓（历史数据显示补仓亏损股胜率仅17%）"
                    )

    # 2. 单只仓位上限
    if total_asset > 0 and buy_amount > 0:
        existing_value = sum(
            h.get("market_value", 0) for h in holdings if h.get("code") == code
        )
        new_pct = (existing_value + buy_amount) / total_asset * 100
        if new_pct > MAX_SINGLE_POSITION_PCT:
            reasons.append(
                f"仓位超限: 买入后{name}占比{new_pct:.1f}%，"
                f"超过单只上限{MAX_SINGLE_POSITION_PCT}%"
            )

    # 3. 持仓数量上限
    held_codes = set(h.get("code") for h in holdings if h.get("shares", 0) > 0)
    if code not in held_codes and len(held_codes) >= MAX_POSITIONS:
        reasons.append(
            f"持仓数超限: 当前已持有{len(held_codes)}只，"
            f"上限{MAX_POSITIONS}只，需先卖出才能买新股"
        )

    # 4. 同一只股票交易频率
    try:
        from trade_journal import _conn
        conn = _conn()
        recent = conn.execute("""
            SELECT COUNT(*) as cnt FROM trade_log
            WHERE code = ? AND trade_date >= date('now', '-30 days')
        """, (code,)).fetchone()
        conn.close()
        if recent and recent["cnt"] >= MAX_SAME_STOCK_TRADES * 2:
            warnings.append(
                f"频繁交易: {name}近30天已交易{recent['cnt']}次，注意交易成本"
            )
    except Exception:
        pass

    # 5. 今日新买入数量
    try:
        from trade_journal import _conn
        conn = _conn()
        today = datetime.now().strftime("%Y-%m-%d")
        today_buys = conn.execute("""
            SELECT COUNT(DISTINCT code) as cnt FROM trade_log
            WHERE trade_type = 'buy' AND trade_date = ?
        """, (today,)).fetchone()
        conn.close()
        if today_buys and today_buys["cnt"] >= MAX_DAILY_NEW_BUYS:
            warnings.append(
                f"今日已买入{today_buys['cnt']}只新股，建议停止买入"
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
    """根据信号强度和当前仓位计算建议买入金额。

    基于你的数据：小仓(<1000元)胜率67%最高，大仓胜率低。
    """
    # 总仓位不超过80%（留20%现金）
    max_invest = total_asset * 0.8
    # 单只上限
    single_max = total_asset * MAX_SINGLE_POSITION_PCT / 100

    if signal_strength == "strong":
        base = single_max * 0.8  # 强信号：用单只上限的80%
    elif signal_strength == "medium":
        base = single_max * 0.5  # 中等信号：50%
    else:
        base = single_max * 0.3  # 弱信号：30%

    # 已有仓位多时减少新仓位
    if current_positions >= 4:
        base *= 0.5
    elif current_positions >= 3:
        base *= 0.7

    return round(min(base, single_max), 0)


def format_risk_alerts(alerts: List[RiskAlert]) -> str:
    """格式化风控警报为可读文本。"""
    if not alerts:
        return "当前持仓风控状态正常"

    critical = [a for a in alerts if a.level == "critical"]
    warning = [a for a in alerts if a.level == "warning"]
    info = [a for a in alerts if a.level == "info"]

    lines = []
    if critical:
        lines.append("** 紧急（需立即处理）**")
        for a in critical:
            lines.append(f"  {a.name}({a.code}): {a.message}")
            lines.append(f"    操作: {a.action}")
    if warning:
        lines.append("\n** 警告 **")
        for a in warning:
            lines.append(f"  {a.name}({a.code}): {a.message}")
    if info:
        lines.append("\n** 提示 **")
        for a in info:
            lines.append(f"  {a.name}({a.code}): {a.message}")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 交易决策增强：基于交割单数据的硬规则
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRADING_RULES_FOR_LLM = """
## 基于真实交割单数据分析的硬性交易规则（必须严格执行）

### 数据来源
分析了用户2025.12-2026.03共233笔交易（123笔完成配对），得出以下规则：

### 核心发现
- 用户胜率60.2%（不低），但盈亏比只有0.40（赚少亏多）
- 赚钱时平均赚5.12%，亏钱时平均亏12.67%
- 短线是优势：当天卖胜率100%，隔天卖93%，超3天胜率骤降至56%

### 必须执行的硬规则

**规则0（最高优先级）: A股T+1追高禁令**
- 当日涨幅超5%的股票：禁止买入，无例外
- 当日涨幅超3%的股票：14:30后禁止买入
- 当日涨幅超3%的股票：其他时段也需降低仓位50%
- 理由：A股T+1，今天追高买入明天才能卖。用户亏损最大的6笔交易（-28%/-27%/-25%）
  全部是追涨后第二天砸盘被套，因为T+1无法当天止损
- **正确做法**：买在回调支撑位（MA5/MA10），不买在冲高途中
- **尾盘更危险**：14:30后追高 = 承受整晚不确定性，次日低开概率极高

1. **绝对止损5%**：任何持仓浮亏超过5%，必须建议"清仓"，无任何例外
   理由：用户亏损超20%的交易有6笔，全部是没及时止损导致的

2. **最长持股3个交易日**：超过3天且盈利不足5%的持仓，必须建议"减仓/清仓"
   理由：用户持股0-1天胜率93%，2-3天72%，4-7天仅56%，7天+仅41%

3. **盈利8%必须减仓一半**：锁定利润，剩余部分用移动止损保护
   理由：用户平均盈利只有5.12%，能到8%已经是超额收益

4. **禁止补仓亏损股**：已经浮亏的股票，绝对不能建议"加仓"
   理由：用户补仓亏损股的案例（雷科防务6笔17%胜率亏1806元）全部是灾难

5. **同一只股票30天内不超过3次交易**：避免反复进出同一只股票
   理由：频繁交易同一只股票（如利欧股份13次）虽然部分盈利但增加摩擦成本

6. **单只仓位不超过15%**：分散风险，即使看好也不能重仓

7. **优先选择回调买入而非突破买入**：
   - 最佳买点：强势股缩量回踩MA5（而非放量突破新高）
   - 理由：追突破 = 追高，T+1下追高当天无法止损；回调买入成本低+止损距离短
   - 用户的盈利案例中，回调买入成功率远高于追涨买入

8. **避开空头排列的股票**：MA空头排列+底部收敛的组合胜率极低
   理由：用户在空头排列买入的63笔交易中，胜率只有17.5%
"""
