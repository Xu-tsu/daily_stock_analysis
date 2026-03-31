"""
LangGraph 状态定义 — 所有节点共享的状态结构
"""
from typing import TypedDict, Optional, Literal, Annotated
from langgraph.graph import add_messages


class PortfolioGraphState(TypedDict):
    """飞书对话式持仓管理的图状态"""

    # ── 对话 ──
    messages: Annotated[list, add_messages]  # 自动累积的消息历史
    session_id: str                          # feishu_{user_id}
    user_text: str                           # 当前用户输入的原始文本

    # ── 意图路由 ──
    intent: Optional[str]
    # 可选值:
    #   "view_portfolio"  — 查看持仓
    #   "buy"             — 买入
    #   "sell"            — 卖出
    #   "clear"           — 清仓
    #   "t0"              — 做T（日内高卖低买）
    #   "rebalance"       — 调仓分析
    #   "scan"            — 市场扫描
    #   "strategy"        — 策略调整
    #   "analyze"         — 个股分析
    #   "trade_history"   — 战绩/复盘
    #   "chat"            — 自由对话

    # ── 交易参数（由意图解析器填充）──
    trade_action: Optional[dict]
    # 格式: {"code": "002506", "name": "协鑫集成", "shares": 500,
    #        "price": 5.4, "action": "buy/sell/clear"}

    # ── 风控检查结果 ──
    risk_check: Optional[dict]
    # 格式: {"allowed": True/False, "warnings": [...], "blocked_reason": "..."}

    # ── 确认工作流 ──
    pending_confirmation: bool   # 等待用户确认
    confirmed: Optional[bool]    # 用户确认/取消

    # ── 做T工作流 ──
    t0_phase: Optional[str]       # "plan" | "confirm_sell" | "monitoring" | "confirm_buy" | "done"
    t0_sell_done: bool
    t0_target_price: Optional[float]  # 目标回补价

    # ── 数据快照 ──
    portfolio: Optional[dict]     # 当前持仓（每次图调用开头加载）

    # ── 输出 ──
    response: str                 # 最终发送给用户的回复文本
