"""
入口节点 — 加载持仓 + 意图分类
"""
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ── 意图正则（复用 portfolio_bot 的模式，扩展更多自然语言）──
INTENT_PATTERNS = [
    # 买入
    (r"买入\s*(\d{6})\s*(\d+)\s*股?\s*([\d.]+)\s*元?", "buy"),
    (r"买\s*(\d{6})", "buy"),
    (r"加仓\s*(\d{6})", "buy"),
    (r"建仓\s*(\d{6})", "buy"),
    # 卖出
    (r"卖出\s*(\d{6})\s*(\d+)\s*股?\s*([\d.]+)\s*元?", "sell"),
    (r"卖\s*(\d{6})", "sell"),
    (r"减仓\s*(\d{6})", "sell"),
    # 清仓
    (r"清仓\s*(\d{6})", "clear"),
    (r"全卖\s*(\d{6})", "clear"),
    # 做T
    (r"做[tT]\s*(\d{6})", "t0"),
    (r"日内[tT]\s*(\d{6})", "t0"),
    (r"高卖低买\s*(\d{6})", "t0"),
    (r"做T", "t0"),
    # 持仓
    (r"^(持仓|仓位|我的持仓|看持仓|查持仓)$", "view_portfolio"),
    (r"^(持仓情况|目前持仓|当前仓位)$", "view_portfolio"),
    # 调仓
    (r"^(调仓|调仓分析|调仓建议)$", "rebalance"),
    # 扫描
    (r"^(扫描|选股|市场扫描|找股票)$", "scan"),
    # 分析
    (r"分析\s*(\d{6})", "analyze"),
    (r"看看\s*(\d{6})", "analyze"),
    (r"(\d{6})\s*(怎么样|如何|咋样|什么情况)", "analyze"),
    # 战绩/复盘
    (r"^(战绩|复盘|交易记录|回测|胜率)$", "trade_history"),
    # 策略
    (r"(策略|止损|止盈|持仓天数|参数|阈值).*(调整|修改|改|设置|设)", "strategy"),
    (r"(调整|修改|改|设置|设).*(策略|止损|止盈|持仓天数|参数|阈值)", "strategy"),
    # 预警
    (r"^(预警|异动|监控)$", "alert"),
    # 主线
    (r"^(主线|主线分析|主线板块)$", "mainline"),
]

# 完整买入/卖出参数的精确正则
BUY_FULL_PATTERN = re.compile(r"买入\s*(\d{6})\s+(\d+)\s*股?\s*([\d.]+)\s*元?")
SELL_FULL_PATTERN = re.compile(r"卖出\s*(\d{6})\s+(\d+)\s*股?\s*([\d.]+)\s*元?")
CLEAR_PATTERN = re.compile(r"清仓\s*(\d{6})")
CODE_PATTERN = re.compile(r"(\d{6})")
T0_PATTERN = re.compile(r"做[tT]\s*(\d{6})?")


def classify_intent(text: str) -> tuple:
    """
    返回 (intent, trade_action_dict_or_None)
    """
    text = text.strip()

    for pattern, intent in INTENT_PATTERNS:
        m = re.search(pattern, text)
        if m:
            trade_action = None

            if intent == "buy":
                fm = BUY_FULL_PATTERN.search(text)
                if fm:
                    trade_action = {
                        "action": "buy",
                        "code": fm.group(1),
                        "shares": int(fm.group(2)),
                        "price": float(fm.group(3)),
                    }
                else:
                    code_m = CODE_PATTERN.search(text)
                    if code_m:
                        trade_action = {"action": "buy", "code": code_m.group(1),
                                        "shares": None, "price": None}

            elif intent == "sell":
                fm = SELL_FULL_PATTERN.search(text)
                if fm:
                    trade_action = {
                        "action": "sell",
                        "code": fm.group(1),
                        "shares": int(fm.group(2)),
                        "price": float(fm.group(3)),
                    }
                else:
                    code_m = CODE_PATTERN.search(text)
                    if code_m:
                        trade_action = {"action": "sell", "code": code_m.group(1),
                                        "shares": None, "price": None}

            elif intent == "clear":
                cm = CLEAR_PATTERN.search(text)
                if cm:
                    trade_action = {"action": "clear", "code": cm.group(1)}

            elif intent == "t0":
                tm = T0_PATTERN.search(text)
                code = tm.group(1) if tm and tm.group(1) else None
                trade_action = {"action": "t0", "code": code}

            elif intent == "analyze":
                code_m = CODE_PATTERN.search(text)
                if code_m:
                    trade_action = {"code": code_m.group(1)}

            return intent, trade_action

    # 无法匹配 → 自由对话
    return "chat", None


def entry_node(state: dict) -> dict:
    """图的入口节点：加载持仓 + 分类意图"""
    from langchain_core.messages import HumanMessage

    user_text = state.get("user_text", "")

    # 如果是确认/取消回复（图被中断后恢复）
    if state.get("pending_confirmation"):
        text_lower = user_text.strip().lower()
        if text_lower in ("确认", "确定", "是", "yes", "ok", "好", "执行"):
            return {"confirmed": True, "pending_confirmation": False}
        elif text_lower in ("取消", "不", "no", "算了", "放弃"):
            return {"confirmed": False, "pending_confirmation": False,
                    "response": "已取消操作。"}
        else:
            return {"response": "请回复「确认」或「取消」。"}

    # 加载持仓
    portfolio = None
    try:
        from portfolio_manager import load_portfolio, sync_portfolio_from_trades
        portfolio = load_portfolio()
        portfolio = sync_portfolio_from_trades(portfolio)
    except Exception as e:
        logger.warning(f"加载持仓失败: {e}")

    # 分类意图
    intent, trade_action = classify_intent(user_text)
    logger.info(f"[LangGraph] 意图: {intent}, 交易参数: {trade_action}")

    return {
        "intent": intent,
        "trade_action": trade_action,
        "portfolio": portfolio,
        "pending_confirmation": False,
        "confirmed": None,
        "risk_check": None,
        "response": "",
    }
