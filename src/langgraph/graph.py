"""
LangGraph 主图 — 飞书对话式持仓管理

意图路由:
  entry_node → route_intent →
    view_portfolio / buy_flow / sell_flow / t0_flow /
    rebalance / scan / analyze / strategy / chat
"""
import logging
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.langgraph.state import PortfolioGraphState
from src.langgraph.nodes.entry import entry_node
from src.langgraph.nodes.portfolio import view_portfolio_node, trade_history_node
from src.langgraph.nodes.trade import (
    risk_check_node, request_confirmation_node, execute_trade_node,
)
from src.langgraph.nodes.t0 import (
    validate_t0_node, plan_t0_node,
    execute_t0_sell_node, execute_t0_buyback_node,
)
from src.langgraph.nodes.rebalance import (
    rebalance_node, scan_node, analyze_stock_node,
)
from src.langgraph.nodes.chat import chat_node, strategy_node

logger = logging.getLogger(__name__)


# ── 路由函数 ──

def route_intent(state: dict) -> str:
    """根据意图路由到对应节点"""
    # 如果已经有 response（比如参数不全的提示），直接结束
    if state.get("response"):
        return "end"

    intent = state.get("intent", "chat")

    route_map = {
        "view_portfolio": "view_portfolio",
        "buy": "risk_check",
        "sell": "risk_check",
        "clear": "risk_check",
        "t0": "validate_t0",
        "rebalance": "rebalance",
        "scan": "scan",
        "analyze": "analyze",
        "trade_history": "trade_history",
        "strategy": "strategy",
        "alert": "chat",      # 暂时用chat处理
        "mainline": "chat",   # 暂时用chat处理
        "chat": "chat",
    }
    return route_map.get(intent, "chat")


def route_after_risk_check(state: dict) -> str:
    """风控检查后的路由"""
    # 如果已有response（参数不全、被拦截等），直接结束
    if state.get("response"):
        return "end"

    risk = state.get("risk_check", {})
    if not risk.get("allowed", True):
        # 风控拦截，生成拒绝消息
        return "confirm"  # confirm节点会处理blocked状态
    return "confirm"


def route_after_confirm(state: dict) -> str:
    """确认消息后的路由"""
    if state.get("pending_confirmation"):
        return "end"  # 等待用户确认，图暂停
    if state.get("response"):
        return "end"  # 被风控拦截，已有拒绝消息
    return "end"


def route_after_entry_for_confirmation(state: dict) -> str:
    """确认流程恢复后的路由"""
    if state.get("confirmed") is True:
        # 检查是否是做T流程
        if state.get("t0_phase") == "confirm_sell":
            return "execute_t0_sell"
        return "execute_trade"
    elif state.get("confirmed") is False:
        return "end"
    # 正常的意图路由
    return "route"


def route_t0_after_validate(state: dict) -> str:
    """做T验证后路由"""
    if state.get("response"):
        return "end"  # 验证失败
    return "plan_t0"


def route_t0_after_sell(state: dict) -> str:
    """做T卖出后路由"""
    return "end"  # 卖出完成，等待用户发"回补"


# ── 构建图 ──

def build_portfolio_graph() -> StateGraph:
    """构建完整的对话式持仓管理图"""

    graph = StateGraph(PortfolioGraphState)

    # 添加节点
    graph.add_node("entry", entry_node)
    graph.add_node("view_portfolio", view_portfolio_node)
    graph.add_node("trade_history", trade_history_node)
    graph.add_node("risk_check", risk_check_node)
    graph.add_node("confirm", request_confirmation_node)
    graph.add_node("execute_trade", execute_trade_node)
    graph.add_node("validate_t0", validate_t0_node)
    graph.add_node("plan_t0", plan_t0_node)
    graph.add_node("execute_t0_sell", execute_t0_sell_node)
    graph.add_node("execute_t0_buyback", execute_t0_buyback_node)
    graph.add_node("rebalance", rebalance_node)
    graph.add_node("scan", scan_node)
    graph.add_node("analyze", analyze_stock_node)
    graph.add_node("strategy", strategy_node)
    graph.add_node("chat", chat_node)

    # 入口
    graph.set_entry_point("entry")

    # entry → 路由（处理确认恢复 + 正常意图分类）
    graph.add_conditional_edges("entry", route_after_entry_for_confirmation, {
        "route": "route_intent_dummy",  # 需要一个中间节点来做路由
        "execute_trade": "execute_trade",
        "execute_t0_sell": "execute_t0_sell",
        "end": END,
    })

    # 添加一个空的路由中间节点
    def route_intent_dummy(state: dict) -> dict:
        return {}
    graph.add_node("route_intent_dummy", route_intent_dummy)

    graph.add_conditional_edges("route_intent_dummy", route_intent, {
        "view_portfolio": "view_portfolio",
        "risk_check": "risk_check",
        "validate_t0": "validate_t0",
        "rebalance": "rebalance",
        "scan": "scan",
        "analyze": "analyze",
        "trade_history": "trade_history",
        "strategy": "strategy",
        "chat": "chat",
        "end": END,
    })

    # 终端节点 → END
    graph.add_edge("view_portfolio", END)
    graph.add_edge("trade_history", END)
    graph.add_edge("rebalance", END)
    graph.add_edge("scan", END)
    graph.add_edge("analyze", END)
    graph.add_edge("strategy", END)
    graph.add_edge("chat", END)

    # 交易流程: risk_check → confirm → (等待) → execute_trade → END
    graph.add_conditional_edges("risk_check", route_after_risk_check, {
        "confirm": "confirm",
        "end": END,
    })
    graph.add_edge("confirm", END)  # pending_confirmation=True 时在这里暂停
    graph.add_edge("execute_trade", END)

    # 做T流程
    graph.add_conditional_edges("validate_t0", route_t0_after_validate, {
        "plan_t0": "plan_t0",
        "end": END,
    })
    graph.add_edge("plan_t0", END)  # pending_confirmation=True 暂停
    graph.add_edge("execute_t0_sell", END)
    graph.add_edge("execute_t0_buyback", END)

    return graph.compile(checkpointer=MemorySaver())


# ── 单例 ──

_graph_instance = None


def get_portfolio_graph():
    """获取图实例（单例）"""
    global _graph_instance
    if _graph_instance is None:
        logger.info("[LangGraph] 初始化对话式持仓管理图...")
        _graph_instance = build_portfolio_graph()
        logger.info("[LangGraph] 图构建完成")
    return _graph_instance


def invoke_graph(user_text: str, session_id: str, thread_id: str) -> str:
    """
    便捷调用接口 — 飞书消息处理器直接调用这个函数

    参数:
        user_text: 用户输入的文本
        session_id: 用户会话ID（feishu_{user_id}）
        thread_id: 线程ID（feishu_{chat_id}_{user_id}），用于确认流程的状态恢复

    返回:
        回复文本
    """
    graph = get_portfolio_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # 检查是否有待确认的状态
        snapshot = graph.get_state(config)
        pending = False
        if snapshot and snapshot.values:
            pending = snapshot.values.get("pending_confirmation", False)

        if pending:
            # 恢复中断的图：传入用户回复
            result = graph.invoke(
                {"user_text": user_text, "messages": []},
                config=config,
            )
        else:
            # 新的对话轮次
            result = graph.invoke(
                {
                    "user_text": user_text,
                    "session_id": session_id,
                    "messages": [],
                    "pending_confirmation": False,
                    "confirmed": None,
                    "t0_phase": None,
                    "t0_sell_done": False,
                    "t0_target_price": None,
                    "response": "",
                },
                config=config,
            )

        response = result.get("response", "")
        if not response:
            response = "操作完成。"
        return response

    except Exception as e:
        logger.error(f"[LangGraph] 图执行异常: {e}", exc_info=True)
        return f"⚠️ 系统错误: {e}"
