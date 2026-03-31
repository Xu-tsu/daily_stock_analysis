"""
自由对话节点 — 处理无法匹配到具体意图的消息
使用本地 LLM 回答股票相关问题
"""
import logging

logger = logging.getLogger(__name__)


def chat_node(state: dict) -> dict:
    """自由对话：用本地LLM回答用户问题"""
    user_text = state.get("user_text", "")
    portfolio = state.get("portfolio")

    if not user_text:
        return {"response": "请输入你的问题。"}

    # 构建上下文
    context_parts = []
    if portfolio and portfolio.get("holdings"):
        context_parts.append("当前持仓：")
        for h in portfolio["holdings"]:
            context_parts.append(
                f"  {h.get('name', h['code'])}({h['code']}) "
                f"{h.get('shares', 0)}股 成本{h.get('cost_price', 0):.2f} "
                f"盈亏{h.get('pnl_pct', 0):+.1f}%"
            )

    context = "\n".join(context_parts) if context_parts else "暂无持仓"

    prompt = f"""你是一位A股短线交易助手，专注低价小盘题材股。用户通过飞书和你对话。

{context}

用户问题: {user_text}

请简洁回答（200字以内），如果涉及操作建议，提醒用户使用具体命令（如"买入 002506 500 5.4"）。"""

    try:
        import litellm
        import os
        model = os.getenv("REBALANCE_LOCAL_MODEL", "ollama/qwen2.5:14b-instruct-q4_K_M")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()
        return {"response": answer}
    except Exception as e:
        logger.error(f"LLM对话失败: {e}")
        return {
            "response": (
                "🤖 AI暂时不可用，你可以直接使用命令：\n"
                "• 持仓 — 查看持仓\n"
                "• 买入 002506 500 5.4 — 买入\n"
                "• 卖出 002506 500 5.4 — 卖出\n"
                "• 清仓 002506 — 清仓\n"
                "• 做T 002506 — 日内做T\n"
                "• 调仓 — 调仓分析\n"
                "• 扫描 — 市场扫描\n"
                "• 分析 002506 — 个股分析\n"
                "• 战绩 — 交易记录"
            )
        }


def strategy_node(state: dict) -> dict:
    """策略参数调整对话"""
    user_text = state.get("user_text", "")

    # 读取当前策略参数
    try:
        from risk_control import (
            HARD_STOP_LOSS_PCT, TRAILING_STOP_TRIGGER, TAKE_PROFIT_HALF_PCT,
            HOLD_DAYS_WARNING, HOLD_DAYS_MAX, CHASE_HIGH_BLOCK_PCT,
            CHASE_HIGH_WARN_PCT, MAX_SINGLE_POSITION_PCT, MAX_POSITIONS,
        )
        current_params = {
            "硬止损线": f"{HARD_STOP_LOSS_PCT}%",
            "移动止损触发": f"盈利{TRAILING_STOP_TRIGGER}%",
            "减仓止盈": f"盈利{TAKE_PROFIT_HALF_PCT}%",
            "持仓预警天数": f"{HOLD_DAYS_WARNING}天",
            "持仓上限天数": f"{HOLD_DAYS_MAX}天",
            "追高警告涨幅": f"{CHASE_HIGH_WARN_PCT}%",
            "追高禁买涨幅": f"{CHASE_HIGH_BLOCK_PCT}%",
            "单只仓位上限": f"{MAX_SINGLE_POSITION_PCT}%",
            "最多持股数": f"{MAX_POSITIONS}只",
        }
    except ImportError:
        current_params = {"状态": "风控模块未加载"}

    lines = ["⚙️ **当前策略参数**", ""]
    for k, v in current_params.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("策略参数调整需要修改 risk_control.py，目前支持查看。")
    lines.append("如需调整，请告诉我想改哪个参数和目标值。")

    return {"response": "\n".join(lines)}
