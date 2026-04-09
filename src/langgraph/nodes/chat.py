"""
自由对话节点 — 处理无法匹配到具体意图的消息

增强版:
- 常见短语（问候/帮助）亚秒直答，不走 LLM
- LLM 优先用云端 Gemini（快），本地 Ollama 备用
- 30s 超时保护，失败返回命令引导
"""
import logging
import os

logger = logging.getLogger(__name__)


def chat_node(state: dict) -> dict:
    """自由对话：用 LLM 回答用户问题（带持仓上下文）"""
    user_text = state.get("user_text", "")
    portfolio = state.get("portfolio")

    if not user_text:
        return {"response": "请输入你的问题。"}

    # ── 快速应答：常见短语直接返回 ──
    quick = _quick_answer(user_text, portfolio)
    if quick:
        return {"response": quick}

    # ── 构建上下文 ──
    context_parts = []
    if portfolio and portfolio.get("holdings"):
        context_parts.append("当前持仓：")
        for h in portfolio["holdings"]:
            context_parts.append(
                f"  {h.get('name', h['code'])}({h['code']}) "
                f"{h.get('shares', 0)}股 成本{h.get('cost_price', 0):.2f} "
                f"现价{h.get('current_price', 0):.2f} "
                f"盈亏{h.get('pnl_pct', 0):+.1f}%"
            )
    context = "\n".join(context_parts) if context_parts else "暂无持仓"

    prompt = f"""你是一位A股短线交易助手，专注低价小盘题材股。用户通过飞书和你对话。

{context}

用户问题: {user_text}

请简洁回答（200字以内），如果涉及操作建议，提醒用户使用具体命令（如"买入 002506 500 5.4"）。"""

    # 按优先级尝试: Azure(快+稳) → Gemini(免费) → 本地Ollama(兜底)
    for model_env, default_model, label, timeout_s in [
        ("REBALANCE_CLOUD_MODEL", "azure/gpt-5.4-nano", "云端Azure", 30),
        ("LITELLM_MODEL", "gemini/gemini-2.5-flash", "云端Gemini", 20),
        ("REBALANCE_LOCAL_MODEL", "ollama/qwen2.5:14b-instruct-q4_K_M", "本地Ollama", 300),
    ]:
        try:
            import litellm
            model = os.getenv(model_env, default_model)
            if not model:
                continue
            extra_kwargs = {}
            if "ollama" in model:
                extra_kwargs["api_base"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_s,
                temperature=0.5,
                **extra_kwargs,
            )
            answer = response.choices[0].message.content.strip()
            return {"response": answer}
        except Exception as e:
            logger.warning(f"LLM对话失败({label} {model_env}): {e}")

    # 全部失败 → 返回命令引导
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
            "• 赤天化呢 — 用名称查询个股\n"
            "• 战绩 — 交易记录"
        )
    }


def _quick_answer(text: str, portfolio: dict = None) -> str | None:
    """对常见短语做即时回答，不走 LLM，亚秒响应。"""
    t = text.strip().lower()

    # 简单问候
    if t in ("你好", "hi", "hello", "嗨", "在吗", "在不在"):
        return (
            "👋 你好！我是你的A股交易助手，可以帮你：\n"
            "• 持仓 — 查看持仓\n"
            "• 调仓 — AI调仓建议\n"
            "• 分析 002506 — 个股分析\n"
            "• 赤天化呢 — 用名称查询\n"
            "• 买入/卖出/清仓/做T — 交易\n"
            "有什么需要？"
        )

    # 帮助
    if t in ("帮助", "help", "?", "？", "菜单", "功能", "命令"):
        return (
            "📋 **可用命令**\n\n"
            "**持仓管理**\n"
            "• 持仓 — 查看当前持仓\n"
            "• 买入 002506 500 5.4 — 买入（代码 数量 价格）\n"
            "• 卖出 002506 500 5.4 — 卖出\n"
            "• 清仓 002506 — 清仓\n"
            "• 做T 002506 — 日内做T\n\n"
            "**也可以用股票名称**\n"
            "• 买入 协鑫集成 500 5.2\n"
            "• 清仓 利欧\n"
            "• 协鑫集成是5.2 有13手 — 修正持仓\n\n"
            "**分析&策略**\n"
            "• 调仓 — AI调仓建议（含多模型辩论）\n"
            "• 扫描 — 市场扫描选股\n"
            "• 分析 002506 — 个股技术分析\n"
            "• 赤天化呢 — 用名称快捷分析\n"
            "• 战绩 — 交易记录\n"
            "• 策略 — 查看/调整策略参数"
        )

    return None


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
