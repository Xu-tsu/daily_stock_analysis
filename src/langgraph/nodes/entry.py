"""
入口节点 — 加载持仓 + 意图分类（增强版）

增强点:
- 支持股票名称解析（"赤天化呢" → analyze 002539）
- 支持持仓更新口令（"协鑫集成是5.2 有13手" → update_portfolio）
- 支持跟上下文的简短追问（"XX呢"、"那XX怎么样"）
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
    (r"^做[tT]$", "t0"),  # 纯 "做T" 无参数
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
    # 券商操作
    (r"^(同步持仓|持仓同步|同步)$", "broker_sync"),
    (r"^(执行质量|执行报告|滑点)$", "broker_quality"),
    (r"^(停止交易|急停|紧急停止)$", "broker_halt"),
    (r"^(恢复交易|恢复下单|解除停止)$", "broker_resume"),
]

# 完整买入/卖出参数的精确正则
BUY_FULL_PATTERN = re.compile(r"买入\s*(\d{6})\s+(\d+)\s*股?\s*([\d.]+)\s*元?")
SELL_FULL_PATTERN = re.compile(r"卖出\s*(\d{6})\s+(\d+)\s*股?\s*([\d.]+)\s*元?")
CLEAR_PATTERN = re.compile(r"清仓\s*(\d{6})")
CODE_PATTERN = re.compile(r"(\d{6})")
T0_PATTERN = re.compile(r"做[tT]\s*(\d{6})?")

# ── 持仓更新口令正则 ──
# "协鑫集成是5.2 有13手" / "利欧 7.5 500股" / "002506成本5.2 1300股"
UPDATE_PORTFOLIO_PATTERN = re.compile(
    r"(?:.*?)"                      # 前缀（股票名/代码）
    r"(?:是|成本|均价|价格)?\s*"
    r"([\d.]+)\s*(?:元)?\s*"        # 价格
    r"(?:有|持有|持)?\s*"
    r"(\d+)\s*(?:手|股)"            # 数量
)

# "XX呢" / "那XX怎么样" / "XX如何" — 股票名称追问
NAME_QUERY_PATTERN = re.compile(
    r"^(?:那|那个)?\s*(.{2,6}?)\s*(?:呢|怎么样|如何|咋样|什么情况|怎样|啥情况|好不好|能买吗|能不能买|可以买吗)[？?]?$"
)

# 带股票名称的买入/卖出/加仓/清仓
NAME_BUY_PATTERN = re.compile(
    r"(?:买入|买|加仓|建仓)\s*(.{2,8}?)(?:\s+(\d+)\s*(?:股|手)?\s*(?:([\d.]+)\s*(?:元)?)?)?$"
)
NAME_SELL_PATTERN = re.compile(
    r"(?:卖出|卖|减仓)\s*(.{2,8}?)(?:\s+(\d+)\s*(?:股|手)?\s*(?:([\d.]+)\s*(?:元)?)?)?$"
)
NAME_CLEAR_PATTERN = re.compile(r"(?:清仓|全卖)\s*(.{2,8})\s*$")
NAME_T0_PATTERN = re.compile(r"做[tT]\s*(.{2,8})\s*$")
NAME_ANALYZE_PATTERN = re.compile(r"(?:分析|看看)\s*(.{2,8})\s*$")


def _resolve_name(name: str, portfolio: dict = None) -> str | None:
    """
    尝试将股票名称解析为6位代码。
    优先级: 持仓名称匹配 → name_to_code_resolver → None
    """
    if not name:
        return None
    name = name.strip()

    # 如果已经是6位数字，直接返回
    if re.match(r"^\d{6}$", name):
        return name

    # 1) 从当前持仓查找（最快、最常用）
    if portfolio and portfolio.get("holdings"):
        for h in portfolio["holdings"]:
            h_name = h.get("name", "")
            h_code = h.get("code", "")
            # 完全匹配 或 包含匹配（"协鑫" 匹配 "协鑫集成"）
            if h_name == name or name in h_name or h_name in name:
                return h_code

    # 2) 通过 name_to_code_resolver 解析（含 AkShare fallback）
    try:
        from src.services.name_to_code_resolver import resolve_name_to_code
        code = resolve_name_to_code(name)
        if code:
            return code
    except Exception as e:
        logger.debug(f"名称解析失败: {name} -> {e}")

    # 3) 通过 stock_mapping 本地字典反查
    try:
        from src.data.stock_mapping import STOCK_NAME_MAP
        for c, n in STOCK_NAME_MAP.items():
            if n == name or name in n or n in name:
                return c
    except Exception:
        pass

    return None


def _try_name_based_intent(text: str, portfolio: dict = None) -> tuple:
    """
    尝试基于股票名称（非代码）的意图识别。
    返回 (intent, trade_action) 或 (None, None) 表示未匹配。
    """
    # "XX呢" / "那XX怎么样" — 追问分析
    m = NAME_QUERY_PATTERN.search(text)
    if m:
        name = m.group(1).strip()
        code = _resolve_name(name, portfolio)
        if code:
            return "analyze", {"code": code, "name": name}

    # "买入 协鑫集成 500 5.2"
    m = NAME_BUY_PATTERN.search(text)
    if m:
        name = m.group(1).strip()
        # 排除纯数字
        if not re.match(r"^\d+$", name):
            code = _resolve_name(name, portfolio)
            if code:
                shares = int(m.group(2)) if m.group(2) else None
                price = float(m.group(3)) if m.group(3) else None
                # "手" 转 "股"
                if shares and "手" in text:
                    shares *= 100
                return "buy", {"action": "buy", "code": code, "shares": shares, "price": price}

    # "卖出 利欧"
    m = NAME_SELL_PATTERN.search(text)
    if m:
        name = m.group(1).strip()
        if not re.match(r"^\d+$", name):
            code = _resolve_name(name, portfolio)
            if code:
                shares = int(m.group(2)) if m.group(2) else None
                price = float(m.group(3)) if m.group(3) else None
                if shares and "手" in text:
                    shares *= 100
                return "sell", {"action": "sell", "code": code, "shares": shares, "price": price}

    # "清仓 雪龙"
    m = NAME_CLEAR_PATTERN.search(text)
    if m:
        name = m.group(1).strip()
        if not re.match(r"^\d+$", name):
            code = _resolve_name(name, portfolio)
            if code:
                return "clear", {"action": "clear", "code": code}

    # "做T 利欧"
    m = NAME_T0_PATTERN.search(text)
    if m:
        name = m.group(1).strip()
        if not re.match(r"^\d+$", name):
            code = _resolve_name(name, portfolio)
            if code:
                return "t0", {"action": "t0", "code": code}

    # "分析 协鑫" / "看看 赤天化"
    m = NAME_ANALYZE_PATTERN.search(text)
    if m:
        name = m.group(1).strip()
        if not re.match(r"^\d+$", name):
            code = _resolve_name(name, portfolio)
            if code:
                return "analyze", {"code": code, "name": name}

    # 持仓更新口令: "协鑫集成是5.2 有13手"
    m = UPDATE_PORTFOLIO_PATTERN.search(text)
    if m:
        price_str, qty_str = m.group(1), m.group(2)
        # 提取名称部分（正则之前的文字）
        prefix = text[:m.start(1)].strip().rstrip("是成本均价价格").strip()
        if prefix:
            code = _resolve_name(prefix, portfolio)
            if code:
                shares = int(qty_str)
                if "手" in text:
                    shares *= 100
                return "update_portfolio", {
                    "action": "update",
                    "code": code,
                    "shares": shares,
                    "price": float(price_str),
                }

    return None, None


def classify_intent(text: str, portfolio: dict = None) -> tuple:
    """
    返回 (intent, trade_action_dict_or_None)

    增强版: 先尝试 code-based 正则，再尝试 name-based 自然语言解析。
    """
    text = text.strip()

    # 第一轮: 原有的 code-based 正则匹配
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

    # 第二轮: name-based 自然语言解析（股票名称 → 代码）
    name_intent, name_action = _try_name_based_intent(text, portfolio)
    if name_intent:
        return name_intent, name_action

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

    # 分类意图（传入持仓用于名称匹配）
    intent, trade_action = classify_intent(user_text, portfolio)
    logger.info(f"[LangGraph] 意图: {intent}, 交易参数: {trade_action}, 文本: {user_text[:40]}")

    return {
        "intent": intent,
        "trade_action": trade_action,
        "portfolio": portfolio,
        "pending_confirmation": False,
        "confirmed": None,
        "risk_check": None,
        "response": "",
    }
