"""
portfolio_bot.py — 通过聊天指令管理持仓 + 触发扫描/调仓

支持的指令（飞书/钉钉发送）:
  买入 002506 500股 5.4元        → 添加/加仓
  卖出 002506 200股 5.8元        → 减仓
  清仓 002506                    → 清掉某只
  持仓                           → 查看当前持仓
  调仓                           → 触发调仓分析
  扫描                           → 触发全市场扫描
  主线                           → 查看主线板块
  回测                           → 查看扫描胜率
  预警                           → 手动触发一次异动检测

放在项目根目录，被 bot 系统调用
"""
import difflib
import json, logging, os, re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from portfolio_manager import load_portfolio, save_portfolio
from src.services.name_to_code_resolver import resolve_name_to_code
from src.services.portfolio_command_llm import interpret_portfolio_command
from src.services.trade_sizing_service import is_valid_a_share_lot

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 指令解析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 匹配：买入 002506 500股 5.4元  或  买入002506 500 5.4
BUY_PATTERN = re.compile(
    r'买入\s*(\d{6})\s*(\d+)\s*股?\s*([\d.]+)\s*元?', re.IGNORECASE
)
SELL_PATTERN = re.compile(
    r'卖出\s*(\d{6})\s*(\d+)\s*股?\s*([\d.]+)\s*元?', re.IGNORECASE
)
CLEAR_PATTERN = re.compile(r'清仓\s*(\d{6})', re.IGNORECASE)
LOT_PATTERN = re.compile(r'(\d+)\s*(手|股)', re.IGNORECASE)
CODE_PATTERN = re.compile(r'(?<!\d)(\d{6})(?!\d)')
PRICE_PATTERN = re.compile(r'(?<![\d.])(\d+(?:\.\d{1,3})?)(?:\s*元)?', re.IGNORECASE)
NAME_TOKEN_PATTERN = re.compile(r'[\u4e00-\u9fffA-Za-z]{2,}')

SHOW_PORTFOLIO_PHRASES = {
    "持仓",
    "查看持仓",
    "持仓查看",
    "我的持仓",
    "看看持仓",
    "查询持仓",
    "持仓查询",
}

PORTFOLIO_COMMAND_PREFIXES = [
    "调仓", "扫描", "主线", "回测", "预警", "分析", "全量分析", "战绩", "复盘", "交易记录",
]

TRADE_ACTION_MAP = {
    "买入": "buy",
    "加仓": "buy",
    "卖出": "sell",
    "减仓": "sell",
    "清仓": "clear",
}

TRADE_FILLER_WORDS = (
    "买入", "加仓", "卖出", "减仓", "清仓",
    "查看持仓", "持仓查看", "我的持仓", "持仓",
    "查看", "查询", "看看", "显示", "一下",
    "按", "以", "在", "于", "加", "元", "股", "手",
    "现在", "现在是", "现价", "目前", "仓位", "仓",
)

LLM_PORTFOLIO_HINT_WORDS = (
    "加仓", "减仓", "清仓", "补仓", "建仓", "买入", "卖出", "卖掉", "卖光", "全卖",
    "仓位", "持仓", "仓", "调仓", "再买", "再补", "做t", "做T", "出掉",
)

_PORTFOLIO_LLM_CACHE: dict[str, Optional[dict]] = {}

FEEDBACK_CUE_WORDS = (
    "卖飞", "止盈后", "清仓后", "卖出后", "减仓后", "买入后", "加仓后", "补仓后",
    "后立刻", "后马上", "后就", "后被", "结果", "反弹", "拉升", "冲到", "回落到",
    "踏空了", "反馈", "复盘",
)

PLAN_CUE_WORDS = ("明天", "明日", "计划", "打算", "准备", "预备")
BATCH_SPLIT_RE = re.compile(r"[，,；;\n]+")


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").replace("\u3000", " ").strip())


def _get_holdings_snapshot() -> list[dict]:
    try:
        portfolio = load_portfolio()
    except Exception:
        return []
    return list(portfolio.get("holdings", []) or [])


def _looks_like_trade_feedback(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False
    if "卖飞" in compact:
        return True
    action_words = ("买入", "加仓", "补仓", "卖出", "减仓", "清仓", "止盈", "止损")
    has_action_context = any(word in compact for word in action_words)
    if not has_action_context:
        return False
    followup_cue = any(word in compact for word in FEEDBACK_CUE_WORDS)
    price_count = len(re.findall(r"\d+(?:\.\d+)?", compact))
    past_tense_cue = any(word in compact for word in ("后", "结果", "被", "后来", "立刻", "马上"))
    return followup_cue or (price_count >= 2 and past_tense_cue)


def _should_try_llm_portfolio_interpretation(text: str) -> bool:
    compact = _compact_text(text)
    if not compact or len(compact) > 160:
        return False

    if _is_show_portfolio_command(compact):
        return True
    if _looks_like_trade_feedback(compact):
        return True

    has_hint_word = any(word in compact for word in LLM_PORTFOLIO_HINT_WORDS)
    has_lot_or_price = LOT_PATTERN.search(compact) is not None or re.search(r"\d+(?:\.\d+)?(?:元|块)?", compact)

    holdings = _get_holdings_snapshot()
    has_holding_reference = any(
        str(item.get("code", "") or "").strip() in compact
        or str(item.get("name", "") or "").strip() in compact
        for item in holdings
        if item
    )

    return bool(has_hint_word and (has_lot_or_price or has_holding_reference or "仓位" in compact or "持仓" in compact))


def _get_llm_portfolio_command(text: str) -> Optional[dict]:
    compact = _compact_text(text)
    if not compact:
        return None
    if compact in _PORTFOLIO_LLM_CACHE:
        return _PORTFOLIO_LLM_CACHE[compact]

    interpretation = interpret_portfolio_command(text, holdings=_get_holdings_snapshot())
    if len(_PORTFOLIO_LLM_CACHE) >= 128:
        _PORTFOLIO_LLM_CACHE.pop(next(iter(_PORTFOLIO_LLM_CACHE)))
    _PORTFOLIO_LLM_CACHE[compact] = interpretation
    return interpretation


def _clear_portfolio_llm_cache() -> None:
    _PORTFOLIO_LLM_CACHE.clear()


def _is_show_portfolio_command(text: str) -> bool:
    compact = _compact_text(text)
    if compact in SHOW_PORTFOLIO_PHRASES:
        return True
    return compact.startswith("持仓") and any(word in compact for word in ("查看", "查询", "看看", "显示"))


def _extract_price(text: str) -> Optional[float]:
    working = text or ""
    code_match = CODE_PATTERN.search(working)
    if code_match:
        working = working.replace(code_match.group(1), " ", 1)

    lot_match = LOT_PATTERN.search(working)
    if lot_match:
        working = f"{working[:lot_match.start()]} {working[lot_match.end():]}"

    matches = list(PRICE_PATTERN.finditer(working))
    if not matches:
        return None

    # 优先取小数价格；如果只有整数，取最后一个数字。
    preferred = next((m for m in matches if "." in m.group(1)), matches[-1])
    try:
        return float(preferred.group(1))
    except (TypeError, ValueError):
        return None


def _extract_stock_identifier(text: str) -> Optional[str]:
    code_match = CODE_PATTERN.search(text or "")
    if code_match:
        return code_match.group(1)

    cleaned = text or ""
    cleaned = LOT_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r'(?<![\d.])\d+(?:\.\d{1,3})?\s*元?', " ", cleaned)
    for word in TRADE_FILLER_WORDS:
        cleaned = cleaned.replace(word, " ")
    cleaned = re.sub(r'[，,。；;:：@/\\+*=（）()【】\[\]_\-—]+', " ", cleaned)

    candidates = [token for token in NAME_TOKEN_PATTERN.findall(cleaned) if token]
    if not candidates:
        return None
    return max(candidates, key=len)


def _looks_like_trade_instruction(text: str) -> bool:
    compact = _compact_text(text)
    has_action = any(word in compact for word in ("买入", "加仓", "卖出", "减仓"))
    has_quantity = LOT_PATTERN.search(compact) is not None
    has_price = _extract_price(compact) is not None
    has_symbol = _extract_stock_identifier(compact) is not None
    return has_action and has_quantity and has_price and has_symbol


def _detect_trade_action(text: str) -> Optional[str]:
    compact = _compact_text(text)
    if _looks_like_trade_feedback(compact):
        return None
    for keyword, action in TRADE_ACTION_MAP.items():
        if compact.startswith(keyword):
            return action

    if "清仓" in compact:
        return "clear"

    if _looks_like_trade_instruction(compact):
        if "加仓" in compact or "买入" in compact:
            return "buy"
        if "减仓" in compact or "卖出" in compact:
            return "sell"
    return None


def _resolve_stock_code(identifier: str) -> Optional[str]:
    if not identifier:
        return None
    if CODE_PATTERN.fullmatch(identifier):
        return identifier
    return resolve_name_to_code(identifier)


def _resolve_portfolio_holding_code(identifier: str, holdings: list[dict]) -> Optional[str]:
    if not identifier:
        return None

    normalized = identifier.strip()
    if not normalized:
        return None

    for holding in holdings:
        code = str(holding.get("code", "") or "").strip()
        name = str(holding.get("name", "") or "").strip()
        if normalized == code or normalized == name:
            return code

    try:
        from pypinyin import lazy_pinyin

        identifier_pinyin = "".join(lazy_pinyin(normalized)).lower()
        if identifier_pinyin:
            for holding in holdings:
                code = str(holding.get("code", "") or "").strip()
                name = str(holding.get("name", "") or "").strip()
                if not name:
                    continue
                holding_pinyin = "".join(lazy_pinyin(name)).lower()
                if identifier_pinyin == holding_pinyin:
                    return code
    except ImportError:
        pass
    except Exception:
        pass

    names = [str(h.get("name", "") or "").strip() for h in holdings if h.get("name")]
    matches = difflib.get_close_matches(normalized, names, n=1, cutoff=0.75)
    if matches:
        matched_name = matches[0]
        for holding in holdings:
            if str(holding.get("name", "") or "").strip() == matched_name:
                return str(holding.get("code", "") or "").strip()
    return None


def _parse_trade_order(text: str) -> Tuple[Optional[dict], Optional[str]]:
    identifier = _extract_stock_identifier(text)
    if not identifier:
        return None, "无法识别股票名称或代码，请写成“协鑫集成 加仓 3手 5.05元”或“买入 002506 300股 5.05元”。"

    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])
    code = _resolve_portfolio_holding_code(identifier, holdings) or _resolve_stock_code(identifier)
    if not code:
        return None, f"无法识别股票名称：{identifier}"

    lot_match = LOT_PATTERN.search(text or "")
    if not lot_match:
        return None, "无法识别买卖数量，请写成“3手”或“300股”。"

    quantity = int(lot_match.group(1))
    unit = lot_match.group(2)
    shares = quantity * 100 if unit == "手" else quantity

    price = _extract_price(text)
    if price is None:
        return None, "无法识别成交价格，请写成“5.05元”或“5.05”。"

    return {
        "code": code,
        "identifier": identifier,
        "shares": shares,
        "price": price,
    }, None


def _parse_clear_target(text: str) -> Tuple[Optional[str], Optional[str]]:
    match = CLEAR_PATTERN.search(text)
    if match:
        return match.group(1), None

    identifier = _extract_stock_identifier(text)
    if not identifier:
        return None, "格式错误，请输入：清仓 002506 或 清仓 协鑫集成"

    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])
    code = _resolve_portfolio_holding_code(identifier, holdings) or _resolve_stock_code(identifier)
    if not code:
        return None, f"无法识别股票名称：{identifier}"
    return code, None


def _refresh_portfolio_summary(portfolio: dict) -> dict:
    holdings = portfolio.get("holdings", []) or []
    total_mv = 0.0
    for item in holdings:
        shares = int(item.get("shares", 0) or 0)
        current_price = float(item.get("current_price", item.get("cost_price", 0)) or 0)
        market_value = round(shares * current_price, 2)
        item["market_value"] = market_value
        total_mv += market_value
    total_asset = round(total_mv + float(portfolio.get("cash", 0) or 0), 2)
    portfolio["total_asset"] = total_asset
    portfolio["actual_position_ratio"] = round(total_mv / total_asset, 4) if total_asset > 0 else 0.0
    return portfolio


def _split_batch_portfolio_segments(text: str) -> list[str]:
    segments = []
    for part in BATCH_SPLIT_RE.split(text or ""):
        segment = part.strip().strip("。")
        if segment:
            segments.append(segment)
    return segments


def _parse_snapshot_segment(text: str) -> Optional[dict]:
    identifier = _extract_stock_identifier(text)
    if not identifier:
        return None

    lot_match = LOT_PATTERN.search(text or "")
    if not lot_match:
        return None

    price = _extract_price(text)
    if price is None or price <= 0:
        return None

    quantity = int(lot_match.group(1))
    unit = lot_match.group(2)
    shares = quantity * 100 if unit == "手" else quantity
    if shares <= 0:
        return None

    return {
        "identifier": identifier,
        "shares": shares,
        "price": float(price),
    }


def _build_feedback_command_from_text(text: str) -> Optional[dict]:
    if not _looks_like_trade_feedback(text):
        return None

    identifier = _extract_stock_identifier(text) or ""
    prices = [
        float(match.group(1))
        for match in PRICE_PATTERN.finditer(text or "")
        if match and match.group(1)
    ]
    reference_price = prices[0] if prices else None
    outcome_price = prices[1] if len(prices) > 1 else None

    compact = _compact_text(text)
    if "清仓" in compact:
        reference_action = "clear"
    elif "减仓" in compact or "卖出" in compact:
        reference_action = "sell"
    elif "加仓" in compact or "补仓" in compact or "买入" in compact:
        reference_action = "buy"
    else:
        reference_action = "other"

    feedback_tag = "other"
    if reference_action in {"clear", "sell"} and outcome_price and reference_price and outcome_price > reference_price:
        feedback_tag = "sold_too_early"
    elif reference_action in {"buy"} and outcome_price and reference_price and outcome_price > reference_price:
        feedback_tag = "dip_buy_success"
    elif reference_action in {"buy"} and outcome_price and reference_price and outcome_price < reference_price:
        feedback_tag = "bought_too_early"

    return {
        "is_portfolio_command": True,
        "action": "feedback",
        "stock_hint": identifier,
        "price": reference_price,
        "outcome_price": outcome_price,
        "reference_action": reference_action,
        "feedback_tag": feedback_tag,
        "summary": text.strip(),
        "needs_clarification": False,
        "confidence": 0.72,
    }


def _looks_like_future_trade_plan(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False
    if not any(word in compact for word in PLAN_CUE_WORDS):
        return False
    if not any(word in compact for word in ("清仓", "卖出", "减仓", "买入", "加仓", "补仓")):
        return False
    return _extract_stock_identifier(text) is not None


def _looks_like_batch_portfolio_sync(text: str) -> bool:
    segments = _split_batch_portfolio_segments(text)
    if len(segments) < 2:
        return False

    recognized = 0
    for segment in segments:
        if _looks_like_trade_feedback(segment):
            recognized += 1
            continue
        if _looks_like_future_trade_plan(segment):
            recognized += 1
            continue
        if _detect_trade_action(segment) is not None:
            recognized += 1
            continue
        if _parse_snapshot_segment(segment):
            recognized += 1
    return recognized >= 2


def _looks_like_trade_instruction(text: str) -> bool:
    compact = _compact_text(text)
    has_action = any(word in compact for word in ("买入", "加仓", "补仓", "卖出", "减仓"))
    has_quantity = LOT_PATTERN.search(compact) is not None
    has_price = _extract_price(compact) is not None
    has_symbol = _extract_stock_identifier(compact) is not None
    return has_action and has_quantity and has_price and has_symbol


def _detect_trade_action(text: str) -> Optional[str]:
    compact = _compact_text(text)
    if _looks_like_trade_feedback(compact):
        return None
    for keyword, action in TRADE_ACTION_MAP.items():
        if compact.startswith(keyword):
            return action

    if "清仓" in compact:
        return "clear"

    if _looks_like_trade_instruction(compact):
        if any(word in compact for word in ("加仓", "补仓", "买入")):
            return "buy"
        if any(word in compact for word in ("减仓", "卖出")):
            return "sell"
    return None


def is_portfolio_command(text: str) -> bool:
    """判断是否是持仓管理指令"""
    text = text.strip()
    if not text:
        return False
    if _is_show_portfolio_command(text):
        return True
    if _looks_like_batch_portfolio_sync(text):
        return True
    if _looks_like_future_trade_plan(text):
        return True
    if any(text.startswith(kw) for kw in PORTFOLIO_COMMAND_PREFIXES):
        return True
    if _detect_trade_action(text) is not None:
        return True
    if _should_try_llm_portfolio_interpretation(text):
        llm_command = _get_llm_portfolio_command(text)
        return bool(llm_command and llm_command.get("is_portfolio_command"))
    return False


def _pick_quote_price(quote: dict) -> float:
    """从行情字典中提取可用价格（兼容不同字段名）。"""
    if not isinstance(quote, dict):
        return 0.0
    for key in ("price", "close", "last", "current_price"):
        try:
            val = float(quote.get(key, 0) or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val > 0:
            return val
    return 0.0


def handle_portfolio_command(text: str) -> str:
    """处理持仓管理指令"""
    text = text.strip()

    try:
        if _looks_like_batch_portfolio_sync(text):
            return _handle_batch_portfolio_sync(text)
        if _looks_like_future_trade_plan(text):
            return _handle_future_trade_plan(text)

        llm_command = None
        if _should_try_llm_portfolio_interpretation(text):
            llm_command = _get_llm_portfolio_command(text)
            llm_result = _handle_llm_portfolio_command(llm_command, raw_text=text)
            if llm_result is not None:
                return llm_result

        trade_action = _detect_trade_action(text)
        if trade_action == "buy":
            return _handle_buy(text)
        elif trade_action == "sell":
            return _handle_sell(text)
        elif trade_action == "clear":
            return _handle_clear(text)
        elif _is_show_portfolio_command(text):
            return _handle_show_portfolio()
        elif text.startswith("调仓"):
            return _handle_rebalance()
        elif text.startswith("全量分析"):
            return _handle_full_analysis()
        elif text.startswith("分析"):
            return _handle_analyze(text)
        elif text.startswith("扫描"):
            return _handle_scan()
        elif text.startswith("主线"):
            return _handle_mainline()
        elif text.startswith("回测"):
            return _handle_backtest()
        elif text.startswith("预警"):
            return _handle_alert()
        elif text.startswith("战绩"):
            return _handle_performance()
        elif text.startswith("复盘") and "分析" not in text:
            return _handle_pattern_review()
        elif text.startswith("交易记录"):
            return _handle_trade_history()
        else:
            return "未识别的指令"
    except Exception as e:
        logger.error(f"处理指令失败: {e}")
        return f"执行失败: {str(e)}"


def _handle_llm_portfolio_command(command: Optional[dict], raw_text: str) -> Optional[str]:
    if not command or not command.get("is_portfolio_command"):
        return None

    action = command.get("action", "other")
    if command.get("needs_clarification"):
        clarification = command.get("clarification") or "识别到了持仓指令，但缺少必要的价格或数量，请补充后再发一次。"
        return clarification

    if action == "show_portfolio":
        return _handle_show_portfolio()
    if action == "rebalance":
        return _handle_rebalance()
    if action == "scan":
        return _handle_scan()
    if action == "mainline":
        return _handle_mainline()
    if action == "backtest":
        return _handle_backtest()
    if action == "alert":
        return _handle_alert()
    if action == "performance":
        return _handle_performance()
    if action == "pattern_review":
        return _handle_pattern_review()
    if action == "trade_history":
        return _handle_trade_history()
    if action == "feedback":
        return _handle_trade_feedback(command, raw_text)
    if action in {"buy", "sell", "clear"}:
        holdings = _get_holdings_snapshot()
        stock_hint = str(command.get("stock_hint", "") or "").strip()
        code = _resolve_portfolio_holding_code(stock_hint, holdings) or _resolve_stock_code(stock_hint)
        if not code:
            return f"无法识别股票名称：{stock_hint or raw_text}"

        if action == "clear":
            return _execute_clear_order(code)

        shares = int(command.get("shares", 0) or 0)
        price = float(command.get("price", 0) or 0)
        if shares <= 0 or price <= 0:
            return "识别到了持仓操作，但缺少有效的数量或价格，请补充后再发一次。"
        if action == "buy":
            return _execute_buy_order(code, shares, price, raw_text=raw_text)
        return _execute_sell_order(code, shares, price)

    return None


def _handle_trade_feedback(command: Optional[dict], raw_text: str) -> str:
    if not command:
        return "识别到了复盘反馈，但缺少可用内容，请补充后再发一次。"

    try:
        from src.services.trade_feedback_service import record_trade_feedback

        result = record_trade_feedback(raw_text=raw_text, parsed_feedback=command, source="feishu")
    except Exception as exc:
        logger.warning("记录交易反馈失败: %s", exc)
        return f"识别到了复盘反馈，但写入失败：{exc}"

    if result.get("needs_clarification"):
        return result.get("clarification") or "识别到了复盘反馈，但还需要补充股票名后才能吸收。"

    symbol = result.get("name") or result.get("code") or "该笔交易"
    tag = result.get("feedback_tag") or "other"
    tag_text = {
        "sold_too_early": "卖飞/止盈偏早",
        "stopped_out_then_rebounded": "止损后反抽",
        "dip_buy_success": "分歧低吸有效",
        "bought_too_early": "左侧接早",
        "chased_high_then_dumped": "追高回撤",
        "good_exit": "兑现节奏正确",
        "other": "执行反馈",
    }.get(tag, "执行反馈")
    guidance = result.get("guidance") or "后续预判会把这条反馈作为纠偏样本。"
    summary = result.get("summary") or raw_text
    return (
        f"✅ 已记录反馈\n"
        f"  {symbol}: {tag_text}\n"
        f"  📝 {summary}\n"
        f"  💡 后续纠偏: {guidance}"
    )


def _handle_future_trade_plan(text: str) -> str:
    identifier = _extract_stock_identifier(text)
    if not identifier:
        return "识别到了明日计划，但没识别出股票名称，请补一句股票名。"

    holdings = _get_holdings_snapshot()
    code = _resolve_portfolio_holding_code(identifier, holdings) or _resolve_stock_code(identifier)
    symbol = identifier
    if code:
        symbol = f"{_get_stock_name(code)}({code})"

    action = "调仓"
    compact = _compact_text(text)
    if "清仓" in compact:
        action = "清仓"
    elif "减仓" in compact or "卖出" in compact:
        action = "减仓"
    elif "加仓" in compact or "补仓" in compact or "买入" in compact:
        action = "加仓"

    return (
        f"📝 已识别为明日计划：{symbol} {action}\n"
        "  这条消息不会修改当前持仓。\n"
        "  如果你要立刻落账，请发送不带“明天/计划”的执行指令。"
    )


def _upsert_holding_snapshot(code: str, shares: int, price: float, raw_text: str = "") -> str:
    if not is_valid_a_share_lot(code, shares):
        return f"❌ {code} 的同步数量不是整手，已跳过。"

    name = _get_stock_name(code)
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])
    existing = next((h for h in holdings if h.get("code") == code), None)

    if existing:
        existing["shares"] = shares
        existing["current_price"] = price
        existing["market_value"] = round(shares * price, 2)
        if not float(existing.get("cost_price", 0) or 0):
            existing["cost_price"] = price
        action = "同步持仓"
    else:
        holdings.append({
            "code": code,
            "name": name,
            "shares": shares,
            "cost_price": price,
            "current_price": price,
            "market_value": round(shares * price, 2),
            "sector": _guess_sector(raw_text, code),
            "buy_date": datetime.now().strftime("%Y-%m-%d"),
            "strategy_tag": "短线",
            "sellable_shares": shares,
        })
        action = "新增持仓"

    portfolio["holdings"] = holdings
    _refresh_portfolio_summary(portfolio)
    save_portfolio(portfolio)
    _clear_portfolio_llm_cache()
    return f"{action}: {name}({code}) {shares}股 @ {price}元"


def _handle_batch_portfolio_sync(text: str) -> str:
    results = []
    unhandled = []
    for segment in _split_batch_portfolio_segments(text):
        handled = False

        feedback_command = _build_feedback_command_from_text(segment)
        if feedback_command:
            results.append(_handle_trade_feedback(feedback_command, segment))
            handled = True
        elif _looks_like_future_trade_plan(segment):
            results.append(_handle_future_trade_plan(segment))
            handled = True
        else:
            action = _detect_trade_action(segment)
            if action == "buy":
                results.append(_handle_buy(segment))
                handled = True
            elif action == "sell":
                results.append(_handle_sell(segment))
                handled = True
            elif action == "clear":
                results.append(_handle_clear(segment))
                handled = True
            else:
                snapshot = _parse_snapshot_segment(segment)
                if snapshot:
                    holdings = _get_holdings_snapshot()
                    code = (
                        _resolve_portfolio_holding_code(snapshot["identifier"], holdings)
                        or _resolve_stock_code(snapshot["identifier"])
                    )
                    if code:
                        results.append(
                            _upsert_holding_snapshot(
                                code=code,
                                shares=int(snapshot["shares"]),
                                price=float(snapshot["price"]),
                                raw_text=segment,
                            )
                        )
                    else:
                        unhandled.append(f"{segment}（无法识别股票）")
                    handled = True

        if not handled:
            unhandled.append(segment)

    if not results and unhandled:
        return "识别到了持仓同步类消息，但没有解析出可执行项，请拆成更短的几句再发。"

    lines = ["✅ 已处理批量持仓同步"]
    for item in results:
        first_line = str(item or "").splitlines()[0]
        lines.append(f"  - {first_line}")
    if unhandled:
        lines.append("⚠️ 未处理片段:")
        for segment in unhandled[:4]:
            lines.append(f"  - {segment}")
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 买入
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _get_stock_name(code: str) -> str:
    """通过腾讯行情获取股票名称"""
    try:
        from macro_data_collector import _fetch_tencent_quote, _stock_code_to_tencent
        tc = _stock_code_to_tencent(code)
        q = _fetch_tencent_quote([tc])
        return q.get(tc, {}).get("name", code)
    except:
        return code


def _handle_buy(text: str) -> str:
    parsed, error = _parse_trade_order(text)
    if error:
        return error

    code = parsed["code"]
    shares = int(parsed["shares"])
    price = float(parsed["price"])
    return _execute_buy_order(code, shares, price, raw_text=text)


def _execute_buy_order(code: str, shares: int, price: float, raw_text: str = "") -> str:
    if not is_valid_a_share_lot(code, shares):
        return "❌ A股买入数量必须是100股整数倍（1手=100股），例如 100股、200股。"
    name = _get_stock_name(code)

    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])

    # 检查是否已持有
    existing = None
    for h in holdings:
        if h["code"] == code:
            existing = h
            break

    cost = round(shares * price, 2)

    if existing:
        # 加仓：重新计算成本价
        old_total = existing["shares"] * existing["cost_price"]
        new_total = old_total + cost
        existing["shares"] += shares
        existing["cost_price"] = round(new_total / existing["shares"], 3)
        existing["current_price"] = price
        existing["market_value"] = round(existing["shares"] * price, 2)
        action = "加仓"
    else:
        # 新建仓位
        # 猜板块
        sector = _guess_sector(raw_text, code)
        holdings.append({
            "code": code,
            "name": name,
            "shares": shares,
            "cost_price": price,
            "current_price": price,
            "market_value": round(shares * price, 2),
            "sector": sector,
            "buy_date": datetime.now().strftime("%Y-%m-%d"),
            "strategy_tag": "短线",
        })
        action = "建仓"

    # 扣现金
    portfolio["cash"] = round(portfolio.get("cash", 0) - cost, 2)
    portfolio["holdings"] = holdings
    _refresh_portfolio_summary(portfolio)
    save_portfolio(portfolio)
    _clear_portfolio_llm_cache()
    try:
        from trade_journal import record_buy

        record_buy(
            code=code,
            name=name,
            shares=shares,
            price=price,
            sector=existing.get("sector", "") if existing else _guess_sector(raw_text, code),
            source="portfolio_bot",
            note=raw_text or action,
        )
    except Exception as exc:
        logger.debug("记录买入交易日志失败: %s", exc)

    return (
        f"✅ {action}成功\n"
        f"  {name}({code}) {shares}股 × {price}元 = {cost}元\n"
        f"  剩余现金: {portfolio['cash']}元"
    )


def _guess_sector(text: str, code: str) -> str:
    """从指令中提取板块，或设为未知"""
    # 简单匹配
    for kw, sector in [
        ("光伏", "光伏"), ("新能源", "新能源"), ("芯片", "半导体"),
        ("电力", "电力"), ("军工", "军工"), ("医药", "医药"),
        ("传媒", "文化传媒"), ("消费", "消费"), ("白酒", "白酒"),
    ]:
        if kw in text:
            return sector
    return "未分类"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 卖出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_sell(text: str) -> str:
    parsed, error = _parse_trade_order(text)
    if error:
        return error

    code = parsed["code"]
    shares = int(parsed["shares"])
    price = float(parsed["price"])
    return _execute_sell_order(code, shares, price)


def _execute_sell_order(code: str, shares: int, price: float) -> str:
    if not is_valid_a_share_lot(code, shares):
        return "❌ A股卖出数量必须是100股整数倍（1手=100股），例如 100股、200股。"

    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])

    target = None
    for h in holdings:
        if h["code"] == code:
            target = h
            break

    if not target:
        return f"❌ 未找到持仓 {code}"

    if shares > target["shares"]:
        return f"❌ 持仓不足，当前只有 {target['shares']} 股"

    income = round(shares * price, 2)
    pnl = round((price - target["cost_price"]) * shares, 2)
    pnl_pct = round((price - target["cost_price"]) / target["cost_price"] * 100, 2)
    trade_name = target.get("name", code)
    trade_cost_price = float(target.get("cost_price", 0) or 0)

    target["shares"] -= shares
    target["current_price"] = price
    target["market_value"] = round(target["shares"] * price, 2)

    # 如果卖光了就删掉
    if target["shares"] <= 0:
        holdings = [h for h in holdings if h["code"] != code]

    portfolio["cash"] = round(portfolio.get("cash", 0) + income, 2)
    portfolio["holdings"] = holdings
    _refresh_portfolio_summary(portfolio)
    save_portfolio(portfolio)
    _clear_portfolio_llm_cache()
    try:
        from trade_journal import record_sell

        record_sell(
            code=code,
            name=trade_name,
            shares=shares,
            price=price,
            cost_price=trade_cost_price,
            source="portfolio_bot",
            note="manual_sell",
        )
    except Exception as exc:
        logger.debug("记录卖出交易日志失败: %s", exc)

    emoji = "🟢" if pnl >= 0 else "🔴"
    return (
        f"✅ 卖出成功\n"
        f"  {target.get('name', code)}({code}) {shares}股 × {price}元 = {income}元\n"
        f"  {emoji} 盈亏: {pnl}元 ({pnl_pct}%)\n"
        f"  剩余持仓: {target['shares']}股\n"
        f"  现金: {portfolio['cash']}元"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 清仓
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_clear(text: str) -> str:
    code, error = _parse_clear_target(text)
    if error:
        return error
    return _execute_clear_order(code)


def _execute_clear_order(code: str) -> str:
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])

    target = None
    for h in holdings:
        if h["code"] == code:
            target = h
            break

    if not target:
        return f"❌ 未找到持仓 {code}"

    # 用当前价估算
    try:
        from macro_data_collector import _fetch_tencent_quote, _stock_code_to_tencent
        tc = _stock_code_to_tencent(code)
        q = _fetch_tencent_quote([tc]).get(tc, {})
        current_price = _pick_quote_price(q) or target.get("current_price", 0)
    except:
        current_price = target.get("current_price", 0)

    income = round(target["shares"] * current_price, 2)
    pnl = round((current_price - target["cost_price"]) * target["shares"], 2)
    cleared_shares = int(target["shares"])
    target_name = target.get("name", code)
    cost_price = float(target.get("cost_price", 0) or 0)

    holdings = [h for h in holdings if h["code"] != code]
    portfolio["cash"] = round(portfolio.get("cash", 0) + income, 2)
    portfolio["holdings"] = holdings
    _refresh_portfolio_summary(portfolio)
    save_portfolio(portfolio)
    _clear_portfolio_llm_cache()
    try:
        from trade_journal import record_sell

        record_sell(
            code=code,
            name=target_name,
            shares=cleared_shares,
            price=current_price,
            cost_price=cost_price,
            source="portfolio_bot",
            note="manual_clear",
        )
    except Exception as exc:
        logger.debug("记录清仓交易日志失败: %s", exc)

    emoji = "🟢" if pnl >= 0 else "🔴"
    return (
        f"✅ 清仓完成\n"
        f"  {target.get('name', code)}({code}) {target['shares']}股\n"
        f"  {emoji} 估算盈亏: {pnl}元\n"
        f"  回收现金: {income}元\n"
        f"  总现金: {portfolio['cash']}元"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 查看持仓
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_show_portfolio() -> str:
    from portfolio_manager import update_current_prices
    from macro_data_collector import _fetch_tencent_quote, _stock_code_to_tencent

    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])
    if not holdings:
        return "📋 当前无持仓"

    # 更新实时价格
    tc_codes = [_stock_code_to_tencent(h["code"]) for h in holdings]
    quotes = _fetch_tencent_quote(tc_codes)
    price_map = {}
    for h in holdings:
        tc = _stock_code_to_tencent(h["code"])
        q = quotes.get(tc, {})
        latest_price = _pick_quote_price(q)
        if latest_price > 0:
            price_map[h["code"]] = latest_price

    portfolio = update_current_prices(portfolio, price_map)

    lines = [f"📋 **当前持仓** ({len(holdings)}只)"]
    lines.append(f"💰 现金: {portfolio.get('cash', 0):.2f}元")
    lines.append(f"📊 总资产: {portfolio.get('total_asset', 0):.2f}元")
    lines.append(f"📈 仓位: {portfolio.get('actual_position_ratio', 0)*100:.1f}%")
    lines.append("")

    total_pnl = 0
    for h in holdings:
        pnl = h.get("pnl_pct", 0)
        total_pnl += (h.get("current_price", 0) - h.get("cost_price", 0)) * h.get("shares", 0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        lines.append(
            f"{emoji} {h.get('name','')}({h['code']}) "
            f"{h.get('shares',0)}股 "
            f"成本:{h.get('cost_price',0):.3f} "
            f"现价:{h.get('current_price',0):.2f} "
            f"{pnl:+.2f}%"
        )

    total_emoji = "🟢" if total_pnl >= 0 else "🔴"
    lines.append(f"\n{total_emoji} 总盈亏: {total_pnl:.2f}元")
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 触发调仓分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_rebalance() -> str:
    try:
        from rebalance_engine import run_rebalance_analysis
        from portfolio_manager import format_rebalance_report
        from src.config import get_config
        config = get_config()
        result = run_rebalance_analysis(config=config)
        if "error" in result:
            return f"❌ 调仓分析失败: {result['error']}"
        return format_rebalance_report(result)
    except Exception as e:
        return f"❌ 调仓分析异常: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 触发全市场扫描
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_scan() -> str:
    try:
        from market_scanner import scan_market
        results = scan_market(max_price=10.0, min_turnover=2.0, top_n=15, mode="trend")
        if not results:
            return "📡 扫描完成，未找到符合条件的候选股"

        lines = [f"📡 **全市场扫描** (Top {len(results)})", ""]
        for s in results:
            lines.append(
                f"  {s['code']} {s['name']} {s['price']:.2f}元 "
                f"涨跌:{s['change_pct']:+.1f}% "
                f"{s['ma_trend']} {s['macd_signal']} "
                f"得分:{s['tech_score']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 扫描失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主线板块
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_mainline() -> str:
    try:
        from data_store import get_sector_mainline, get_consecutive_inflow_stocks
        sectors = get_sector_mainline(min_days=2)
        stocks = get_consecutive_inflow_stocks(min_days=2, min_total=300)

        lines = ["🔥 **主线板块分析**", ""]
        if sectors:
            lines.append("板块（连续资金流入）:")
            for s in sectors[:5]:
                lines.append(
                    f"  {s['sector_name']} — {s['inflow_days']}天 "
                    f"累计:{s['total_net']:.0f}万"
                )
        else:
            lines.append("板块数据不足（需积累2+天）")

        if stocks:
            lines.append("\n个股（连续主力流入）:")
            for s in stocks[:5]:
                lines.append(
                    f"  {s['code']} {s['name']} — {s['inflow_days']}天 "
                    f"累计:{s['total_net']:.0f}万"
                )

        return "\n".join(lines)
    except Exception as e:
        return f"❌ 主线分析失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 回测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_backtest() -> str:
    try:
        from data_store import get_backtest_summary
        summary = get_backtest_summary(30)
        if not summary:
            return "📊 暂无回测数据（需积累数天扫描结果）"

        lines = ["📊 **扫描策略回测** (近30天)", ""]
        for mode, s in summary.items():
            lines.append(
                f"  [{mode}] 样本:{s['total']} "
                f"3日胜率:{s['win_rate_3d']}% "
                f"5日胜率:{s['win_rate_5d']}% "
                f"均收益:{s['avg_return_3d']}%"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 回测查询失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 手动预警
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_alert() -> str:
    try:
        from market_monitor import check_market_anomaly, format_anomaly_alert
        result = check_market_anomaly()
        alert = format_anomaly_alert(result)
        if alert:
            return alert
        return "✅ 当前无异动"
    except Exception as e:
        return f"❌ 预警检测失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 增强分析（新+旧结合）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYZE_PATTERN = re.compile(r'分析\s*(\d{6})', re.IGNORECASE)

def _handle_analyze(text: str) -> str:
    """增强版个股分析：技术面+资金面+新闻+AI"""
    m = ANALYZE_PATTERN.search(text)
    if not m:
        return "格式：分析 002506"
    code = m.group(1)
    try:
        from analysis_bridge import enhanced_stock_analysis, format_enhanced_report
        data = enhanced_stock_analysis(code)
        return format_enhanced_report(data)
    except Exception as e:
        return f"❌ 分析失败: {str(e)}"


def _handle_full_analysis() -> str:
    """全量分析：持仓+扫描+调仓"""
    try:
        from analysis_bridge import run_full_enhanced_analysis
        return run_full_enhanced_analysis()
    except Exception as e:
        return f"❌ 全量分析失败: {str(e)}"
