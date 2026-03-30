# -*- coding: utf-8 -*-
"""Natural-language portfolio command interpreter backed by a local LLM."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

from src.config import (
    detect_local_ollama_models,
    extra_litellm_params,
    get_api_keys_for_model,
    get_config,
)
from src.services.trade_feedback_service import get_recent_trade_context

logger = logging.getLogger(__name__)


class _LiteLLMPlaceholder:
    """Patchable placeholder before litellm is imported."""

    completion = None


litellm = sys.modules.get("litellm") or _LiteLLMPlaceholder()

COMMAND_SCHEMA = {
    "is_portfolio_command": False,
    "action": "other",
    "stock_hint": "",
    "shares": None,
    "price": None,
    "outcome_price": None,
    "reference_action": "",
    "feedback_tag": "other",
    "summary": "",
    "needs_clarification": False,
    "clarification": "",
    "confidence": 0.0,
    "reason": "",
}

VALID_ACTIONS = {
    "buy",
    "sell",
    "clear",
    "feedback",
    "show_portfolio",
    "rebalance",
    "scan",
    "mainline",
    "backtest",
    "alert",
    "performance",
    "pattern_review",
    "trade_history",
    "other",
}

ACTION_ALIASES = {
    "portfolio": "show_portfolio",
    "show": "show_portfolio",
    "view_portfolio": "show_portfolio",
    "view": "show_portfolio",
    "holdings": "show_portfolio",
    "持仓": "show_portfolio",
    "调仓": "rebalance",
    "扫描": "scan",
    "主线": "mainline",
    "回测": "backtest",
    "预警": "alert",
    "feedback": "feedback",
    "复盘反馈": "feedback",
    "反馈": "feedback",
}

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

PORTFOLIO_COMMAND_SYSTEM_PROMPT = """你是 A 股持仓机器人里的“口令识别器”，只做一件事：把用户自然语言识别成结构化命令。
你必须严格只返回一个 JSON 对象，不要 markdown，不要解释，不要多余文字。

支持动作：
- buy: 买入 / 加仓 / 补仓 / 再买
- sell: 卖出 / 减仓 / 先出一部分
- clear: 清仓 / 全卖 / 卖光
- feedback: 用户在复盘上一次交易结果，希望系统吸收经验并改进下次判断
- show_portfolio: 查看持仓 / 看看仓位
- rebalance / scan / mainline / backtest / alert / performance / pattern_review / trade_history
- other: 不是持仓机器人命令，只是聊天、问行情、问机会

输出 JSON schema：
{
  "is_portfolio_command": true,
  "action": "buy|sell|clear|feedback|show_portfolio|rebalance|scan|mainline|backtest|alert|performance|pattern_review|trade_history|other",
  "stock_hint": "股票名称或代码，没有则空字符串",
  "shares": 300,
  "price": 5.05,
  "outcome_price": 7.77,
  "reference_action": "buy|sell|clear|hold|other",
  "feedback_tag": "sold_too_early|stopped_out_then_rebounded|dip_buy_success|bought_too_early|chased_high_then_dumped|good_exit|other",
  "summary": "一句话总结这条反馈",
  "needs_clarification": false,
  "clarification": "",
  "confidence": 0.96,
  "reason": "一句话说明判断依据"
}

规则：
1. 只要用户是在执行持仓动作、查看当前持仓/仓位、或者对上一笔交易做复盘反馈，就判定 is_portfolio_command=true。
2. 普通聊天、问行情、问龙头、问机会、让你推荐股票，不算持仓命令，返回 action=other。
3. A 股数量统一输出为 shares（股数），不要输出“手”；3手必须输出成 300。
4. 用户说“全卖”“卖光”“清掉”这类，action=clear，shares 置空。
5. 如果用户描述的是过去已经发生的交易结果，例如“7.61 清仓后立刻被拉升到 7.77”“止损后马上反弹”，这是 feedback，不是新的卖出命令。
6. 对语音转文字口误、同音字、近音字，要结合当前持仓列表和最近交易记录优先推断，例如“写信继承”优先理解成“协鑫集成”。
7. 如果是明确的持仓命令，但缺少必要信息（例如加仓没说价格，减仓没说数量），则 needs_clarification=true，并给出 clarification。
8. 如果是查看仓位/持仓，不需要股票名、股数、价格。
9. feedback 的 summary 要总结“这次交易哪里做对/做错了”；feedback_tag 选最贴切的一个。
10. confidence 取值 0 到 1。
"""


def _strip_json_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = _JSON_FENCE_RE.sub("", cleaned).strip()
    return cleaned


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _normalize_action(action: Any) -> str:
    if not isinstance(action, str):
        return "other"
    normalized = action.strip().lower()
    if normalized in ACTION_ALIASES:
        normalized = ACTION_ALIASES[normalized]
    if normalized not in VALID_ACTIONS:
        return "other"
    return normalized


def _normalize_response(data: dict, raw_text: str) -> dict:
    normalized = dict(COMMAND_SCHEMA)

    action = _normalize_action(data.get("action"))
    normalized["action"] = action
    normalized["is_portfolio_command"] = _bool_value(data.get("is_portfolio_command")) or action != "other"

    stock_hint = data.get("stock_hint")
    normalized["stock_hint"] = stock_hint.strip() if isinstance(stock_hint, str) else ""

    shares = _to_int(data.get("shares"))
    if shares is not None and shares > 0 and "手" in raw_text and shares < 100:
        shares *= 100
    normalized["shares"] = shares if shares and shares > 0 else None

    price = _to_float(data.get("price"))
    normalized["price"] = round(price, 3) if price and price > 0 else None

    outcome_price = _to_float(data.get("outcome_price"))
    normalized["outcome_price"] = round(outcome_price, 3) if outcome_price and outcome_price > 0 else None

    reference_action = data.get("reference_action")
    normalized["reference_action"] = reference_action.strip().lower() if isinstance(reference_action, str) else ""

    feedback_tag = data.get("feedback_tag")
    normalized["feedback_tag"] = feedback_tag.strip() if isinstance(feedback_tag, str) and feedback_tag.strip() else "other"

    summary = data.get("summary")
    normalized["summary"] = summary.strip() if isinstance(summary, str) else ""

    normalized["needs_clarification"] = _bool_value(data.get("needs_clarification"))
    clarification = data.get("clarification")
    normalized["clarification"] = clarification.strip() if isinstance(clarification, str) else ""
    if normalized["needs_clarification"] and not normalized["clarification"]:
        normalized["clarification"] = "识别到了持仓操作，但缺少必要的价格或数量，请补充后再发一次。"

    confidence = _to_float(data.get("confidence"))
    normalized["confidence"] = max(0.0, min(1.0, confidence or 0.0))

    reason = data.get("reason")
    normalized["reason"] = reason.strip() if isinstance(reason, str) else ""
    return normalized


def _parse_llm_response(content: str, raw_text: str) -> Optional[dict]:
    cleaned = _strip_json_fence(content)
    if not cleaned:
        return None

    parsed: Optional[dict] = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            repaired = repair_json(cleaned, return_objects=True)
            if isinstance(repaired, dict):
                parsed = repaired
        except Exception:
            parsed = None

    if not isinstance(parsed, dict):
        return None
    return _normalize_response(parsed, raw_text)


def _resolve_local_command_model() -> str:
    cfg = get_config()
    candidates: List[str] = []

    explicit = os.getenv("REBALANCE_LOCAL_MODEL", "").strip()
    if explicit:
        candidates.append(explicit)

    primary = str(getattr(cfg, "litellm_model", "") or "").strip()
    if primary.startswith("ollama/"):
        candidates.append(primary)

    for fallback in list(getattr(cfg, "litellm_fallback_models", []) or []):
        fallback = str(fallback or "").strip()
        if fallback.startswith("ollama/"):
            candidates.append(fallback)

    for detected in detect_local_ollama_models():
        if detected:
            candidates.append(detected)

    seen = set()
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            return model
    return ""


def _get_litellm_client():
    global litellm
    if getattr(litellm, "completion", None) is not None:
        return litellm
    try:
        litellm = importlib.import_module("litellm")
    except ModuleNotFoundError:
        return None
    return litellm if getattr(litellm, "completion", None) is not None else None


def _call_local_command_llm(text: str, holdings: Optional[List[dict]] = None) -> Optional[tuple[str, str]]:
    model = _resolve_local_command_model()
    if not model:
        logger.info("[PortfolioLLM] 未检测到可用本地模型，跳过自然语言持仓识别")
        return None

    litellm_client = _get_litellm_client()
    if litellm_client is None:
        logger.debug("[PortfolioLLM] litellm completion unavailable")
        return None

    cfg = get_config()
    holdings = holdings or []
    recent_trades = get_recent_trade_context(limit=8)

    holdings_brief = [
        {
            "code": str(item.get("code", "") or "").strip(),
            "name": str(item.get("name", "") or "").strip(),
            "shares": int(item.get("shares", 0) or 0),
        }
        for item in holdings
    ]
    recent_trades_brief = [
        {
            "trade_date": str(item.get("trade_date", "") or "").strip(),
            "trade_type": str(item.get("trade_type", "") or "").strip(),
            "code": str(item.get("code", "") or "").strip(),
            "name": str(item.get("name", "") or "").strip(),
            "shares": int(item.get("shares", 0) or 0),
            "price": round(float(item.get("price", 0) or 0), 3),
        }
        for item in recent_trades
    ]

    user_prompt = (
        "当前持仓列表（优先用它来消歧和修正语音识别口误）：\n"
        f"{json.dumps(holdings_brief, ensure_ascii=False)}\n\n"
        "最近交易记录（优先用于理解“7.61 清仓后拉到 7.77”这类反馈）：\n"
        f"{json.dumps(recent_trades_brief, ensure_ascii=False)}\n\n"
        f"用户原始消息：{text}"
    )

    call_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": PORTFOLIO_COMMAND_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 320,
        "timeout": 25,
    }
    keys = get_api_keys_for_model(model, cfg)
    if keys:
        call_kwargs["api_key"] = keys[0]
    call_kwargs.update(extra_litellm_params(model, cfg))

    response = litellm_client.completion(**call_kwargs)
    try:
        content = response.choices[0].message.content
    except Exception:
        content = None
    if not content:
        return None
    return content, model


def interpret_portfolio_command(text: str, holdings: Optional[List[dict]] = None) -> Optional[dict]:
    """Interpret a free-form portfolio command using the configured local model."""
    raw_text = (text or "").strip()
    if not raw_text:
        return None

    try:
        result = _call_local_command_llm(raw_text, holdings=holdings)
        if not result:
            return None
        content, model = result
        parsed = _parse_llm_response(content, raw_text)
        if not parsed:
            logger.warning("[PortfolioLLM] 本地模型返回无法解析的内容: %s", content)
            return None
        parsed["model"] = model
        logger.info(
            "[PortfolioLLM] action=%s stock=%s shares=%s price=%s outcome=%s clarification=%s model=%s confidence=%.2f",
            parsed.get("action"),
            parsed.get("stock_hint"),
            parsed.get("shares"),
            parsed.get("price"),
            parsed.get("outcome_price"),
            parsed.get("needs_clarification"),
            model,
            parsed.get("confidence", 0.0),
        )
        return parsed
    except Exception as exc:
        logger.warning("[PortfolioLLM] 本地模型识别持仓口令失败: %s", exc)
        return None

