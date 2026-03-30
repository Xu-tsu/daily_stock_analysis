# -*- coding: utf-8 -*-
"""Persist manual trade feedback and surface it back into future prompts."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CODE_RE = re.compile(r"^\d{6}$")

FEEDBACK_TAG_ALIASES = {
    "sold_too_early": "sold_too_early",
    "sell_too_early": "sold_too_early",
    "卖飞": "sold_too_early",
    "清仓过早": "sold_too_early",
    "止盈过早": "sold_too_early",
    "stopped_out_then_rebounded": "stopped_out_then_rebounded",
    "stop_out_then_rebound": "stopped_out_then_rebounded",
    "止损后拉回": "stopped_out_then_rebounded",
    "止损后反弹": "stopped_out_then_rebounded",
    "dip_buy_success": "dip_buy_success",
    "低吸成功": "dip_buy_success",
    "分歧低吸成功": "dip_buy_success",
    "bought_too_early": "bought_too_early",
    "抄底过早": "bought_too_early",
    "左侧过早": "bought_too_early",
    "chased_high_then_dumped": "chased_high_then_dumped",
    "追高被砸": "chased_high_then_dumped",
    "追高回撤": "chased_high_then_dumped",
    "good_exit": "good_exit",
    "卖出正确": "good_exit",
    "退出正确": "good_exit",
    "other": "other",
    "其他": "other",
}

FEEDBACK_GUIDANCE = {
    "sold_too_early": "近期有卖飞样本；下次若主线和盘口仍强，优先分批止盈并保留底仓做T，不再一刀切清仓。",
    "stopped_out_then_rebounded": "近期有止损后反抽样本；下次先区分量化洗盘和趋势破坏，关键位确认失守后再执行。",
    "dip_buy_success": "近期分歧低吸反馈较好；在主线未坏且板块回流时，可更重视回踩承接而不是追涨。",
    "bought_too_early": "近期有左侧接早样本；下次低吸前先确认板块回流、竞价承接和量价二次确认。",
    "chased_high_then_dumped": "近期有追高回撤样本；下次避免情绪高潮接力，优先等回踩或换手确认。",
    "good_exit": "近期也有正确兑现样本；拥挤高潮和板块转弱时，继续保持主动兑现纪律。",
    "other": "把这条反馈作为执行纠偏样本，和市场环境、板块强弱一起综合使用。",
}


def _db_path() -> str:
    return os.getenv("SCANNER_DB_PATH", "data/scanner_history.db")


def _conn() -> sqlite3.Connection:
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_reference_action(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()
    if normalized in {"reduce"}:
        return "sell"
    if normalized in {"sell", "clear", "buy", "hold"}:
        return normalized
    return ""


def _normalize_feedback_tag(value: Any) -> str:
    if not isinstance(value, str):
        return "other"
    normalized = value.strip().lower()
    if not normalized:
        return "other"
    return FEEDBACK_TAG_ALIASES.get(normalized, FEEDBACK_TAG_ALIASES.get(value.strip(), "other"))


def init_trade_feedback_table() -> None:
    conn = _conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trade_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_date TEXT NOT NULL,
            code TEXT,
            name TEXT,
            reference_action TEXT,
            reference_price REAL,
            outcome_price REAL,
            outcome_delta REAL,
            outcome_delta_pct REAL,
            feedback_tag TEXT,
            related_trade_id INTEGER,
            raw_text TEXT NOT NULL,
            summary TEXT,
            source TEXT DEFAULT 'feishu',
            created_at TEXT DEFAULT (datetime('now','localtime'))
        );

        CREATE INDEX IF NOT EXISTS idx_trade_feedback_created_at
        ON trade_feedback(created_at);

        CREATE INDEX IF NOT EXISTS idx_trade_feedback_code
        ON trade_feedback(code);
        """
    )
    conn.commit()
    conn.close()


def get_recent_trade_context(limit: int = 12) -> list[dict]:
    try:
        from trade_journal import init_trade_tables

        init_trade_tables()
    except Exception:
        pass

    conn = _conn()
    try:
        rows = conn.execute(
            """
            SELECT id, trade_date, trade_type, code, name, shares, price, amount, pnl, pnl_pct, created_at
            FROM trade_log
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _resolve_stock_code(stock_hint: str) -> str:
    if not stock_hint:
        return ""
    normalized = stock_hint.strip()
    if CODE_RE.fullmatch(normalized):
        return normalized
    try:
        from src.services.name_to_code_resolver import resolve_name_to_code

        return resolve_name_to_code(normalized) or ""
    except Exception:
        return ""


def _resolve_related_trade(
    *,
    stock_hint: str,
    reference_action: str,
    reference_price: Optional[float],
    recent_trades: list[dict],
) -> Optional[dict]:
    candidates = list(recent_trades or [])
    if not candidates:
        return None

    if stock_hint:
        normalized = stock_hint.strip()
        filtered = [
            item
            for item in candidates
            if normalized in {
                str(item.get("code", "") or "").strip(),
                str(item.get("name", "") or "").strip(),
            }
        ]
        if filtered:
            candidates = filtered

    if reference_action in {"sell", "clear"}:
        sell_candidates = [item for item in candidates if str(item.get("trade_type", "")).lower() == "sell"]
        if sell_candidates:
            candidates = sell_candidates
    elif reference_action == "buy":
        buy_candidates = [item for item in candidates if str(item.get("trade_type", "")).lower() == "buy"]
        if buy_candidates:
            candidates = buy_candidates

    if reference_price and reference_price > 0:
        candidates = sorted(
            candidates,
            key=lambda item: abs(float(item.get("price", 0) or 0) - reference_price),
        )
        best = candidates[0]
        best_diff = abs(float(best.get("price", 0) or 0) - reference_price)
        tolerance = max(0.05, reference_price * 0.02)
        if best_diff <= tolerance:
            return best
        return None

    return candidates[0] if len(candidates) == 1 else None


def _infer_feedback_tag(
    *,
    raw_text: str,
    reference_action: str,
    reference_price: Optional[float],
    outcome_price: Optional[float],
) -> str:
    compact = (raw_text or "").replace(" ", "")
    if "卖飞" in compact:
        return "sold_too_early"
    if "止损" in compact and any(word in compact for word in ("反弹", "拉回", "拉升", "被拉")):
        return "stopped_out_then_rebounded"
    if any(word in compact for word in ("追高", "高开追", "冲高买")) and any(
        word in compact for word in ("回落", "被砸", "跳水")
    ):
        return "chased_high_then_dumped"
    if any(word in compact for word in ("低吸", "回踩买", "分歧买")) and any(
        word in compact for word in ("拉升", "涨停", "冲高")
    ):
        return "dip_buy_success"

    if not reference_price or not outcome_price:
        return "other"

    delta_pct = ((outcome_price - reference_price) / reference_price) * 100
    if reference_action in {"sell", "clear"}:
        if delta_pct >= 1.0:
            return "sold_too_early"
        return "good_exit"
    if reference_action == "buy":
        if delta_pct >= 2.0:
            return "dip_buy_success"
        if delta_pct <= -2.0:
            return "bought_too_early"
    return "other"


def _compose_summary(
    *,
    name: str,
    code: str,
    reference_action: str,
    reference_price: Optional[float],
    outcome_price: Optional[float],
    feedback_tag: str,
    raw_text: str,
) -> str:
    symbol = name or code or "该笔交易"
    action_text = {
        "buy": "买入",
        "sell": "卖出",
        "clear": "清仓",
        "hold": "持有",
    }.get(reference_action, "操作")
    if reference_price and outcome_price:
        delta_pct = ((outcome_price - reference_price) / reference_price) * 100
        tag_text = {
            "sold_too_early": "卖飞了",
            "stopped_out_then_rebounded": "止损后被拉回",
            "dip_buy_success": "低吸后兑现成功",
            "bought_too_early": "抄底偏早",
            "chased_high_then_dumped": "追高后回撤",
            "good_exit": "卖出时机基本正确",
            "other": "形成了一条复盘反馈",
        }.get(feedback_tag, "形成了一条复盘反馈")
        return (
            f"{symbol}{action_text}参考价 {reference_price:.3f}，随后走到 {outcome_price:.3f}"
            f"（{delta_pct:+.2f}%），说明这次{tag_text}。"
        )
    if raw_text:
        return raw_text.strip()
    return f"{symbol}{action_text}后新增了一条人工反馈。"


def build_feedback_guidance(feedback_tag: str) -> str:
    return FEEDBACK_GUIDANCE.get(feedback_tag or "other", FEEDBACK_GUIDANCE["other"])


def record_trade_feedback(
    *,
    raw_text: str,
    parsed_feedback: dict,
    source: str = "feishu",
) -> dict:
    init_trade_feedback_table()

    stock_hint = str(parsed_feedback.get("stock_hint", "") or "").strip()
    reference_action = _normalize_reference_action(parsed_feedback.get("reference_action"))
    reference_price = _to_float(parsed_feedback.get("price"))
    outcome_price = _to_float(parsed_feedback.get("outcome_price"))

    recent_trades = get_recent_trade_context(limit=12)
    related_trade = _resolve_related_trade(
        stock_hint=stock_hint,
        reference_action=reference_action,
        reference_price=reference_price,
        recent_trades=recent_trades,
    )

    code = stock_hint if CODE_RE.fullmatch(stock_hint) else ""
    name = stock_hint if stock_hint and not CODE_RE.fullmatch(stock_hint) else ""
    if related_trade:
        code = code or str(related_trade.get("code", "") or "").strip()
        name = name or str(related_trade.get("name", "") or "").strip()
        if not reference_price:
            reference_price = _to_float(related_trade.get("price"))
    elif stock_hint and not code and not name:
        code = _resolve_stock_code(stock_hint)

    if not code and not name:
        return {
            "saved": False,
            "needs_clarification": True,
            "clarification": "识别到了复盘反馈，但暂时无法判断对应的是哪只股票。下次可以补一句股票名，或者先让系统记录买卖后再反馈。",
        }

    feedback_tag = _normalize_feedback_tag(parsed_feedback.get("feedback_tag"))
    if feedback_tag == "other":
        feedback_tag = _infer_feedback_tag(
            raw_text=raw_text,
            reference_action=reference_action,
            reference_price=reference_price,
            outcome_price=outcome_price,
        )

    outcome_delta = None
    outcome_delta_pct = None
    if reference_price and outcome_price:
        outcome_delta = round(outcome_price - reference_price, 4)
        outcome_delta_pct = round(outcome_delta / reference_price * 100, 2)

    summary = str(parsed_feedback.get("summary", "") or "").strip()
    if not summary:
        summary = _compose_summary(
            name=name,
            code=code,
            reference_action=reference_action,
            reference_price=reference_price,
            outcome_price=outcome_price,
            feedback_tag=feedback_tag,
            raw_text=raw_text,
        )

    conn = _conn()
    conn.execute(
        """
        INSERT INTO trade_feedback (
            feedback_date, code, name, reference_action, reference_price,
            outcome_price, outcome_delta, outcome_delta_pct, feedback_tag,
            related_trade_id, raw_text, summary, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            code or None,
            name or None,
            reference_action or None,
            reference_price,
            outcome_price,
            outcome_delta,
            outcome_delta_pct,
            feedback_tag,
            related_trade.get("id") if related_trade else None,
            raw_text.strip(),
            summary,
            source,
        ),
    )
    feedback_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()

    guidance = build_feedback_guidance(feedback_tag)
    logger.info(
        "[TradeFeedback] saved id=%s code=%s tag=%s ref=%s outcome=%s",
        feedback_id,
        code or name,
        feedback_tag,
        reference_price,
        outcome_price,
    )
    return {
        "saved": True,
        "id": feedback_id,
        "code": code,
        "name": name,
        "reference_action": reference_action,
        "reference_price": reference_price,
        "outcome_price": outcome_price,
        "outcome_delta_pct": outcome_delta_pct,
        "feedback_tag": feedback_tag,
        "summary": summary,
        "guidance": guidance,
    }


def get_recent_feedback(limit: int = 6) -> list[dict]:
    init_trade_feedback_table()
    conn = _conn()
    rows = conn.execute(
        """
        SELECT id, feedback_date, code, name, reference_action, reference_price,
               outcome_price, outcome_delta, outcome_delta_pct, feedback_tag,
               raw_text, summary, source, created_at
        FROM trade_feedback
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def format_feedback_for_prompt(limit: int = 6) -> str:
    rows = get_recent_feedback(limit=limit)
    if not rows:
        return ""

    lines = ["这些是真实盘后通过飞书回灌的人工反馈，请把它们视为执行纠偏样本，而不是孤立噪音。"]
    counts = Counter()
    for row in rows:
        tag = _normalize_feedback_tag(row.get("feedback_tag"))
        counts[tag] += 1
        symbol = row.get("name") or row.get("code") or "未识别标的"
        metrics = []
        if row.get("reference_price"):
            metrics.append(f"参考价 {float(row['reference_price']):.3f}")
        if row.get("outcome_price"):
            metrics.append(f"随后 {float(row['outcome_price']):.3f}")
        if row.get("outcome_delta_pct") not in (None, ""):
            metrics.append(f"偏差 {float(row['outcome_delta_pct']):+.2f}%")
        metric_text = "；".join(metrics)
        lines.append(
            f"- {row.get('feedback_date', '')} {symbol}: {row.get('summary') or row.get('raw_text')}"
            + (f"（{metric_text}）" if metric_text else "")
        )

    guidance_lines = []
    for tag, _count in counts.most_common():
        guidance = build_feedback_guidance(tag)
        if guidance and guidance not in guidance_lines:
            guidance_lines.append(guidance)
    if guidance_lines:
        lines.append("请优先吸收这些纠偏：")
        for item in guidance_lines[:4]:
            lines.append(f"- {item}")

    return "\n".join(lines)
