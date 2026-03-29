# -*- coding: utf-8 -*-
"""Helpers for A-share lot sizing and fee-aware execution estimates."""

from __future__ import annotations

import re
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.config import get_config

CN_LOT_SIZE = 100
DEFAULT_CN_FEE_RATE = 0.00031
DEFAULT_CN_SELL_TAX_RATE = 0.001
DEFAULT_CN_MIN_FEE = 5.0


@dataclass(frozen=True)
class TradeCostProfile:
    market: str = "cn"
    fee_rate: float = DEFAULT_CN_FEE_RATE
    sell_tax_rate: float = DEFAULT_CN_SELL_TAX_RATE
    min_fee: float = DEFAULT_CN_MIN_FEE
    source: str = "default"
    sample_size: int = 0


def is_a_share_code(code: Any) -> bool:
    text = str(code or "").strip()
    return text.isdigit() and len(text) == 6


def is_valid_a_share_lot(code: Any, shares: Any) -> bool:
    if not is_a_share_code(code):
        return True
    try:
        qty = int(shares)
    except (TypeError, ValueError):
        return False
    return qty > 0 and qty % CN_LOT_SIZE == 0


def round_down_to_lot(shares: Any, lot_size: int = CN_LOT_SIZE) -> int:
    try:
        qty = int(float(shares or 0))
    except (TypeError, ValueError):
        return 0
    if qty < lot_size:
        return 0
    return qty // lot_size * lot_size


def estimate_a_share_trade(
    *,
    shares: int,
    price: float,
    side: str,
    profile: Optional[TradeCostProfile] = None,
) -> Dict[str, float]:
    profile = profile or TradeCostProfile()
    qty = max(int(shares or 0), 0)
    px = max(float(price or 0.0), 0.0)
    gross = round(qty * px, 2)
    if gross <= 0:
        return {
            "gross_amount": 0.0,
            "fee": 0.0,
            "tax": 0.0,
            "cash_out": 0.0,
            "net_cash": 0.0,
        }

    fee = max(gross * float(profile.fee_rate), float(profile.min_fee))
    tax = gross * float(profile.sell_tax_rate) if str(side).lower() == "sell" else 0.0
    fee = round(fee, 2)
    tax = round(tax, 2)
    return {
        "gross_amount": gross,
        "fee": fee,
        "tax": tax,
        "cash_out": round(gross + fee + tax, 2),
        "net_cash": round(gross - fee - tax, 2),
    }


def max_affordable_a_share_shares(
    *,
    cash_budget: float,
    price: float,
    profile: Optional[TradeCostProfile] = None,
    lot_size: int = CN_LOT_SIZE,
) -> int:
    budget = float(cash_budget or 0.0)
    px = float(price or 0.0)
    if budget <= 0 or px <= 0:
        return 0

    candidate = round_down_to_lot(int(budget // px), lot_size=lot_size)
    while candidate >= lot_size:
        cost = estimate_a_share_trade(
            shares=candidate,
            price=px,
            side="buy",
            profile=profile,
        )
        if cost["cash_out"] <= budget + 1e-8:
            return candidate
        candidate -= lot_size
    return 0


def load_cn_trade_cost_profile(
    db_paths: Optional[Sequence[str]] = None,
) -> TradeCostProfile:
    config_path = str(getattr(get_config(), "database_path", "") or "").strip()
    paths = []
    for raw_path in [*(db_paths or []), config_path, "data/scanner_history.db"]:
        path = str(raw_path or "").strip()
        if path and path not in paths:
            paths.append(path)

    samples: List[Dict[str, float]] = []
    source = "default"

    for path in paths:
        if not Path(path).exists():
            continue
        rows, row_source = _load_trade_cost_rows(path)
        if rows:
            samples.extend(rows)
            if source == "default":
                source = row_source

    if not samples:
        return TradeCostProfile()

    fee_values = [float(row["fee"]) for row in samples if float(row["fee"]) > 0]
    fee_rows = [
        (float(row["gross"]), float(row["fee"]))
        for row in samples
        if float(row["gross"]) > 0 and float(row["fee"]) > 0
    ]
    tax_rows = [
        (float(row["gross"]), float(row["tax"]))
        for row in samples
        if str(row.get("side", "")).lower() == "sell"
        and float(row["gross"]) > 0
        and float(row["tax"]) > 0
    ]

    min_fee = DEFAULT_CN_MIN_FEE
    if fee_values:
        min_fee = min(max(min(fee_values), 0.0), 50.0)

    variable_fee_rates = [
        fee / gross
        for gross, fee in fee_rows
        if gross > 0 and fee > 0 and (gross >= 20000 or fee > min_fee * 1.15)
    ]
    variable_fee_rates = [rate for rate in variable_fee_rates if 0 < rate < 0.005]
    fee_rate = statistics.median(variable_fee_rates) if variable_fee_rates else DEFAULT_CN_FEE_RATE

    tax_rates = [tax / gross for gross, tax in tax_rows if gross > 0 and 0 < tax / gross < 0.01]
    sell_tax_rate = statistics.median(tax_rates) if tax_rates else DEFAULT_CN_SELL_TAX_RATE

    return TradeCostProfile(
        fee_rate=float(fee_rate),
        sell_tax_rate=float(sell_tax_rate),
        min_fee=float(min_fee),
        source=source,
        sample_size=len(samples),
    )


def extract_amount_hint(*texts: Any) -> Optional[float]:
    amounts: List[float] = []
    for text in texts:
        if not text:
            continue
        for match in re.findall(r"(\d+(?:\.\d+)?)\s*(?:元|块)", str(text)):
            try:
                amounts.append(float(match))
            except ValueError:
                continue
    return max(amounts) if amounts else None


def extract_percent_hint(*texts: Any, prefer: str = "max") -> Optional[float]:
    values: List[float] = []
    for text in texts:
        if not text:
            continue
        for match in re.findall(r"(\d+(?:\.\d+)?)\s*%", str(text)):
            try:
                values.append(float(match))
            except ValueError:
                continue
    if not values:
        return None
    return max(values) if prefer == "max" else min(values)


def extract_reference_price(*texts: Any, fallback: Optional[float] = None) -> Optional[float]:
    for text in texts:
        if text is None or text == "":
            continue
        if isinstance(text, (int, float)):
            value = float(text)
            if value > 0:
                return value
            continue
        values: List[float] = []
        for item in re.findall(r"\d+(?:\.\d+)?", str(text)):
            try:
                number = float(item)
            except ValueError:
                continue
            if number > 0:
                values.append(number)
        if values:
            return round(sum(values[:2]) / min(len(values), 2), 4)
    if fallback is None:
        return None
    try:
        fallback_value = float(fallback)
    except (TypeError, ValueError):
        return None
    return fallback_value if fallback_value > 0 else None


def annotate_a_share_trade_suggestions(
    *,
    actions: Iterable[Dict[str, Any]],
    holdings: Iterable[Dict[str, Any]],
    cash: float,
    total_asset: float,
    candidates: Optional[Iterable[Dict[str, Any]]] = None,
    max_single_position_pct: float = 15.0,
    buy_budget_fallback_ratio: float = 0.05,
    profile: Optional[TradeCostProfile] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], TradeCostProfile]:
    cost_profile = profile or load_cn_trade_cost_profile()
    holdings_by_code = {
        str(item.get("code", "") or ""): dict(item)
        for item in holdings
        if item.get("code")
    }
    available_cash = float(cash or 0.0)
    total_asset_value = float(total_asset or 0.0)
    if total_asset_value <= 0:
        holding_market_value = sum(float(item.get("market_value", 0) or 0.0) for item in holdings_by_code.values())
        total_asset_value = holding_market_value + available_cash

    action_rows = [dict(item) for item in actions]
    candidate_rows = [dict(item) for item in (candidates or [])]

    for row in action_rows + candidate_rows:
        row.setdefault("lot_size", CN_LOT_SIZE)
        row.setdefault("cost_profile_source", cost_profile.source)
        row.setdefault("cost_profile_samples", cost_profile.sample_size)
        row.setdefault("suggested_shares", 0)
        row.setdefault("suggested_lots", 0)

    for row in action_rows:
        action = str(row.get("action", "")).lower()
        code = str(row.get("code", "") or "")
        if not is_a_share_code(code):
            continue

        holding = holdings_by_code.get(code, {})
        reference_price = extract_reference_price(
            row.get("reference_price"),
            holding.get("current_price"),
            row.get("target_buy_price"),
            row.get("target_sell_price"),
            row.get("buy_price_range"),
            fallback=holding.get("cost_price"),
        )
        row["reference_price"] = reference_price

        if action in {"sell", "reduce"}:
            sellable = int(holding.get("sellable_shares", holding.get("shares", 0)) or 0)
            target_pct = (
                100.0
                if action == "sell"
                else (extract_percent_hint(row.get("ratio"), row.get("detail"), prefer="max") or 50.0)
            )
            raw_shares = sellable if target_pct >= 100 else int(sellable * target_pct / 100.0)
            suggested_shares = round_down_to_lot(raw_shares)
            row["suggested_shares"] = suggested_shares
            row["suggested_lots"] = suggested_shares // CN_LOT_SIZE if suggested_shares else 0
            if suggested_shares <= 0:
                row["quantity_reason"] = "当前可卖股数不足 1 手，按 A 股整手规则暂不生成卖出数量。"
                continue

            estimate = estimate_a_share_trade(
                shares=suggested_shares,
                price=reference_price or 0.0,
                side="sell",
                profile=cost_profile,
            )
            row["estimated_amount"] = estimate["gross_amount"]
            row["estimated_fee"] = estimate["fee"]
            row["estimated_tax"] = estimate["tax"]
            row["estimated_net_cash"] = estimate["net_cash"]
            available_cash += estimate["net_cash"]

            remainder = sellable - suggested_shares
            if remainder > 0:
                row["quantity_reason"] = f"剩余 {remainder} 股未纳入建议，避免出现非整手卖出。"
            continue

        if action != "buy":
            continue

        holding_value = float(holding.get("market_value", 0) or 0.0)
        percent_hint = extract_percent_hint(row.get("ratio"), row.get("detail"), prefer="min")
        budget_hint = extract_amount_hint(row.get("ratio"), row.get("detail"), row.get("reason"))
        if budget_hint is None and percent_hint is not None and holding_value > 0:
            budget_hint = holding_value * percent_hint / 100.0
        if budget_hint is None:
            budget_hint = min(
                available_cash,
                max(total_asset_value * buy_budget_fallback_ratio, 0.0),
            )
        max_position_value = max(total_asset_value * max_single_position_pct / 100.0 - holding_value, 0.0)
        budget_cap = max_position_value if max_position_value > 0 else available_cash
        budget = min(float(budget_hint or 0.0), available_cash, budget_cap)
        row["budget_hint"] = round(float(budget_hint or 0.0), 2)
        row["allocation_budget"] = round(max(budget, 0.0), 2)

        if not reference_price or reference_price <= 0:
            row["quantity_reason"] = "缺少有效参考价格，暂不生成买入数量。"
            continue

        suggested_shares = max_affordable_a_share_shares(
            cash_budget=budget,
            price=reference_price,
            profile=cost_profile,
        )
        row["suggested_shares"] = suggested_shares
        row["suggested_lots"] = suggested_shares // CN_LOT_SIZE if suggested_shares else 0
        if suggested_shares <= 0:
            row["quantity_reason"] = "预算不足以覆盖 A 股 1 手成交额和最低手续费，暂不生成买入数量。"
            continue

        estimate = estimate_a_share_trade(
            shares=suggested_shares,
            price=reference_price,
            side="buy",
            profile=cost_profile,
        )
        row["estimated_amount"] = estimate["gross_amount"]
        row["estimated_fee"] = estimate["fee"]
        row["estimated_tax"] = estimate["tax"]
        row["estimated_cash_out"] = estimate["cash_out"]
        available_cash = max(available_cash - estimate["cash_out"], 0.0)

    if candidate_rows:
        eligible_indices = []
        for idx, row in enumerate(candidate_rows):
            code = str(row.get("code", "") or "")
            if not is_a_share_code(code):
                continue
            if code in holdings_by_code:
                row["quantity_reason"] = "该标的已在持仓中，候选列表不重复给新开仓数量。"
                continue
            eligible_indices.append(idx)

        for offset, idx in enumerate(eligible_indices):
            row = candidate_rows[idx]
            slots_left = max(len(eligible_indices) - offset, 1)
            reference_price = extract_reference_price(
                row.get("reference_price"),
                row.get("buy_price_range"),
                row.get("price"),
            )
            row["reference_price"] = reference_price

            candidate_cap = (
                total_asset_value * min(max_single_position_pct, 10.0) / 100.0
                if total_asset_value > 0
                else available_cash
            )
            budget = min(available_cash / slots_left, candidate_cap)
            row["allocation_budget"] = round(max(budget, 0.0), 2)

            if not reference_price or reference_price <= 0:
                row["quantity_reason"] = "缺少有效参考价格，暂不生成开仓数量。"
                continue

            suggested_shares = max_affordable_a_share_shares(
                cash_budget=budget,
                price=reference_price,
                profile=cost_profile,
            )
            row["suggested_shares"] = suggested_shares
            row["suggested_lots"] = suggested_shares // CN_LOT_SIZE if suggested_shares else 0
            if suggested_shares <= 0:
                row["quantity_reason"] = "剩余可用资金不足以覆盖 1 手成交额和手续费。"
                continue

            estimate = estimate_a_share_trade(
                shares=suggested_shares,
                price=reference_price,
                side="buy",
                profile=cost_profile,
            )
            row["estimated_amount"] = estimate["gross_amount"]
            row["estimated_fee"] = estimate["fee"]
            row["estimated_tax"] = estimate["tax"]
            row["estimated_cash_out"] = estimate["cash_out"]
            available_cash = max(available_cash - estimate["cash_out"], 0.0)

    return action_rows, candidate_rows, cost_profile


def _load_trade_cost_rows(path: str) -> Tuple[List[Dict[str, float]], str]:
    rows: List[Dict[str, float]] = []
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        tables = {
            item[0]
            for item in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "portfolio_trades" in tables:
            for quantity, price, fee, tax, side in cur.execute(
                """
                SELECT quantity, price, fee, tax, side
                FROM portfolio_trades
                WHERE market = 'cn' AND quantity > 0 AND price > 0
                """
            ).fetchall():
                gross = float(quantity or 0.0) * float(price or 0.0)
                rows.append(
                    {
                        "gross": gross,
                        "fee": max(float(fee or 0.0), 0.0),
                        "tax": max(float(tax or 0.0), 0.0),
                        "side": str(side or "").strip().lower(),
                    }
                )
            if rows:
                return _normalize_cost_rows(rows), "portfolio_trades"

        if "trade_log" in tables:
            columns = {
                item[1]
                for item in cur.execute("PRAGMA table_info(trade_log)").fetchall()
            }
            if {"fee", "tax"}.issubset(columns):
                for shares, price, fee, tax, side in cur.execute(
                    """
                    SELECT shares, price, fee, tax, trade_type
                    FROM trade_log
                    WHERE shares > 0 AND price > 0
                    """
                ).fetchall():
                    gross = float(shares or 0.0) * float(price or 0.0)
                    rows.append(
                        {
                            "gross": gross,
                            "fee": max(float(fee or 0.0), 0.0),
                            "tax": max(float(tax or 0.0), 0.0),
                            "side": str(side or "").strip().lower(),
                        }
                    )
                if rows:
                    return _normalize_cost_rows(rows), "trade_log"
    finally:
        conn.close()
    return [], "default"


def _normalize_cost_rows(rows: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    normalized: List[Dict[str, float]] = []
    for row in rows:
        gross = float(row.get("gross", 0.0) or 0.0)
        fee = float(row.get("fee", 0.0) or 0.0)
        tax = float(row.get("tax", 0.0) or 0.0)
        side = str(row.get("side", "") or "").strip().lower()
        if gross <= 0 or fee < 0 or tax < 0:
            continue
        normalized.append(
            {
                "gross": gross,
                "fee": fee,
                "tax": tax,
                "side": side,
            }
        )
    return normalized
