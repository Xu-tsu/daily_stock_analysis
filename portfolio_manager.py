"""
portfolio_manager.py
Minimal portfolio persistence and report formatting helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_PORTFOLIO_FILE = Path("data/portfolio.json")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_portfolio(path: str | Path = DEFAULT_PORTFOLIO_FILE) -> Dict:
    p = Path(path)
    if not p.exists():
        return {"cash": 0.0, "holdings": []}
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("cash", 0.0)
    data.setdefault("holdings", [])
    return data


def save_portfolio(portfolio: Dict, path: str | Path = DEFAULT_PORTFOLIO_FILE) -> None:
    p = Path(path)
    _ensure_parent(p)
    with p.open("w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def update_current_prices(portfolio: Dict, price_map: Dict[str, float]) -> Dict:
    holdings = portfolio.get("holdings", [])
    total_mv = 0.0
    for h in holdings:
        code = str(h.get("code", ""))
        if code in price_map and price_map[code] > 0:
            h["current_price"] = float(price_map[code])
        cp = float(h.get("current_price", 0) or 0)
        shares = float(h.get("shares", 0) or 0)
        cost = float(h.get("cost_price", 0) or 0)
        h["market_value"] = round(cp * shares, 2)
        h["pnl_pct"] = round(((cp - cost) / cost * 100), 2) if cost > 0 else 0.0
        total_mv += h["market_value"]
    cash = float(portfolio.get("cash", 0) or 0)
    portfolio["total_asset"] = round(cash + total_mv, 2)
    portfolio["actual_position_ratio"] = round(total_mv / (cash + total_mv), 4) if (cash + total_mv) > 0 else 0.0
    portfolio["holdings"] = holdings
    return portfolio


def format_rebalance_report(result: Dict) -> str:
    if not result:
        return "❌ 无调仓结果"
    if "error" in result:
        return f"❌ 调仓失败: {result['error']}"
    lines: List[str] = ["📊 调仓建议"]
    lines.append(f"总资产: {result.get('total_asset', 0):,.2f}")
    lines.append(f"现金: {result.get('cash', 0):,.2f}")
    lines.append(f"仓位: {result.get('position_ratio', 0)*100:.1f}%")
    lines.append("")
    suggestions = result.get("suggestions", [])
    if not suggestions:
        lines.append("暂无调仓建议，建议维持仓位。")
    else:
        for i, s in enumerate(suggestions, 1):
            lines.append(f"{i}. {s}")
    return "\n".join(lines)
