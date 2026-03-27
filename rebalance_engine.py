"""
rebalance_engine.py
Lightweight rebalance analysis engine.
"""

from __future__ import annotations

from typing import Any, Dict, List

from macro_data_collector import _fetch_tencent_quote, _stock_code_to_tencent
from portfolio_manager import load_portfolio, update_current_prices


def _extract_latest_price(ff: Dict[str, Any]) -> float:
    """
    Price fallback chain:
    1) daily[-1].close
    2) daily[-1].price
    3) top-level price
    """
    if not isinstance(ff, dict):
        return 0.0
    daily = ff.get("daily", [])
    if isinstance(daily, list) and daily:
        last = daily[-1] if isinstance(daily[-1], dict) else {}
        for key in ("close", "price"):
            try:
                v = float(last.get(key, 0) or 0)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    try:
        p = float(ff.get("price", 0) or 0)
        return p if p > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _collect_macro_data(portfolio: Dict) -> Dict[str, Any]:
    holdings = portfolio.get("holdings", [])
    tc_codes = [_stock_code_to_tencent(h.get("code", "")) for h in holdings if h.get("code")]
    quotes = _fetch_tencent_quote(tc_codes) if tc_codes else {}
    holdings_fund_flow: Dict[str, Dict[str, Any]] = {}
    for h in holdings:
        code = str(h.get("code", ""))
        tc = _stock_code_to_tencent(code)
        q = quotes.get(tc, {})
        holdings_fund_flow[code] = {
            "price": q.get("price", 0),
            "daily": [{"price": q.get("price", 0), "close": q.get("price", 0)}],
        }
    return {"holdings_fund_flow": holdings_fund_flow}


def run_rebalance_analysis(config: Any = None) -> Dict[str, Any]:
    _ = config
    try:
        portfolio = load_portfolio()
        holdings = portfolio.get("holdings", [])
        if not holdings:
            return {"error": "持仓为空"}

        macro_data = _collect_macro_data(portfolio)

        # 更新持仓实时价格
        price_map: Dict[str, float] = {}
        for code, ff in macro_data.get("holdings_fund_flow", {}).items():
            latest = _extract_latest_price(ff)
            if latest > 0:
                price_map[code] = latest

        portfolio = update_current_prices(portfolio, price_map)

        suggestions: List[str] = []
        for h in portfolio.get("holdings", []):
            code = h.get("code", "")
            pnl = float(h.get("pnl_pct", 0) or 0)
            if pnl <= -8:
                suggestions.append(f"{code}: 跌幅较大，建议评估止损或减仓。")
            elif pnl >= 12:
                suggestions.append(f"{code}: 浮盈较高，建议分批止盈。")
            else:
                suggestions.append(f"{code}: 暂维持观察。")

        total_asset = float(portfolio.get("total_asset", 0) or 0)
        cash = float(portfolio.get("cash", 0) or 0)
        ratio = float(portfolio.get("actual_position_ratio", 0) or 0)

        return {
            "total_asset": total_asset,
            "cash": cash,
            "position_ratio": ratio,
            "suggestions": suggestions,
            "price_map": price_map,
        }
    except Exception as e:
        return {"error": str(e)}
