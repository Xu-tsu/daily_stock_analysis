"""
持仓更新节点 — 处理口语化持仓修正

例如: "协鑫集成是5.2 有13手" → 更新 002506 的成本价和数量
"""
import logging

logger = logging.getLogger(__name__)


def update_portfolio_node(state: dict) -> dict:
    """口语持仓修正：根据用户描述直接更新 portfolio.json"""
    trade_action = state.get("trade_action", {})
    if not trade_action:
        return {"response": "无法解析持仓更新参数，请用：\n协鑫集成是5.2 有13手"}

    code = trade_action.get("code")
    new_price = trade_action.get("price")
    new_shares = trade_action.get("shares")

    if not code:
        return {"response": "未识别到股票代码，请重试。"}

    try:
        from portfolio_manager import load_portfolio, save_portfolio

        portfolio = load_portfolio()
        holdings = portfolio.get("holdings", [])

        # 查找目标持仓
        target = None
        for h in holdings:
            if h.get("code") == code:
                target = h
                break

        if target is None:
            # 持仓里没有，当作新建仓处理
            name = trade_action.get("name", code)
            try:
                from market_scanner import get_stock_name_fast
                name = get_stock_name_fast(code) or code
            except Exception:
                pass

            # "手" → "股"（如果未在 entry 转换）
            shares = new_shares
            new_holding = {
                "code": code,
                "name": name,
                "shares": shares,
                "cost_price": new_price if new_price else 0,
                "current_price": new_price if new_price else 0,
                "market_value": shares * (new_price or 0),
                "pnl": 0,
                "pnl_pct": 0,
                "sector": "未分类",
                "buy_date": _today_str(),
                "strategy_tag": "短线",
            }
            holdings.append(new_holding)
            portfolio["holdings"] = holdings
            save_portfolio(portfolio)

            return {
                "response": (
                    f"✅ 已新增持仓：\n"
                    f"  {name}({code}) {shares}股 成本{new_price}元"
                )
            }

        # 已有持仓 → 更新
        old_shares = target.get("shares", 0)
        old_price = target.get("cost_price", 0)
        name = target.get("name", code)

        changes = []
        if new_shares is not None and new_shares != old_shares:
            target["shares"] = new_shares
            changes.append(f"数量 {old_shares}→{new_shares}股")
        if new_price is not None and abs(new_price - old_price) > 0.001:
            target["cost_price"] = new_price
            changes.append(f"成本 {old_price:.3f}→{new_price}元")

        if not changes:
            return {
                "response": f"📋 {name}({code}) 持仓信息已是最新，无需更新。"
            }

        # 重算市值
        shares = target.get("shares", 0)
        price = target.get("current_price", target.get("cost_price", 0))
        target["market_value"] = shares * price
        if target.get("cost_price", 0) > 0:
            target["pnl"] = (price - target["cost_price"]) * shares
            target["pnl_pct"] = (price / target["cost_price"] - 1) * 100

        save_portfolio(portfolio)

        return {
            "response": (
                f"✅ 已更新 {name}({code})：\n"
                f"  {'；'.join(changes)}\n"
                f"  当前: {target['shares']}股 成本{target['cost_price']:.3f}元"
            )
        }

    except Exception as e:
        logger.error(f"持仓更新异常: {e}", exc_info=True)
        return {"response": f"❌ 更新失败: {e}"}


def _today_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
