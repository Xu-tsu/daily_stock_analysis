"""
做T (T+0 日内交易) 工作流节点

做T流程：
1. 验证：该股有T+1可卖余额（昨天或更早买的）
2. 计划：分析当前价格和支撑位，建议卖出价和回补价
3. 确认卖出 → 执行卖出
4. 监控：等待价格回落到目标价
5. 确认回补 → 执行买入
6. 汇总：计算做T盈亏
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def validate_t0_node(state: dict) -> dict:
    """验证做T条件：是否有可卖余额"""
    trade = state.get("trade_action", {})
    portfolio = state.get("portfolio")
    code = trade.get("code")

    if not code:
        # 没指定代码，列出可做T的持仓
        if not portfolio or not portfolio.get("holdings"):
            return {"response": "当前没有持仓，无法做T。"}

        lines = ["📋 **可做T的持仓**（需要有T+1可卖余额）："]
        has_any = False
        for h in portfolio["holdings"]:
            sellable = h.get("sellable_shares", h.get("shares", 0))
            if sellable > 0:
                has_any = True
                lines.append(
                    f"  {h.get('name', h['code'])}({h['code']}) "
                    f"可卖{sellable}股 | 成本{h.get('cost_price', 0):.2f}"
                )
        if not has_any:
            return {"response": "所有持仓都是今天买入的，T+1不可卖，无法做T。"}

        lines.append("")
        lines.append("请指定股票：做T 002506")
        return {"response": "\n".join(lines)}

    # 查找持仓
    holding = None
    if portfolio:
        for h in portfolio.get("holdings", []):
            if h["code"] == code:
                holding = h
                break

    if not holding:
        return {"response": f"未找到 {code} 的持仓，无法做T。"}

    sellable = holding.get("sellable_shares", holding.get("shares", 0))
    if sellable <= 0:
        return {
            "response": (
                f"⛔ {holding.get('name', code)}({code}) 今天无可卖余额，"
                f"全部{holding.get('shares', 0)}股为今日买入，T+1不可卖。"
            )
        }

    return {
        "trade_action": {
            **trade,
            "code": code,
            "name": holding.get("name", code),
            "sellable_shares": sellable,
            "total_shares": holding.get("shares", 0),
            "cost_price": holding.get("cost_price", 0),
            "current_price": holding.get("current_price", 0),
        },
        "t0_phase": "plan",
    }


def plan_t0_node(state: dict) -> dict:
    """规划做T方案：建议卖出数量、价格和回补目标价"""
    trade = state.get("trade_action", {})
    code = trade.get("code")
    name = trade.get("name", code)
    sellable = trade.get("sellable_shares", 0)
    cost = trade.get("cost_price", 0)
    current = trade.get("current_price", 0)

    # 获取实时数据
    try:
        from market_scanner import get_realtime_quote_tencent
        qt = get_realtime_quote_tencent(code)
        if qt and qt.get("price"):
            current = float(qt["price"])
            trade["current_price"] = current
    except Exception:
        pass

    # 简单策略：卖在当前价（或略高），目标回补价 = 当前价 - 1~2%
    sell_price = round(current, 2)
    buy_back_target = round(current * 0.985, 2)  # 目标跌1.5%回补
    t0_shares = sellable  # 默认全卖

    # 预计收益
    spread = sell_price - buy_back_target
    expected_profit = round(spread * t0_shares, 2)

    lines = [
        f"📊 **做T方案** — {name}({code})",
        f"",
        f"可卖余额: {sellable}股 | 成本: {cost:.2f} | 现价: {current:.2f}",
        f"",
        f"🔴 **Step 1 卖出**: {t0_shares}股 × {sell_price:.2f}元",
        f"🟢 **Step 2 回补**: {t0_shares}股 × 目标{buy_back_target:.2f}元",
        f"💰 **预计收益**: 每股赚{spread:.2f}元，共{expected_profit:+,.0f}元",
        f"",
        f"回复「确认」执行卖出，「取消」放弃",
    ]

    return {
        "response": "\n".join(lines),
        "trade_action": {
            **trade,
            "t0_sell_price": sell_price,
            "t0_sell_shares": t0_shares,
            "t0_buy_back_target": buy_back_target,
        },
        "t0_phase": "confirm_sell",
        "t0_target_price": buy_back_target,
        "pending_confirmation": True,
    }


def execute_t0_sell_node(state: dict) -> dict:
    """执行做T的卖出步骤"""
    if not state.get("confirmed"):
        return {"response": "已取消做T。", "t0_phase": None, "pending_confirmation": False}

    trade = state.get("trade_action", {})
    code = trade.get("code")
    name = trade.get("name", code)
    sell_shares = trade.get("t0_sell_shares", 0)
    sell_price = trade.get("t0_sell_price", 0)
    target = trade.get("t0_buy_back_target", 0)

    # 执行卖出
    try:
        from portfolio_manager import load_portfolio, save_portfolio
        portfolio = load_portfolio()
        for h in portfolio["holdings"]:
            if h["code"] == code:
                h["shares"] -= sell_shares
                h["market_value"] = round(h.get("current_price", 0) * h["shares"], 2)
                if h["shares"] <= 0:
                    portfolio["holdings"].remove(h)
                portfolio["cash"] = portfolio.get("cash", 0) + round(sell_price * sell_shares, 2)
                save_portfolio(portfolio)
                break

        # 记录卖出
        from src.langgraph.nodes.trade import _log_trade
        _log_trade(code, name, "sell", sell_shares, sell_price,
                   buy_price=trade.get("cost_price", 0))

        return {
            "response": (
                f"✅ 做T卖出完成: {name}({code}) {sell_shares}股 × {sell_price:.2f}元\n"
                f"⏳ 等待回补目标价 {target:.2f}元\n"
                f"到价后发送「回补」或「回补 {code}」执行买回"
            ),
            "t0_phase": "monitoring",
            "t0_sell_done": True,
            "pending_confirmation": False,
            "portfolio": portfolio,
        }
    except Exception as e:
        logger.error(f"做T卖出失败: {e}")
        return {"response": f"❌ 做T卖出失败: {e}", "pending_confirmation": False}


def execute_t0_buyback_node(state: dict) -> dict:
    """执行做T的回补买入"""
    trade = state.get("trade_action", {})
    code = trade.get("code")
    name = trade.get("name", code)
    buy_shares = trade.get("t0_sell_shares", 0)
    target_price = state.get("t0_target_price", 0)

    # 获取当前价
    current_price = target_price
    try:
        from market_scanner import get_realtime_quote_tencent
        qt = get_realtime_quote_tencent(code)
        if qt and qt.get("price"):
            current_price = float(qt["price"])
    except Exception:
        pass

    # 执行买入
    try:
        from portfolio_manager import load_portfolio, save_portfolio
        portfolio = load_portfolio()

        found = False
        for h in portfolio["holdings"]:
            if h["code"] == code:
                old_shares = h["shares"]
                old_cost = h["cost_price"]
                new_shares = old_shares + buy_shares
                h["cost_price"] = round(
                    (old_cost * old_shares + current_price * buy_shares) / new_shares, 3
                )
                h["shares"] = new_shares
                h["current_price"] = current_price
                h["market_value"] = round(current_price * new_shares, 2)
                found = True
                break

        if not found:
            portfolio["holdings"].append({
                "code": code,
                "name": name,
                "shares": buy_shares,
                "cost_price": current_price,
                "current_price": current_price,
                "market_value": round(current_price * buy_shares, 2),
                "sector": "",
                "buy_date": datetime.now().strftime("%Y-%m-%d"),
                "strategy_tag": "t0_buyback",
            })

        portfolio["cash"] = portfolio.get("cash", 0) - round(current_price * buy_shares, 2)
        save_portfolio(portfolio)

        # 记录买入
        from src.langgraph.nodes.trade import _log_trade
        _log_trade(code, name, "buy", buy_shares, current_price)

        # 计算做T盈亏
        sell_price = trade.get("t0_sell_price", 0)
        spread = sell_price - current_price
        profit = round(spread * buy_shares, 2)
        emoji = "📈" if profit >= 0 else "📉"

        return {
            "response": (
                f"✅ 做T回补完成: {name}({code}) {buy_shares}股 × {current_price:.2f}元\n"
                f"{emoji} **做T结果**: 卖{sell_price:.2f} → 买{current_price:.2f}，"
                f"每股{'+' if spread>=0 else ''}{spread:.2f}元，"
                f"共{'+' if profit>=0 else ''}{profit:,.0f}元"
            ),
            "t0_phase": "done",
            "pending_confirmation": False,
            "portfolio": portfolio,
        }
    except Exception as e:
        logger.error(f"做T回补失败: {e}")
        return {"response": f"❌ 做T回补失败: {e}", "pending_confirmation": False}
