"""
持仓查看节点 — 格式化当前持仓信息
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def view_portfolio_node(state: dict) -> dict:
    """查看当前持仓"""
    portfolio = state.get("portfolio")
    if not portfolio or not portfolio.get("holdings"):
        return {"response": "当前没有持仓。"}

    holdings = portfolio["holdings"]
    cash = portfolio.get("cash", 0)
    total = portfolio.get("total_asset", 0)
    position_ratio = portfolio.get("actual_position_ratio", 0)
    today = datetime.now()

    lines = []
    lines.append("📊 **当前持仓**")
    lines.append(f"💰 总资产: {total:,.0f}元 | 现金: {cash:,.0f}元 | 仓位: {position_ratio*100:.1f}%")
    lines.append("")

    total_pnl = 0
    for h in holdings:
        code = h["code"]
        name = h.get("name", code)
        shares = h.get("shares", 0)
        sellable = h.get("sellable_shares", shares)
        cost = h.get("cost_price", 0)
        current = h.get("current_price", 0)
        pnl_pct = h.get("pnl_pct", 0)
        mv = h.get("market_value", 0)
        sector = h.get("sector", "")
        buy_date = h.get("buy_date", "")

        # 持仓天数
        hold_days = "?"
        if buy_date:
            try:
                bd = datetime.strptime(buy_date[:10], "%Y-%m-%d")
                hold_days = (today - bd).days
            except (ValueError, TypeError):
                pass

        # 盈亏
        pnl_amount = (current - cost) * shares if cost and current else 0
        total_pnl += pnl_amount

        # emoji
        if pnl_pct > 0:
            emoji = "🟢"
        elif pnl_pct < -5:
            emoji = "🔴"
        elif pnl_pct < 0:
            emoji = "🟡"
        else:
            emoji = "⚪"

        t1_note = ""
        if sellable < shares:
            t1_note = f" (T+1: 可卖{sellable}股)"

        lines.append(
            f"{emoji} **{name}**({code}) [{sector}]"
        )
        lines.append(
            f"   {shares}股{t1_note} | 成本{cost:.2f} → 现价{current:.2f}"
        )
        lines.append(
            f"   盈亏: {pnl_pct:+.2f}% ({pnl_amount:+,.0f}元) | 持仓{hold_days}天 | 市值{mv:,.0f}元"
        )

    lines.append("")
    pnl_emoji = "📈" if total_pnl >= 0 else "📉"
    lines.append(f"{pnl_emoji} 总浮盈亏: {total_pnl:+,.0f}元")

    return {"response": "\n".join(lines)}


def trade_history_node(state: dict) -> dict:
    """查看交易战绩"""
    try:
        from trade_journal import get_stats, format_performance
        stats = get_stats()
        if not stats:
            return {"response": "暂无交易记录。"}
        text = format_performance(stats)
        return {"response": text}
    except Exception as e:
        logger.warning(f"获取交易记录失败: {e}")
        return {"response": f"获取交易记录失败: {e}"}
