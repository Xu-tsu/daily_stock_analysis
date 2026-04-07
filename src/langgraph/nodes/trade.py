"""
交易节点 — 买入/卖出/清仓 + 风控检查 + 确认工作流

券商集成:
  BROKER_ENABLED=true 时，确认后先在 THS 模拟盘下单，
  成交后才更新本地 portfolio.json，保证本地/远程一致。
"""
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

BROKER_ENABLED = os.getenv("BROKER_ENABLED", "false").lower() == "true"


def _get_stock_name(code: str) -> str:
    """获取股票名称"""
    try:
        from market_scanner import get_stock_name_fast
        return get_stock_name_fast(code) or code
    except Exception:
        return code


def risk_check_node(state: dict) -> dict:
    """风控检查节点 — 买入前必经"""
    trade = state.get("trade_action")
    portfolio = state.get("portfolio")
    if not trade:
        return {"response": "交易参数缺失，请使用格式：买入 002506 500 5.4"}

    code = trade.get("code", "")
    action = trade.get("action", "")
    shares = trade.get("shares")
    price = trade.get("price")
    name = _get_stock_name(code)
    trade["name"] = name

    # 参数不全时提示
    if action == "buy" and (not shares or not price):
        return {
            "response": f"请补全参数：买入 {code} <股数> <价格>\n例如：买入 {code} 500 5.4",
            "trade_action": trade,
        }
    if action == "sell" and (not shares or not price):
        return {
            "response": f"请补全参数：卖出 {code} <股数> <价格>\n例如：卖出 {code} 500 5.4",
            "trade_action": trade,
        }

    risk_result = {"allowed": True, "warnings": [], "blocked_reason": None}

    if action == "buy":
        # T+1 追高检查
        try:
            from risk_control import check_buy_permission
            holdings = portfolio.get("holdings", []) if portfolio else []
            # 获取当日涨幅
            change_pct = 0
            try:
                from market_scanner import get_realtime_quote_tencent
                qt = get_realtime_quote_tencent(code)
                if qt and qt.get("change_pct"):
                    change_pct = float(qt["change_pct"])
                    if qt.get("price"):
                        price = float(qt["price"])
                        trade["price"] = price
            except Exception:
                pass

            perm = check_buy_permission(
                code=code, name=name, holdings=holdings,
                total_asset=portfolio.get("total_asset", 0) if portfolio else 0,
                buy_amount=shares * price if shares and price else 0,
                current_change_pct=change_pct,
            )
            if not perm.get("allowed", True):
                risk_result["allowed"] = False
                risk_result["blocked_reason"] = perm.get("reason", "风控拦截")
            if perm.get("warnings"):
                risk_result["warnings"] = perm["warnings"]
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"风控检查异常: {e}")

    elif action == "sell" or action == "clear":
        # 检查可卖余额
        if portfolio:
            for h in portfolio.get("holdings", []):
                if h["code"] == code:
                    sellable = h.get("sellable_shares", h.get("shares", 0))
                    total_shares = h.get("shares", 0)
                    if action == "clear":
                        trade["shares"] = sellable
                        trade["price"] = h.get("current_price", 0)
                        if sellable < total_shares:
                            risk_result["warnings"].append(
                                f"T+1约束：总{total_shares}股中仅{sellable}股可卖，"
                                f"剩余{total_shares - sellable}股今日买入不可卖"
                            )
                    elif shares and shares > sellable:
                        risk_result["allowed"] = False
                        risk_result["blocked_reason"] = (
                            f"可卖余额不足：要卖{shares}股但只有{sellable}股可卖"
                            f"（总{total_shares}股中{total_shares - sellable}股为今日买入）"
                        )
                    break
            else:
                if action == "clear":
                    return {"response": f"未找到 {code} 的持仓，无法清仓。"}

    return {
        "risk_check": risk_result,
        "trade_action": trade,
    }


def request_confirmation_node(state: dict) -> dict:
    """生成确认消息，等待用户确认"""
    trade = state.get("trade_action", {})
    risk = state.get("risk_check", {})

    if not risk.get("allowed", True):
        return {"response": f"⛔ 风控拦截：{risk.get('blocked_reason', '未知原因')}"}

    code = trade.get("code", "")
    name = trade.get("name", code)
    action = trade.get("action", "")
    shares = trade.get("shares", 0)
    price = trade.get("price", 0)
    amount = shares * price if shares and price else 0

    action_cn = {"buy": "买入", "sell": "卖出", "clear": "清仓"}.get(action, action)

    lines = [f"📋 确认{action_cn}："]
    lines.append(f"  {name}({code})")
    lines.append(f"  {shares}股 × {price:.2f}元 = {amount:,.0f}元")

    # 显示风控警告
    for w in risk.get("warnings", []):
        lines.append(f"  ⚠️ {w}")

    lines.append("")
    lines.append("回复「确认」执行，「取消」放弃")

    return {
        "response": "\n".join(lines),
        "pending_confirmation": True,
    }


def _execute_via_broker(code: str, name: str, action: str, shares: int, price: float) -> dict:
    """通过券商执行交易，返回 {"success": bool, "result": OrderResult, "message": str}"""
    try:
        from src.broker import get_broker
        from src.broker.safety import check_order_allowed

        broker = get_broker()
        if not broker or not broker.is_connected():
            return {"success": False, "result": None,
                    "message": "券商未连接，仅更新本地持仓"}

        # 安全检查
        safety = check_order_allowed(code, price, shares, total_asset=0)
        if not safety.get("allowed", True):
            return {"success": False, "result": None,
                    "message": f"安全限制: {safety.get('reason', '未知')}"}

        if action == "buy":
            result = broker.buy(code, price, shares)
        else:
            result = broker.sell(code, price, shares)

        # 记录执行质量
        try:
            from src.broker.execution_tracker import record_execution
            record_execution(result, session_id="langgraph", mode="broker+local")
        except Exception as e:
            logger.warning(f"记录执行质量失败: {e}")

        if result.is_success:
            return {"success": True, "result": result,
                    "message": f"券商成交: {result.actual_shares}股@{result.actual_price:.2f}"}
        else:
            return {"success": False, "result": result,
                    "message": f"券商拒绝: {result.message}"}

    except ImportError:
        return {"success": False, "result": None, "message": "券商模块未安装"}
    except Exception as e:
        logger.error(f"券商执行异常: {e}")
        return {"success": False, "result": None, "message": f"券商异常: {e}"}


def execute_trade_node(state: dict) -> dict:
    """执行交易 — 仅在确认后触发

    BROKER_ENABLED=true 时:
      1. 先在 THS 模拟盘下单
      2. 成交后用实际成交价更新本地 portfolio
      3. 券商拒绝则不更新本地，返回错误
    """
    confirmed = state.get("confirmed")
    if not confirmed:
        return {"response": "已取消操作。"}

    trade = state.get("trade_action", {})
    portfolio = state.get("portfolio")
    if not trade or not portfolio:
        return {"response": "交易参数丢失，请重新输入。"}

    code = trade.get("code", "")
    name = trade.get("name", code)
    action = trade.get("action", "")
    shares = trade.get("shares", 0)
    price = trade.get("price", 0)

    # ── 券商执行（如果启用）──
    broker_msg = ""
    broker_order_result = None
    if BROKER_ENABLED:
        broker_ret = _execute_via_broker(code, name, action, shares, price)
        broker_msg = broker_ret["message"]
        if broker_ret.get("result"):
            broker_order_result = broker_ret["result"].to_dict()
        if broker_ret["success"]:
            # 用实际成交价替换请求价
            actual = broker_ret["result"]
            if actual.actual_price > 0:
                price = actual.actual_price
            if actual.actual_shares > 0:
                shares = actual.actual_shares
            logger.info(f"[券商] 成交: {name} {action} {shares}股@{price}")
        else:
            # 券商拒绝 → 不更新本地
            if broker_ret.get("result") and broker_ret["result"].status == "rejected":
                return {
                    "response": f"⛔ 券商拒绝下单: {broker_msg}",
                    "pending_confirmation": False,
                    "broker_order_result": broker_order_result,
                }
            # 其他失败（断连等）→ 仅本地记账 + 提示
            broker_msg = f"\n⚠️ {broker_msg}，仅本地记账"

    try:
        from portfolio_manager import load_portfolio, save_portfolio

        # 重新加载最新持仓（避免并发问题）
        portfolio = load_portfolio()

        if action == "buy":
            # 查找是否已有该股持仓
            found = False
            for h in portfolio.get("holdings", []):
                if h["code"] == code:
                    old_shares = h["shares"]
                    old_cost = h["cost_price"]
                    new_shares = old_shares + shares
                    h["cost_price"] = round(
                        (old_cost * old_shares + price * shares) / new_shares, 3
                    )
                    h["shares"] = new_shares
                    h["current_price"] = price
                    h["market_value"] = round(price * new_shares, 2)
                    found = True
                    break
            if not found:
                portfolio["holdings"].append({
                    "code": code,
                    "name": name,
                    "shares": shares,
                    "cost_price": price,
                    "current_price": price,
                    "market_value": round(price * shares, 2),
                    "sector": "",
                    "buy_date": datetime.now().strftime("%Y-%m-%d"),
                    "strategy_tag": "langgraph",
                })
            portfolio["cash"] = portfolio.get("cash", 0) - round(price * shares, 2)
            save_portfolio(portfolio)

            # 记录到 trade_log
            _log_trade(code, name, "buy", shares, price)

            return {
                "response": f"✅ 已买入 {name}({code}) {shares}股 × {price:.2f}元 = {shares*price:,.0f}元{broker_msg}",
                "portfolio": portfolio,
                "pending_confirmation": False,
                "broker_order_result": broker_order_result,
            }

        elif action in ("sell", "clear"):
            for h in portfolio.get("holdings", []):
                if h["code"] == code:
                    sell_shares = shares if shares else h.get("shares", 0)
                    if sell_shares >= h["shares"]:
                        # 全卖
                        pnl = round((price - h["cost_price"]) * h["shares"], 2)
                        portfolio["holdings"].remove(h)
                        portfolio["cash"] = portfolio.get("cash", 0) + round(price * h["shares"], 2)
                        save_portfolio(portfolio)
                        _log_trade(code, name, "sell", h["shares"], price,
                                   buy_price=h["cost_price"])
                        emoji = "📈" if pnl >= 0 else "📉"
                        return {
                            "response": (
                                f"✅ 已清仓 {name}({code}) {h['shares']}股 × {price:.2f}元\n"
                                f"{emoji} 盈亏: {pnl:+,.0f}元{broker_msg}"
                            ),
                            "portfolio": portfolio,
                            "pending_confirmation": False,
                            "broker_order_result": broker_order_result,
                        }
                    else:
                        # 部分卖
                        pnl = round((price - h["cost_price"]) * sell_shares, 2)
                        h["shares"] -= sell_shares
                        h["market_value"] = round(h["current_price"] * h["shares"], 2)
                        portfolio["cash"] = portfolio.get("cash", 0) + round(price * sell_shares, 2)
                        save_portfolio(portfolio)
                        _log_trade(code, name, "sell", sell_shares, price,
                                   buy_price=h["cost_price"])
                        emoji = "📈" if pnl >= 0 else "📉"
                        return {
                            "response": (
                                f"✅ 已卖出 {name}({code}) {sell_shares}股 × {price:.2f}元\n"
                                f"{emoji} 盈亏: {pnl:+,.0f}元 | 剩余{h['shares']}股{broker_msg}"
                            ),
                            "portfolio": portfolio,
                            "pending_confirmation": False,
                            "broker_order_result": broker_order_result,
                        }
            return {"response": f"未找到 {code} 的持仓。", "pending_confirmation": False}

    except Exception as e:
        logger.error(f"执行交易失败: {e}")
        return {"response": f"❌ 执行失败: {e}", "pending_confirmation": False}


def _log_trade(code: str, name: str, trade_type: str, shares: int,
               price: float, buy_price: float = 0):
    """记录交易到 trade_log"""
    try:
        import sqlite3
        conn = sqlite3.connect("data/scanner_history.db")
        conn.execute("""
            INSERT INTO trade_log (trade_date, trade_type, code, name, shares, price, amount, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d"),
            trade_type, code, name, shares, price,
            round(price * shares, 2), "langgraph",
        ))
        if trade_type == "sell" and buy_price:
            conn.execute("""
                UPDATE trade_log SET buy_price=?, pnl=?, pnl_pct=?
                WHERE id = last_insert_rowid()
            """, (
                buy_price,
                round((price - buy_price) * shares, 2),
                round((price - buy_price) / buy_price * 100, 2) if buy_price else 0,
            ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"记录交易日志失败: {e}")
