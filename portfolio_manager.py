"""
portfolio_manager.py — 持仓管理与调仓建议模块
放在项目根目录，与 analyzer_service.py 同级
"""
import json, os, logging, sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = os.getenv("PORTFOLIO_FILE", "data/portfolio.json")


def _fmt_price(v, digits: int = 2):
    """把 AI / 实时报价来的价格格式化成 2 位小数；非数值或区间字符串原样返回。"""
    if v is None:
        return v
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return v

# ──────────────────────────────────────────────
# 1. 持仓数据读写
# ──────────────────────────────────────────────
DEFAULT_PORTFOLIO = {
    "updated_at": "",
    "cash": 0,
    "total_asset": 0,
    "target_position_ratio": 0.7,
    "holdings": []
    # 每条 holding:
    # {
    #   "code": "600519", "name": "贵州茅台",
    #   "shares": 100, "cost_price": 1580.0,
    #   "current_price": 0, "market_value": 0,
    #   "sector": "白酒", "buy_date": "2026-01-15",
    #   "strategy_tag": "价值长持"
    # }
}

def load_portfolio() -> dict:
    """加载本地持仓 JSON"""
    p = Path(PORTFOLIO_FILE)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning(f"持仓文件 {PORTFOLIO_FILE} 不存在，使用空持仓")
    return DEFAULT_PORTFOLIO.copy()


def sync_portfolio_from_broker(portfolio: dict) -> dict:
    """以券商（THS）真实持仓为准，覆盖 portfolio.json。

    broker.get_positions() 返回的是同花顺客户端刷新后的实时数据：
    - shares:          真实持仓股数（成交后已更新）
    - sellable_shares: T+1 可卖余额（THS 客户端自动计算）
    - cost_price:      成本价（含手续费均摊）
    - current_price:   最新市价
    - market_value:    最新市值
    - pnl / pnl_pct:  盈亏

    合并逻辑：
    - broker 有的股票 → 覆盖 shares/cost/price/sellable
    - broker 没有但 portfolio.json 有 → 说明已清仓，移除
    - buy_date / sector / strategy_tag 保留 portfolio.json 里的（broker 不返回这些）
    """
    if os.getenv("BROKER_ENABLED", "false").lower() not in ("true", "1", "yes"):
        return portfolio

    try:
        from src.broker import get_broker
        broker = get_broker()
        if not broker or not broker.is_connected():
            logger.debug("[持仓同步] broker 未连接，跳过")
            return portfolio
    except Exception as e:
        logger.debug(f"[持仓同步] broker 初始化失败: {e}")
        return portfolio

    try:
        positions = broker.get_positions()
        balance = broker.get_balance()
    except Exception as e:
        logger.warning(f"[持仓同步] 获取持仓/资金失败: {e}")
        return portfolio

    if not positions and not balance:
        return portfolio

    # 建立旧持仓索引（保留 buy_date 等元数据）
    old_map = {h["code"]: h for h in portfolio.get("holdings", [])}

    # 以 broker 数据为准重建 holdings
    new_holdings = []
    total_mv = 0
    for pos in positions:
        if pos.shares <= 0:
            continue
        old = old_map.get(pos.code, {})
        h = {
            "code": pos.code,
            "name": pos.name or old.get("name", pos.code),
            "shares": pos.shares,
            "sellable_shares": pos.sellable_shares,
            "cost_price": pos.cost_price,
            "current_price": pos.current_price,
            "market_value": pos.market_value,
            "pnl_pct": pos.pnl_pct,
            # 保留 portfolio.json 的元数据（broker 不返回这些）
            "sector": old.get("sector", ""),
            "buy_date": old.get("buy_date", ""),
            "strategy_tag": old.get("strategy_tag", ""),
        }
        new_holdings.append(h)
        total_mv += pos.market_value

    # 记录变化
    old_codes = set(old_map.keys())
    new_codes = {p.code for p in positions if p.shares > 0}
    removed = old_codes - new_codes
    added = new_codes - old_codes
    if removed:
        logger.info(f"[持仓同步] 已清仓（从broker确认）: {removed}")
    if added:
        logger.info(f"[持仓同步] 新增持仓（从broker确认）: {added}")

    # 更新 portfolio
    portfolio["holdings"] = new_holdings
    if balance:
        portfolio["cash"] = balance.cash
        portfolio["total_asset"] = balance.total_asset
    else:
        portfolio["total_asset"] = round(total_mv + portfolio.get("cash", 0), 2)
    portfolio["actual_position_ratio"] = (
        round(total_mv / portfolio["total_asset"], 4)
        if portfolio["total_asset"] > 0 else 0
    )

    # 对比后有变化才记录
    changed_count = 0
    for h in new_holdings:
        old = old_map.get(h["code"], {})
        if (old.get("shares", 0) != h["shares"]
                or abs(old.get("current_price", 0) - h["current_price"]) > 0.001):
            changed_count += 1
            logger.info(
                f"  [broker→portfolio] {h['name']}({h['code']}): "
                f"股数{old.get('shares',0)}→{h['shares']} "
                f"可卖{h['sellable_shares']} "
                f"价格{old.get('current_price',0):.2f}→{h['current_price']:.2f}"
            )

    if changed_count or removed or added:
        save_portfolio(portfolio)
        logger.info(f"[持仓同步] 已从broker刷新 {len(new_holdings)} 只持仓，{changed_count} 只有变化")
    else:
        logger.debug("[持仓同步] broker数据与本地一致，无需更新")

    return portfolio

def save_portfolio(portfolio: dict):
    """保存持仓到本地"""
    p = Path(PORTFOLIO_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    portfolio["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)
    logger.info(f"持仓已保存到 {PORTFOLIO_FILE}")

def sync_portfolio_from_trades(portfolio: dict) -> dict:
    """从 trade_log 自动校准每只持仓的 buy_date、shares、cost_price。

    解决的核心问题：portfolio.json 是手动维护的，buy_date 经常与实际
    交割记录不一致，导致持仓天数计算错误，进而导致错误的清仓建议。

    逻辑：
    1. 对每只持仓股，从 trade_log 中取出所有 buy/sell 记录
    2. 用 FIFO 配对法计算当前未平仓的部分
    3. 用最早的未平仓买入日期作为 buy_date
    4. 用加权平均价作为 cost_price
    """
    DB_PATH = "data/scanner_history.db"
    if not os.path.exists(DB_PATH):
        logger.warning(f"trade_log 数据库 {DB_PATH} 不存在，跳过同步")
        return portfolio

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        logger.warning(f"连接 trade_log 数据库失败: {e}")
        return portfolio

    synced = 0
    for h in portfolio.get("holdings", []):
        code = h["code"]
        try:
            rows = conn.execute("""
                SELECT trade_date, trade_type, shares, price, amount
                FROM trade_log WHERE code = ?
                ORDER BY trade_date ASC, id ASC
            """, (code,)).fetchall()

            if not rows:
                continue

            # FIFO 法：按时间顺序累积买入，减去卖出
            buy_queue = []  # [(date, shares, price), ...]
            for r in rows:
                if r["trade_type"] == "buy":
                    buy_queue.append({
                        "date": r["trade_date"],
                        "shares": abs(r["shares"]),
                        "price": r["price"],
                    })
                elif r["trade_type"] == "sell":
                    sell_remaining = abs(r["shares"])
                    while sell_remaining > 0 and buy_queue:
                        if buy_queue[0]["shares"] <= sell_remaining:
                            sell_remaining -= buy_queue[0]["shares"]
                            buy_queue.pop(0)
                        else:
                            buy_queue[0]["shares"] -= sell_remaining
                            sell_remaining = 0

            if not buy_queue:
                # 所有买入都已被卖出覆盖 — 但portfolio里还有持仓
                # 可能是通过其他渠道买入的，不动它
                continue

            # 用未平仓买入记录计算
            total_shares = sum(b["shares"] for b in buy_queue)
            weighted_cost = sum(b["shares"] * b["price"] for b in buy_queue) / total_shares
            earliest_buy_date = buy_queue[0]["date"]

            # T+1 可卖余额：排除今天买入的股数
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_bought = sum(b["shares"] for b in buy_queue if b["date"] == today_str)
            sellable_shares = total_shares - today_bought

            old_date = h.get("buy_date", "")
            old_shares = h.get("shares", 0)

            # 更新 buy_date（始终以 trade_log 为准）
            if old_date != earliest_buy_date:
                logger.info(
                    f"  同步 {h.get('name', code)}: buy_date {old_date} → {earliest_buy_date}"
                )
                h["buy_date"] = earliest_buy_date
                synced += 1

            # 如果 trade_log 的股数与 portfolio 差别大，也校准
            if abs(total_shares - old_shares) > 0 and total_shares > 0:
                logger.info(
                    f"  同步 {h.get('name', code)}: shares {old_shares} → {total_shares}, "
                    f"cost {h.get('cost_price', 0):.3f} → {weighted_cost:.3f}"
                )
                h["shares"] = total_shares
                h["cost_price"] = round(weighted_cost, 3)
                synced += 1

            # 更新 T+1 可卖余额
            h["sellable_shares"] = sellable_shares
            if today_bought > 0:
                logger.info(
                    f"  T+1 {h.get('name', code)}: 总{total_shares}股, "
                    f"今日买入{today_bought}股不可卖, 可卖{sellable_shares}股"
                )

        except Exception as e:
            logger.warning(f"同步 {code} 失败: {e}")

    conn.close()

    if synced > 0:
        logger.info(f"持仓同步完成：修正了 {synced} 个字段")
        save_portfolio(portfolio)
    else:
        logger.info("持仓同步检查完成：数据一致，无需修正")

    return portfolio


def update_current_prices(portfolio: dict, price_map: dict) -> dict:
    """用实时价格更新持仓市值
    price_map: {"600519": 1820.0, "300750": 225.0, ...}
    """
    total_mv = 0
    for h in portfolio["holdings"]:
        code = h["code"]
        if code in price_map:
            # 价格精度：转债 3 位，股票 2 位（避免实时报价带来的 5-8 位小数污染 AI prompt）
            _d = 3 if str(code).startswith(("110", "113", "123", "127", "128", "118")) else 2
            h["current_price"] = round(float(price_map[code]), _d)
            h["market_value"] = round(h["current_price"] * h["shares"], 2)
            h["pnl_pct"] = round(
                (h["current_price"] - h["cost_price"]) / h["cost_price"] * 100, 2
            )
        total_mv += h.get("market_value", 0)
    portfolio["total_asset"] = round(total_mv + portfolio.get("cash", 0), 2)
    portfolio["actual_position_ratio"] = (
        round(total_mv / portfolio["total_asset"], 4)
        if portfolio["total_asset"] > 0 else 0
    )
    # 持久化到磁盘，避免下次读取时仍是陈旧价格
    save_portfolio(portfolio)
    return portfolio


# ──────────────────────────────────────────────
# 2. 调仓建议格式化（给通知模块用）
# ──────────────────────────────────────────────
def _append_execution_plan(lines: list, item: dict, indent: str = "   ") -> None:
    shares = int(item.get("suggested_shares", 0) or 0)
    lots = int(item.get("suggested_lots", 0) or 0)
    reference_price = float(item.get("reference_price", 0) or 0)
    quantity_reason = item.get("quantity_reason", "")

    if shares > 0:
        plan = f"{indent}📐 建议数量: {shares}股"
        if lots > 0:
            plan += f" ({lots}手)"
        if reference_price > 0:
            plan += f" @ {reference_price:.3f}元"
        lines.append(plan)

        fee = float(item.get("estimated_fee", 0) or 0)
        tax = float(item.get("estimated_tax", 0) or 0)
        if item.get("action") == "buy" or "estimated_cash_out" in item:
            amount = float(item.get("estimated_amount", 0) or 0)
            cash_out = float(item.get("estimated_cash_out", 0) or 0)
            lines.append(
                f"{indent}💰 预计成交额 {amount:.2f}元 | 手续费 {fee:.2f}元 | 税费 {tax:.2f}元 | 总支出 {cash_out:.2f}元"
            )
        elif "estimated_net_cash" in item:
            amount = float(item.get("estimated_amount", 0) or 0)
            net_cash = float(item.get("estimated_net_cash", 0) or 0)
            lines.append(
                f"{indent}💰 预计成交额 {amount:.2f}元 | 手续费 {fee:.2f}元 | 印花税 {tax:.2f}元 | 预计到账 {net_cash:.2f}元"
            )
    elif quantity_reason:
        lines.append(f"{indent}📐 数量约束: {quantity_reason}")

def _truncate_report_text(value, limit: int = 150) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = " ".join(text.split())
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _append_agent_discussion_legacy(lines: list, discussion: dict) -> None:
    if not discussion:
        return

    lines.append("馃 **澶氭ā鍨嬭京璁?*:")

    summary = _truncate_report_text(discussion.get("summary", ""), 180)
    if summary:
        lines.append(f"   鎽樿: {summary}")

    for item in discussion.get("rounds", []):
        label = item.get("agent_label", "Agent")
        role = item.get("role_label", "")
        model = item.get("model", "")
        signal = _truncate_report_text(item.get("signal_label", ""), 90)

        header_parts = [label]
        if role:
            header_parts.append(role)
        header = " | ".join(header_parts)
        if model:
            header += f" [{model}]"
        if signal:
            header += f" 鈫?{signal}"
        lines.append(f"   鈥?{header}")

        reasoning = _truncate_report_text(item.get("reasoning", ""), 180)
        if reasoning:
            lines.append(f"     {reasoning}")

    disagreements = discussion.get("disagreements", [])
    if disagreements:
        lines.append("   鈿旓笍 鍒嗘鐒︾偣:")
        for item in disagreements[:5]:
            lines.append(f"   鈥?{_truncate_report_text(item, 140)}")

    lines.append("")


def _append_agent_discussion(lines: list, discussion: dict) -> None:
    if not discussion:
        return

    lines.append("🤖 **多模型讨论**:")

    summary = _truncate_report_text(discussion.get("summary", ""), 180)
    if summary:
        lines.append(f"   摘要: {summary}")

    for item in discussion.get("rounds", []):
        label = item.get("agent_label", "Agent")
        role = item.get("role_label", "")
        model = item.get("model", "")
        signal = _truncate_report_text(item.get("signal_label", ""), 90)

        header_parts = [label]
        if role:
            header_parts.append(role)
        header = " | ".join(header_parts)
        if model:
            header += f" [{model}]"
        if signal:
            header += f" → {signal}"
        lines.append(f"   • {header}")

        reasoning = _truncate_report_text(item.get("reasoning", ""), 180)
        if reasoning:
            lines.append(f"     {reasoning}")

    disagreements = discussion.get("disagreements", [])
    if disagreements:
        lines.append("   ⚔️ 分歧焦点:")
        for item in disagreements[:5]:
            lines.append(f"   • {_truncate_report_text(item, 140)}")

    lines.append("")


def _format_rebalance_report_legacy(rebalance: dict) -> str:
    """把 LLM 返回的调仓 JSON 格式化为推送文本"""
    lines = []
    lines.append("📊 **调仓建议报告**")
    lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # 总仓位建议
    pos = rebalance.get("overall_position_advice", "")
    if pos:
        lines.append(f"🎯 **仓位建议**: {pos}")
        lines.append("")

    # 大盘判断
    market = rebalance.get("market_assessment", "")
    if market:
        lines.append(f"📈 **大盘研判**: {market}")
        lines.append("")

    # 板块分析
    sector = rebalance.get("sector_assessment", "")
    if sector:
        lines.append(f"🔥 **板块轮动**: {sector}")
        lines.append("")

    # 辩论摘要（如果有）
    debate = rebalance.get("debate_summary", "")
    if debate:
        lines.append(f"🤖 **多模型辩论**: {debate}")
        lines.append("")

    # 逐只操作
    actions = rebalance.get("actions", [])
    if actions:
        lines.append("📋 **操作建议**:")
        for a in actions:
            emoji = {"buy": "🟢", "hold": "🟡", "reduce": "🟠", "sell": "🔴"}.get(
                a.get("action", "hold"), "⚪"
            )
            action_cn = {
                "buy": "加仓", "hold": "持有",
                "reduce": "减仓", "sell": "清仓"
            }.get(a.get("action", "hold"), a.get("action", ""))
            lines.append(
                f"{emoji} **{a.get('name', a.get('code', ''))}** → {action_cn}"
            )
            if a.get("detail"):
                lines.append(f"   {a['detail']}")
            if a.get("reason"):
                lines.append(f"   💡 {a['reason']}")
            _append_execution_plan(lines, a)
            # 卖点/止损价
            tp = a.get("target_sell_price")
            sl = a.get("stop_loss_price")
            st = a.get("sell_timing")
            if tp or sl:
                price_info = []
                if tp:
                    price_info.append(f"目标卖出价: {_fmt_price(tp)}")
                if sl:
                    price_info.append(f"止损价: {_fmt_price(sl)}")
                lines.append(f"   🎯 {' | '.join(price_info)}")
            if st:
                lines.append(f"   ⏰ 卖出时机: {st}")
        lines.append("")

    # 候选换股
    candidates = rebalance.get("new_candidates", [])
    if candidates:
        lines.append("🔍 **换股候选**:")
        for c in candidates:
            line = f"  • {c.get('name', '')}({c.get('code', '')}) — {c.get('reason', '')}"
            bp = c.get("buy_price_range")
            tp = c.get("target_sell_price")
            sl = c.get("stop_loss_price")
            if bp or tp or sl:
                prices = []
                if bp:
                    prices.append(f"买入区间:{bp}")
                if tp:
                    prices.append(f"目标:{_fmt_price(tp)}")
                if sl:
                    prices.append(f"止损:{_fmt_price(sl)}")
                line += f"\n    🎯 {' | '.join(prices)}"
            lines.append(line)
            _append_execution_plan(lines, c, indent="    ")
        lines.append("")

    # 风险提示
    risk = rebalance.get("risk_warning", "")
    if risk:
        lines.append(f"⚠️ **风险提示**: {risk}")

    return "\n".join(lines)


def format_rebalance_report(rebalance: dict) -> str:
    """把调仓 JSON 格式化为可推送的文本报告。"""
    lines = [
        "📊 **调仓建议报告**",
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    pos = rebalance.get("overall_position_advice", "")
    if pos:
        lines.append(f"🎯 **仓位建议**: {pos}")
        lines.append("")

    market = rebalance.get("market_assessment", "")
    if market:
        lines.append(f"📈 **大盘研判**: {market}")
        lines.append("")

    sector = rebalance.get("sector_assessment", "")
    if sector:
        lines.append(f"🔥 **板块轮动**: {sector}")
        lines.append("")

    debate = rebalance.get("debate_summary", "")
    if debate:
        lines.append(f"🤖 **多模型辩论**: {debate}")
        lines.append("")

    _append_agent_discussion(lines, rebalance.get("agent_discussion", {}))

    actions = rebalance.get("actions", [])
    if actions:
        lines.append("📋 **操作建议**:")
        for action in actions:
            emoji = {
                "buy": "🟢",
                "hold": "🟡",
                "reduce": "🟠",
                "sell": "🔴",
            }.get(action.get("action", "hold"), "ℹ️")
            action_cn = {
                "buy": "加仓",
                "hold": "持有",
                "reduce": "减仓",
                "sell": "清仓",
            }.get(action.get("action", "hold"), action.get("action", ""))
            lines.append(f"{emoji} **{action.get('name', action.get('code', ''))}** → {action_cn}")
            if action.get("detail"):
                lines.append(f"   {action['detail']}")
            if action.get("reason"):
                lines.append(f"   💡 {action['reason']}")
            _append_execution_plan(lines, action)

            target_price = action.get("target_sell_price")
            stop_loss = action.get("stop_loss_price")
            sell_timing = action.get("sell_timing")
            if target_price or stop_loss:
                price_info = []
                if target_price:
                    price_info.append(f"目标卖出价: {_fmt_price(target_price)}")
                if stop_loss:
                    price_info.append(f"止损价: {_fmt_price(stop_loss)}")
                lines.append(f"   🎯 {' | '.join(price_info)}")
            if sell_timing:
                lines.append(f"   ⏰ 卖出时机: {sell_timing}")
        lines.append("")

    candidates = rebalance.get("new_candidates", [])
    if candidates:
        lines.append("🔍 **换股候选**:")
        for candidate in candidates:
            line = f"  • {candidate.get('name', '')}({candidate.get('code', '')}) — {candidate.get('reason', '')}"
            relay_role = candidate.get("relay_role")
            if relay_role:
                line += f"\n    🔗 角色: {relay_role}"
            buy_range = candidate.get("buy_price_range")
            target_price = candidate.get("target_sell_price")
            stop_loss = candidate.get("stop_loss_price")
            if buy_range or target_price or stop_loss:
                prices = []
                if buy_range:
                    prices.append(f"买入区间:{buy_range}")
                if target_price:
                    prices.append(f"目标:{_fmt_price(target_price)}")
                if stop_loss:
                    prices.append(f"止损:{_fmt_price(stop_loss)}")
                line += f"\n    🎯 {' | '.join(prices)}"
            timing_note = candidate.get("timing_note")
            if timing_note:
                line += f"\n    ⏱️ 时机: {timing_note}"
            lines.append(line)
            _append_execution_plan(lines, candidate, indent="    ")
        lines.append("")

    risk = rebalance.get("risk_warning", "")
    if risk:
        lines.append(f"⚠️ **风险提示**: {risk}")

    # ── 降级日志展示 ──
    degradation_summary = rebalance.get("degradation_summary", "")
    degradation_log = rebalance.get("degradation_log", [])
    if degradation_log:
        lines.append("")
        lines.append(f"🔧 **流程状态**: {degradation_summary}")
        severity_emoji = {"critical": "🔴", "error": "🟠", "warning": "🟡"}
        for d in degradation_log:
            emoji = severity_emoji.get(d.get("severity", "warning"), "ℹ️")
            lines.append(f"{emoji} [{d.get('step','')}] {d.get('reason','')}")
            fix = d.get("fix_suggestion", "")
            if fix:
                first_fix = fix.split("\n")[0]
                lines.append(f"   🔧 {first_fix}")
    elif degradation_summary:
        lines.append("")
        lines.append(f"🔧 **流程状态**: {degradation_summary}")

    return "\n".join(lines)
