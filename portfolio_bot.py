"""
portfolio_bot.py — 通过聊天指令管理持仓 + 触发扫描/调仓

支持的指令（飞书/钉钉发送）:
  买入 002506 500股 5.4元        → 添加/加仓
  卖出 002506 200股 5.8元        → 减仓
  清仓 002506                    → 清掉某只
  持仓                           → 查看当前持仓
  调仓                           → 触发调仓分析
  扫描                           → 触发全市场扫描
  主线                           → 查看主线板块
  回测                           → 查看扫描胜率
  预警                           → 手动触发一次异动检测

放在项目根目录，被 bot 系统调用
"""
import json, logging, os, re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from portfolio_manager import load_portfolio, save_portfolio

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 指令解析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 匹配：买入 002506 500股 5.4元  或  买入002506 500 5.4
BUY_PATTERN = re.compile(
    r'买入\s*(\d{6})\s*(\d+)\s*股?\s*([\d.]+)\s*元?', re.IGNORECASE
)
SELL_PATTERN = re.compile(
    r'卖出\s*(\d{6})\s*(\d+)\s*股?\s*([\d.]+)\s*元?', re.IGNORECASE
)
CLEAR_PATTERN = re.compile(r'清仓\s*(\d{6})', re.IGNORECASE)


def is_portfolio_command(text: str) -> bool:
    """判断是否是持仓管理指令"""
    text = text.strip()
    keywords = [
        "买入", "卖出", "清仓", "持仓", "调仓", "扫描", "主线",
        "回测", "预警", "分析", "全量分析", "战绩", "复盘", "交易记录",
    ]
    return any(text.startswith(kw) for kw in keywords)


def _pick_quote_price(quote: dict) -> float:
    """从行情字典中提取可用价格（兼容不同字段名）。"""
    if not isinstance(quote, dict):
        return 0.0
    for key in ("price", "close", "last", "current_price"):
        try:
            val = float(quote.get(key, 0) or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val > 0:
            return val
    return 0.0


def handle_portfolio_command(text: str) -> str:
    """处理持仓管理指令"""
    text = text.strip()

    try:
        if text.startswith("买入"):
            return _handle_buy(text)
        elif text.startswith("卖出"):
            return _handle_sell(text)
        elif text.startswith("清仓"):
            return _handle_clear(text)
        elif text.startswith("持仓"):
            return _handle_show_portfolio()
        elif text.startswith("调仓"):
            return _handle_rebalance()
        elif text.startswith("全量分析"):
            return _handle_full_analysis()
        elif text.startswith("分析"):
            return _handle_analyze(text)
        elif text.startswith("扫描"):
            return _handle_scan()
        elif text.startswith("主线"):
            return _handle_mainline()
        elif text.startswith("回测"):
            return _handle_backtest()
        elif text.startswith("预警"):
            return _handle_alert()
        elif text.startswith("战绩"):
            return _handle_performance()
        elif text.startswith("复盘") and "分析" not in text:
            return _handle_pattern_review()
        elif text.startswith("交易记录"):
            return _handle_trade_history()
        else:
            return "未识别的指令"
    except Exception as e:
        logger.error(f"处理指令失败: {e}")
        return f"执行失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 买入
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _get_stock_name(code: str) -> str:
    """通过腾讯行情获取股票名称"""
    try:
        from macro_data_collector import _fetch_tencent_quote, _stock_code_to_tencent
        tc = _stock_code_to_tencent(code)
        q = _fetch_tencent_quote([tc])
        return q.get(tc, {}).get("name", code)
    except:
        return code


def _handle_buy(text: str) -> str:
    m = BUY_PATTERN.search(text)
    if not m:
        return "格式错误，请输入：买入 002506 500股 5.4元"

    code = m.group(1)
    shares = int(m.group(2))
    price = float(m.group(3))
    name = _get_stock_name(code)

    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])

    # 检查是否已持有
    existing = None
    for h in holdings:
        if h["code"] == code:
            existing = h
            break

    cost = round(shares * price, 2)

    if existing:
        # 加仓：重新计算成本价
        old_total = existing["shares"] * existing["cost_price"]
        new_total = old_total + cost
        existing["shares"] += shares
        existing["cost_price"] = round(new_total / existing["shares"], 3)
        existing["current_price"] = price
        existing["market_value"] = round(existing["shares"] * price, 2)
        action = "加仓"
    else:
        # 新建仓位
        # 猜板块
        sector = _guess_sector(text, code)
        holdings.append({
            "code": code,
            "name": name,
            "shares": shares,
            "cost_price": price,
            "current_price": price,
            "market_value": round(shares * price, 2),
            "sector": sector,
            "buy_date": datetime.now().strftime("%Y-%m-%d"),
            "strategy_tag": "短线",
        })
        action = "建仓"

    # 扣现金
    portfolio["cash"] = round(portfolio.get("cash", 0) - cost, 2)
    portfolio["holdings"] = holdings
    save_portfolio(portfolio)

    return (
        f"✅ {action}成功\n"
        f"  {name}({code}) {shares}股 × {price}元 = {cost}元\n"
        f"  剩余现金: {portfolio['cash']}元"
    )


def _guess_sector(text: str, code: str) -> str:
    """从指令中提取板块，或设为未知"""
    # 简单匹配
    for kw, sector in [
        ("光伏", "光伏"), ("新能源", "新能源"), ("芯片", "半导体"),
        ("电力", "电力"), ("军工", "军工"), ("医药", "医药"),
        ("传媒", "文化传媒"), ("消费", "消费"), ("白酒", "白酒"),
    ]:
        if kw in text:
            return sector
    return "未分类"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 卖出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_sell(text: str) -> str:
    m = SELL_PATTERN.search(text)
    if not m:
        return "格式错误，请输入：卖出 002506 200股 5.8元"

    code = m.group(1)
    shares = int(m.group(2))
    price = float(m.group(3))

    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])

    target = None
    for h in holdings:
        if h["code"] == code:
            target = h
            break

    if not target:
        return f"❌ 未找到持仓 {code}"

    if shares > target["shares"]:
        return f"❌ 持仓不足，当前只有 {target['shares']} 股"

    income = round(shares * price, 2)
    pnl = round((price - target["cost_price"]) * shares, 2)
    pnl_pct = round((price - target["cost_price"]) / target["cost_price"] * 100, 2)

    target["shares"] -= shares
    target["current_price"] = price
    target["market_value"] = round(target["shares"] * price, 2)

    # 如果卖光了就删掉
    if target["shares"] <= 0:
        holdings = [h for h in holdings if h["code"] != code]

    portfolio["cash"] = round(portfolio.get("cash", 0) + income, 2)
    portfolio["holdings"] = holdings
    save_portfolio(portfolio)

    emoji = "🟢" if pnl >= 0 else "🔴"
    return (
        f"✅ 卖出成功\n"
        f"  {target.get('name', code)}({code}) {shares}股 × {price}元 = {income}元\n"
        f"  {emoji} 盈亏: {pnl}元 ({pnl_pct}%)\n"
        f"  剩余持仓: {target['shares']}股\n"
        f"  现金: {portfolio['cash']}元"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 清仓
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_clear(text: str) -> str:
    m = CLEAR_PATTERN.search(text)
    if not m:
        return "格式错误，请输入：清仓 002506"

    code = m.group(1)
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])

    target = None
    for h in holdings:
        if h["code"] == code:
            target = h
            break

    if not target:
        return f"❌ 未找到持仓 {code}"

    # 用当前价估算
    try:
        from macro_data_collector import _fetch_tencent_quote, _stock_code_to_tencent
        tc = _stock_code_to_tencent(code)
        q = _fetch_tencent_quote([tc]).get(tc, {})
        current_price = _pick_quote_price(q) or target.get("current_price", 0)
    except:
        current_price = target.get("current_price", 0)

    income = round(target["shares"] * current_price, 2)
    pnl = round((current_price - target["cost_price"]) * target["shares"], 2)

    holdings = [h for h in holdings if h["code"] != code]
    portfolio["cash"] = round(portfolio.get("cash", 0) + income, 2)
    portfolio["holdings"] = holdings
    save_portfolio(portfolio)

    emoji = "🟢" if pnl >= 0 else "🔴"
    return (
        f"✅ 清仓完成\n"
        f"  {target.get('name', code)}({code}) {target['shares']}股\n"
        f"  {emoji} 估算盈亏: {pnl}元\n"
        f"  回收现金: {income}元\n"
        f"  总现金: {portfolio['cash']}元"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 查看持仓
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_show_portfolio() -> str:
    from portfolio_manager import update_current_prices
    from macro_data_collector import _fetch_tencent_quote, _stock_code_to_tencent

    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])
    if not holdings:
        return "📋 当前无持仓"

    # 更新实时价格
    tc_codes = [_stock_code_to_tencent(h["code"]) for h in holdings]
    quotes = _fetch_tencent_quote(tc_codes)
    price_map = {}
    for h in holdings:
        tc = _stock_code_to_tencent(h["code"])
        q = quotes.get(tc, {})
        latest_price = _pick_quote_price(q)
        if latest_price > 0:
            price_map[h["code"]] = latest_price

    portfolio = update_current_prices(portfolio, price_map)

    lines = [f"📋 **当前持仓** ({len(holdings)}只)"]
    lines.append(f"💰 现金: {portfolio.get('cash', 0):.2f}元")
    lines.append(f"📊 总资产: {portfolio.get('total_asset', 0):.2f}元")
    lines.append(f"📈 仓位: {portfolio.get('actual_position_ratio', 0)*100:.1f}%")
    lines.append("")

    total_pnl = 0
    for h in holdings:
        pnl = h.get("pnl_pct", 0)
        total_pnl += (h.get("current_price", 0) - h.get("cost_price", 0)) * h.get("shares", 0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        lines.append(
            f"{emoji} {h.get('name','')}({h['code']}) "
            f"{h.get('shares',0)}股 "
            f"成本:{h.get('cost_price',0):.3f} "
            f"现价:{h.get('current_price',0):.2f} "
            f"{pnl:+.2f}%"
        )

    total_emoji = "🟢" if total_pnl >= 0 else "🔴"
    lines.append(f"\n{total_emoji} 总盈亏: {total_pnl:.2f}元")
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 触发调仓分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_rebalance() -> str:
    try:
        from rebalance_engine import run_rebalance_analysis
        from portfolio_manager import format_rebalance_report
        from src.config import get_config
        config = get_config()
        result = run_rebalance_analysis(config=config)
        if "error" in result:
            return f"❌ 调仓分析失败: {result['error']}"
        return format_rebalance_report(result)
    except Exception as e:
        return f"❌ 调仓分析异常: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 触发全市场扫描
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_scan() -> str:
    try:
        from market_scanner import scan_market
        results = scan_market(max_price=10.0, min_turnover=2.0, top_n=15, mode="trend")
        if not results:
            return "📡 扫描完成，未找到符合条件的候选股"

        lines = [f"📡 **全市场扫描** (Top {len(results)})", ""]
        for s in results:
            lines.append(
                f"  {s['code']} {s['name']} {s['price']:.2f}元 "
                f"涨跌:{s['change_pct']:+.1f}% "
                f"{s['ma_trend']} {s['macd_signal']} "
                f"得分:{s['tech_score']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 扫描失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主线板块
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_mainline() -> str:
    try:
        from data_store import get_sector_mainline, get_consecutive_inflow_stocks
        sectors = get_sector_mainline(min_days=2)
        stocks = get_consecutive_inflow_stocks(min_days=2, min_total=300)

        lines = ["🔥 **主线板块分析**", ""]
        if sectors:
            lines.append("板块（连续资金流入）:")
            for s in sectors[:5]:
                lines.append(
                    f"  {s['sector_name']} — {s['inflow_days']}天 "
                    f"累计:{s['total_net']:.0f}万"
                )
        else:
            lines.append("板块数据不足（需积累2+天）")

        if stocks:
            lines.append("\n个股（连续主力流入）:")
            for s in stocks[:5]:
                lines.append(
                    f"  {s['code']} {s['name']} — {s['inflow_days']}天 "
                    f"累计:{s['total_net']:.0f}万"
                )

        return "\n".join(lines)
    except Exception as e:
        return f"❌ 主线分析失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 回测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_backtest() -> str:
    try:
        from data_store import get_backtest_summary
        summary = get_backtest_summary(30)
        if not summary:
            return "📊 暂无回测数据（需积累数天扫描结果）"

        lines = ["📊 **扫描策略回测** (近30天)", ""]
        for mode, s in summary.items():
            lines.append(
                f"  [{mode}] 样本:{s['total']} "
                f"3日胜率:{s['win_rate_3d']}% "
                f"5日胜率:{s['win_rate_5d']}% "
                f"均收益:{s['avg_return_3d']}%"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 回测查询失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 手动预警
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _handle_alert() -> str:
    try:
        from market_monitor import check_market_anomaly, format_anomaly_alert
        result = check_market_anomaly()
        alert = format_anomaly_alert(result)
        if alert:
            return alert
        return "✅ 当前无异动"
    except Exception as e:
        return f"❌ 预警检测失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 增强分析（新+旧结合）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYZE_PATTERN = re.compile(r'分析\s*(\d{6})', re.IGNORECASE)

def _handle_analyze(text: str) -> str:
    """增强版个股分析：技术面+资金面+新闻+AI"""
    m = ANALYZE_PATTERN.search(text)
    if not m:
        return "格式：分析 002506"
    code = m.group(1)
    try:
        from analysis_bridge import enhanced_stock_analysis, format_enhanced_report
        data = enhanced_stock_analysis(code)
        return format_enhanced_report(data)
    except Exception as e:
        return f"❌ 分析失败: {str(e)}"


def _handle_full_analysis() -> str:
    """全量分析：持仓+扫描+调仓"""
    try:
        from analysis_bridge import run_full_enhanced_analysis
        return run_full_enhanced_analysis()
    except Exception as e:
        return f"❌ 全量分析失败: {str(e)}"
