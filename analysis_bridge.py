"""
analysis_bridge.py — 新旧分析系统整合桥接
放在项目根目录

功能：
  1. 增强原项目的个股分析（注入资金流+板块+新闻数据）
  2. 替换原项目的 scanner（用我们的海外可用版）
  3. 提供统一的"全量分析"入口
  4. 模型路由：轻任务走本地Ollama，重任务走Gemini
"""
import json, logging, os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 增强版个股分析（原分析 + 新数据）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def enhanced_stock_analysis(stock_code: str, config=None) -> dict:
    """
    把原项目的 analyze_stock + 我们的新数据源合并

    流程：
      1. 腾讯实时行情（海外可用）
      2. K线技术分析（MA/MACD/RSI/量价）
      3. 同花顺资金流（如果在Top50内）
      4. SearXNG 个股新闻
      5. 原项目的 LLM 分析（Gemini）
    """
    result = {"code": stock_code, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # 1. 腾讯实时行情
    try:
        from macro_data_collector import _fetch_tencent_quote, _stock_code_to_tencent
        tc = _stock_code_to_tencent(stock_code)
        quotes = _fetch_tencent_quote([tc])
        q = quotes.get(tc, {})
        result["realtime"] = q
        result["name"] = q.get("name", stock_code)
        result["price"] = q.get("price", 0)
        result["change_pct"] = q.get("change_pct", 0)
    except Exception as e:
        logger.warning(f"实时行情获取失败: {e}")

    # 2. K线技术分析
    try:
        from market_scanner import _fetch_kline, analyze_kline
        df = _fetch_kline(stock_code, 120)
        if not df.empty:
            ta = analyze_kline(df)
            result["technical"] = ta
    except Exception as e:
        logger.warning(f"K线分析失败: {e}")

    # 3. 同花顺资金流
    try:
        from ths_scraper import fetch_single_stock_fund_flow
        fund = fetch_single_stock_fund_flow(stock_code)
        if fund and "error" not in fund:
            result["fund_flow"] = fund
    except Exception as e:
        logger.debug(f"同花顺资金流获取失败: {e}")

    # 4. 个股新闻
    try:
        from macro_data_collector import fetch_stock_news
        name = result.get("name", "")
        news = fetch_stock_news(stock_code, name, limit=5)
        result["news"] = news
    except Exception as e:
        logger.debug(f"个股新闻获取失败: {e}")

    # 5. 原项目 LLM 分析（如果可用）
    try:
        from analyzer_service import analyze_stock
        if config is None:
            from src.config import get_config
            config = get_config()
        original = analyze_stock(stock_code, config=config)
        if original:
            result["original_analysis"] = {
                "sentiment_score": getattr(original, "sentiment_score", 0),
                "operation_advice": getattr(original, "operation_advice", ""),
                "trend_prediction": getattr(original, "trend_prediction", ""),
            }
            db = getattr(original, "dashboard", None)
            if db:
                result["dashboard"] = {
                    "core_conclusion": getattr(db, "core_conclusion", None),
                }
    except Exception as e:
        logger.warning(f"原项目分析调用失败: {e}")

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 格式化增强分析报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def format_enhanced_report(data: dict) -> str:
    """把增强分析结果格式化为可读报告"""
    lines = []
    name = data.get("name", data.get("code", ""))
    code = data.get("code", "")
    price = data.get("price", 0)
    chg = data.get("change_pct", 0)

    emoji = "🟢" if chg > 0 else "🔴" if chg < 0 else "⚪"
    lines.append(f"{emoji} **{name}({code})** {price:.2f}元 {chg:+.2f}%")
    lines.append("")

    # 技术分析
    ta = data.get("technical", {})
    if ta:
        lines.append(f"📊 **技术面**")
        lines.append(f"  MA趋势: {ta.get('ma_trend', '—')}")
        lines.append(f"  MACD: {ta.get('macd_signal', '—')}")
        lines.append(f"  RSI: {ta.get('rsi', '—')}")
        lines.append(f"  量价: {ta.get('vol_pattern', '—')} (量比:{ta.get('vol_ratio', '—')})")
        lines.append(f"  乖离率: {ta.get('bias5', '—')}%")
        lines.append(f"  支撑/压力: {ta.get('support', '—')} / {ta.get('resistance', '—')}")
        lines.append(f"  技术得分: {ta.get('score', 0)}/90")
        lines.append("")

    # 资金流
    fund = data.get("fund_flow", {})
    if fund and fund.get("source") == "ths_rank":
        lines.append(f"💰 **资金面** (同花顺)")
        lines.append(f"  主力净流入: {fund.get('main_net', 0):.0f}万")
        lines.append(f"  主力净占比: {fund.get('main_net_pct', 0):.1f}%")
        lines.append(f"  趋势: {fund.get('fund_trend', '—')}")
        lines.append("")

    # 原项目分析
    orig = data.get("original_analysis", {})
    if orig:
        lines.append(f"🤖 **AI分析** (Gemini)")
        lines.append(f"  情绪得分: {orig.get('sentiment_score', '—')}")
        lines.append(f"  操作建议: {orig.get('operation_advice', '—')}")
        lines.append(f"  趋势预测: {orig.get('trend_prediction', '—')}")
        db = data.get("dashboard", {})
        if db and db.get("core_conclusion"):
            cc = db["core_conclusion"]
            if isinstance(cc, dict):
                lines.append(f"  核心结论: {cc.get('one_sentence', '')}")
        lines.append("")

    # 新闻
    news = data.get("news", [])
    if news:
        lines.append(f"📰 **最新消息** ({len(news)}条)")
        for n in news[:3]:
            lines.append(f"  • {n.get('title', '')[:50]}")
        lines.append("")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 替换原项目 scanner 的桥接
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def scan_market_bridge(
    mode: str = "all",
    top_n: int = 15,
    max_cap: float = 300,
    max_bias: float = 8.0,
    min_turnover: float = 1.0,
) -> dict:
    """
    替换原项目 scan_strong_stocks 的桥接函数
    兼容原项目的返回格式
    """
    try:
        from market_scanner import scan_market
        candidates = scan_market(
            max_price=100.0,  # 不限价格（原项目的scanner不限价）
            min_turnover=min_turnover,
            max_market_cap=max_cap,
            top_n=top_n,
            mode="trend" if mode in ("all", "trend") else mode,
        )

        if not candidates:
            return {
                "success": True, "count": 0,
                "message": f"未找到符合条件的股票（模式：{mode}）",
                "stocks": [],
            }

        # 转换为原项目格式
        stocks = []
        codes = []
        for s in candidates:
            stocks.append({
                "code": s["code"],
                "name": s["name"],
                "price": s["price"],
                "change_pct": s["change_pct"],
                "turnover": s["turnover_rate"],
                "market_cap": s["market_cap"],
                "strategy": s.get("ma_trend", ""),
                "bias_pct": s.get("bias5", 0),
                "tech_score": s.get("tech_score", 0),
                "macd_signal": s.get("macd_signal", ""),
                "rsi": s.get("rsi", 0),
                "vol_pattern": s.get("vol_pattern", ""),
            })
            codes.append(s["code"])

        return {
            "success": True,
            "count": len(stocks),
            "message": f"找到 {len(stocks)} 只（模式：{mode}）",
            "codes_list": ",".join(codes),
            "stocks": stocks,
        }
    except Exception as e:
        logger.error(f"扫描桥接失败: {e}")
        return {"success": False, "count": 0, "message": str(e), "stocks": []}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 全量分析入口（一键跑全部）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_full_enhanced_analysis(config=None) -> str:
    """
    一键执行全量分析：
      1. 全市场扫描 → 候选股
      2. 对持仓股做增强分析
      3. 多Agent调仓建议
      4. 汇总报告
    """
    if config is None:
        from src.config import get_config
        config = get_config()

    lines = []
    lines.append(f"📊 **全量分析报告**")
    lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # 1. 持仓股增强分析
    try:
        from portfolio_manager import load_portfolio
        portfolio = load_portfolio()
        holdings = portfolio.get("holdings", [])

        if holdings:
            lines.append(f"═══ 持仓分析 ({len(holdings)}只) ═══")
            lines.append("")
            for h in holdings:
                code = h["code"]
                logger.info(f"增强分析: {h.get('name', code)}...")
                data = enhanced_stock_analysis(code, config=config)
                report = format_enhanced_report(data)
                lines.append(report)
                lines.append("---")
    except Exception as e:
        lines.append(f"持仓分析失败: {e}")

    # 2. 全市场扫描
    try:
        lines.append("")
        lines.append("═══ 全市场扫描 Top10 ═══")
        lines.append("")
        from market_scanner import scan_market
        candidates = scan_market(max_price=50.0, min_turnover=3.0, top_n=10, mode="dragon")
        for s in candidates:
            lines.append(
                f"  {s['code']} {s['name']} {s['price']:.2f}元 "
                f"{s['change_pct']:+.1f}% | {s['ma_trend']} {s['macd_signal']} "
                f"得分:{s['tech_score']}"
            )
    except Exception as e:
        lines.append(f"全市场扫描失败: {e}")

    # 3. 调仓建议
    try:
        lines.append("")
        lines.append("═══ 调仓建议 ═══")
        lines.append("")
        from rebalance_engine import run_rebalance_analysis
        from portfolio_manager import format_rebalance_report
        result = run_rebalance_analysis(config=config)
        if "error" not in result:
            lines.append(format_rebalance_report(result))
        else:
            lines.append(f"调仓分析失败: {result['error']}")
    except Exception as e:
        lines.append(f"调仓分析失败: {e}")

    return "\n".join(lines)