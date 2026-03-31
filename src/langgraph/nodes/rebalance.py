"""
调仓分析节点 — 包装现有的多模型辩论引擎
"""
import logging
import threading

logger = logging.getLogger(__name__)


def rebalance_node(state: dict) -> dict:
    """执行调仓分析（异步，因为耗时较长）"""
    try:
        from rebalance_engine import run_rebalance_analysis
        from portfolio_manager import format_rebalance_report

        # 先返回"分析中"提示
        logger.info("[LangGraph] 开始调仓分析...")
        result = run_rebalance_analysis()

        if result.get("error"):
            return {"response": f"⚠️ 调仓分析失败: {result['error']}"}

        report = format_rebalance_report(result)
        return {"response": report}

    except Exception as e:
        logger.error(f"调仓分析异常: {e}")
        return {"response": f"❌ 调仓分析异常: {e}"}


def scan_node(state: dict) -> dict:
    """市场扫描节点"""
    try:
        from market_scanner import scan_market

        logger.info("[LangGraph] 开始市场扫描...")
        results = scan_market(mode="auto", top_n=10)

        if not results:
            return {"response": "扫描完成，未找到符合条件的股票。"}

        lines = ["🔍 **市场扫描结果**（Top 10）", ""]
        for i, r in enumerate(results[:10], 1):
            code = r.get("code", "")
            name = r.get("name", code)
            score = r.get("score", 0)
            price = r.get("price", 0)
            change = r.get("change_pct", 0)
            reason = r.get("reason", "")

            emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            lines.append(
                f"{i}. {emoji} **{name}**({code}) "
                f"{price:.2f}元 {change:+.1f}% | 评分{score}"
            )
            if reason:
                lines.append(f"   💡 {reason[:60]}")

        return {"response": "\n".join(lines)}

    except Exception as e:
        logger.error(f"市场扫描异常: {e}")
        return {"response": f"❌ 扫描异常: {e}"}


def analyze_stock_node(state: dict) -> dict:
    """个股分析节点"""
    trade = state.get("trade_action", {})
    code = trade.get("code") if trade else None

    if not code:
        return {"response": "请指定股票代码：分析 002506"}

    try:
        from market_scanner import analyze_kline
        try:
            from src.analyzer import get_stock_name_multi_source
            name = get_stock_name_multi_source(code) or code
        except Exception:
            from src.data.stock_mapping import STOCK_NAME_MAP
            name = STOCK_NAME_MAP.get(code, code)
        logger.info(f"[LangGraph] 分析 {name}({code})...")
        result = analyze_kline(code)

        if not result:
            return {"response": f"无法获取 {name}({code}) 的分析数据。"}

        score = result.get("score", 0)
        ma_trend = result.get("ma_trend", "未知")
        macd = result.get("macd_signal", "未知")
        rsi = result.get("rsi", 0)
        vol = result.get("vol_pattern", "未知")
        bias5 = result.get("bias5", 0)

        lines = [
            f"📊 **{name}**({code}) 技术分析",
            f"",
            f"综合评分: {score}/100",
            f"均线趋势: {ma_trend}",
            f"MACD信号: {macd}",
            f"RSI: {rsi:.1f}",
            f"成交量: {vol}",
            f"MA5乖离率: {bias5:.2f}%",
        ]

        # 简单建议
        if score >= 70:
            lines.append(f"\n💡 评分较高，可关注买入机会（注意T+1，涨幅>3%慎追）")
        elif score >= 50:
            lines.append(f"\n💡 评分中性，建议观望或轻仓试探")
        else:
            lines.append(f"\n💡 评分偏低，不建议买入")

        return {"response": "\n".join(lines)}

    except Exception as e:
        logger.error(f"个股分析异常: {e}")
        return {"response": f"❌ 分析异常: {e}"}
