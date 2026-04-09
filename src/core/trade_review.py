# -*- coding: utf-8 -*-
"""
trade_review.py — 盘后复盘笔记系统

职责：
1. 收集当日交易数据、持仓盈亏、日内峰值、AI决策记录
2. 调用 AI 分析盈利/亏损原因，生成结构化复盘笔记
3. 保存到文件 + 推送通知

数据来源：
- trade_journal: 当日买卖记录 + 历史模式分析
- intraday_tracker: 日内峰值统计（执行价 vs 最优价偏差）
- rebalance_history: AI多Agent决策记录
- portfolio / broker: 当前持仓盈亏快照
"""

import json, logging, os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TZ_CN = timezone(timedelta(hours=8))
REVIEW_DIR = Path("data/trade_reviews")


def run_trade_review(
    trade_date: Optional[str] = None,
    send_notification: bool = True,
) -> Optional[str]:
    """
    执行盘后复盘分析，返回复盘报告文本。

    Args:
        trade_date: 复盘日期 YYYY-MM-DD，默认今天
        send_notification: 是否推送通知
    """
    if not trade_date:
        trade_date = datetime.now(TZ_CN).strftime("%Y-%m-%d")

    logger.info(f"[复盘] 开始 {trade_date} 复盘分析...")

    # ── 1. 收集数据 ──
    data = _collect_review_data(trade_date)

    if not data.get("has_activity"):
        # 即使没有买卖操作，有持仓也要复盘（分析浮盈浮亏+资金流向）
        if not data.get("holdings_snapshot") and not data.get("fund_flow", {}).get("holding_stocks_flow"):
            msg = f"📝 {trade_date} 无交易活动且无持仓，跳过复盘"
            logger.info(msg)
            return msg
        logger.info(f"[复盘] {trade_date} 无买卖操作，但有持仓，生成持仓复盘")

    # ── 2. AI 分析 ──
    ai_review = _ai_analyze(data, trade_date)

    # ── 3. 组装报告 ──
    report = _format_review_report(data, ai_review, trade_date)

    # ── 4. 保存 ──
    _save_review(report, data, ai_review, trade_date)

    # ── 5. 推送 ──
    if send_notification:
        _send_review_notification(report)

    logger.info(f"[复盘] {trade_date} 复盘完成")
    return report


def _collect_review_data(trade_date: str) -> dict:
    """收集复盘所需的全部数据"""
    data = {
        "trade_date": trade_date,
        "has_activity": False,
        "trades": [],
        "holdings_snapshot": [],
        "peak_stats": [],
        "rebalance_actions": [],
        "performance": {},
        "pattern_analysis": {},
        "market_context": {},
    }

    # 1. 当日交易记录
    try:
        from trade_journal import _conn
        conn = _conn()
        rows = conn.execute("""
            SELECT t.trade_date, t.trade_type, t.code, t.name, t.shares, t.price,
                   t.amount, t.pnl, t.pnl_pct, t.hold_days,
                   t.ma_trend, t.macd_signal, t.rsi, t.tech_score, t.sector, t.source, t.note,
                   c.sh_index, c.sh_change_pct, c.stock_position_pct, c.vol_ratio,
                   c.main_net_inflow, c.sector_rank
            FROM trade_log t
            LEFT JOIN trade_market_context c ON c.trade_log_id = t.id
            WHERE t.trade_date = ?
            ORDER BY t.created_at
        """, (trade_date,)).fetchall()
        conn.close()
        data["trades"] = [dict(r) for r in rows]
        if data["trades"]:
            data["has_activity"] = True
    except Exception as e:
        logger.warning(f"[复盘] 读取交易记录失败: {e}")

    # 2. 当前持仓快照
    try:
        if os.getenv("BROKER_ENABLED", "false").lower() in ("true", "1", "yes"):
            from src.broker import get_broker
            broker = get_broker()
            if broker and broker.is_connected():
                positions = broker.get_positions()
                for pos in positions:
                    data["holdings_snapshot"].append({
                        "code": pos.code,
                        "name": pos.name,
                        "shares": pos.shares,
                        "sellable_shares": pos.sellable_shares,
                        "cost_price": pos.cost_price,
                        "current_price": pos.current_price,
                        "pnl_pct": round((pos.current_price - pos.cost_price) / pos.cost_price * 100, 2) if pos.cost_price > 0 else 0,
                        "market_value": round(pos.shares * pos.current_price, 2),
                    })
                data["has_activity"] = True
    except Exception:
        pass

    if not data["holdings_snapshot"]:
        try:
            from portfolio_manager import load_portfolio
            portfolio = load_portfolio()
            for h in portfolio.get("holdings", []):
                cost = h.get("cost_price", 0)
                cur = h.get("current_price", 0)
                data["holdings_snapshot"].append({
                    "code": h.get("code", ""),
                    "name": h.get("name", ""),
                    "shares": h.get("shares", 0),
                    "cost_price": cost,
                    "current_price": cur,
                    "pnl_pct": round((cur - cost) / cost * 100, 2) if cost > 0 else 0,
                    "market_value": h.get("market_value", 0),
                })
            if data["holdings_snapshot"]:
                data["has_activity"] = True
        except Exception:
            pass

    # 3. 日内峰值统计
    try:
        from trade_journal import _conn as tj_conn
        conn = tj_conn()
        peaks = conn.execute("""
            SELECT code, name, open_price, close_price, day_high, day_low,
                   high_time, low_time, executed_price, executed_direction, executed_time,
                   price_vs_high_pct, price_vs_low_pct,
                   ai_target_price, ai_stop_price, target_reached, stop_reached
            FROM intraday_peak_stats
            WHERE trade_date = ?
        """, (trade_date,)).fetchall()
        conn.close()
        data["peak_stats"] = [dict(p) for p in peaks]
    except Exception as e:
        logger.debug(f"[复盘] 峰值统计读取失败: {e}")

    # 4. 当日调仓决策
    try:
        rebalance_dir = Path("data/rebalance_history")
        date_prefix = trade_date.replace("-", "")
        for f in sorted(rebalance_dir.glob(f"rebalance_{date_prefix}_*.json"), reverse=True):
            with open(f, "r", encoding="utf-8") as fp:
                rb = json.load(fp)
            data["rebalance_actions"] = rb.get("actions", [])
            data["rebalance_market"] = rb.get("market_assessment", "")
            data["rebalance_sector"] = rb.get("sector_assessment", "")
            data["rebalance_risk"] = rb.get("risk_warning", "")
            # 取最近一次即可
            break
    except Exception as e:
        logger.debug(f"[复盘] 调仓记录读取失败: {e}")

    # 5. 战绩统计
    try:
        from trade_journal import get_performance_summary
        data["performance"] = get_performance_summary(days=30)
    except Exception:
        pass

    # 6. 盈亏模式
    try:
        from trade_journal import analyze_winning_patterns
        data["pattern_analysis"] = analyze_winning_patterns(days=90)
    except Exception:
        pass

    # 7. 大盘环境（从交易记录中提取）
    for t in data["trades"]:
        if t.get("sh_index"):
            data["market_context"] = {
                "sh_index": t.get("sh_index"),
                "sh_change_pct": t.get("sh_change_pct"),
            }
            break

    # 8. 资金流向（板块 + 个股）
    data["fund_flow"] = {}
    try:
        from data_store import get_fund_flow_latest, get_sector_flow_latest
        stock_flow = get_fund_flow_latest(trade_date=trade_date, top_n=10)
        sector_flow = get_sector_flow_latest(trade_date=trade_date, top_n=10)
        data["fund_flow"]["top_inflow_stocks"] = stock_flow.get("inflow", [])
        data["fund_flow"]["top_outflow_stocks"] = stock_flow.get("outflow", [])
        data["fund_flow"]["top_inflow_sectors"] = sector_flow.get("inflow", [])
        data["fund_flow"]["top_outflow_sectors"] = sector_flow.get("outflow", [])
    except Exception as e:
        logger.debug(f"[复盘] 资金流向读取失败: {e}")

    # 8b. 持仓个股的资金流向
    try:
        from ths_scraper import fetch_stock_fund_flow_rank
        all_flow = fetch_stock_fund_flow_rank(top_n=200)
        holding_codes = {h["code"] for h in data["holdings_snapshot"] if h.get("code")}
        holding_flow = [f for f in all_flow if f.get("code") in holding_codes]
        data["fund_flow"]["holding_stocks_flow"] = holding_flow
    except Exception as e:
        logger.debug(f"[复盘] 持仓资金流向失败: {e}")

    # 9. 市场环境（如果交易记录中没有，从扫描器获取）
    if not data.get("market_context", {}).get("sh_index"):
        try:
            from market_scanner import detect_market_regime
            regime = detect_market_regime()
            data["market_context"]["regime"] = regime.get("regime", "unknown")
            data["market_context"]["regime_score"] = regime.get("score", 0)
            data["market_context"]["regime_detail"] = regime.get("detail", "")
        except Exception:
            pass

    return data


def _ai_analyze(data: dict, trade_date: str) -> str:
    """调用 AI 生成复盘分析"""
    prompt = _build_review_prompt(data, trade_date)

    # 按优先级: Azure(快+稳) → Gemini(免费) → Gemini-lite → 本地Ollama
    try:
        import litellm

        models_to_try = []
        for env_key in ["REBALANCE_CLOUD_MODEL", "LITELLM_MODEL", "REBALANCE_CLOUD_FALLBACK", "LITELLM_FALLBACK_MODELS"]:
            val = os.getenv(env_key, "")
            for m in val.split(","):
                m = m.strip()
                if m and m not in models_to_try:
                    models_to_try.append(m)

        # 云端模型
        for model in models_to_try:
            try:
                logger.info(f"[复盘] 尝试 {model}...")
                extra = {}
                if "ollama" in model:
                    extra["api_base"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    extra["timeout"] = 300
                else:
                    extra["timeout"] = 120
                resp = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=3000,
                    **extra,
                )
                text = resp.choices[0].message.content.strip()
                if text:
                    logger.info(f"[复盘] AI分析完成 (model={model})")
                    return text
            except Exception as e:
                logger.warning(f"[复盘] {model} 调用失败: {e}")
                continue

        # 兜底本地
        local_model = os.getenv("REBALANCE_LOCAL_MODEL", "")
        if local_model and local_model not in models_to_try:
            try:
                logger.info(f"[复盘] 兜底本地 {local_model}...")
                resp = litellm.completion(
                    model=local_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=3000,
                    timeout=300,
                    api_base=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                )
                text = resp.choices[0].message.content.strip()
                if text:
                    return text
            except Exception as e:
                logger.warning(f"[复盘] 本地模型也失败: {e}")

    except Exception as e:
        logger.warning(f"[复盘] AI分析失败，使用纯数据复盘: {e}")

    return ""


def _build_review_prompt(data: dict, trade_date: str) -> str:
    """构建复盘 AI prompt"""
    sections = []

    # 当日交易
    if data["trades"]:
        trade_lines = []
        for t in data["trades"]:
            direction = "买入" if t["trade_type"] == "buy" else "卖出"
            pnl_str = ""
            if t["trade_type"] == "sell" and t.get("pnl") is not None:
                pnl_str = f" 盈亏:{t['pnl']}元({t.get('pnl_pct', 0)}%) 持仓{t.get('hold_days', 0)}天"
            trade_lines.append(
                f"  {direction} {t.get('name', '')}({t['code']}) "
                f"{t.get('shares', 0)}股 @ {t.get('price', 0)}元{pnl_str} "
                f"[MA:{t.get('ma_trend', '')} MACD:{t.get('macd_signal', '')} RSI:{t.get('rsi', '')}]"
            )
        sections.append("【当日交易】\n" + "\n".join(trade_lines))

    # 持仓快照
    if data["holdings_snapshot"]:
        hold_lines = []
        for h in data["holdings_snapshot"]:
            emoji = "盈" if h["pnl_pct"] >= 0 else "亏"
            hold_lines.append(
                f"  {h['name']}({h['code']}) {h['shares']}股 "
                f"成本:{h['cost_price']} 现价:{h['current_price']} "
                f"{emoji}{h['pnl_pct']:+.2f}% 市值:{h['market_value']}"
            )
        sections.append("【当前持仓】\n" + "\n".join(hold_lines))

    # 日内峰值
    if data["peak_stats"]:
        peak_lines = []
        for p in data["peak_stats"]:
            exec_info = ""
            if p.get("executed_price", 0) > 0:
                d = "买" if p["executed_direction"] == "buy" else "卖"
                gap = p["price_vs_low_pct"] if p["executed_direction"] == "buy" else p["price_vs_high_pct"]
                exec_info = f" | 执行:{d}@{p['executed_price']} 偏差:{gap:+.1f}%"
            peak_lines.append(
                f"  {p.get('name', '')}({p['code']}) "
                f"开:{p.get('open_price', 0)} 收:{p.get('close_price', 0)} "
                f"高:{p.get('day_high', 0)}({p.get('high_time', '')}) "
                f"低:{p.get('day_low', 0)}({p.get('low_time', '')})"
                f"{exec_info}"
            )
        sections.append("【日内峰值】\n" + "\n".join(peak_lines))

    # AI决策记录
    if data.get("rebalance_actions"):
        action_lines = []
        for a in data["rebalance_actions"]:
            action_lines.append(
                f"  {a.get('name', '')}({a.get('code', '')}) "
                f"建议:{a.get('action', '')} "
                f"目标价:{a.get('target_sell_price', '')} "
                f"止损价:{a.get('stop_loss_price', '')} "
                f"理由:{a.get('detail', '')[:80]}"
            )
        sections.append("【AI调仓建议】\n" + "\n".join(action_lines))

    # 大盘环境
    if data.get("market_context", {}).get("sh_index"):
        mc = data["market_context"]
        sections.append(
            f"【大盘环境】\n"
            f"  上证指数: {mc.get('sh_index', '')} 涨跌: {mc.get('sh_change_pct', '')}%"
        )

    # 近30天战绩
    perf = data.get("performance", {})
    if perf.get("total_trades", 0) > 0:
        sections.append(
            f"【近30天战绩】\n"
            f"  交易{perf['total_trades']}笔 胜率:{perf.get('win_rate', 0)}% "
            f"总盈亏:{perf.get('total_pnl', 0)}元 平均收益:{perf.get('avg_pnl_pct', 0)}%"
        )

    # 资金流向
    fund_flow = data.get("fund_flow", {})
    if fund_flow:
        flow_lines = []
        # 板块资金
        top_in_sectors = fund_flow.get("top_inflow_sectors", [])
        top_out_sectors = fund_flow.get("top_outflow_sectors", [])
        if top_in_sectors:
            flow_lines.append("  主力流入板块: " + ", ".join(
                f"{s.get('name', '')}({s.get('net', 0):.0f}万)" for s in top_in_sectors[:5]
            ))
        if top_out_sectors:
            flow_lines.append("  主力流出板块: " + ", ".join(
                f"{s.get('name', '')}({s.get('net', 0):.0f}万)" for s in top_out_sectors[:5]
            ))
        # 个股资金
        top_in_stocks = fund_flow.get("top_inflow_stocks", [])
        top_out_stocks = fund_flow.get("top_outflow_stocks", [])
        if top_in_stocks:
            flow_lines.append("  主力流入个股: " + ", ".join(
                f"{s.get('name', '')}({s.get('net', 0):.0f}万)" for s in top_in_stocks[:5]
            ))
        if top_out_stocks:
            flow_lines.append("  主力流出个股: " + ", ".join(
                f"{s.get('name', '')}({s.get('net', 0):.0f}万)" for s in top_out_stocks[:5]
            ))
        # 持仓个股资金流
        holding_flow = fund_flow.get("holding_stocks_flow", [])
        if holding_flow:
            flow_lines.append("  持仓股资金流:")
            for hf in holding_flow:
                net = hf.get("main_net", 0)
                emoji = "流入" if net > 0 else "流出"
                flow_lines.append(f"    {hf.get('name', '')}({hf.get('code', '')}) 主力{emoji} {abs(net):.0f}万")
        if flow_lines:
            sections.append("【资金流向】\n" + "\n".join(flow_lines))

    # 市场环境
    mc = data.get("market_context", {})
    mc_lines = []
    if mc.get("sh_index"):
        mc_lines.append(f"  上证指数: {mc['sh_index']} 涨跌: {mc.get('sh_change_pct', '')}%")
    if mc.get("regime"):
        regime_map = {"bull": "牛市", "sideways": "震荡市", "bear": "熊市"}
        mc_lines.append(f"  市场环境: {regime_map.get(mc['regime'], mc['regime'])} (score={mc.get('regime_score', 0)})")
        if mc.get("regime_detail"):
            mc_lines.append(f"  环境细节: {mc['regime_detail']}")
    if mc_lines:
        sections.append("【市场环境】\n" + "\n".join(mc_lines))

    data_block = "\n\n".join(sections)

    return f"""你是一位经验丰富的A股短线交易复盘专家。请根据以下 {trade_date} 的交易数据，生成一份详细的复盘笔记。

{data_block}

请按以下结构输出复盘笔记（每部分都要结合具体数据分析，不要空泛）：

## 一、今日操作总结
- 列出当日所有买卖操作及结果
- 总盈亏金额和百分比

## 二、资金流向分析
- 今日主力资金流入了哪些板块/个股？
- 持仓股的主力资金是流入还是流出？
- 资金流向与操作方向是否一致？（顺势还是逆势？）
- 明日资金流向预判

## 三、盈利交易分析
- 哪些交易盈利了？为什么盈利？
- 买入时机是否合理（结合MA/MACD/RSI）？
- 卖出时机是否最优（对比日内峰值）？
- 值得保持的操作习惯

## 四、亏损交易分析
- 哪些交易亏损了？根本原因是什么？
- 是追高被套？是大盘拖累？还是个股利空？
- 是买入时机问题、还是没有及时止损？
- 市场环境（牛市/震荡/熊市）对操作的影响

## 五、持仓诊断
- 当前持仓中哪些需要关注？
- 浮亏持仓：继续持有的理由是否还成立？止损位在哪？
- 浮盈持仓：是否应止盈？目标位在哪？
- 持仓与当前市场环境是否匹配？

## 六、执行力评分（满分10分）
- 对比AI建议和实际执行的差异
- 执行价格与日内最优价格的偏差
- 是否存在犹豫/拖延/冲动操作
- 给出具体分数和理由

## 七、改进方案（重点！）
- 今天最大的教训是什么？
- 明天具体应该怎么操作？（持仓处理+新机会）
- 仓位管理需要调整吗？（结合市场环境）
- 给出3条具体可执行的改进建议
- 下次遇到类似情况应该如何应对？

请用简洁专业的语言，每条分析必须引用具体数据（价格、涨跌幅、资金流向数值等），不要泛泛而谈。"""


def _format_review_report(data: dict, ai_review: str, trade_date: str) -> str:
    """组装最终复盘报告"""
    lines = [f"📝 **{trade_date} 盘后复盘笔记**", ""]

    # 概要
    trades = data.get("trades", [])
    buys = [t for t in trades if t["trade_type"] == "buy"]
    sells = [t for t in trades if t["trade_type"] == "sell"]
    total_pnl = sum(t.get("pnl", 0) or 0 for t in sells)

    lines.append(f"📊 今日交易: {len(buys)}笔买入 + {len(sells)}笔卖出")
    if sells:
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        lines.append(f"{pnl_emoji} 已实现盈亏: {total_pnl:.2f}元")

    # 持仓总览
    holdings = data.get("holdings_snapshot", [])
    if holdings:
        total_unrealized = sum(
            (h["current_price"] - h["cost_price"]) * h["shares"]
            for h in holdings if h.get("cost_price", 0) > 0
        )
        unr_emoji = "🟢" if total_unrealized >= 0 else "🔴"
        lines.append(f"{unr_emoji} 持仓浮动盈亏: {total_unrealized:.2f}元 ({len(holdings)}只)")

    # 峰值偏差
    peak_stats = data.get("peak_stats", [])
    exec_peaks = [p for p in peak_stats if p.get("executed_price", 0) > 0]
    if exec_peaks:
        gaps = []
        for p in exec_peaks:
            if p["executed_direction"] == "buy":
                gaps.append(abs(p.get("price_vs_low_pct", 0)))
            else:
                gaps.append(abs(p.get("price_vs_high_pct", 0)))
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        lines.append(f"📐 执行偏差: 平均偏离最优价 {avg_gap:.1f}%")

    lines.append("")
    lines.append("─" * 30)
    lines.append("")

    # AI 分析
    if ai_review:
        lines.append(ai_review)
    else:
        # 无 AI 时用纯数据复盘
        lines.append(_fallback_data_review(data))

    return "\n".join(lines)


def _fallback_data_review(data: dict) -> str:
    """AI不可用时的纯数据复盘"""
    lines = []

    trades = data.get("trades", [])
    if trades:
        lines.append("**交易明细:**")
        for t in trades:
            direction = "买入" if t["trade_type"] == "buy" else "卖出"
            pnl_str = ""
            if t["trade_type"] == "sell" and t.get("pnl") is not None:
                emoji = "🟢" if t["pnl"] >= 0 else "🔴"
                pnl_str = f" {emoji}{t['pnl']:+.2f}元({t.get('pnl_pct', 0):+.2f}%)"
            lines.append(f"  {direction} {t.get('name', '')}({t['code']}) {t.get('shares', 0)}股@{t.get('price', 0)}{pnl_str}")

    holdings = data.get("holdings_snapshot", [])
    if holdings:
        lines.append("\n**持仓快照:**")
        for h in holdings:
            emoji = "🟢" if h["pnl_pct"] >= 0 else "🔴"
            lines.append(f"  {h['name']}({h['code']}) {emoji}{h['pnl_pct']:+.2f}% 市值:{h['market_value']}")

    peak_stats = data.get("peak_stats", [])
    exec_peaks = [p for p in peak_stats if p.get("executed_price", 0) > 0]
    if exec_peaks:
        lines.append("\n**执行偏差:**")
        for p in exec_peaks:
            d = "买" if p["executed_direction"] == "buy" else "卖"
            gap = p["price_vs_low_pct"] if p["executed_direction"] == "buy" else p["price_vs_high_pct"]
            lines.append(
                f"  {p.get('name', '')} {d}@{p['executed_price']} "
                f"日高:{p.get('day_high', 0)} 日低:{p.get('day_low', 0)} "
                f"偏差:{gap:+.1f}%"
            )

    perf = data.get("performance", {})
    if perf.get("total_trades", 0) > 0:
        lines.append(
            f"\n**近30天:** {perf['total_trades']}笔 "
            f"胜率:{perf.get('win_rate', 0)}% "
            f"总盈亏:{perf.get('total_pnl', 0)}元"
        )

    return "\n".join(lines) if lines else "暂无足够数据生成复盘"


def _save_review(report: str, data: dict, ai_review: str, trade_date: str):
    """保存复盘笔记到文件"""
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # 保存可读的 markdown 报告
    md_path = REVIEW_DIR / f"review_{trade_date.replace('-', '')}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"[复盘] 报告已保存: {md_path}")

    # 保存结构化 JSON（便于后续 AI 引用）
    json_path = REVIEW_DIR / f"review_{trade_date.replace('-', '')}.json"
    review_data = {
        "trade_date": trade_date,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trades_count": len(data.get("trades", [])),
        "holdings_count": len(data.get("holdings_snapshot", [])),
        "trades": data.get("trades", []),
        "holdings_snapshot": data.get("holdings_snapshot", []),
        "peak_stats": data.get("peak_stats", []),
        "rebalance_actions": data.get("rebalance_actions", []),
        "performance_30d": data.get("performance", {}),
        "ai_review": ai_review,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, ensure_ascii=False, indent=2)


def _send_review_notification(report: str):
    """推送复盘报告"""
    try:
        from src.notification import NotificationService
        notifier = NotificationService()
        if notifier.is_available():
            notifier.send(report)
            logger.info("[复盘] 通知已推送")
    except Exception as e:
        logger.warning(f"[复盘] 推送失败: {e}")


def format_review_brief(trade_date: Optional[str] = None) -> str:
    """读取已保存的复盘笔记，返回格式化文本（飞书指令用）"""
    if not trade_date:
        trade_date = datetime.now(TZ_CN).strftime("%Y-%m-%d")

    md_path = REVIEW_DIR / f"review_{trade_date.replace('-', '')}.md"
    if md_path.exists():
        return md_path.read_text(encoding="utf-8")

    # 没有已保存的，实时生成
    return run_trade_review(trade_date=trade_date, send_notification=False) or "暂无复盘数据"


def get_recent_reviews(days: int = 5) -> list:
    """获取最近几天的复盘摘要（供AI参考）"""
    reviews = []
    if not REVIEW_DIR.exists():
        return reviews

    for f in sorted(REVIEW_DIR.glob("review_*.json"), reverse=True)[:days]:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            reviews.append({
                "trade_date": data.get("trade_date", ""),
                "trades_count": data.get("trades_count", 0),
                "holdings_count": data.get("holdings_count", 0),
                "ai_review_brief": (data.get("ai_review", "") or "")[:200],
            })
        except Exception:
            continue
    return reviews
