# -*- coding: utf-8 -*-
"""
limit_up_analyzer.py — 涨停股分析 + 关联股挖掘

功能：
1. 获取当日涨停池（akshare stock_zt_pool_em）
2. 获取龙虎榜明细（机构/游资买卖）
3. 对每只涨停股搜索涨停原因（网络搜索）
4. 分析涨停逻辑，找出关联概念和受益股
5. 结合资金流向，AI 生成综合分析报告
6. 推送到飞书/钉钉/邮件等通知渠道
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TZ_CN = timezone(timedelta(hours=8))
REPORT_DIR = Path("data/limit_up_reports")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 数据采集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fetch_limit_up_pool(trade_date: str) -> List[dict]:
    """获取涨停池（东方财富）"""
    try:
        import akshare as ak
        date_str = trade_date.replace("-", "")
        df = ak.stock_zt_pool_em(date=date_str)
        if df is None or df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            code = str(row.get("代码", "")).zfill(6)
            results.append({
                "code": code,
                "name": str(row.get("名称", "")),
                "price": float(row.get("最新价", 0) or 0),
                "change_pct": float(row.get("涨跌幅", 0) or 0),
                "turnover_rate": float(row.get("换手率", 0) or 0),
                "amount": float(row.get("成交额", 0) or 0),
                "market_cap": float(row.get("流通市值", 0) or 0),
                "zt_reason": str(row.get("涨停原因", "") or ""),
                "first_zt_time": str(row.get("首次封板时间", "") or ""),
                "last_zt_time": str(row.get("最后封板时间", "") or ""),
                "open_count": int(row.get("开板次数", 0) or 0),
                "zt_stats": str(row.get("涨停统计", "") or ""),
                "continuous_zt": int(row.get("连板数", 0) or 0),
                "industry": str(row.get("所属行业", "") or ""),
            })
        logger.info(f"[涨停分析] 获取涨停池 {len(results)} 只")
        return results
    except Exception as e:
        logger.warning(f"[涨停分析] 涨停池获取失败: {e}")
        return []


def _fetch_dragon_tiger_list(trade_date: str) -> List[dict]:
    """获取龙虎榜明细"""
    try:
        import akshare as ak
        date_str = trade_date.replace("-", "")
        df = ak.stock_lhb_detail_em(start_date=date_str, end_date=date_str)
        if df is None or df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            code = str(row.get("代码", "")).zfill(6)
            results.append({
                "code": code,
                "name": str(row.get("名称", "")),
                "reason": str(row.get("上榜原因", "") or ""),
                "buy_amount": float(row.get("买入额", 0) or 0),
                "sell_amount": float(row.get("卖出额", 0) or 0),
                "net_amount": float(row.get("净买额", 0) or 0),
                "change_pct": float(row.get("涨跌幅", 0) or 0),
                "turnover_rate": float(row.get("换手率", 0) or 0),
                "amount": float(row.get("成交额", 0) or 0),
            })
        logger.info(f"[涨停分析] 龙虎榜 {len(results)} 条")
        return results
    except Exception as e:
        logger.warning(f"[涨停分析] 龙虎榜获取失败: {e}")
        return []


def _fetch_lhb_stock_detail(code: str, trade_date: str) -> List[dict]:
    """获取单只股票龙虎榜买卖席位明细"""
    try:
        import akshare as ak
        date_str = trade_date.replace("-", "")
        df = ak.stock_lhb_stock_detail_em(
            symbol=code,
            date=date_str,
            flag="买入"
        )
        seats = []
        if df is not None and not df.empty:
            for _, row in df.head(5).iterrows():
                seats.append({
                    "trader": str(row.get("营业部名称", "") or ""),
                    "buy_amount": float(row.get("买入金额", 0) or 0),
                    "sell_amount": float(row.get("卖出金额", 0) or 0),
                    "net_amount": float(row.get("净买额", 0) or 0),
                })
        return seats
    except Exception:
        return []


def _fetch_concept_boards(code: str) -> List[str]:
    """获取股票所属概念板块"""
    try:
        from data_provider.base import get_provider
        provider = get_provider()
        boards = provider.get_belong_boards(code)
        return [
            b.get("board_name", "")
            for b in (boards or [])
            if b.get("board_type") == "概念" and b.get("board_name")
        ]
    except Exception:
        pass

    # 兜底：akshare
    try:
        import akshare as ak
        df = ak.stock_board_concept_name_em()
        if df is None or df.empty:
            return []
        concepts = []
        for _, row in df.iterrows():
            concept_name = str(row.get("板块名称", ""))
            try:
                detail = ak.stock_board_concept_cons_em(symbol=concept_name)
                if detail is not None and not detail.empty:
                    codes_in = detail["代码"].astype(str).str.zfill(6).tolist()
                    if code in codes_in:
                        concepts.append(concept_name)
            except Exception:
                continue
            if len(concepts) >= 5:
                break
        return concepts
    except Exception:
        return []


def _fetch_concept_related_stocks(concept: str, limit: int = 10) -> List[dict]:
    """获取同概念板块的股票列表"""
    try:
        import akshare as ak
        df = ak.stock_board_concept_cons_em(symbol=concept)
        if df is None or df.empty:
            return []
        results = []
        for _, row in df.head(limit).iterrows():
            results.append({
                "code": str(row.get("代码", "")).zfill(6),
                "name": str(row.get("名称", "")),
                "price": float(row.get("最新价", 0) or 0),
                "change_pct": float(row.get("涨跌幅", 0) or 0),
            })
        return results
    except Exception:
        return []


def _search_limit_up_reason(name: str, code: str) -> str:
    """网络搜索涨停原因"""
    try:
        from src.search_service import SearchService
        svc = SearchService()
        result = svc.search_stock_news(
            stock_code=code,
            stock_name=name,
            max_results=3,
            focus_keywords=["涨停", "利好", "消息", "政策"],
        )
        if result and result.results:
            snippets = []
            for r in result.results[:3]:
                title = getattr(r, "title", "") or ""
                snippet = getattr(r, "content", "") or getattr(r, "snippet", "") or ""
                if title:
                    snippets.append(f"{title}: {snippet[:100]}")
            return "\n".join(snippets)
    except Exception as e:
        logger.debug(f"[涨停分析] 搜索 {name} 失败: {e}")
    return ""


def _fetch_fund_flow_top(top_n: int = 30) -> Dict[str, dict]:
    """获取资金流向排行，返回 {code: flow_data}"""
    try:
        from ths_scraper import fetch_stock_fund_flow_rank
        flows = fetch_stock_fund_flow_rank(top_n=top_n)
        return {f.get("code", ""): f for f in (flows or []) if f.get("code")}
    except Exception:
        return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 数据组装 + 并行采集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _enrich_limit_up_stocks(
    zt_stocks: List[dict],
    lhb_map: Dict[str, dict],
    flow_map: Dict[str, dict],
    trade_date: str,
    max_detail: int = 15,
) -> List[dict]:
    """并行为涨停股补充搜索/概念/龙虎榜席位"""
    enriched = []

    # 只分析前 max_detail 只（按连板数+成交额排序）
    sorted_stocks = sorted(
        zt_stocks,
        key=lambda x: (x.get("continuous_zt", 0), x.get("amount", 0)),
        reverse=True,
    )[:max_detail]

    def _enrich_one(stock):
        code = stock["code"]
        name = stock["name"]

        # 龙虎榜
        stock["lhb"] = lhb_map.get(code, {})
        if stock["lhb"]:
            stock["lhb_seats"] = _fetch_lhb_stock_detail(code, trade_date)

        # 资金流
        stock["fund_flow"] = flow_map.get(code, {})

        # 概念板块
        stock["concepts"] = _fetch_concept_boards(code)

        # 网络搜索涨停原因
        if not stock.get("zt_reason"):
            stock["search_reason"] = _search_limit_up_reason(name, code)
        else:
            stock["search_reason"] = ""

        return stock

    workers = min(6, len(sorted_stocks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_enrich_one, s): s for s in sorted_stocks}
        for f in as_completed(futures):
            try:
                enriched.append(f.result())
            except Exception as e:
                stock = futures[f]
                logger.warning(f"[涨停分析] {stock.get('name', '')} 补充失败: {e}")
                enriched.append(stock)

    # 按连板数+成交额重新排序
    enriched.sort(key=lambda x: (x.get("continuous_zt", 0), x.get("amount", 0)), reverse=True)
    return enriched


def _find_related_stocks(enriched_stocks: List[dict]) -> Dict[str, List[dict]]:
    """根据涨停股的共同概念，找出关联受益股"""
    # 统计概念出现频次
    concept_count: Dict[str, int] = {}
    concept_stocks: Dict[str, List[str]] = {}
    for s in enriched_stocks:
        for c in s.get("concepts", []):
            concept_count[c] = concept_count.get(c, 0) + 1
            concept_stocks.setdefault(c, []).append(s["name"])

    # 筛选至少2只涨停股共享的概念
    hot_concepts = {
        c: names for c, names in concept_stocks.items()
        if concept_count.get(c, 0) >= 2
    }

    # 获取这些概念的成分股（排除已涨停的）
    zt_codes = {s["code"] for s in enriched_stocks}
    related = {}
    for concept in list(hot_concepts.keys())[:5]:
        members = _fetch_concept_related_stocks(concept, limit=15)
        # 排除涨停股，按涨幅排序（找还没涨的潜力股）
        candidates = [
            m for m in members
            if m["code"] not in zt_codes and m.get("change_pct", 0) < 9
        ]
        candidates.sort(key=lambda x: x.get("change_pct", 0), reverse=True)
        if candidates:
            related[concept] = {
                "zt_members": hot_concepts[concept],
                "candidates": candidates[:8],
            }

    return related


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. AI 分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_analysis_prompt(
    enriched: List[dict],
    related: Dict[str, List[dict]],
    trade_date: str,
) -> str:
    """构建 AI 分析 prompt"""
    sections = []

    # 涨停概览
    zt_lines = []
    for s in enriched:
        zt_tag = f"【{s['continuous_zt']}连板】" if s.get("continuous_zt", 0) > 1 else ""
        reason = s.get("zt_reason", "") or ""
        search = s.get("search_reason", "") or ""
        reason_text = reason or search[:120] or "未知"

        line = (
            f"  {zt_tag}{s['name']}({s['code']}) "
            f"行业:{s.get('industry', '')} "
            f"封板:{s.get('first_zt_time', '')} "
            f"开板:{s.get('open_count', 0)}次 "
            f"换手:{s.get('turnover_rate', 0):.1f}% "
            f"成交额:{s.get('amount', 0)/1e8:.1f}亿"
        )
        if reason_text:
            line += f"\n    原因: {reason_text}"

        # 龙虎榜
        lhb = s.get("lhb", {})
        if lhb:
            line += f"\n    龙虎榜: 净买{lhb.get('net_amount', 0)/1e4:.0f}万"
        seats = s.get("lhb_seats", [])
        if seats:
            top_seats = [f"{st['trader'][:8]}(净买{st['net_amount']/1e4:.0f}万)" for st in seats[:3]]
            line += f"\n    席位: {', '.join(top_seats)}"

        # 资金流
        flow = s.get("fund_flow", {})
        if flow.get("main_net"):
            line += f"\n    主力净流入: {flow['main_net']:.0f}万"

        # 概念
        concepts = s.get("concepts", [])
        if concepts:
            line += f"\n    概念: {', '.join(concepts[:5])}"

        zt_lines.append(line)

    sections.append(f"【{trade_date} 涨停池 ({len(enriched)}只)】\n" + "\n\n".join(zt_lines))

    # 关联概念
    if related:
        rel_lines = []
        for concept, info in related.items():
            zt_names = ", ".join(info["zt_members"])
            cands = info["candidates"]
            cand_str = ", ".join(
                f"{c['name']}({c['code']}){c['change_pct']:+.1f}%"
                for c in cands[:5]
            )
            rel_lines.append(
                f"  概念【{concept}】涨停成员: {zt_names}\n"
                f"    关联受益: {cand_str}"
            )
        sections.append("【热门概念 & 关联股】\n" + "\n".join(rel_lines))

    data_block = "\n\n".join(sections)

    return f"""你是一位A股短线游资风格的盘后分析师。请根据以下 {trade_date} 涨停数据，生成一份深度涨停复盘分析报告。

{data_block}

请按以下结构输出：

## 一、涨停情绪总览
- 今日涨停数量、连板高度、情绪温度（冰点/修复/高潮/分歧）
- 涨停集中在哪些行业/概念？资金主攻方向是什么？

## 二、核心主线分析
- 识别今日1-3条核心主线（最强概念/题材）
- 每条主线：龙头是谁？逻辑是什么？持续性判断？
- 龙虎榜机构/游资动向：哪些知名席位在买？说明什么？

## 三、连板股深度分析
- 连板股的逻辑和空间判断
- 首板股中哪些有晋级（二板）潜力？为什么？

## 四、明日关联机会
- 基于今日涨停逻辑，哪些关联股明天可能受益？
- 给出具体股票代码和买入逻辑
- 注意风险提示：哪些是补涨末端不宜追高

## 五、风险提示
- 哪些涨停股是末端行情（见顶信号）？
- 市场整体情绪风险评估
- 明天需要规避的方向

语言要求：简洁犀利，直接给结论和操作建议。每只股票的分析要有数据支撑。"""


def _ai_analyze_limit_up(prompt: str) -> str:
    """调用 AI 分析"""
    try:
        import litellm
        cloud_model = os.getenv("REBALANCE_CLOUD_MODEL") or os.getenv("LITELLM_MODEL", "")
        fallback = os.getenv("REBALANCE_CLOUD_FALLBACK") or os.getenv("LITELLM_FALLBACK_MODELS", "")
        models = [m.strip() for m in [cloud_model, fallback] if m.strip()]

        for model in models:
            try:
                resp = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=4000,
                    timeout=120,
                )
                text = resp.choices[0].message.content.strip()
                if text:
                    logger.info(f"[涨停分析] AI完成 (model={model}, {len(text)}字)")
                    return text
            except Exception as e:
                logger.warning(f"[涨停分析] {model} 失败: {e}")
                continue
    except Exception as e:
        logger.warning(f"[涨停分析] AI调用失败: {e}")
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 报告生成 + 发送
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _format_report(
    enriched: List[dict],
    related: Dict[str, List[dict]],
    ai_analysis: str,
    trade_date: str,
) -> str:
    """组装最终报告"""
    lines = [f"🔥 **{trade_date} 涨停深度分析**", ""]

    # 概要统计
    total = len(enriched)
    continuous = [s for s in enriched if s.get("continuous_zt", 0) > 1]
    max_board = max((s.get("continuous_zt", 0) for s in enriched), default=0)
    on_lhb = sum(1 for s in enriched if s.get("lhb"))

    lines.append(f"📊 涨停: {total}只 | 连板: {len(continuous)}只 | 最高: {max_board}连板 | 龙虎榜: {on_lhb}只")

    # 行业分布
    industry_count: Dict[str, int] = {}
    for s in enriched:
        ind = s.get("industry", "未知") or "未知"
        industry_count[ind] = industry_count.get(ind, 0) + 1
    top_industries = sorted(industry_count.items(), key=lambda x: x[1], reverse=True)[:5]
    if top_industries:
        ind_str = " | ".join(f"{k}({v})" for k, v in top_industries)
        lines.append(f"🏭 行业分布: {ind_str}")

    # 热门概念
    if related:
        concept_str = " | ".join(f"{c}({len(info['zt_members'])}只)" for c, info in list(related.items())[:5])
        lines.append(f"💡 热门概念: {concept_str}")

    lines.append("")
    lines.append("─" * 30)
    lines.append("")

    # AI 分析
    if ai_analysis:
        lines.append(ai_analysis)
    else:
        # 纯数据 fallback
        lines.append("**涨停股明细:**")
        for s in enriched[:15]:
            zt_tag = f"[{s['continuous_zt']}板]" if s.get("continuous_zt", 0) > 1 else ""
            reason = s.get("zt_reason", "") or s.get("search_reason", "")[:60] or ""
            lines.append(
                f"  {zt_tag}{s['name']}({s['code']}) "
                f"{s.get('industry', '')} "
                f"换手:{s.get('turnover_rate', 0):.1f}%"
                + (f" | {reason}" if reason else "")
            )

        if related:
            lines.append("\n**关联机会:**")
            for concept, info in list(related.items())[:3]:
                cands = ", ".join(
                    f"{c['name']}{c['change_pct']:+.1f}%"
                    for c in info["candidates"][:5]
                )
                lines.append(f"  {concept}: {cands}")

    return "\n".join(lines)


def _save_report(report: str, enriched: List[dict], trade_date: str):
    """保存报告"""
    import json
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    md_path = REPORT_DIR / f"limit_up_{trade_date.replace('-', '')}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    json_path = REPORT_DIR / f"limit_up_{trade_date.replace('-', '')}.json"
    json_data = {
        "trade_date": trade_date,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_limit_up": len(enriched),
        "stocks": enriched,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"[涨停分析] 报告已保存: {md_path}")


def _send_report(report: str):
    """推送报告"""
    try:
        from src.notification import NotificationService
        notifier = NotificationService()
        if notifier.is_available():
            notifier.send(report)
            logger.info("[涨停分析] 报告已推送")
    except Exception as e:
        logger.warning(f"[涨停分析] 推送失败: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_limit_up_analysis(
    trade_date: Optional[str] = None,
    send_notification: bool = True,
    max_detail: int = 15,
) -> Optional[str]:
    """
    执行涨停股深度分析，返回报告文本。

    Args:
        trade_date: 分析日期 YYYY-MM-DD，默认今天
        send_notification: 是否推送通知
        max_detail: 详细分析的涨停股数量上限
    """
    if not trade_date:
        trade_date = datetime.now(TZ_CN).strftime("%Y-%m-%d")

    logger.info(f"[涨停分析] 开始 {trade_date} 涨停分析...")

    # 1. 并行获取基础数据
    zt_stocks = []
    lhb_list = []
    flow_map = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_zt = pool.submit(_fetch_limit_up_pool, trade_date)
        f_lhb = pool.submit(_fetch_dragon_tiger_list, trade_date)
        f_flow = pool.submit(_fetch_fund_flow_top, 50)

        zt_stocks = f_zt.result()
        lhb_list = f_lhb.result()
        flow_map = f_flow.result()

    if not zt_stocks:
        msg = f"📊 {trade_date} 未获取到涨停数据"
        logger.info(msg)
        return msg

    # 龙虎榜转为 {code: data}
    lhb_map = {}
    for item in lhb_list:
        code = item.get("code", "")
        if code:
            lhb_map[code] = item

    # 2. 并行补充详情（搜索+概念+席位）
    enriched = _enrich_limit_up_stocks(zt_stocks, lhb_map, flow_map, trade_date, max_detail)

    # 3. 挖掘关联股
    related = _find_related_stocks(enriched)

    # 4. AI 分析
    prompt = _build_analysis_prompt(enriched, related, trade_date)
    ai_analysis = _ai_analyze_limit_up(prompt)

    # 5. 生成报告
    report = _format_report(enriched, related, ai_analysis, trade_date)

    # 6. 保存
    _save_report(report, enriched, trade_date)

    # 7. 推送
    if send_notification:
        _send_report(report)

    logger.info(f"[涨停分析] {trade_date} 完成，涨停{len(zt_stocks)}只")
    return report


def get_limit_up_report(trade_date: Optional[str] = None) -> str:
    """读取已保存报告或实时生成（飞书指令用）"""
    if not trade_date:
        trade_date = datetime.now(TZ_CN).strftime("%Y-%m-%d")

    md_path = REPORT_DIR / f"limit_up_{trade_date.replace('-', '')}.md"
    if md_path.exists():
        return md_path.read_text(encoding="utf-8")

    return run_limit_up_analysis(trade_date=trade_date, send_notification=False) or "暂无涨停数据"
