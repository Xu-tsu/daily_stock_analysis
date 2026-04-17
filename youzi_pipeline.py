# -*- coding: utf-8 -*-
"""游资风格选股编排器 + 全程报告。

对接 dual_trader.run_stock_scan 的现有 candidate 列表：
  1. 构造每个 candidate 的 TimeframeContext（日/5日/周/月）
  2. 构造 NewsContext（热点概念排名 / 事件加分 / 近期亏损）
  3. 跑所有激活的游资风格（youzi_styles.score_all_styles）
  4. 聚合投票（aggregate_verdict）+ V2 仓位协同
  5. 产出完整报告：
        - reports/youzi_pick_YYYYMMDD_HHMM.md   （人看）
        - reports/youzi_pick_YYYYMMDD_HHMM.json （机器看 / 复盘）

主入口:
    enrich_with_youzi(results, hot_concepts, regime, adaptive_coeff=1.0,
                      event_defense=False, active_styles=None)
    → (enriched_results, trace_dict, report_paths)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from youzi_styles import (
    NewsContext,
    STYLE_REGISTRY,
    aggregate_verdict,
    list_styles,
    score_all_styles,
)
from youzi_timeframe import TimeframeContext, build_timeframe_context

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────
# V2 & Regime 协同
# ──────────────────────────────────────────────────────────

def _get_v2_plan(regime, adaptive_coeff: float = 1.0, event_defense: bool = False):
    try:
        from live_strategy_v2 import get_position_plan
        return get_position_plan(regime, adaptive_coeff, event_defense)
    except Exception as e:
        logger.warning(f"[youzi] V2 plan unavailable: {e}")
        return {
            "regime": "SIDE", "max_positions": 1,
            "single_pct": 0.3, "total_pct": 0.3,
            "stop_loss": -3.0, "tp_half": 4.0, "tp_full": 8.0,
        }


def _normalize_regime_str(regime) -> str:
    try:
        from live_strategy_v2 import normalize_regime
        return normalize_regime(regime)
    except Exception:
        if isinstance(regime, dict):
            return (regime.get("regime") or "SIDE").upper()
        return (regime or "SIDE").upper()


# ──────────────────────────────────────────────────────────
# NewsContext 构造
# ──────────────────────────────────────────────────────────

def _build_news_ctx(
    candidate: Dict,
    hot_concepts: Optional[List[Dict]],
) -> NewsContext:
    code = str(candidate.get("code", "")).replace("sh", "").replace("sz", "")[-6:]

    in_hot = False
    hot_names: List[str] = []
    rank = 0
    if hot_concepts:
        for idx, hc in enumerate(hot_concepts, 1):
            stock_codes = set()
            for s in hc.get("stocks", []):
                cc = str(s.get("code", "")).strip()[-6:]
                if cc:
                    stock_codes.add(cc)
            if code in stock_codes:
                in_hot = True
                hot_names.append(str(hc.get("name", "")))
                if rank == 0 or idx < rank:
                    rank = idx
    return NewsContext(
        in_hot_concept=in_hot,
        hot_concept_names=hot_names,
        hot_concept_rank=rank,
        event_boost=bool(candidate.get("event_boost")),
        event_bonus=int(candidate.get("event_bonus", 0) or 0),
        recent_loser=bool(candidate.get("recent_loser")),
        news_score=(40 if in_hot else 0) + int(candidate.get("event_bonus", 0) or 0),
    )


# ──────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────

def enrich_with_youzi(
    results: List[Dict],
    hot_concepts: Optional[List[Dict]] = None,
    regime=None,
    adaptive_coeff: float = 1.0,
    event_defense: bool = False,
    active_styles: Optional[List[str]] = None,
    save_report: bool = True,
    top_n_report: int = 15,
) -> Tuple[List[Dict], Dict, Dict[str, str]]:
    """对 scan_market 的 results 做游资风格增强 + V2 协同 + 报告落盘。

    Returns:
        enriched: 每条 candidate 追加字段：
            youzi_results       : {style_name: dict(YouziStyleResult)}
            youzi_aggregate     : {buy_votes, weighted_score, consensus, ...}
            youzi_buy_votes     : List[str]
            youzi_weighted_score: float
            youzi_verdict       : "buy" | "watch" | "skip" | "veto"
            tech_score          : 合并（base + 权重累积 delta）
        trace  : 整体过程 dict，可用于推送
        paths  : {"md": "...", "json": "..."} 或空 dict
    """
    t0 = time.time()
    regime_key = _normalize_regime_str(regime)
    v2_plan = _get_v2_plan(regime, adaptive_coeff, event_defense)
    active = active_styles or list_styles()

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  [YOUZI] 游资风格评分  regime={regime_key} styles={active}")
    logger.info(f"  [YOUZI] V2 plan: max_pos={v2_plan['max_positions']} "
                f"single={v2_plan['single_pct']*100:.1f}% total={v2_plan['total_pct']*100:.1f}%")
    logger.info("=" * 60)

    enriched: List[Dict] = []
    per_candidate_records: List[Dict] = []

    # regime -> 风格权重（不同 regime 下不同游资更靠谱）
    REGIME_WEIGHTS = {
        "BULL":  {"chen_xiaoqun": 1.3, "zhao_laoge": 1.2, "zhang_jiahu": 1.2, "ouyang": 0.8},
        "SIDE":  {"chen_xiaoqun": 1.0, "zhao_laoge": 1.0, "zhang_jiahu": 0.7, "ouyang": 1.3},
        "BEAR":  {"chen_xiaoqun": 0.5, "zhao_laoge": 0.5, "zhang_jiahu": 0.3, "ouyang": 1.2},
        "CRASH": {"chen_xiaoqun": 0.2, "zhao_laoge": 0.2, "zhang_jiahu": 0.0, "ouyang": 0.5},
    }
    weights = REGIME_WEIGHTS.get(regime_key, REGIME_WEIGHTS["SIDE"])

    for cand in results:
        code = str(cand.get("code", "")).replace("sh", "").replace("sz", "")[-6:]
        # 1. 多周期
        try:
            tf = build_timeframe_context(code)
        except Exception as e:
            logger.warning(f"[youzi] timeframe failed {code}: {e}")
            tf = TimeframeContext(code=code, ok=False, error=str(e))

        # 2. 消息面
        news = _build_news_ctx(cand, hot_concepts)

        # 3. 跑所有风格
        per_style = score_all_styles(cand, tf, news, regime=regime_key, active=active)

        # 4. 聚合
        agg = aggregate_verdict(per_style, weights=weights)

        # 5. 合并回 tech_score：加权 delta / 2，避免与原分冲击太猛
        total_delta = 0.0
        weight_sum = 0.0
        for name, r in per_style.items():
            if r.verdict == "veto":
                continue
            w = weights.get(name, 1.0)
            total_delta += r.score_delta * w
            weight_sum += w
        avg_delta = (total_delta / weight_sum) if weight_sum > 0 else 0.0

        new_tech_score = int(round(int(cand.get("tech_score", 0) or 0) + avg_delta))

        enriched_cand = dict(cand)
        enriched_cand["youzi_results"] = {k: v.to_dict() for k, v in per_style.items()}
        enriched_cand["youzi_aggregate"] = agg
        enriched_cand["youzi_buy_votes"] = agg["buy_votes"]
        enriched_cand["youzi_weighted_score"] = agg["weighted_score"]
        enriched_cand["youzi_verdict"] = (
            "veto" if agg["vetoed_by"] and not agg["buy_votes"]
            else ("buy" if agg["buy_votes"]
                  else ("watch" if agg["watch_votes"] else "skip"))
        )
        enriched_cand["timeframe"] = tf.to_dict()
        enriched_cand["news_ctx"] = news.to_dict()
        enriched_cand["tech_score"] = new_tech_score

        enriched.append(enriched_cand)
        per_candidate_records.append({
            "code": code,
            "name": cand.get("name", ""),
            "base_score": int(cand.get("tech_score", 0) or 0),
            "final_tech_score": new_tech_score,
            "weighted_delta": round(avg_delta, 1),
            "per_style": {k: v.to_dict() for k, v in per_style.items()},
            "aggregate": agg,
            "timeframe": tf.to_dict(),
            "news": news.to_dict(),
        })

    # 重新按 tech_score 排序
    enriched.sort(key=lambda x: x.get("tech_score", 0), reverse=True)

    # ── 决策层：结合 V2 选出 final buy list ──
    max_pos = int(v2_plan.get("max_positions", 1) or 1)
    buy_list: List[Dict] = []
    for cand in enriched:
        if len(buy_list) >= max_pos:
            break
        v = cand.get("youzi_verdict")
        if v == "veto":
            continue
        # 至少要有 1 家风格 buy 或 weighted_score >= 80
        if v == "buy" or cand["youzi_weighted_score"] >= 80:
            buy_list.append(cand)

    elapsed = round(time.time() - t0, 1)
    trace = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": elapsed,
        "regime": regime_key,
        "adaptive_coeff": adaptive_coeff,
        "event_defense": event_defense,
        "v2_plan": v2_plan,
        "active_styles": active,
        "regime_weights": weights,
        "candidate_count": len(results),
        "buy_count": len(buy_list),
        "candidates": per_candidate_records[:top_n_report],
        "buy_list": [
            {
                "code": c["code"], "name": c.get("name"),
                "final_score": c["tech_score"],
                "buy_votes": c["youzi_buy_votes"],
                "weighted_score": c["youzi_weighted_score"],
                "verdict": c["youzi_verdict"],
            }
            for c in buy_list
        ],
    }

    paths: Dict[str, str] = {}
    if save_report:
        try:
            paths = _save_report(trace, enriched[:top_n_report])
            logger.info(f"  [YOUZI] report saved: {paths.get('md')}")
        except Exception as e:
            logger.warning(f"[youzi] save report failed: {e}")

    # 持久化快照，供 youzi_review.py T+N 复盘
    try:
        from youzi_watchlist import save_snapshot
        save_snapshot(
            enriched=enriched,
            buy_list=buy_list,
            regime=regime_key,
            v2_plan=v2_plan,
        )
    except Exception as e:
        logger.debug(f"[youzi] watchlist save failed: {e}")

    logger.info(f"  [YOUZI] done in {elapsed}s | candidates={len(enriched)} buy={len(buy_list)}")
    return enriched, trace, paths


# ──────────────────────────────────────────────────────────
# 报告渲染
# ──────────────────────────────────────────────────────────

def _save_report(trace: Dict, top_candidates: List[Dict]) -> Dict[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    md_path = REPORTS_DIR / f"youzi_pick_{ts}.md"
    json_path = REPORTS_DIR / f"youzi_pick_{ts}.json"

    # JSON 原始
    json_payload = dict(trace)
    json_payload["top_candidates_full"] = [
        {
            "code": c.get("code"),
            "name": c.get("name"),
            "price": c.get("price"),
            "change_pct": c.get("change_pct"),
            "tech_score": c.get("tech_score"),
            "signal_type": c.get("signal_type"),
            "market_cap": c.get("market_cap"),
            "turnover_rate": c.get("turnover_rate"),
            "youzi_verdict": c.get("youzi_verdict"),
            "youzi_buy_votes": c.get("youzi_buy_votes"),
            "youzi_weighted_score": c.get("youzi_weighted_score"),
            "timeframe": c.get("timeframe"),
            "news_ctx": c.get("news_ctx"),
            "youzi_results": c.get("youzi_results"),
        }
        for c in top_candidates
    ]
    def _json_default(o):
        # numpy scalars / bools / arrays → python native
        try:
            import numpy as _np
            if isinstance(o, (_np.bool_,)):
                return bool(o)
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, (_np.ndarray,)):
                return o.tolist()
        except Exception:
            pass
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return str(o)

    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8"
    )

    # Markdown 人类视图
    md = _render_markdown(trace, top_candidates)
    md_path.write_text(md, encoding="utf-8")

    return {"md": str(md_path), "json": str(json_path)}


def _render_markdown(trace: Dict, cands: List[Dict]) -> str:
    lines = []
    lines.append(f"# 游资风格选股报告 · {trace['timestamp']}")
    lines.append("")
    lines.append(f"- **市场 regime**: `{trace['regime']}`")
    lines.append(f"- **V2 仓位计划**: max_positions={trace['v2_plan']['max_positions']}, "
                 f"single_pct={trace['v2_plan']['single_pct']*100:.1f}%, "
                 f"total_pct={trace['v2_plan']['total_pct']*100:.1f}%, "
                 f"adaptive_coeff={trace['adaptive_coeff']}, "
                 f"event_defense={trace['event_defense']}")
    lines.append(f"- **激活风格**: {', '.join(trace['active_styles'])}")
    lines.append(f"- **风格权重 (regime={trace['regime']})**: "
                 + ", ".join(f"{k}={v}" for k, v in trace['regime_weights'].items()))
    lines.append(f"- **扫描候选数**: {trace['candidate_count']}  → **最终买入**: {trace['buy_count']}  "
                 f"(耗时 {trace['elapsed_s']}s)")
    lines.append("")

    # 买入列表
    lines.append("## 💰 最终买入清单")
    lines.append("")
    if trace["buy_list"]:
        lines.append("| # | 代码 | 名称 | 综合分 | 加权分 | 多数派风格 | 结论 |")
        lines.append("|---|------|------|--------|--------|-----------|------|")
        for i, b in enumerate(trace["buy_list"], 1):
            lines.append(f"| {i} | `{b['code']}` | {b['name']} | **{b['final_score']}** | "
                         f"{b['weighted_score']} | {','.join(b['buy_votes']) or '-'} | {b['verdict']} |")
    else:
        lines.append("> 本轮无风格共识买入信号。")
    lines.append("")

    # 详细候选评分过程
    lines.append("## 🔍 候选明细（Top 15）")
    lines.append("")
    for i, c in enumerate(cands, 1):
        name = c.get("name") or "?"
        code = c.get("code")
        price = c.get("price", 0)
        chg = c.get("change_pct", 0)
        sig = c.get("signal_type", "")
        mc = c.get("market_cap", 0)
        if mc > 10000:
            mc_yi = mc / 1e8
        else:
            mc_yi = mc
        tf = c.get("timeframe", {}) or {}
        news = c.get("news_ctx", {}) or {}

        lines.append(f"### {i}. {name}({code}) · 价 {price:.2f} · 涨幅 {chg:+.1f}% · 市值 {mc_yi:.0f}亿")
        lines.append("")
        lines.append(f"- **信号**: `{sig}` · tech_score = **{c.get('tech_score')}**")
        lines.append(f"- **结论**: `{c.get('youzi_verdict')}` · 买入投票: "
                     f"{','.join(c.get('youzi_buy_votes', [])) or '无'} · "
                     f"加权分 {c.get('youzi_weighted_score')}")
        lines.append(f"- **多周期**: 日线{'多头' if tf.get('daily_long_arrangement') else ('空头' if tf.get('daily_short_arrangement') else '震荡')}"
                     f" / 周线 {tf.get('weekly_trend','-')} (RSI={tf.get('weekly_rsi','-')})"
                     f" / 月线 {tf.get('monthly_trend','-')}"
                     f" / 距年高 {tf.get('pct_from_52w_high','-')}%"
                     f" / 连板 {tf.get('consec_limit',0)} · 连阳 {tf.get('consec_up',0)}")
        lines.append(f"- **消息面**: 热点={'是' if news.get('in_hot_concept') else '否'}"
                     f" · 概念TOP={news.get('hot_concept_rank',0)}"
                     f" · 概念={','.join(news.get('hot_concept_names',[])) or '-'}"
                     f" · 事件加分={news.get('event_bonus',0)}"
                     f" · 近期亏损={'是' if news.get('recent_loser') else '否'}")
        lines.append("")
        lines.append("<details><summary>各游资打分过程</summary>")
        lines.append("")
        yr = c.get("youzi_results", {}) or {}
        for sname, r in yr.items():
            lines.append(f"- **{r.get('display_name', sname)}** → `{r.get('verdict')}` "
                         f"(base={r.get('base_score')}, Δ={r.get('score_delta'):+d}, final={r.get('final_score')})")
            for rsn in r.get("reasons", []):
                lines.append(f"    - {rsn}")
            for veto in r.get("vetoes", []):
                lines.append(f"    - ❌ VETO: {veto}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("---")
    lines.append("_生成器: youzi_pipeline.enrich_with_youzi / 报告路径: "
                 f"{REPORTS_DIR}_")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# 推送报告（可选：合入飞书/邮件/钉钉）
# ──────────────────────────────────────────────────────────

def dispatch_report(trace: Dict, md_path: Optional[str] = None) -> None:
    """把报告推到飞书 / 邮件。失败不抛出，只打日志。"""
    text_summary = (
        f"【游资选股 · {trace['timestamp']}】\n"
        f"regime={trace['regime']}  候选{trace['candidate_count']}只  买入{trace['buy_count']}只\n"
    )
    if trace["buy_list"]:
        for b in trace["buy_list"]:
            text_summary += f"  · {b['name']}({b['code']}) 分{b['final_score']} "\
                            f"[{','.join(b['buy_votes']) or '-'}]\n"
    else:
        text_summary += "  · 无共识买入\n"

    # 飞书
    try:
        from src.notification import send_feishu_text
        send_feishu_text(text_summary)
    except Exception as e:
        logger.debug(f"[youzi] feishu dispatch skipped: {e}")

    # 邮件（发送 MD 摘要）
    try:
        if md_path and os.path.exists(md_path):
            from src.notification import send_email
            body = Path(md_path).read_text(encoding="utf-8")
            send_email(subject=f"[游资选股] {trace['timestamp']}", body=body)
    except Exception as e:
        logger.debug(f"[youzi] email dispatch skipped: {e}")


# ──────────────────────────────────────────────────────────
# CLI 手动测试
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="游资风格选股·手动测试")
    parser.add_argument("--regime", default="SIDE", help="BULL/SIDE/BEAR/CRASH")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from market_scanner import scan_market
    r = scan_market(mode="dragon", max_price=50.0, min_turnover=3.0, top_n=args.top)

    try:
        from news_scanner import get_top_concepts
        hot = get_top_concepts(n=10)
    except Exception:
        hot = []

    enriched, trace, paths = enrich_with_youzi(
        r, hot_concepts=hot, regime=args.regime, adaptive_coeff=1.0,
    )
    print(json.dumps({"buy_list": trace["buy_list"], "paths": paths}, ensure_ascii=False, indent=2))
