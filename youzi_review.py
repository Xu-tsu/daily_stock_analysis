# -*- coding: utf-8 -*-
"""游资选股 T+N 复盘。

功能：
  1. 读 youzi_watchlist.json 里的历史 snapshot
  2. 对每一条 candidate，拉取 snapshot 之后的真实日线
     - 1 天后收益
     - 3 天后收益
     - 5 天后收益
     - 5 天内最大涨幅 / 最大回撤
  3. 三张对比表：
     - bought vs. not-bought  → 是否"错过大肉"或"救了狗命"
     - 风格正确率             → 每个游资 buy 标签的命中率
     - vetoed-but-gained      → 被否决但实际大涨的（用来反思)
  4. 输出 reports/youzi_review_YYYYMMDD.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from youzi_watchlist import list_snapshots, snapshots_needing_review

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────
# 行情回访
# ──────────────────────────────────────────────────────────

def _fetch_future_kline(code: str, since_date: str, days: int = 10) -> pd.DataFrame:
    """拉取 since_date 之后的 K 线（含当日及以后 days 天）。"""
    try:
        from market_scanner import _fetch_kline
        df = _fetch_kline(code, days=days + 30)
    except Exception as e:
        logger.debug(f"[review] fetch_kline failed {code}: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    # 归一化索引
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("date", "Date", "trade_date"):
            if col in df.columns:
                df = df.copy()
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
    df = df.sort_index()
    # since_date 当日及以后
    try:
        since = pd.Timestamp(since_date[:10])
    except Exception:
        return df
    return df[df.index >= since].head(days + 1)


def _summarize_move(entry_price: float, future: pd.DataFrame) -> Dict:
    """基于 entry_price 和未来 K 线，计算不同时点的收益 + 极值。"""
    if future is None or future.empty or entry_price <= 0:
        return {"valid": False}

    close = future["close"].astype(float).values
    high = future["high"].astype(float).values if "high" in future.columns else close
    low = future["low"].astype(float).values if "low" in future.columns else close

    def pct(later: float) -> float:
        return round((later / entry_price - 1) * 100, 2)

    ret_1d = pct(close[1]) if len(close) > 1 else 0.0
    ret_3d = pct(close[3]) if len(close) > 3 else (pct(close[-1]) if len(close) > 1 else 0.0)
    ret_5d = pct(close[5]) if len(close) > 5 else (pct(close[-1]) if len(close) > 1 else 0.0)

    max_gain = pct(float(np.max(high[1:]))) if len(high) > 1 else 0.0
    max_drawdown = pct(float(np.min(low[1:]))) if len(low) > 1 else 0.0

    return {
        "valid": True,
        "ret_1d": ret_1d,
        "ret_3d": ret_3d,
        "ret_5d": ret_5d,
        "max_gain": max_gain,
        "max_drawdown": max_drawdown,
        "days_observed": int(len(close) - 1),
    }


# ──────────────────────────────────────────────────────────
# 复盘主入口
# ──────────────────────────────────────────────────────────

@dataclass
class StyleStat:
    style: str
    picks: int = 0          # 该风格 verdict=buy 的次数
    wins: int = 0           # 3日正收益次数
    big_wins: int = 0       # 3日 >= 5%
    losses: int = 0         # 3日负收益
    avg_ret_3d: float = 0.0
    avg_ret_5d: float = 0.0
    _rets_3d: List[float] = field(default_factory=list)
    _rets_5d: List[float] = field(default_factory=list)

    def finalize(self):
        if self._rets_3d:
            self.avg_ret_3d = round(float(np.mean(self._rets_3d)), 2)
        if self._rets_5d:
            self.avg_ret_5d = round(float(np.mean(self._rets_5d)), 2)


def review_snapshot(snapshot: Dict) -> Dict:
    """对一个 snapshot 做 T+N 回访。"""
    since = snapshot.get("timestamp", "")
    day_count = 6  # 观察 5 个交易日（含当日）
    results: List[Dict] = []

    # 每个风格的统计容器
    style_stats: Dict[str, StyleStat] = {}
    buy_stats = StyleStat(style="_BOUGHT_")
    nobuy_stats = StyleStat(style="_NOT_BOUGHT_")
    vetoed_stats = StyleStat(style="_VETOED_")

    for c in snapshot.get("candidates", []):
        code = c["code"]
        entry_price = c.get("entry_price", 0)
        if entry_price <= 0:
            continue

        fut = _fetch_future_kline(code, since, days=day_count)
        move = _summarize_move(entry_price, fut)
        if not move.get("valid"):
            continue

        record = {
            "code": code,
            "name": c.get("name", ""),
            "entry_price": entry_price,
            "acted": c.get("acted", False),
            "youzi_verdict": c.get("youzi_verdict"),
            "youzi_buy_votes": c.get("youzi_buy_votes", []),
            "youzi_vetoed_by": c.get("youzi_vetoed_by", []),
            "per_style_verdicts": c.get("per_style_verdicts", {}),
            "hot_concepts": c.get("hot_concepts", []),
            "signal_type": c.get("signal_type", ""),
            **move,
        }
        results.append(record)

        # 归类统计
        group = buy_stats if c.get("acted") else nobuy_stats
        group.picks += 1
        group._rets_3d.append(move["ret_3d"])
        group._rets_5d.append(move["ret_5d"])
        if move["ret_3d"] > 0:
            group.wins += 1
            if move["ret_3d"] >= 5:
                group.big_wins += 1
        elif move["ret_3d"] < 0:
            group.losses += 1

        # 被 veto 且实际大涨
        if c.get("youzi_verdict") == "veto":
            vetoed_stats.picks += 1
            vetoed_stats._rets_3d.append(move["ret_3d"])
            vetoed_stats._rets_5d.append(move["ret_5d"])
            if move["ret_3d"] > 0:
                vetoed_stats.wins += 1
                if move["ret_3d"] >= 5:
                    vetoed_stats.big_wins += 1

        # 分风格统计：该风格判 buy 的票命中率
        for sname, verdict in (c.get("per_style_verdicts") or {}).items():
            if verdict != "buy":
                continue
            st = style_stats.setdefault(sname, StyleStat(style=sname))
            st.picks += 1
            st._rets_3d.append(move["ret_3d"])
            st._rets_5d.append(move["ret_5d"])
            if move["ret_3d"] > 0:
                st.wins += 1
                if move["ret_3d"] >= 5:
                    st.big_wins += 1
            elif move["ret_3d"] < 0:
                st.losses += 1

    for s in (*style_stats.values(), buy_stats, nobuy_stats, vetoed_stats):
        s.finalize()

    return {
        "snapshot_ts": since,
        "regime": snapshot.get("regime"),
        "candidate_count": len(results),
        "records": results,
        "bought_summary": buy_stats.__dict__,
        "not_bought_summary": nobuy_stats.__dict__,
        "vetoed_summary": vetoed_stats.__dict__,
        "style_stats": {k: v.__dict__ for k, v in style_stats.items()},
    }


def review_recent(days_back: int = 7, horizon_days: int = 5) -> Dict:
    """复盘最近 N 天内已有 >= horizon_days 观察窗口的 snapshot（可合并多次）。"""
    snaps = snapshots_needing_review(horizon_days=horizon_days)
    # 只取最近 days_back 天的
    cutoff = datetime.now() - timedelta(days=days_back + horizon_days)
    snaps = [
        s for s in snaps
        if datetime.strptime(s["timestamp"], "%Y-%m-%d %H:%M:%S") >= cutoff
    ]
    if not snaps:
        return {"empty": True, "reason": f"无可复盘快照（需要 >= {horizon_days}个交易日观察窗口）"}

    reviews = [review_snapshot(s) for s in snaps]

    # 聚合跨 snapshot 的总体统计
    merged = {
        "snapshots_reviewed": len(reviews),
        "first_ts": snaps[0]["timestamp"],
        "last_ts": snaps[-1]["timestamp"],
        "reviews": reviews,
    }
    return merged


# ──────────────────────────────────────────────────────────
# 渲染 Markdown
# ──────────────────────────────────────────────────────────

def render_review_markdown(review: Dict) -> str:
    if review.get("empty"):
        return f"# 游资选股复盘\n\n> {review.get('reason')}\n"

    lines = []
    lines.append(f"# 游资选股·复盘报告  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"- **复盘快照数**: {review['snapshots_reviewed']}")
    lines.append(f"- **覆盖区间**: {review['first_ts']} ~ {review['last_ts']}")
    lines.append("")

    # 每个 snapshot
    for idx, rv in enumerate(review["reviews"], 1):
        lines.append("---")
        lines.append(f"## 快照 #{idx} · {rv['snapshot_ts']} · regime={rv.get('regime')}")
        lines.append("")

        bs = rv["bought_summary"]
        ns = rv["not_bought_summary"]
        vs = rv["vetoed_summary"]

        lines.append("### 📊 分组对比")
        lines.append("")
        lines.append("| 分组 | 样本 | 胜 | 败 | 大胜(>5%) | 平均3日 | 平均5日 |")
        lines.append("|------|------|----|----|-----------|---------|---------|")
        for label, s in [("已买入", bs), ("未买入候选", ns), ("被Veto", vs)]:
            lines.append(f"| {label} | {s['picks']} | {s['wins']} | {s['losses']} "
                         f"| {s['big_wins']} | {s['avg_ret_3d']:+.2f}% | {s['avg_ret_5d']:+.2f}% |")

        diff = bs["avg_ret_3d"] - ns["avg_ret_3d"]
        if bs["picks"] > 0 and ns["picks"] > 0:
            if diff > 1.5:
                lines.append(f"\n> ✅ 选股正确：买入组 3日收益比未买组高 **{diff:+.2f}pct**")
            elif diff < -1.5:
                lines.append(f"\n> ❌ 选股失准：买入组 3日收益比未买组低 **{diff:+.2f}pct**（重新审视评分权重）")
            else:
                lines.append(f"\n> ⚠️ 选股表现相当（差 {diff:+.2f}pct），需要更多样本验证")
        lines.append("")

        # 风格命中率
        lines.append("### 🎯 各游资风格命中率（仅统计 verdict=buy 的票）")
        lines.append("")
        lines.append("| 风格 | buy数 | 胜率 | 大胜率 | 平均3日 |")
        lines.append("|------|-------|------|--------|---------|")
        for sname, st in rv["style_stats"].items():
            if st["picks"] == 0:
                continue
            wr = st["wins"] / st["picks"] * 100
            bwr = st["big_wins"] / st["picks"] * 100
            lines.append(f"| {sname} | {st['picks']} | {wr:.0f}% | {bwr:.0f}% | {st['avg_ret_3d']:+.2f}% |")
        lines.append("")

        # 明细 top 赢家 / top 输家
        winners = sorted(rv["records"], key=lambda r: r["ret_3d"], reverse=True)[:5]
        losers = sorted(rv["records"], key=lambda r: r["ret_3d"])[:5]

        lines.append("### 🏆 涨幅榜（未来3日）Top 5")
        lines.append("")
        lines.append("| # | 名称 | 代码 | 入价 | 3日 | 5日 | 最大涨 | 入选结论 | 买入风格 |")
        lines.append("|---|------|------|------|-----|-----|--------|----------|----------|")
        for i, r in enumerate(winners, 1):
            mark = "💰" if r.get("acted") else ("🚫" if r.get("youzi_verdict") == "veto" else "👀")
            lines.append(f"| {i} | {r['name']} {mark} | {r['code']} | {r['entry_price']:.2f} | "
                         f"{r['ret_3d']:+.2f}% | {r['ret_5d']:+.2f}% | {r['max_gain']:+.2f}% | "
                         f"{r.get('youzi_verdict')} | {','.join(r.get('youzi_buy_votes', [])) or '-'} |")
        lines.append("")

        lines.append("### 🩸 跌幅榜（未来3日）Top 5")
        lines.append("")
        lines.append("| # | 名称 | 代码 | 入价 | 3日 | 5日 | 最大跌 | 入选结论 | 买入风格 |")
        lines.append("|---|------|------|------|-----|-----|--------|----------|----------|")
        for i, r in enumerate(losers, 1):
            mark = "💰" if r.get("acted") else ("🚫" if r.get("youzi_verdict") == "veto" else "👀")
            lines.append(f"| {i} | {r['name']} {mark} | {r['code']} | {r['entry_price']:.2f} | "
                         f"{r['ret_3d']:+.2f}% | {r['ret_5d']:+.2f}% | {r['max_drawdown']:+.2f}% | "
                         f"{r.get('youzi_verdict')} | {','.join(r.get('youzi_buy_votes', [])) or '-'} |")
        lines.append("")

        # Veto 失误
        veto_gainers = [
            r for r in rv["records"]
            if r.get("youzi_verdict") == "veto" and r.get("ret_3d", 0) >= 5
        ]
        if veto_gainers:
            lines.append("### ⚠️ 被Veto但3日涨 >5% 的（反思规则过严）")
            lines.append("")
            lines.append("| 名称 | 代码 | 3日 | 被否决者 | 热点 |")
            lines.append("|------|------|-----|----------|------|")
            for r in sorted(veto_gainers, key=lambda x: x["ret_3d"], reverse=True)[:8]:
                lines.append(f"| {r['name']} | {r['code']} | {r['ret_3d']:+.2f}% | "
                             f"{','.join(r.get('youzi_vetoed_by', []))} | "
                             f"{','.join(r.get('hot_concepts', [])) or '-'} |")
            lines.append("")

    lines.append("---")
    lines.append("_生成器: youzi_review.review_recent / 数据源: data/youzi_watchlist.json_")
    return "\n".join(lines)


def save_review(review: Dict) -> str:
    md = render_review_markdown(review)
    path = REPORTS_DIR / f"youzi_review_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    path.write_text(md, encoding="utf-8")
    return str(path)


def run_and_dispatch(horizon_days: int = 3) -> Optional[str]:
    """完整入口：跑复盘 → 落盘 → 推送飞书/邮件 → 返回 md path。"""
    review = review_recent(days_back=15, horizon_days=horizon_days)
    if review.get("empty"):
        logger.info(f"[youzi_review] {review.get('reason')}")
        return None

    md_path = save_review(review)
    logger.info(f"[youzi_review] saved {md_path}")

    # 推送
    try:
        from src.notification import send_feishu_text
        head = f"【游资选股·复盘】\n覆盖 {review['snapshots_reviewed']} 个快照 "\
               f"({review['first_ts'][:10]} ~ {review['last_ts'][:10]})\n详见日志附件"
        send_feishu_text(head)
    except Exception as e:
        logger.debug(f"[youzi_review] feishu skipped: {e}")

    try:
        from src.notification import send_email
        body = Path(md_path).read_text(encoding="utf-8")
        send_email(subject=f"[游资复盘] {datetime.now().strftime('%Y-%m-%d')}", body=body)
    except Exception as e:
        logger.debug(f"[youzi_review] email skipped: {e}")

    return md_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="游资选股复盘")
    parser.add_argument("--horizon", type=int, default=3,
                        help="至少观察多少个交易日才复盘（默认 3）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = run_and_dispatch(horizon_days=args.horizon)
    print("review md:", path)
