# -*- coding: utf-8 -*-
"""把 backtest_youzi_2026 的 JSON 输出 → 每日详细报告。

用法:
    python tools/render_youzi_daily.py reports/youzi_backtest_2026ytd_YYYYMMDD_HHMM.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def render(json_path: str) -> str:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    snapshots = data["snapshots"]
    trades = data["trades"]

    # 按日期把 trades 归组
    trades_by_date = defaultdict(list)
    for t in trades:
        trades_by_date[t["date"]].append(t)

    # 重建持仓演化（从 trades 顺序回放）
    holdings: dict[str, dict] = {}  # code -> {name, shares, cost}
    history_holdings_by_date: dict[str, list[dict]] = {}

    # 先确定每日末持仓
    date_order = [s["date"] for s in snapshots]
    date_set = set(date_order)

    # 逐日回放
    trades_sorted = sorted(trades, key=lambda t: (t["date"], 0 if t["direction"] == "sell" else 1))
    # 但 sell 在 buy 前（同日先卖后买），同时按 trades 原始顺序即可（引擎日内先 sells 后 buys）
    trades_sorted = trades

    idx_ptr = 0
    for d in date_order:
        # 处理当日所有交易
        while idx_ptr < len(trades_sorted) and trades_sorted[idx_ptr]["date"] == d:
            t = trades_sorted[idx_ptr]
            c = t["code"]
            if t["direction"] == "buy":
                holdings[c] = {
                    "name": t["name"],
                    "shares": t["shares"],
                    "cost": t["price"],
                }
            else:  # sell
                h = holdings.get(c)
                if h:
                    h["shares"] -= t["shares"]
                    if h["shares"] <= 0:
                        del holdings[c]
            idx_ptr += 1
        history_holdings_by_date[d] = [
            {"code": c, **h} for c, h in holdings.items()
        ]

    out = []
    out.append("=" * 96)
    out.append("  游资风格回测 · 2026 YTD · 每日详细明细")
    out.append("=" * 96)
    out.append(
        f"  初始资金: ¥{200000:,.0f}   "
        f"最终资产: ¥{snapshots[-1]['total_asset']:,.0f}   "
        f"累计: {(snapshots[-1]['total_asset']-200000)/200000*100:+.2f}%"
    )
    out.append("")

    # 逐日明细
    for s in snapshots:
        d = s["date"]
        cum = (s["total_asset"] - 200000) / 200000 * 100
        header = (
            f"── {d}  {s['regime']:5s}  总资产 ¥{s['total_asset']:>10,.0f}  "
            f"日内 {s['daily_return_pct']:+6.2f}%  累计 {cum:+6.2f}%  持仓 {s['holdings_count']} ──"
        )
        out.append(header)

        tdays = trades_by_date.get(d, [])
        sells = [t for t in tdays if t["direction"] == "sell"]
        buys  = [t for t in tdays if t["direction"] == "buy"]

        if sells:
            for t in sells:
                # 从 reason 里抽 pnl
                pnl_str = ""
                r = t.get("reason", "") or ""
                # reason 形如 "TP_HALF+5.3%" / "SL-5.4%" / "EXPIRE2d"
                out.append(
                    f"    [卖] {t['code']} {t['name'][:6]:<6}  "
                    f"{t['shares']:>5}股 @¥{t['price']:>6.2f}  "
                    f"金额¥{t['amount']:>9,.0f}  原因={r}"
                )
        if buys:
            for t in buys:
                out.append(
                    f"    [买] {t['code']} {t['name'][:6]:<6}  "
                    f"{t['shares']:>5}股 @¥{t['price']:>6.2f}  "
                    f"金额¥{t['amount']:>9,.0f}  原因={t.get('reason','')}"
                )

        # 持仓收盘快照
        hold_snap = history_holdings_by_date.get(d, [])
        if hold_snap:
            items = []
            for h in hold_snap:
                items.append(f"{h['code']}×{h['shares']}@¥{h['cost']:.2f}")
            out.append(f"    持仓: {', '.join(items)}")
        elif s["holdings_count"] == 0:
            out.append(f"    持仓: (空仓)")

        out.append("")

    # 汇总交易对
    out.append("=" * 96)
    out.append("  完整交易对（买-卖配对）")
    out.append("=" * 96)
    out.append(f"  {'代码':<8} {'名称':<8} {'买日':<12} {'买价':>7} {'卖日':<12} {'卖价':>7} {'持天':>4} {'盈亏%':>7} {'原因':<20}")
    out.append("  " + "─" * 92)
    # 按 code 配对（FIFO）
    per_code: dict[str, list] = defaultdict(list)
    for t in trades:
        per_code[t["code"]].append(t)
    pairs = []
    for code, lst in per_code.items():
        buy_stack = []
        for t in lst:
            if t["direction"] == "buy":
                buy_stack.append(t)
            else:
                if not buy_stack: continue
                b = buy_stack[-1]
                # 部分或全部
                sell_shares = t["shares"]
                consumed = min(sell_shares, b["shares"])
                pnl = (t["price"] - b["price"]) / b["price"] * 100 if b["price"] else 0
                # 持天
                from datetime import datetime as _dt
                try:
                    days = (_dt.strptime(t["date"], "%Y-%m-%d") - _dt.strptime(b["date"], "%Y-%m-%d")).days
                except Exception:
                    days = 0
                pairs.append({
                    "code": code, "name": b["name"],
                    "buy_date": b["date"], "buy_price": b["price"],
                    "sell_date": t["date"], "sell_price": t["price"],
                    "shares": consumed, "pnl_pct": pnl,
                    "reason": t.get("reason", ""),
                })
                b["shares"] -= consumed
                if b["shares"] <= 0: buy_stack.pop()

    pairs.sort(key=lambda p: p["buy_date"])
    for p in pairs:
        out.append(
            f"  {p['code']:<8} {p['name'][:8]:<8} "
            f"{p['buy_date']:<12} {p['buy_price']:>7.2f} "
            f"{p['sell_date']:<12} {p['sell_price']:>7.2f} "
            f"{0:>4} {p['pnl_pct']:>+6.2f}% {p['reason'][:20]:<20}"
        )
    out.append("")

    # 胜负统计
    wins = [p for p in pairs if p["pnl_pct"] > 0]
    losses = [p for p in pairs if p["pnl_pct"] <= 0]
    out.append(f"  统计: 共 {len(pairs)} 笔  胜 {len(wins)}  负 {len(losses)}  "
               f"胜率 {len(wins)/max(1,len(pairs))*100:.1f}%")
    if wins:
        out.append(f"        平均盈利 {sum(p['pnl_pct'] for p in wins)/len(wins):+.2f}%")
    if losses:
        out.append(f"        平均亏损 {sum(p['pnl_pct'] for p in losses)/len(losses):+.2f}%")

    # 风格归因
    attr = data.get("style_attribution", {})
    out.append("")
    out.append("  ─── 风格归因 ───")
    for sn, v in attr.items():
        bc = v.get("buys", 0)
        wc = v.get("wins", 0)
        pnl = v.get("pnl_sum", 0)
        wr = wc / bc * 100 if bc else 0
        avg = pnl / bc if bc else 0
        out.append(f"    {sn:<15} buy={bc:<3} 胜={wc:<3} 胜率={wr:5.1f}%  平均={avg:+.2f}%")

    return "\n".join(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        from glob import glob
        files = sorted(glob("reports/youzi_backtest_2026ytd_*.json"))
        if not files:
            sys.exit("no json found")
        path = files[-1]
    txt = render(path)
    print(txt)
    # 同时写出一份 md
    out_md = Path(path).with_suffix(".daily.md")
    out_md.write_text(txt, encoding="utf-8")
    print(f"\n[已保存] {out_md}")
