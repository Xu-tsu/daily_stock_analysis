# -*- coding: utf-8 -*-
"""事后最优（oracle）龙头 vs 当前选股诊断。

核心逻辑：
    对 backtest 跑过的每一天 T：
      1. 用 calc_indicators(df, T_idx) 生成与实盘相同的候选特征（T 日收盘态）
      2. 偷看未来：用 T+1 开盘 → T+2~T+4 最高价计算"真实收益"
      3. 取未来收益 TopN 作为"事后龙头"
      4. 把 oracle 龙头的特征分布 vs 我们实际买入的特征分布对比
      5. 找出 signal_type、tech_score、今日涨幅、换手、chg_5d、均线形态上的偏差
      6. 基于偏差给出调参建议 & 做一次仿真：如果按 oracle 特征的中位数设门槛，
         选股命中率会变成多少

⚠ 偷看未来仅用于诊断/参数反推，不参与实盘或回测决策。
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, quantiles
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_adaptive_v2 import calc_indicators


HORIZON = 3  # 看未来 3 日
TOP_N = 5    # 每天取未来收益 top_n 作为 oracle 龙头


def _forward_return(df: pd.DataFrame, idx: int, horizon: int = HORIZON) -> Optional[float]:
    """买入 T+1 开盘，卖出 max(high) in T+1..T+horizon。"""
    if idx + 1 >= len(df):
        return None
    t1 = df.iloc[idx + 1]
    op = float(t1.get("open", 0))
    if op <= 0:
        return None
    end = min(idx + 1 + horizon, len(df))
    highs = df.iloc[idx + 1:end]["high"].astype(float)
    if highs.empty:
        return None
    peak = float(highs.max())
    return (peak - op) / op * 100


def _forward_open2close(df: pd.DataFrame, idx: int, horizon: int = HORIZON) -> Optional[float]:
    """T+1 开盘 → T+horizon 收盘（更贴近实际可达）。"""
    if idx + 1 >= len(df):
        return None
    t1 = df.iloc[idx + 1]
    op = float(t1.get("open", 0))
    if op <= 0:
        return None
    end_idx = min(idx + horizon, len(df) - 1)
    close_end = float(df.iloc[end_idx].get("close", 0))
    if close_end <= 0:
        return None
    return (close_end - op) / op * 100


def _derive_signal(ind: Dict) -> str:
    if ind["today_limit"] and ind.get("consec_limit", 0) >= 1:
        return "board_hit"
    if ind["yest_limit"] and ind["today_chg"] > 3:
        return "relay"
    if ind.get("vol_breakout"):
        return "breakout"
    if ind["chg_3d"] >= 10 and ind["today_chg"] > 2:
        return "momentum"
    return "trend"


def build_oracle_table(all_data: Dict[str, pd.DataFrame],
                       start: str, end: str,
                       top_n: int = TOP_N) -> List[dict]:
    """对每一天，扫全市场，按未来 3 日开→峰收益 Top-N 记录。"""
    # 构造 (date -> code -> df_idx) 的快速查询
    trading_dates = set()
    for df in all_data.values():
        trading_dates.update(df["date"].astype(str).str[:10].tolist())
    days = sorted(d for d in trading_dates if start <= d <= end)

    records = []
    for d in days:
        day_candidates = []
        for code, df in all_data.items():
            ser = df["date"].astype(str).str[:10] == d
            if not ser.any():
                continue
            idx = int(df.index[ser][0])
            if idx < 30:
                continue
            ind = calc_indicators(df, idx)
            if ind is None:
                continue
            price = float(ind["price"])
            if price < 3 or price > 50:
                continue
            if ind["turnover"] < 1:
                continue
            fwd_peak = _forward_return(df, idx, HORIZON)
            fwd_o2c  = _forward_open2close(df, idx, HORIZON)
            if fwd_peak is None:
                continue
            day_candidates.append({
                "date": d, "code": code, "name": str(df.iloc[idx].get("stock_name", code)),
                "price": price,
                "today_chg": ind["today_chg"],
                "turnover": ind["turnover"],
                "chg_3d": ind["chg_3d"],
                "chg_5d": ind["chg_5d"],
                "vol_ratio": ind["vol_ratio"],
                "rsi": ind["rsi"],
                "ma5_slope": ind["ma5_slope"],
                "ma10_slope": ind["ma10_slope"],
                "ma_bull": bool(ind["ma5_slope"] > 0 and ind["ma10_slope"] > 0),
                "signal": _derive_signal(ind),
                "today_limit": bool(ind["today_limit"]),
                "yest_limit": bool(ind["yest_limit"]),
                "consec_limit": int(ind.get("consec_limit", 0)),
                "vol_breakout": bool(ind.get("vol_breakout", False)),
                "fwd_peak_3d": fwd_peak,
                "fwd_o2c_3d": fwd_o2c or 0.0,
            })
        day_candidates.sort(key=lambda x: x["fwd_peak_3d"], reverse=True)
        for rank, c in enumerate(day_candidates[:top_n], 1):
            c["rank"] = rank
            records.append(c)
    return records


def feature_summary(picks: List[dict], label: str) -> Dict[str, float]:
    def s(key):
        vs = [p[key] for p in picks if p.get(key) is not None and not isinstance(p[key], bool)]
        return {
            "n": len(vs),
            "mean": float(np.mean(vs)) if vs else 0,
            "median": float(np.median(vs)) if vs else 0,
            "p25": float(np.percentile(vs, 25)) if vs else 0,
            "p75": float(np.percentile(vs, 75)) if vs else 0,
        }
    def frac(key):
        vs = [p[key] for p in picks if key in p]
        return sum(1 for v in vs if v) / max(1, len(vs))
    out = {"label": label, "n_samples": len(picks)}
    for k in ("price","today_chg","turnover","chg_3d","chg_5d","vol_ratio","rsi",
              "ma5_slope","ma10_slope","fwd_peak_3d","fwd_o2c_3d"):
        out[k] = s(k)
    out["pct_today_limit"]  = frac("today_limit")
    out["pct_yest_limit"]   = frac("yest_limit")
    out["pct_vol_breakout"] = frac("vol_breakout")
    out["pct_ma_bull"]      = frac("ma_bull")
    sig_counts = defaultdict(int)
    for p in picks:
        sig_counts[p.get("signal","?")] += 1
    out["signal_dist"] = dict(sig_counts)
    return out


def extract_actual_picks(trades_json_path: str,
                         all_data: Dict[str, pd.DataFrame]) -> List[dict]:
    """从回测 JSON 的 buy trades 还原实际买入那天的特征。"""
    data = json.loads(Path(trades_json_path).read_text(encoding="utf-8"))
    trades = data["trades"]
    picks = []
    for t in trades:
        if t["direction"] != "buy":
            continue
        code = t["code"]
        buy_date = t["date"]
        df = all_data.get(code)
        if df is None:
            continue
        # buy 在 T+1 开盘成交，特征是 T 日收盘态 → 找 T 日索引
        # buy_date 就是执行日（T+1），对应的决策日是前一个交易日
        mask = df["date"].astype(str).str[:10] == buy_date
        if not mask.any(): continue
        exec_idx = int(df.index[mask][0])
        dec_idx = exec_idx - 1
        if dec_idx < 30: continue
        ind = calc_indicators(df, dec_idx)
        if ind is None: continue
        fwd_peak = _forward_return(df, dec_idx, HORIZON)
        fwd_o2c  = _forward_open2close(df, dec_idx, HORIZON)
        picks.append({
            "date": str(df.iloc[dec_idx]["date"])[:10],
            "exec_date": buy_date,
            "code": code, "name": t.get("name", code),
            "price": float(ind["price"]),
            "today_chg": ind["today_chg"],
            "turnover": ind["turnover"],
            "chg_3d": ind["chg_3d"],
            "chg_5d": ind["chg_5d"],
            "vol_ratio": ind["vol_ratio"],
            "rsi": ind["rsi"],
            "ma5_slope": ind["ma5_slope"],
            "ma10_slope": ind["ma10_slope"],
            "ma_bull": bool(ind["ma5_slope"] > 0 and ind["ma10_slope"] > 0),
            "signal": _derive_signal(ind),
            "today_limit": bool(ind["today_limit"]),
            "yest_limit": bool(ind["yest_limit"]),
            "consec_limit": int(ind.get("consec_limit", 0)),
            "vol_breakout": bool(ind.get("vol_breakout", False)),
            "fwd_peak_3d": fwd_peak or 0,
            "fwd_o2c_3d": fwd_o2c or 0,
        })
    return picks


def render_comparison(oracle: List[dict], actual: List[dict]) -> str:
    o = feature_summary(oracle, "事后龙头")
    a = feature_summary(actual, "实际买入")
    lines = []
    lines.append("="*96)
    lines.append("  Oracle 诊断 · 事后龙头 vs 实际买入 特征分布对比")
    lines.append("="*96)
    lines.append(f"  Oracle 样本: {o['n_samples']} 笔 (每天 Top{TOP_N})")
    lines.append(f"  实际样本:   {a['n_samples']} 笔")
    lines.append("")

    def fmt_row(key, unit=""):
        ov = o[key]; av = a[key]
        gap = av["median"] - ov["median"]
        return (f"  {key:<15} | 龙头 中位 {ov['median']:>7.2f}{unit} [P25 {ov['p25']:>6.2f} - P75 {ov['p75']:>6.2f}] "
                f"| 实际 中位 {av['median']:>7.2f}{unit} [P25 {av['p25']:>6.2f} - P75 {av['p75']:>6.2f}] "
                f"| Δ {gap:>+6.2f}")
    lines.append("  ── 核心数值特征（中位数 & 四分位）──")
    for k, u in [("today_chg","%"), ("turnover","%"), ("chg_3d","%"),
                 ("chg_5d","%"), ("vol_ratio","x"), ("rsi",""),
                 ("price","Y"), ("ma5_slope","%"), ("ma10_slope","%"),
                 ("fwd_peak_3d","%"), ("fwd_o2c_3d","%")]:
        lines.append(fmt_row(k, u))
    lines.append("")
    lines.append("  ── 布尔特征占比 ──")
    for k in ("pct_today_limit","pct_yest_limit","pct_vol_breakout","pct_ma_bull"):
        lines.append(f"  {k:<18}   龙头 {o[k]*100:>5.1f}%   实际 {a[k]*100:>5.1f}%   Δ {(a[k]-o[k])*100:>+6.1f}pp")
    lines.append("")
    lines.append("  ── Signal 分布 ──")
    all_sigs = sorted(set(list(o["signal_dist"].keys()) + list(a["signal_dist"].keys())))
    lines.append(f"  {'signal':<14} {'龙头 n / 占比':<18} {'实际 n / 占比':<18}")
    for s in all_sigs:
        on = o["signal_dist"].get(s, 0); an = a["signal_dist"].get(s, 0)
        op = on / max(1, o["n_samples"]) * 100
        ap = an / max(1, a["n_samples"]) * 100
        lines.append(f"  {s:<14} {on:>4} / {op:>5.1f}%        {an:>4} / {ap:>5.1f}%")
    lines.append("")

    # ── 调参建议 ──
    lines.append("="*96)
    lines.append("  调参建议（基于 oracle 中位数 → 实际中位数的偏差）")
    lines.append("="*96)

    suggestions = []
    today_chg_gap = a["today_chg"]["median"] - o["today_chg"]["median"]
    if today_chg_gap > 1.5:
        suggestions.append(
            f"⚠ 实际买入 today_chg 中位 +{a['today_chg']['median']:.1f}% 高于龙头 "
            f"+{o['today_chg']['median']:.1f}% → 在追高。"
            f"\n   建议：yozui_styles 里对 today_chg > {o['today_chg']['p75']:.1f}% 扣分或加大 veto 范围。"
        )

    chg5_gap = a["chg_5d"]["median"] - o["chg_5d"]["median"]
    if chg5_gap > 2:
        suggestions.append(
            f"⚠ 实际买入 chg_5d 中位 +{a['chg_5d']['median']:.1f}% 高于龙头 "
            f"+{o['chg_5d']['median']:.1f}% → 追了已涨多的票。"
            f"\n   建议：将 _base_tech_score 的 chg_5d 惩罚从 >50 调到 >{o['chg_5d']['p75']:.0f}，更严格。"
        )

    turn_gap = a["turnover"]["median"] - o["turnover"]["median"]
    if abs(turn_gap) > 1.5:
        direction = "低" if turn_gap < 0 else "高"
        suggestions.append(
            f"⚠ 实际买入 turnover 中位 {a['turnover']['median']:.1f}% 比龙头"
            f"{direction} {abs(turn_gap):.1f}pp → 换手筛选区间不匹配。"
            f"\n   建议：把甜蜜区定为 [{o['turnover']['p25']:.1f}%, {o['turnover']['p75']:.1f}%]。"
        )

    ma_gap = a["pct_ma_bull"] - o["pct_ma_bull"]
    if ma_gap > 0.05:
        suggestions.append(
            f"⚠ 实际买入 MA 多头占比 {a['pct_ma_bull']*100:.0f}% 高于龙头 "
            f"{o['pct_ma_bull']*100:.0f}% → 过分依赖多头排列错过反转龙头。"
            f"\n   建议：减弱 daily_short_arrangement 的 VETO 强度，允许空头排列反包。"
        )
    elif ma_gap < -0.05:
        suggestions.append(
            f"⚠ 实际买入 MA 多头占比 {a['pct_ma_bull']*100:.0f}% 低于龙头 "
            f"{o['pct_ma_bull']*100:.0f}% → 买了太多空头排列股。"
            f"\n   建议：加强 daily_short_arrangement 的 VETO。"
        )

    lim_gap = a["pct_today_limit"] - o["pct_today_limit"]
    if lim_gap > 0.1:
        suggestions.append(
            f"⚠ 实际买入涨停占比 {a['pct_today_limit']*100:.0f}% 远高于龙头 "
            f"{o['pct_today_limit']*100:.0f}% → 把板追得太多。"
            f"\n   建议：限制每天最多买 1 只涨停票，其余用反包 / 突破。"
        )
    elif lim_gap < -0.1:
        suggestions.append(
            f"⚠ 实际买入涨停占比 {a['pct_today_limit']*100:.0f}% 低于龙头 "
            f"{o['pct_today_limit']*100:.0f}% → 错过涨停龙头。"
            f"\n   建议：放宽涨停打板过滤。"
        )

    vb_gap = a["pct_vol_breakout"] - o["pct_vol_breakout"]
    if vb_gap < -0.05:
        suggestions.append(
            f"⚠ 实际买入放量突破占比 {a['pct_vol_breakout']*100:.0f}% 低于龙头 "
            f"{o['pct_vol_breakout']*100:.0f}% → 放量突破信号没被充分利用。"
            f"\n   建议：zhang_jiahu 对 vol_breakout 加大权重（+20~+30）。"
        )

    if not suggestions:
        suggestions.append("（各项特征差距 < 阈值，未发现明显系统偏差）")
    for s in suggestions:
        lines.append(s)

    lines.append("")
    # 龙头前 5 天样例
    lines.append("="*96)
    lines.append(f"  事后龙头样例（每日 Top1 · 前 10 天）")
    lines.append("="*96)
    lines.append(f"  {'日期':<12} {'代码':<8} {'名称':<10} {'today%':>7} {'换手%':>7} {'chg3d%':>7} {'chg5d%':>7} {'vol_r':>6} {'signal':<10} {'未来3d峰%':>9}")
    cur_date = None
    shown = 0
    for r in oracle:
        if r.get("rank") != 1: continue
        if shown >= 15: break
        lines.append(
            f"  {r['date']:<12} {r['code']:<8} {r['name'][:6]:<10} "
            f"{r['today_chg']:>+7.2f} {r['turnover']:>7.2f} {r['chg_3d']:>+7.2f} "
            f"{r['chg_5d']:>+7.2f} {r['vol_ratio']:>6.2f} {r['signal']:<10} "
            f"{r['fwd_peak_3d']:>+9.2f}"
        )
        shown += 1
    lines.append("")

    # 实际低收益样本（最差 5 笔）
    lines.append("="*96)
    lines.append(f"  实际买入最差 10 笔（按未来 3d 峰值计，如果达到峰值时卖）")
    lines.append("="*96)
    lines.append(f"  {'决策日':<12} {'买入日':<12} {'代码':<8} {'today%':>7} {'换手%':>7} {'chg3d%':>7} {'chg5d%':>7} {'signal':<10} {'未来3d峰%':>9}")
    sorted_actual = sorted(actual, key=lambda x: x["fwd_peak_3d"])[:10]
    for r in sorted_actual:
        lines.append(
            f"  {r['date']:<12} {r['exec_date']:<12} {r['code']:<8} "
            f"{r['today_chg']:>+7.2f} {r['turnover']:>7.2f} {r['chg_3d']:>+7.2f} "
            f"{r['chg_5d']:>+7.2f} {r['signal']:<10} {r['fwd_peak_3d']:>+9.2f}"
        )

    return "\n".join(lines)


def main(json_path: str, start: str = "2026-01-05", end: str = "2026-04-09"):
    cache = Path("data/backtest_cache_2026ytd.pkl")
    all_data = pickle.loads(cache.read_bytes())
    print(f"[*] Loaded {len(all_data)} stocks")

    print(f"[*] Building oracle top-{TOP_N}/day ...")
    oracle = build_oracle_table(all_data, start, end, top_n=TOP_N)
    print(f"    → {len(oracle)} oracle records")

    print(f"[*] Extracting actual picks from {json_path} ...")
    actual = extract_actual_picks(json_path, all_data)
    print(f"    → {len(actual)} actual picks")

    report = render_comparison(oracle, actual)
    print(report)

    out_md = Path(json_path).with_suffix(".oracle.md")
    out_md.write_text(report, encoding="utf-8")
    print(f"\n[saved] {out_md}")

    # 落盘 oracle 原始表，便于后续调参回测
    out_json = Path(json_path).with_suffix(".oracle.json")
    out_json.write_text(
        json.dumps({"oracle": oracle, "actual": actual}, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    jp = sys.argv[1] if len(sys.argv) > 1 else "reports/youzi_backtest_2026ytd_20260417_0227.json"
    main(jp)
