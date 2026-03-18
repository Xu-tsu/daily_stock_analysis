# -*- coding: utf-8 -*-
"""
A股全市场扫描器 - 多策略筛选
策略一：多头趋势（MA5>MA10，股价在MA20上方）
策略二：跌停低吸（近期2-3个跌停后企稳反弹）
策略三：超跌反弹（RSI超卖+缩量企稳）

用法：
    python scanner.py                    # 默认全策略扫描
    python scanner.py --mode trend       # 只扫趋势股
    python scanner.py --mode dip         # 只扫跌停低吸
    python scanner.py --mode oversold    # 只扫超跌反弹
    python scanner.py --top 10           # 只输出前10只
    python scanner.py --update-env       # 自动更新.env的STOCK_LIST
"""

import argparse
import os
import numpy as np
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd


def get_all_stocks():
    """用腾讯接口获取全市场行情"""
    import requests
    import time

    print("正在通过腾讯接口获取全市场行情...")

    # 生成全部A股代码（沪市60xxxx + 深市00xxxx/30xxxx）
    all_codes = []
    for i in range(0, 5000):
        all_codes.append(f"sz{i:06d}")
    for i in range(600000, 606000):
        all_codes.append(f"sh{i:06d}")
    for i in range(300000, 303000):
        all_codes.append(f"sz{i:06d}")
    for i in range(301000, 302000):
        all_codes.append(f"sz{i:06d}")
    for i in range(688000, 689000):
        all_codes.append(f"sh{i:06d}")

    rows = []
    batch_size = 50
    total_batches = (len(all_codes) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(all_codes), batch_size):
        batch = all_codes[batch_idx:batch_idx + batch_size]
        symbols = ",".join([f"s_{c}" for c in batch])

        if (batch_idx // batch_size) % 50 == 0:
            print(f"  进度: {batch_idx // batch_size}/{total_batches}")

        try:
            url = f"http://qt.gtimg.cn/q={','.join(batch)}"
            resp = requests.get(url, timeout=10)
            lines = resp.text.strip().split("\n")

            for line in lines:
                if "~" not in line or 'v_pv_none' in line:
                    continue
                try:
                    parts = line.split("~")
                    if len(parts) < 46:
                        continue
                    name = parts[1]
                    code = parts[2]
                    price = float(parts[3]) if parts[3] else 0
                    if price <= 0:
                        continue
                    change_pct = float(parts[32]) if parts[32] else 0
                    volume = float(parts[6]) if parts[6] else 0
                    turnover = float(parts[38]) if parts[38] else 0
                    cap_str = parts[44] if len(parts) > 44 else "0"
                    cap = float(cap_str) * 10000 if cap_str else 0  # 万元转元

                    rows.append({
                        "代码": code,
                        "名称": name,
                        "最新价": price,
                        "涨跌幅": change_pct,
                        "成交量": volume,
                        "换手率": turnover,
                        "流通市值": cap,
                    })
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            continue

        time.sleep(0.1)

    if rows:
        df = pd.DataFrame(rows)
        # 去掉价格为0的
        df = df[df["最新价"] > 0]
        print(f"腾讯接口获取到 {len(df)} 只股票")
        return df

    print("腾讯接口获取失败")
    return None


def get_stock_history(code, days=60):
    """获取单只股票历史K线"""
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        return df
    except:
        return None


def calculate_indicators(df):
    """计算技术指标：均线、RSI、成交量均线"""
    if df is None or len(df) < 20:
        return None
    df = df.copy()
    df["MA5"] = df["收盘"].rolling(5).mean()
    df["MA10"] = df["收盘"].rolling(10).mean()
    df["MA20"] = df["收盘"].rolling(20).mean()

    # RSI6
    delta = df["收盘"].diff()
    gain = delta.where(delta > 0, 0).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI6"] = 100 - (100 / (1 + rs))

    # 成交量均线
    df["VOL_MA5"] = df["成交量"].rolling(5).mean()
    df["VOL_MA20"] = df["成交量"].rolling(20).mean()

    return df


def basic_filter(df, min_turnover=1.0, max_cap=300):
    """基础过滤：排除ST、停牌、换手率过低、市值过大"""
    # 排除ST和退市
    df = df[~df["名称"].str.contains("ST|退", na=False)]
    # 排除停牌
    df = df[df["成交量"] > 0]
    # 代码长度
    df = df[df["代码"].str.len() == 6]

    # 换手率
    if "换手率" in df.columns:
        df["换手率"] = pd.to_numeric(df["换手率"], errors="coerce")
        df = df[df["换手率"] >= min_turnover]

    # 流通市值（亿）
    if "流通市值" in df.columns:
        df["流通市值"] = pd.to_numeric(df["流通市值"], errors="coerce")
        df = df[df["流通市值"] <= max_cap * 1e8]

    # 涨跌幅范围
    if "涨跌幅" in df.columns:
        df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
        df = df[df["涨跌幅"] > -11]
        df = df[df["涨跌幅"] < 11]

    return df


# ============================================================
# 策略一：趋势股（放宽版）
# ============================================================
def scan_trend(df, max_bias=8.0, top_n=15):
    """扫描趋势股：股价在MA20上方，MA5>=MA10"""
    print("\n📈 扫描趋势股...")
    # 按涨幅+换手率排序取前300只
    df["综合分"] = df["涨跌幅"].rank(pct=True) * 0.6 + df["换手率"].rank(pct=True) * 0.4
    subset = df.nlargest(300, "综合分")

    candidates = []
    total = len(subset)
    for idx, (_, row) in enumerate(subset.iterrows()):
        code, name, price, change_pct, turnover, cap = _extract_row(row)
        if idx % 30 == 0:
            print(f"  趋势扫描进度: {idx}/{total}")

        hist = get_stock_history(code, days=40)
        hist = calculate_indicators(hist)
        if hist is None:
            continue

        last = hist.iloc[-1]
        ma5, ma10, ma20 = last.get("MA5"), last.get("MA10"), last.get("MA20")
        if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20):
            continue

        # 放宽条件：股价在MA20上方，MA5>=MA10
        if not (price > ma20 and ma5 >= ma10):
            continue

        bias = (price - ma5) / ma5 * 100 if ma5 > 0 else 999
        if abs(bias) > max_bias:
            continue

        spread = (ma5 - ma20) / ma20 * 100 if ma20 > 0 else 0

        candidates.append({
            "代码": code, "名称": name, "现价": round(price, 2),
            "涨跌幅": round(change_pct, 2), "换手率": round(turnover, 2),
            "流通市值(亿)": round(cap, 2),
            "MA5": round(ma5, 2), "MA10": round(ma10, 2), "MA20": round(ma20, 2),
            "乖离率": round(bias, 2), "均线发散度": round(spread, 2),
            "策略": "趋势",
        })

    candidates.sort(key=lambda x: x["均线发散度"], reverse=True)
    return candidates[:top_n]


# ============================================================
# 策略二：跌停低吸（你最喜欢的）
# ============================================================
def scan_limit_down_dip(df, top_n=15):
    """
    扫描跌停低吸：近10个交易日内有2-3个跌停（跌幅<=-9.5%），
    且最近1-2天企稳（跌幅收窄或翻红），出现低吸机会。
    """
    print("\n📉 扫描跌停低吸...")
    # 先选近期跌幅较大的（近5日累计跌幅大的优先看）
    subset = df.copy()
    # 选今日没有跌停的（还能买进）
    subset = subset[subset["涨跌幅"] > -9.5]

    candidates = []
    total = min(len(subset), 500)
    # 按换手率排序（跌停后放量说明有资金接盘）
    subset = subset.nlargest(500, "换手率") if len(subset) > 500 else subset

    for idx, (_, row) in enumerate(subset.iterrows()):
        code, name, price, change_pct, turnover, cap = _extract_row(row)
        if idx % 50 == 0:
            print(f"  跌停低吸进度: {idx}/{total}")

        hist = get_stock_history(code, days=30)
        hist = calculate_indicators(hist)
        if hist is None or len(hist) < 10:
            continue

        recent = hist.tail(10)

        # 统计近10天跌停次数（跌幅<=-9.5%视为跌停）
        limit_down_count = (recent["涨跌幅"] <= -9.5).sum()

        # 核心条件：近10天有2-3个跌停
        if limit_down_count < 2 or limit_down_count > 4:
            continue

        # 最近一天不能是跌停（否则买不进）
        last_change = recent.iloc[-1]["涨跌幅"]
        if last_change <= -9.5:
            continue

        # 企稳信号：最近1-2天跌幅收窄或翻红
        last_2 = recent.tail(2)["涨跌幅"]
        is_stabilizing = (last_2.iloc[-1] > last_2.iloc[0]) or (last_2.iloc[-1] > -3)

        if not is_stabilizing:
            continue

        # RSI超卖加分
        rsi = recent.iloc[-1].get("RSI6", 50)
        rsi_score = max(0, 30 - rsi) if not pd.isna(rsi) else 0  # RSI越低分越高

        # 缩量企稳加分（今日量小于5日均量说明抛压减小）
        vol = recent.iloc[-1]["成交量"]
        vol_ma5 = recent.iloc[-1].get("VOL_MA5", vol)
        vol_shrink = vol < vol_ma5 * 0.8 if not pd.isna(vol_ma5) else False

        # 计算从最低点反弹幅度
        recent_low = recent["最低"].min()
        bounce_pct = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0

        # 综合评分
        score = (
            limit_down_count * 15 +      # 跌停次数越多超跌越深
            rsi_score * 2 +               # RSI越低越好
            (10 if vol_shrink else 0) +   # 缩量企稳加分
            (10 if is_stabilizing else 0) + # 企稳信号加分
            min(bounce_pct * 2, 20)       # 反弹幅度（不超过20分）
        )

        last_row = hist.iloc[-1]
        ma5 = round(last_row.get("MA5", 0), 2) if not pd.isna(last_row.get("MA5", 0)) else 0
        ma10 = round(last_row.get("MA10", 0), 2) if not pd.isna(last_row.get("MA10", 0)) else 0
        ma20 = round(last_row.get("MA20", 0), 2) if not pd.isna(last_row.get("MA20", 0)) else 0

        candidates.append({
            "代码": code, "名称": name, "现价": round(price, 2),
            "涨跌幅": round(change_pct, 2), "换手率": round(turnover, 2),
            "流通市值(亿)": round(cap, 2),
            "MA5": ma5, "MA10": ma10, "MA20": ma20,
            "跌停次数": int(limit_down_count),
            "RSI6": round(rsi, 1) if not pd.isna(rsi) else 0,
            "反弹幅度": round(bounce_pct, 2),
            "缩量企稳": "是" if vol_shrink else "否",
            "综合评分": round(score, 1),
            "乖离率": round((price - ma5) / ma5 * 100, 2) if ma5 > 0 else 0,
            "均线发散度": 0,
            "策略": "跌停低吸",
        })

    candidates.sort(key=lambda x: x["综合评分"], reverse=True)
    return candidates[:top_n]


# ============================================================
# 策略三：超跌反弹
# ============================================================
def scan_oversold_bounce(df, top_n=15):
    """扫描超跌反弹：近期大跌后RSI超卖+缩量，等待反弹"""
    print("\n💎 扫描超跌反弹...")
    # 选今日跌幅不太大的（还有买入机会）
    subset = df[df["涨跌幅"] > -7].copy()
    subset = subset.nlargest(400, "换手率") if len(subset) > 400 else subset

    candidates = []
    total = len(subset)

    for idx, (_, row) in enumerate(subset.iterrows()):
        code, name, price, change_pct, turnover, cap = _extract_row(row)
        if idx % 50 == 0:
            print(f"  超跌反弹进度: {idx}/{total}")

        hist = get_stock_history(code, days=40)
        hist = calculate_indicators(hist)
        if hist is None or len(hist) < 20:
            continue

        last = hist.iloc[-1]
        rsi6 = last.get("RSI6", 50)
        if pd.isna(rsi6) or rsi6 > 25:  # RSI6<25才算超卖
            continue

        # 近20日最高到现在的跌幅
        high_20 = hist.tail(20)["最高"].max()
        drop_pct = (price - high_20) / high_20 * 100 if high_20 > 0 else 0

        if drop_pct > -15:  # 至少跌15%才算超跌
            continue

        # 缩量
        vol = last["成交量"]
        vol_ma20 = last.get("VOL_MA20", vol)
        vol_shrink = vol < vol_ma20 * 0.6 if not pd.isna(vol_ma20) else False

        ma5 = round(last.get("MA5", 0), 2) if not pd.isna(last.get("MA5", 0)) else 0
        ma10 = round(last.get("MA10", 0), 2) if not pd.isna(last.get("MA10", 0)) else 0
        ma20 = round(last.get("MA20", 0), 2) if not pd.isna(last.get("MA20", 0)) else 0

        score = abs(drop_pct) + (30 - rsi6) + (15 if vol_shrink else 0)

        candidates.append({
            "代码": code, "名称": name, "现价": round(price, 2),
            "涨跌幅": round(change_pct, 2), "换手率": round(turnover, 2),
            "流通市值(亿)": round(cap, 2),
            "MA5": ma5, "MA10": ma10, "MA20": ma20,
            "RSI6": round(rsi6, 1),
            "20日跌幅": round(drop_pct, 2),
            "缩量": "是" if vol_shrink else "否",
            "综合评分": round(score, 1),
            "乖离率": round((price - ma5) / ma5 * 100, 2) if ma5 > 0 else 0,
            "均线发散度": 0,
            "策略": "超跌反弹",
        })

    candidates.sort(key=lambda x: x["综合评分"], reverse=True)
    return candidates[:top_n]


# ============================================================
# 辅助函数
# ============================================================
def _extract_row(row):
    """从DataFrame行提取标准字段"""
    code = row["代码"]
    name = row["名称"]
    price = float(row["最新价"]) if pd.notna(row["最新价"]) else 0
    change_pct = float(row["涨跌幅"]) if pd.notna(row["涨跌幅"]) else 0
    turnover = float(row["换手率"]) if pd.notna(row.get("换手率", 0)) else 0
    cap = float(row.get("流通市值", 0)) / 1e8 if pd.notna(row.get("流通市值", 0)) else 0
    return code, name, price, change_pct, turnover, cap


def scan_market(max_cap=300, min_turnover=1.0, max_bias=8.0, top_n=30, mode="all"):
    """
    全市场多策略扫描入口
    mode: all / trend / dip / oversold
    """
    df = get_all_stocks()
    if df is None:
        return []

    print("正在进行基础过滤...")
    df = basic_filter(df, min_turnover=min_turnover, max_cap=max_cap)
    print(f"基础过滤后剩余 {len(df)} 只")

    all_candidates = []

    if mode in ("all", "dip"):
        dip_results = scan_limit_down_dip(df, top_n=top_n)
        all_candidates.extend(dip_results)
        print(f"  跌停低吸: 找到 {len(dip_results)} 只")

    if mode in ("all", "oversold"):
        oversold_results = scan_oversold_bounce(df, top_n=top_n)
        all_candidates.extend(oversold_results)
        print(f"  超跌反弹: 找到 {len(oversold_results)} 只")

    if mode in ("all", "trend"):
        trend_results = scan_trend(df, max_bias=max_bias, top_n=top_n)
        all_candidates.extend(trend_results)
        print(f"  趋势股: 找到 {len(trend_results)} 只")

    # 去重（同一股票可能命中多个策略）
    seen = set()
    unique = []
    for c in all_candidates:
        if c["代码"] not in seen:
            seen.add(c["代码"])
            unique.append(c)

    return unique[:top_n]


def print_results(candidates):
    """打印扫描结果"""
    if not candidates:
        print("\n未找到符合条件的股票")
        return

    print(f"\n{'=' * 90}")
    print(f"全市场扫描结果 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"共找到 {len(candidates)} 只符合条件的股票")
    print(f"{'=' * 90}")
    print(f"{'排名':>4} {'策略':<8} {'代码':<8} {'名称':<10} {'现价':>8} {'涨跌%':>7} {'换手%':>7} {'乖离%':>7}")
    print("-" * 90)

    for i, s in enumerate(candidates, 1):
        strategy = s.get("策略", "")
        extra = ""
        if strategy == "跌停低吸":
            extra = f" | 跌停{s.get('跌停次数', 0)}次 RSI{s.get('RSI6', 0)} 反弹{s.get('反弹幅度', 0)}%"
        elif strategy == "超跌反弹":
            extra = f" | RSI{s.get('RSI6', 0)} 20日跌{s.get('20日跌幅', 0)}%"
        print(f"{i:>4} {strategy:<8} {s['代码']:<8} {s['名称']:<10} {s['现价']:>8.2f} {s['涨跌幅']:>6.2f}% {s['换手率']:>6.2f}% {s['乖离率']:>6.2f}%{extra}")

    codes = ",".join([s["代码"] for s in candidates])
    print(f"\n股票代码列表：")
    print(codes)


def update_env(candidates, env_path=".env"):
    """将扫描结果更新到.env文件"""
    if not candidates:
        print("无候选股票，跳过更新")
        return
    codes = ",".join([s["代码"] for s in candidates])
    if not os.path.exists(env_path):
        print(f"{env_path} 不存在")
        return
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()
    import re
    new_content = re.sub(r"STOCK_LIST=.*", f"STOCK_LIST={codes}", content)
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"\n已更新 {env_path} 中的 STOCK_LIST = {codes}")


def main():
    parser = argparse.ArgumentParser(description="A股全市场多策略扫描器")
    parser.add_argument("--top", type=int, default=20, help="输出前N只（默认20）")
    parser.add_argument("--max-cap", type=float, default=300, help="最大流通市值（亿，默认300）")
    parser.add_argument("--min-turnover", type=float, default=1.0, help="最小换手率（%%，默认1）")
    parser.add_argument("--max-bias", type=float, default=8.0, help="最大乖离率（%%，默认8）")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "trend", "dip", "oversold"],
                        help="扫描模式：all/trend/dip/oversold")
    parser.add_argument("--update-env", action="store_true", help="自动更新.env的STOCK_LIST")
    parser.add_argument("--append-env", action="store_true", help="追加到现有STOCK_LIST（不覆盖）")
    args = parser.parse_args()

    print(f"扫描模式: {args.mode} | 市值<{args.max_cap}亿 | 换手>{args.min_turnover}% | 乖离<{args.max_bias}% | TOP{args.top}")

    candidates = scan_market(
        max_cap=args.max_cap,
        min_turnover=args.min_turnover,
        max_bias=args.max_bias,
        top_n=args.top,
        mode=args.mode,
    )

    print_results(candidates)

    if args.update_env:
        update_env(candidates)
    elif args.append_env and candidates:
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("STOCK_LIST="):
                        existing = line.strip().split("=", 1)[1].split(",")
                        new_codes = [s["代码"] for s in candidates]
                        merged = list(dict.fromkeys(existing + new_codes))
                        candidates_merged = [{"代码": c} for c in merged]
                        update_env(candidates_merged, env_path)
                        break


if __name__ == "__main__":
    main()