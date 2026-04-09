"""
data_mining.py — 数据挖掘：找出日收益5%+的模式特征
用2025年全量数据分析：什么样的股票买入后第二天能暴涨
"""
import logging
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# 加载缓存数据
with open("data/backtest_cache_2025.pkl", "rb") as f:
    all_data = pickle.load(f)

logger.info(f"加载 {len(all_data)} 只股票")

# 收集每只股票每天的特征和次日收益
records = []

for code, df in all_data.items():
    if len(df) < 30:
        continue

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    dates = df["date"].values

    for i in range(25, len(df) - 1):
        # 次日收益（次日开盘买、次日收盘卖 OR 今日收盘买、次日收盘卖）
        ret_next_close = (close[i+1] / close[i] - 1) * 100  # 今收→明收
        ret_next_open2close = (close[i+1] / df["open"].values[i+1] - 1) * 100  # 明开→明收
        ret_next_open_gap = (df["open"].values[i+1] / close[i] - 1) * 100  # 隔夜跳空

        # === 今日特征 ===
        today_chg = (close[i] / close[i-1] - 1) * 100
        today_range = (high[i] - low[i]) / close[i-1] * 100

        # 近N日动量
        chg_3d = (close[i] / close[i-3] - 1) * 100
        chg_5d = (close[i] / close[i-5] - 1) * 100

        # 均线
        ma5 = np.mean(close[i-4:i+1])
        ma10 = np.mean(close[i-9:i+1])
        ma20 = np.mean(close[i-19:i+1])

        # 连阳天数
        consec_up = 0
        for j in range(i, max(i-10, 0), -1):
            if close[j] > close[j-1]:
                consec_up += 1
            else:
                break

        # 连涨停天数（涨幅>9.5%）
        consec_limit = 0
        for j in range(i, max(i-10, 0), -1):
            day_chg = (close[j] / close[j-1] - 1) * 100
            if day_chg > 9.5:
                consec_limit += 1
            else:
                break

        # 放量程度
        vol_ma5 = np.mean(volume[i-4:i+1])
        vol_ma20 = np.mean(volume[i-19:i+1])
        vol_ratio = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1
        vol_today_ratio = volume[i] / vol_ma20 if vol_ma20 > 0 else 1

        # 偏离度
        bias5 = (close[i] - ma5) / ma5 * 100
        bias20 = (close[i] - ma20) / ma20 * 100

        # RSI
        deltas = np.diff(close[i-14:i+1])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.001
        rsi = 100 - 100 / (1 + gain / loss)

        # 价格位置（20日高低点）
        h20 = np.max(high[i-19:i+1])
        l20 = np.min(low[i-19:i+1])
        price_pos = (close[i] - l20) / (h20 - l20) if h20 > l20 else 0.5

        # 突破新高
        is_10d_high = close[i] >= np.max(high[i-9:i]) * 0.98
        is_20d_high = close[i] >= np.max(high[i-19:i]) * 0.98

        # 换手率
        turnover = float(df.iloc[i].get("turnover_rate", 0))

        # 价格
        price = close[i]

        # 涨停接力（昨天涨停，今天继续）
        prev_chg = (close[i-1] / close[i-2] - 1) * 100 if i >= 2 else 0
        is_prev_limit = prev_chg > 9.5

        records.append({
            "date": dates[i],
            "code": code,
            "price": price,
            "today_chg": today_chg,
            "today_range": today_range,
            "chg_3d": chg_3d,
            "chg_5d": chg_5d,
            "consec_up": consec_up,
            "consec_limit": consec_limit,
            "vol_ratio": vol_ratio,
            "vol_today_ratio": vol_today_ratio,
            "bias5": bias5,
            "bias20": bias20,
            "rsi": rsi,
            "price_pos": price_pos,
            "is_10d_high": is_10d_high,
            "is_20d_high": is_20d_high,
            "turnover": turnover,
            "ma5_gt_ma10": ma5 > ma10,
            "ma10_gt_ma20": ma10 > ma20,
            "is_prev_limit": is_prev_limit,
            # 目标变量
            "ret_next_close": ret_next_close,
            "ret_next_gap": ret_next_open_gap,
            "ret_next_o2c": ret_next_open2close,
        })

df_all = pd.DataFrame(records)
logger.info(f"总样本: {len(df_all)}")

# ═══════════════════════════════════════════
# 分析什么特征组合→次日高收益
# ═══════════════════════════════════════════

print("\n" + "=" * 70)
print("  DATA MINING: What predicts next-day 5%+ returns?")
print("=" * 70)

# 1. 基础统计
print(f"\n[Overall] Avg next-day return: {df_all['ret_next_close'].mean():.3f}%")
print(f"  5%+ rate: {(df_all['ret_next_close'] > 5).mean()*100:.2f}%")
print(f"  3%+ rate: {(df_all['ret_next_close'] > 3).mean()*100:.2f}%")

# 2. 各特征的预测力
print(f"\n{'Feature':<25} {'Condition':<20} {'N':>8} {'Avg Ret%':>10} {'5%+ Rate':>10} {'3%+ Rate':>10}")
print("-" * 85)

tests = [
    ("today_chg", "涨停 (>9.5%)", df_all["today_chg"] > 9.5),
    ("today_chg", "强势 5-9.5%", (df_all["today_chg"] >= 5) & (df_all["today_chg"] < 9.5)),
    ("today_chg", "上涨 2-5%", (df_all["today_chg"] >= 2) & (df_all["today_chg"] < 5)),
    ("today_chg", "微涨 0-2%", (df_all["today_chg"] >= 0) & (df_all["today_chg"] < 2)),
    ("today_chg", "下跌 <0%", df_all["today_chg"] < 0),
    ("", "", pd.Series([False]*len(df_all))),  # separator
    ("consec_up", "连阳>=3天", df_all["consec_up"] >= 3),
    ("consec_up", "连阳>=5天", df_all["consec_up"] >= 5),
    ("consec_limit", "连板>=1天", df_all["consec_limit"] >= 1),
    ("consec_limit", "连板>=2天", df_all["consec_limit"] >= 2),
    ("is_prev_limit", "昨日涨停", df_all["is_prev_limit"] == True),
    ("", "", pd.Series([False]*len(df_all))),
    ("vol_today_ratio", "今日放量>2x", df_all["vol_today_ratio"] > 2),
    ("vol_today_ratio", "今日放量>3x", df_all["vol_today_ratio"] > 3),
    ("vol_ratio", "5日放量>1.5x", df_all["vol_ratio"] > 1.5),
    ("", "", pd.Series([False]*len(df_all))),
    ("chg_3d", "3日涨5-15%", (df_all["chg_3d"] >= 5) & (df_all["chg_3d"] < 15)),
    ("chg_3d", "3日涨15%+", df_all["chg_3d"] >= 15),
    ("chg_5d", "5日涨10-20%", (df_all["chg_5d"] >= 10) & (df_all["chg_5d"] < 20)),
    ("chg_5d", "5日涨20%+", df_all["chg_5d"] >= 20),
    ("", "", pd.Series([False]*len(df_all))),
    ("is_20d_high", "创20日新高", df_all["is_20d_high"] == True),
    ("price_pos", "20日高位>90%", df_all["price_pos"] > 0.9),
    ("rsi", "RSI>70", df_all["rsi"] > 70),
    ("rsi", "RSI>80", df_all["rsi"] > 80),
    ("", "", pd.Series([False]*len(df_all))),
    ("turnover", "换手>10%", df_all["turnover"] > 10),
    ("turnover", "换手>15%", df_all["turnover"] > 15),
    ("turnover", "换手>20%", df_all["turnover"] > 20),
    ("price", "价格<5元", df_all["price"] < 5),
    ("price", "价格5-10元", (df_all["price"] >= 5) & (df_all["price"] < 10)),
    ("price", "价格10-20元", (df_all["price"] >= 10) & (df_all["price"] < 20)),
    ("price", "价格20-50元", (df_all["price"] >= 20) & (df_all["price"] < 50)),
]

for feat, cond_name, mask in tests:
    if feat == "":
        print()
        continue
    n = mask.sum()
    if n < 10:
        continue
    sub = df_all[mask]
    avg_ret = sub["ret_next_close"].mean()
    rate_5 = (sub["ret_next_close"] > 5).mean() * 100
    rate_3 = (sub["ret_next_close"] > 3).mean() * 100
    print(f"  {feat:<23} {cond_name:<20} {n:>8} {avg_ret:>+10.3f} {rate_5:>9.1f}% {rate_3:>9.1f}%")

# 3. 组合特征（龙头战法核心信号）
print(f"\n{'='*70}")
print("  COMBO SIGNALS (Dragon Head Patterns)")
print(f"{'='*70}")

combos = [
    ("涨停+放量", (df_all["today_chg"] > 9.5) & (df_all["vol_today_ratio"] > 2)),
    ("涨停+首板(非连板)", (df_all["today_chg"] > 9.5) & (df_all["consec_limit"] == 1)),
    ("涨停+二连板", (df_all["today_chg"] > 9.5) & (df_all["consec_limit"] == 2)),
    ("涨停+高换手>10%", (df_all["today_chg"] > 9.5) & (df_all["turnover"] > 10)),
    ("涨停+高换手>15%", (df_all["today_chg"] > 9.5) & (df_all["turnover"] > 15)),
    ("强势5-9%+放量2x", (df_all["today_chg"].between(5, 9.5)) & (df_all["vol_today_ratio"] > 2)),
    ("强势5-9%+新高", (df_all["today_chg"].between(5, 9.5)) & (df_all["is_20d_high"])),
    ("强势5-9%+连阳3+", (df_all["today_chg"].between(5, 9.5)) & (df_all["consec_up"] >= 3)),
    ("3日涨10%+今日5%+放量", (df_all["chg_3d"] > 10) & (df_all["today_chg"] > 5) & (df_all["vol_today_ratio"] > 1.5)),
    ("连阳5+放量+新高", (df_all["consec_up"] >= 5) & (df_all["vol_ratio"] > 1.5) & (df_all["is_20d_high"])),
    ("昨涨停+今涨5%+", (df_all["is_prev_limit"]) & (df_all["today_chg"] > 5)),
    ("昨涨停+今涨0-5%", (df_all["is_prev_limit"]) & (df_all["today_chg"].between(0, 5))),
    ("昨涨停+高换手>8%", (df_all["is_prev_limit"]) & (df_all["turnover"] > 8)),
    ("今5%+放量3x+新高", (df_all["today_chg"] > 5) & (df_all["vol_today_ratio"] > 3) & (df_all["is_20d_high"])),
    ("多头+连阳3+放量+涨3-7%", (df_all["ma5_gt_ma10"]) & (df_all["ma10_gt_ma20"]) & (df_all["consec_up"] >= 3) & (df_all["vol_ratio"] > 1.3) & (df_all["today_chg"].between(3, 7))),
]

print(f"\n{'Combo':<35} {'N':>6} {'Avg%':>8} {'Gap%':>8} {'5%+':>8} {'3%+':>8} {'WR>0':>8}")
print("-" * 80)
for name, mask in combos:
    n = mask.sum()
    if n < 5:
        print(f"  {name:<33} {n:>6}  (too few)")
        continue
    sub = df_all[mask]
    avg = sub["ret_next_close"].mean()
    gap = sub["ret_next_gap"].mean()
    r5 = (sub["ret_next_close"] > 5).mean() * 100
    r3 = (sub["ret_next_close"] > 3).mean() * 100
    wr = (sub["ret_next_close"] > 0).mean() * 100
    print(f"  {name:<33} {n:>6} {avg:>+8.2f} {gap:>+8.2f} {r5:>7.1f}% {r3:>7.1f}% {wr:>7.1f}%")

# 4. 隔夜跳空分析
print(f"\n{'='*70}")
print("  OVERNIGHT GAP ANALYSIS (close→next open)")
print(f"{'='*70}")

gap_tests = [
    ("涨停股", df_all["today_chg"] > 9.5),
    ("涨停+首板", (df_all["today_chg"] > 9.5) & (df_all["consec_limit"] == 1)),
    ("涨停+连板2+", (df_all["today_chg"] > 9.5) & (df_all["consec_limit"] >= 2)),
    ("强势5-9.5%", df_all["today_chg"].between(5, 9.5)),
    ("上涨2-5%", df_all["today_chg"].between(2, 5)),
]

print(f"\n{'Pattern':<25} {'N':>6} {'Avg Gap%':>10} {'Gap>2%':>10} {'Gap>0%':>10}")
print("-" * 60)
for name, mask in gap_tests:
    n = mask.sum()
    if n < 5:
        continue
    sub = df_all[mask]
    avg_gap = sub["ret_next_gap"].mean()
    r2 = (sub["ret_next_gap"] > 2).mean() * 100
    r0 = (sub["ret_next_gap"] > 0).mean() * 100
    print(f"  {name:<23} {n:>6} {avg_gap:>+10.3f} {r2:>9.1f}% {r0:>9.1f}%")

print("\nDone.")
