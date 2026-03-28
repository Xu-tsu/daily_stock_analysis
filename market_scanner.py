"""
market_scanner.py — A股全市场扫描 + K线技术分析
海外可用版（腾讯行情 + Tushare股票列表）

流程:
  1. Tushare 获取全量A股列表（~5000只，免费接口）
  2. 腾讯行情批量获取实时数据（100只/批，~50批）
  3. 第一轮筛选：价格/换手率/涨跌幅
  4. 对候选股获取K线并做技术分析（MA/MACD/RSI/量价）
  5. 第二轮筛选：技术形态打分排名
  6. 输出最终候选列表

使用:
  python market_scanner.py                       # 默认扫描
  python market_scanner.py --max-price 10        # 最高价格
  python market_scanner.py --top 30              # 输出前30只
  python market_scanner.py --mode trend          # 趋势模式
"""
import argparse, json, logging, math, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 腾讯行情工具（复用 macro_data_collector 的）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _code_to_tc(code: str) -> str:
    if code.startswith("6") or code.startswith("9"):
        return f"sh{code}"
    return f"sz{code}"


def _fetch_one_batch(batch_codes: list, batch_idx: int) -> dict:
    """获取单批腾讯行情（供线程池调用）"""
    import re
    results = {}
    tc_codes = [_code_to_tc(c) for c in batch_codes]
    code_str = ",".join(tc_codes)
    try:
        r = requests.get(
            f"https://qt.gtimg.cn/q={code_str}",
            timeout=15, headers=HEADERS,
        )
        r.encoding = "gbk"
        for line in r.text.strip().split("\n"):
            m = re.match(r'v_(\w+)="(.+)"', line.strip().rstrip(";"))
            if not m:
                continue
            fields = m.group(2).split("~")
            if len(fields) < 45:
                continue
            raw_code = fields[2]
            results[raw_code] = {
                "name": fields[1],
                "code": raw_code,
                "price": float(fields[3]) if fields[3] else 0,
                "prev_close": float(fields[4]) if fields[4] else 0,
                "open": float(fields[5]) if fields[5] else 0,
                "volume": int(fields[6]) if fields[6] else 0,
                "change_pct": float(fields[32]) if fields[32] else 0,
                "high": float(fields[33]) if fields[33] else 0,
                "low": float(fields[34]) if fields[34] else 0,
                "amount": float(fields[37]) if fields[37] else 0,
                "turnover_rate": float(fields[38]) if fields[38] else 0,
                "market_cap": float(fields[44]) if fields[44] else 0,
                "amplitude": float(fields[43]) if fields[43] else 0,
            }
    except Exception as e:
        logger.warning(f"腾讯批量行情第 {batch_idx+1} 批失败: {e}")
    return results


def _batch_tencent_quotes(codes: list, batch_size: int = 80, max_workers: int = 10) -> dict:
    """批量获取腾讯行情，多线程并发（每批最多80只）"""
    batches = []
    for i in range(0, len(codes), batch_size):
        batches.append(codes[i:i + batch_size])

    total = len(batches)
    logger.info(f"  共 {total} 批请求，{max_workers} 线程并发...")
    all_results = {}
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one_batch, batch, idx): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            results = future.result()
            all_results.update(results)
            done_count += 1
            if done_count % 50 == 0 or done_count == total:
                logger.info(f"  行情进度: {done_count}/{total} 批完成，已获取 {len(all_results)} 只")

    return all_results


def _fetch_kline(code: str, days: int = 120) -> pd.DataFrame:
    """获取单只股票腾讯日K线"""
    tc = _code_to_tc(code)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tc},day,,,{days},qfq"
    try:
        r = requests.get(url, timeout=15, headers=HEADERS)
        data = r.json()
        sd = data.get("data", {})
        if isinstance(sd, dict):
            sd = sd.get(tc, {})
        if not isinstance(sd, dict):
            return pd.DataFrame()
        kline = sd.get("qfqday", sd.get("day", []))
        if not kline:
            return pd.DataFrame()
        cols = ["date", "open", "close", "high", "low", "volume"]
        df = pd.DataFrame(kline, columns=cols[:len(kline[0])])
        for c in ["open", "close", "high", "low", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except:
        return pd.DataFrame()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 获取全量A股列表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_all_stock_codes() -> list:
    """获取全量A股代码列表（与原项目scanner相同的腾讯接口方式）"""
    # 直接生成代码范围，腾讯行情会自动过滤不存在的
    all_codes = []
    # 深市 000000-004999
    for i in range(0, 5000):
        all_codes.append(f"{i:06d}")
    # 沪市主板 600000-605999
    for i in range(600000, 606000):
        all_codes.append(f"{i:06d}")
    # 深市创业板 300000-302999（已包含301000-301999）
    for i in range(300000, 303000):
        all_codes.append(f"{i:06d}")
    # 沪市科创板 688000-688999
    for i in range(688000, 689000):
        all_codes.append(f"{i:06d}")

    logger.info(f"[代码生成] 共 {len(all_codes)} 个候选代码（与原项目一致）")
    return all_codes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# K线技术分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_kline(df: pd.DataFrame) -> dict:
    """
    对K线做完整技术分析
    返回：MA排列、MACD、RSI、量价关系、得分
    """
    if df is None or len(df) < 20:
        return {"score": 0, "error": "数据不足"}

    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series([0] * len(df))
    high = df["high"]
    low = df["low"]
    latest = close.iloc[-1]

    result = {}

    # === MA 均线 ===
    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20

    if ma5 > ma10 > ma20:
        result["ma_trend"] = "多头排列"
        ma_score = 30
    elif ma5 > ma10:
        result["ma_trend"] = "短期多头"
        ma_score = 20
    elif ma5 < ma10 < ma20:
        result["ma_trend"] = "空头排列"
        ma_score = -10  # 负分！用户数据：空头排列胜率仅17.5%
    else:
        result["ma_trend"] = "震荡"
        ma_score = 10

    # === 乖离率 BIAS ===
    bias5 = (latest - ma5) / ma5 * 100 if ma5 else 0
    bias20 = (latest - ma20) / ma20 * 100 if ma20 else 0
    result["bias5"] = round(bias5, 2)
    result["bias20"] = round(bias20, 2)
    # 低乖离率（回踩到位）加分
    bias_score = 10 if -3 < bias5 < 2 else 5 if bias5 < 5 else 0

    # === RSI ===
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50
    result["rsi"] = round(rsi_val, 1)
    # RSI 30-60 加分（非超买）
    rsi_score = 15 if 30 < rsi_val < 60 else 10 if rsi_val < 30 else 0

    # === MACD ===
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    macd_bar = 2 * (dif - dea)
    result["macd_dif"] = round(dif.iloc[-1], 3)
    result["macd_dea"] = round(dea.iloc[-1], 3)
    result["macd_bar"] = round(macd_bar.iloc[-1], 3)
    # 金叉或即将金叉加分
    if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2]:
        result["macd_signal"] = "金叉"
        macd_score = 20
    elif dif.iloc[-1] > dea.iloc[-1]:
        result["macd_signal"] = "多头"
        macd_score = 15
    elif macd_bar.iloc[-1] > macd_bar.iloc[-2]:
        result["macd_signal"] = "底部收敛"
        macd_score = 10
    else:
        result["macd_signal"] = "空头"
        macd_score = 0

    # === 量价关系 ===
    vol_ma5 = volume.rolling(5).mean().iloc[-1]
    vol_ma20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else vol_ma5
    vol_ratio = round(vol_ma5 / vol_ma20, 2) if vol_ma20 > 0 else 1
    result["vol_ratio"] = vol_ratio
    # 缩量回踩（量比0.6-0.9）或温和放量（1.0-1.5）加分
    if 0.6 <= vol_ratio <= 0.9:
        result["vol_pattern"] = "缩量回踩"
        vol_score = 15
    elif 1.0 <= vol_ratio <= 1.5:
        result["vol_pattern"] = "温和放量"
        vol_score = 10
    elif vol_ratio > 2.0:
        result["vol_pattern"] = "异常放量"
        vol_score = 5
    else:
        result["vol_pattern"] = "缩量"
        vol_score = 5

    # === 支撑压力 ===
    recent_20 = close.tail(20)
    result["support"] = round(recent_20.min(), 2)
    result["resistance"] = round(recent_20.max(), 2)

    # === 近N日涨跌 ===
    if len(close) >= 5:
        result["chg_5d"] = round((latest / close.iloc[-6] - 1) * 100, 2) if close.iloc[-6] > 0 else 0
    if len(close) >= 10:
        result["chg_10d"] = round((latest / close.iloc[-11] - 1) * 100, 2) if close.iloc[-11] > 0 else 0

    # === T+1 安全分（回调买入加分，追高扣分）===
    # 价格贴近MA5支撑=止损距离短=T+1风险低
    t1_score = 0
    if abs(bias5) < 1.5:
        t1_score = 10  # 贴近MA5，止损距离最短
    elif bias5 < -3:
        t1_score = 5   # 远低于MA5，超跌但风险大
    elif bias5 > 5:
        t1_score = -10  # 远高于MA5，追高危险
    result["t1_safety"] = t1_score

    # === 总分 ===
    result["score"] = ma_score + bias_score + rsi_score + macd_score + vol_score + t1_score
    result["max_score"] = 100

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全市场扫描主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _load_winning_patterns() -> dict:
    """加载交易盈利模式（如果有足够数据的话）"""
    try:
        from trade_journal import analyze_winning_patterns
        patterns = analyze_winning_patterns(days=90)
        if "optimal_strategy" in patterns:
            return patterns["optimal_strategy"]
    except Exception:
        pass
    return {}


def _load_fund_flow_stocks() -> dict:
    """加载当日主力资金流入的股票集合，返回 {code: main_net}"""
    try:
        from ths_scraper import fetch_stock_fund_flow_rank
        stocks = fetch_stock_fund_flow_rank(top_n=50)
        return {s["code"]: s.get("main_net", 0) for s in stocks if s.get("main_net", 0) > 0}
    except Exception:
        return {}


def _load_mainline_sectors() -> set:
    """加载当前主线板块名称集合"""
    try:
        from data_store import get_sector_mainline
        sectors = get_sector_mainline(min_days=2)
        return {s["sector_name"] for s in sectors[:10]}
    except Exception:
        return set()


def scan_market(
    max_price: float = 10.0,
    min_turnover: float = 2.0,
    max_market_cap: float = 100.0,
    min_change: float = -3.0,
    max_change: float = 9.5,
    top_n: int = 30,
    mode: str = "trend",
) -> list:
    """
    全市场扫描

    流程:
      1. 获取全量股票列表
      2. 批量获取实时行情
      3. 第一轮筛选（价格/换手率/市值/涨跌幅）
      4. 对候选股做K线技术分析
      5. 按技术得分排序

    mode:
      trend  — 趋势股（MA多头+缩量回踩）
      breakout — 突破股（放量突破近期高点）
      oversold — 超跌反弹（RSI低+MACD底背离）
      sub_dragon — 副龙头（短期题材+低位+资金流入+即将拉升）
    """
    start_time = time.time()

    # 预加载辅助数据（用于打分增强）
    winning_patterns = _load_winning_patterns()
    fund_flow_stocks = {}
    mainline_sectors = set()
    if mode == "sub_dragon":
        logger.info("[预加载] 获取资金流和主线板块数据...")
        fund_flow_stocks = _load_fund_flow_stocks()
        mainline_sectors = _load_mainline_sectors()

    # Step 1: 获取股票列表
    logger.info("[扫描 1/4] 获取A股列表...")
    all_codes = get_all_stock_codes()

    # Step 2: 批量获取实时行情
    logger.info(f"[扫描 2/4] 批量获取 {len(all_codes)} 只股票实时行情...")
    quotes = _batch_tencent_quotes(all_codes)
    valid_quotes = {k: v for k, v in quotes.items() if v.get("price", 0) > 0}
    logger.info(f"  有效行情: {len(valid_quotes)} 只")

    # Step 3: 第一轮筛选
    logger.info("[扫描 3/4] 第一轮筛选...")
    filtered = []
    for code, q in valid_quotes.items():
        price = q.get("price", 0)
        turnover = q.get("turnover_rate", 0)
        mcap = q.get("market_cap", 0)
        chg = q.get("change_pct", 0)
        name = q.get("name", "")

        # 排除ST、退市、新股
        if "ST" in name or "退" in name or "N" == name[0:1]:
            continue

        # 基础筛选
        if price > max_price or price <= 0:
            continue
        if turnover < min_turnover:
            continue
        if mcap > max_market_cap and mcap > 0:
            continue
        if chg < min_change or chg > max_change:
            continue

        filtered.append({"code": code, "quote": q})

    logger.info(f"  第一轮通过: {len(filtered)} 只")

    # Step 4: K线技术分析（对筛选后的候选股）
    logger.info(f"[扫描 4/4] K线技术分析 ({len(filtered)} 只)...")
    results = []
    for i, item in enumerate(filtered):
        code = item["code"]
        q = item["quote"]

        if i > 0 and i % 50 == 0:
            logger.info(f"  已分析 {i}/{len(filtered)}...")
            time.sleep(0.5)  # 控制频率

        df = _fetch_kline(code, days=120)
        if df.empty:
            continue

        ta = analyze_kline(df)
        if ta.get("score", 0) == 0:
            continue

        # ============ 按模式加分 ============
        bonus = 0
        chg_today = q.get("change_pct", 0)

        # ── T+1 追高惩罚（所有模式通用，最高优先级）──
        # A股今天买明天才能卖，追高=承受隔夜风险无法止损
        if chg_today >= 7:
            bonus -= 30   # 涨停板附近，T+1极度危险
        elif chg_today >= 5:
            bonus -= 20   # 涨幅过大，禁止追高
        elif chg_today >= 3:
            bonus -= 10   # 追高警告
        # 回调买入奖励：当日微跌但趋势向上 = 最佳T+1买点
        if -2 < chg_today < 0.5 and ta.get("ma_trend") in ("多头排列", "短期多头"):
            bonus += 15   # 强势股回调，明日大概率反弹

        if mode == "trend":
            if ta.get("ma_trend") == "多头排列":
                bonus += 10
            if ta.get("vol_pattern") == "缩量回踩":
                bonus += 15  # 缩量回踩是T+1最安全的买点（提升权重）
            elif ta.get("vol_pattern") == "温和放量":
                bonus += 5
            if -2 < ta.get("bias5", 99) < 1:
                bonus += 10  # 贴近均线=止损距离短

        elif mode == "breakout":
            # 突破模式降权：T+1下追突破风险极高
            if ta.get("vol_ratio", 0) > 1.5:
                bonus += 5   # 降低（原来10分）
            if chg_today > 3:
                bonus -= 5   # 反而扣分（原来+10）：当日大涨的突破不追

        elif mode == "oversold":
            if ta.get("rsi", 50) < 30:
                bonus += 15
            if ta.get("macd_signal") == "底部收敛":
                bonus += 10
            # 超跌反弹也不能追高买：等确认后再入
            if chg_today > 5:
                bonus -= 10  # 超跌反弹首日涨太多不追

        elif mode == "sub_dragon":
            # 副龙头策略：低位 + 题材催化 + 资金流入 + 技术底部
            if code in fund_flow_stocks:
                net = fund_flow_stocks[code]
                bonus += 15 if net > 5000 else 10 if net > 1000 else 5
            chg_10d = ta.get("chg_10d", 0)
            if chg_10d < -5 and 0 < chg_today < 3:
                bonus += 10  # 超跌+温和反弹（非暴涨）
            elif chg_10d < -3 and chg_today > 0:
                bonus += 5
            if ta.get("macd_signal") in ("金叉", "底部收敛"):
                bonus += 10
            if q.get("turnover_rate", 0) > 5:
                bonus += 5
            if ta.get("rsi", 50) < 45:
                bonus += 5

        # ── 空头排列惩罚（用户数据：空头排列胜率仅17.5%）──
        if ta.get("ma_trend") == "空头排列":
            bonus -= 15

        # 盈利模式加分（基于历史交易数据学习）
        if winning_patterns:
            best_ma = winning_patterns.get("best_ma_trend", "")
            best_macd = winning_patterns.get("best_macd_signal", "")
            if best_ma and ta.get("ma_trend") == best_ma:
                bonus += 5
            if best_macd and ta.get("macd_signal") == best_macd:
                bonus += 5

        # 向量知识库加分：检索历史相似场景的胜率
        try:
            from knowledge_base import query_similar_trades
            query = f"MA趋势:{ta.get('ma_trend','')} MACD:{ta.get('macd_signal','')}"
            similar = query_similar_trades(query, n_results=5)
            if similar:
                wins = sum(1 for s in similar if s["metadata"].get("result") == "盈利")
                if wins >= 3:
                    bonus += 5  # 历史相似场景多数盈利
                elif wins <= 1:
                    bonus -= 5  # 历史相似场景多数亏损
        except Exception:
            pass

        final_score = ta.get("score", 0) + bonus

        entry = {
            "code": code,
            "name": q.get("name", ""),
            "price": q.get("price", 0),
            "change_pct": q.get("change_pct", 0),
            "turnover_rate": q.get("turnover_rate", 0),
            "market_cap": q.get("market_cap", 0),
            "amount": q.get("amount", 0),
            "ma_trend": ta.get("ma_trend", ""),
            "macd_signal": ta.get("macd_signal", ""),
            "rsi": ta.get("rsi", 0),
            "bias5": ta.get("bias5", 0),
            "vol_pattern": ta.get("vol_pattern", ""),
            "vol_ratio": ta.get("vol_ratio", 0),
            "support": ta.get("support", 0),
            "resistance": ta.get("resistance", 0),
            "chg_5d": ta.get("chg_5d", 0),
            "chg_10d": ta.get("chg_10d", 0),
            "tech_score": final_score,
        }
        # 副龙头模式额外信息
        if mode == "sub_dragon" and code in fund_flow_stocks:
            entry["main_net"] = fund_flow_stocks[code]
        results.append(entry)

        time.sleep(0.1)  # 控制K线请求频率

    # 排序
    results.sort(key=lambda x: x["tech_score"], reverse=True)
    final = results[:top_n]

    elapsed = round(time.time() - start_time, 1)
    logger.info(f"\n扫描完成！耗时 {elapsed} 秒")
    logger.info(f"  总股票: {len(all_codes)}")
    logger.info(f"  有效行情: {len(valid_quotes)}")
    logger.info(f"  第一轮通过: {len(filtered)}")
    logger.info(f"  技术分析: {len(results)} 只有效")
    logger.info(f"  最终候选: {len(final)} 只")

    return final


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI 入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股全市场扫描器")
    parser.add_argument("--max-price", type=float, default=10.0, help="最高价格")
    parser.add_argument("--min-turnover", type=float, default=2.0, help="最低换手率%")
    parser.add_argument("--max-cap", type=float, default=100.0, help="最大市值(亿)")
    parser.add_argument("--top", type=int, default=20, help="输出前N只")
    parser.add_argument("--mode", default="trend", choices=["trend", "breakout", "oversold"])
    parser.add_argument("--output", type=str, default=None, help="输出JSON文件路径")
    args = parser.parse_args()

    from src.config import get_config
    from src.logging_config import setup_logging
    config = get_config()
    setup_logging(log_prefix="scanner", debug=False, log_dir=config.log_dir)

    results = scan_market(
        max_price=args.max_price,
        min_turnover=args.min_turnover,
        max_market_cap=args.max_cap,
        top_n=args.top,
        mode=args.mode,
    )

    print(f"\n{'='*80}")
    print(f"  A股全市场扫描结果 ({args.mode}模式) — Top {len(results)}")
    print(f"{'='*80}")
    print(f"{'代码':>8} {'名称':<8} {'价格':>6} {'涨跌%':>6} {'换手%':>6} {'MA趋势':<8} {'MACD':<8} {'RSI':>5} {'量价':<8} {'得分':>4}")
    print("-" * 80)
    for s in results:
        print(
            f"{s['code']:>8} {s['name']:<8} {s['price']:>6.2f} "
            f"{s['change_pct']:>6.2f} {s['turnover_rate']:>6.2f} "
            f"{s['ma_trend']:<8} {s['macd_signal']:<8} "
            f"{s['rsi']:>5.1f} {s['vol_pattern']:<8} {s['tech_score']:>4}"
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 {args.output}")