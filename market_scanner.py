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
        _ALL_KCOLS = ["date", "open", "close", "high", "low", "volume", "turnover"]
        ncols = len(kline[0])
        cols = _ALL_KCOLS[:ncols] if ncols <= len(_ALL_KCOLS) else [f"c{i}" for i in range(ncols)]
        df = pd.DataFrame(kline, columns=cols)
        for c in ["open", "close", "high", "low", "volume", "turnover"]:
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
# 市场环境检测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_market_regime() -> dict:
    """
    检测当前市场环境：bull / sideways / bear

    依据:
      1. 上证指数 MA5/MA20/MA60 排列
      2. 近5日/20日涨跌幅
      3. 市场宽度（涨跌家数比，通过批量行情估算）

    返回: {"regime": "bull"|"sideways"|"bear", "score": int, "detail": str}
    """
    regime_score = 0   # >30 牛市, -30~30 震荡, <-30 熊市
    details = []

    try:
        # 获取上证指数K线
        df = _fetch_kline("000001", days=80)  # 上证指数在腾讯用sh000001
        if df is None or len(df) < 60:
            # fallback: 用沪深300
            df = _fetch_kline("399300", days=80)

        if df is not None and len(df) >= 60:
            close = df["close"].values

            # MA排列
            ma5 = np.mean(close[-5:])
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:])

            if ma5 > ma20 > ma60:
                regime_score += 30
                details.append("MA多头排列(+30)")
            elif ma5 > ma20:
                regime_score += 15
                details.append("短期多头(+15)")
            elif ma5 < ma20 < ma60:
                regime_score -= 30
                details.append("MA空头排列(-30)")
            elif ma5 < ma20:
                regime_score -= 15
                details.append("短期空头(-15)")
            else:
                details.append("MA震荡(0)")

            # 近5日涨跌幅
            chg_5d = (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0
            if chg_5d > 3:
                regime_score += 20
                details.append(f"5日涨{chg_5d:.1f}%(+20)")
            elif chg_5d > 1:
                regime_score += 10
                details.append(f"5日涨{chg_5d:.1f}%(+10)")
            elif chg_5d < -3:
                regime_score -= 20
                details.append(f"5日跌{chg_5d:.1f}%(-20)")
            elif chg_5d < -1:
                regime_score -= 10
                details.append(f"5日跌{chg_5d:.1f}%(-10)")

            # 近20日涨跌幅
            chg_20d = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0
            if chg_20d > 8:
                regime_score += 15
                details.append(f"20日涨{chg_20d:.1f}%(+15)")
            elif chg_20d > 3:
                regime_score += 5
                details.append(f"20日涨{chg_20d:.1f}%(+5)")
            elif chg_20d < -8:
                regime_score -= 15
                details.append(f"20日跌{chg_20d:.1f}%(-15)")
            elif chg_20d < -3:
                regime_score -= 5
                details.append(f"20日跌{chg_20d:.1f}%(-5)")

            # 指数是否在60日线上方
            if close[-1] > ma60:
                regime_score += 10
                details.append("站上60日线(+10)")
            else:
                regime_score -= 10
                details.append("跌破60日线(-10)")

    except Exception as e:
        logger.warning(f"[市场环境] 指数分析失败: {e}")
        details.append("指数分析失败")

    # 判定
    if regime_score >= 30:
        regime = "bull"
    elif regime_score <= -30:
        regime = "bear"
    else:
        regime = "sideways"

    detail_str = ", ".join(details)
    logger.info(f"[市场环境] {regime.upper()} (score={regime_score}) | {detail_str}")

    return {
        "regime": regime,
        "score": regime_score,
        "detail": detail_str,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# K线技术分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_kline(df: pd.DataFrame, mode: str = "dragon") -> dict:
    """技术评分（支持多模式）

    mode:
      dragon   — 龙头打板（涨停/连板/接力）
      trend    — 趋势选股（MA/RSI/MACD/量价综合评分）
      oversold — 超跌反弹
      breakout — 放量突破
    """
    if df is None or len(df) < 20:
        return {"score": 0, "error": "数据不足"}

    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series([0] * len(df))
    high = df["high"]
    low = df["low"]
    latest = close.iloc[-1]
    close_arr = close.values
    high_arr = high.values
    vol_arr = volume.values

    result = {}

    # ── 基础指标 ──
    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]

    today_chg = (latest / close_arr[-2] - 1) * 100 if len(close_arr) >= 2 else 0
    result["today_chg"] = round(today_chg, 2)

    prev_chg = (close_arr[-2] / close_arr[-3] - 1) * 100 if len(close_arr) >= 3 else 0
    is_prev_limit = prev_chg > 9.5

    # 连板天数
    consec_limit = 0
    for j in range(-1, max(-8, -len(close_arr)), -1):
        day_chg = (close_arr[j] / close_arr[j-1] - 1) * 100 if j-1 >= -len(close_arr) else 0
        if day_chg > 9.5:
            consec_limit += 1
        else:
            break

    # 连阳天数
    consec_up = 0
    for j in range(-1, max(-8, -len(close_arr)), -1):
        if close_arr[j] > close_arr[j-1]:
            consec_up += 1
        else:
            break

    # 量比
    vol_ma20 = np.mean(vol_arr[-20:]) if len(vol_arr) >= 20 else np.mean(vol_arr[-5:])
    vol_today_ratio = vol_arr[-1] / vol_ma20 if vol_ma20 > 0 else 1
    result["vol_ratio"] = round(vol_today_ratio, 2)

    # 3日/5日涨幅
    chg_3d = (latest / close_arr[-4] - 1) * 100 if len(close_arr) >= 4 else 0
    chg_5d = (latest / close_arr[-6] - 1) * 100 if len(close_arr) >= 6 else 0
    result["chg_3d"] = round(chg_3d, 2)

    # 20日新高
    h20 = np.max(high_arr[-20:]) if len(high_arr) >= 20 else np.max(high_arr[-10:])
    is_20d_high = latest >= h20 * 0.98

    # 换手率（如果有）
    turnover = 0
    if "turnover_rate" in df.columns:
        turnover = float(df["turnover_rate"].iloc[-1])
        if pd.isna(turnover):
            turnover = 0

    score = 0
    signal_type = "none"

    # 昨日跌幅（用于地天板判断）
    prev_chg_down = prev_chg  # 已经算过了
    # 前天涨幅
    prev2_chg = (close_arr[-3] / close_arr[-4] - 1) * 100 if len(close_arr) >= 4 else 0

    # ═══ 策略S: 地天板 / 天地天板 ═══
    # 昨日跌停（或大跌≥8%）+ 今日涨停 → 地天板，超强反转信号
    if today_chg > 9.5 and prev_chg_down < -8:
        score = 50
        if prev2_chg > 9.5:
            # 天地天板：前天涨停→昨天跌停→今天涨停，30%振幅
            score += 40
            signal_type = "sky_floor_sky"
        else:
            signal_type = "floor_to_sky"
        if vol_today_ratio > 1.5:
            score += 10    # 放量反包更强
        if turnover > 8:
            score += 5     # 换手充分，筹码交换

    # 昨日涨停 + 今日大跌（量化砸盘）→ 标记为潜在地天板机会
    elif prev_chg > 9.5 and today_chg < -5:
        # 不评分（当前在坑里），但标记供持仓判断用
        result["floor_trap"] = True   # 可能的量化砸盘坑
        result["potential_reversal"] = True

    # ═══ 策略A: 涨停打板 ═══
    elif today_chg > 9.5:
        score = 30
        if consec_limit >= 2:
            score += 40     # 二连板最强
        elif consec_limit == 1:
            score += 25     # 首板
        if 5 <= turnover <= 15:
            score += 15
        elif turnover > 15:
            score -= 10
        if vol_today_ratio < 2:
            score += 10     # 缩量涨停
        signal_type = "board_hit"

    # ═══ 策略B: 涨停接力 ═══
    elif is_prev_limit and today_chg > 3:
        score = 25
        if today_chg >= 5:
            score += 35
        elif today_chg >= 3:
            score += 20
        if turnover > 8:
            score += 10
        signal_type = "relay"

    # ═══ 策略C: 爆量突破 ═══
    elif today_chg >= 5 and vol_today_ratio > 2 and is_20d_high:
        score = 20
        if chg_3d > 10:
            score += 25
        elif chg_3d > 5:
            score += 15
        if consec_up >= 3:
            score += 15
        if ma5 > ma10 > ma20:
            score += 10
        signal_type = "breakout"

    # ═══ 策略D: 强势动量 ═══
    elif chg_3d >= 15 and today_chg > 2:
        score = 20
        if chg_5d > 20:
            score += 25
        elif chg_5d > 10:
            score += 15
        if consec_up >= 4:
            score += 15
        if is_20d_high:
            score += 10
        signal_type = "momentum"

    # ═══ 非龙头模式：趋势/超跌/突破 综合评分 ═══
    if mode != "dragon" and score == 0:
        signal_type = "trend"

        # RSI
        rsi_tmp = 50
        if len(close_arr) >= 15:
            _deltas = np.diff(close_arr[-15:])
            _gain = np.mean(_deltas[_deltas > 0]) if np.any(_deltas > 0) else 0
            _loss = -np.mean(_deltas[_deltas < 0]) if np.any(_deltas < 0) else 0.001
            rsi_tmp = 100 - 100 / (1 + _gain / _loss)

        # Bias5
        _bias5 = (latest - ma5) / ma5 * 100 if ma5 > 0 else 0

        # MA趋势评分
        if ma5 > ma10 > ma20:
            score += 15      # 多头排列
        elif ma5 > ma10:
            score += 10      # 短期多头
        elif ma5 < ma10 < ma20:
            score += 5       # 空头（均值回归机会）
        else:
            score += 8       # 震荡

        # RSI评分
        if rsi_tmp < 30:
            score += 20      # 超卖
        elif rsi_tmp < 40:
            score += 15
        elif rsi_tmp < 50:
            score += 10
        elif rsi_tmp < 60:
            score += 5
        elif rsi_tmp > 70:
            score -= 10      # 超买

        # Bias5评分（偏离均线）
        if _bias5 < -5:
            score += 15      # 深度偏离
        elif _bias5 < -3:
            score += 12
        elif _bias5 < -1:
            score += 8
        elif _bias5 < 2:
            score += 5
        elif _bias5 > 5:
            score -= 10      # 过度偏离

        # 量比评分
        if 0.8 <= vol_today_ratio <= 1.5:
            score += 10      # 温和放量
        elif vol_today_ratio > 1.5:
            score += 5       # 放量
        elif vol_today_ratio > 2.5:
            score -= 5       # 过度放量

        # MACD
        _ema12 = close.ewm(span=12).mean()
        _ema26 = close.ewm(span=26).mean()
        _dif = _ema12 - _ema26
        _dea = _dif.ewm(span=9).mean()
        if _dif.iloc[-1] > _dea.iloc[-1] and _dif.iloc[-2] <= _dea.iloc[-2]:
            score += 10      # 金叉
        elif _dif.iloc[-1] > _dea.iloc[-1]:
            score += 5       # 多头

        # 20日价格位置
        _h20 = np.max(high_arr[-20:]) if len(high_arr) >= 20 else np.max(high_arr)
        _l20 = np.min(low.values[-20:]) if len(low) >= 20 else np.min(low.values)
        _range20 = _h20 - _l20
        _pos = (latest - _l20) / _range20 if _range20 > 0 else 0.5
        if _pos < 0.3:
            score += 10      # 低位
        elif _pos < 0.5:
            score += 5
        elif _pos > 0.9:
            score -= 10      # 高位

        # ── 龙头预埋加分（提前埋伏有连板预期的票）──
        # 条件：近期有涨停历史 + 当前回调到位 + 量能未散
        has_recent_limit = False
        for j in range(-2, max(-8, -len(close_arr)), -1):
            _d_chg = (close_arr[j] / close_arr[j-1] - 1) * 100 if j-1 >= -len(close_arr) else 0
            if _d_chg > 9.5:
                has_recent_limit = True
                break

        if has_recent_limit:
            # 近7日内有过涨停，现在回调 → 龙头预埋
            if today_chg < 3 and _pos < 0.7:
                score += 15
                signal_type = "pre_dragon"  # 龙头预埋
            elif today_chg < 5:
                score += 8
                signal_type = "pre_dragon"

        # 连阳蓄势（3天以上连阳但每天涨幅不大 → 可能要爆发）
        if consec_up >= 3 and today_chg < 5 and chg_3d < 10:
            score += 8

        # 涨幅限制（追高惩罚）
        if today_chg >= 7:
            score -= 15
        elif today_chg >= 5:
            score -= 5
        elif today_chg < -5:
            score += 5       # 深跌反弹机会

    # 价格过滤
    if latest > 50:
        score -= 20

    # ── 辅助信息 ──
    result["score"] = max(0, score)
    result["max_score"] = 100
    result["signal_type"] = signal_type
    result["consec_limit"] = consec_limit
    result["consec_up"] = consec_up
    result["t1_safety"] = 0

    result["ma_trend"] = (
        "多头排列" if ma5 > ma10 > ma20
        else "短期多头" if ma5 > ma10
        else "空头排列" if ma5 < ma10 < ma20
        else "震荡"
    )
    result["vol_pattern"] = (
        "放量上攻" if vol_today_ratio > 1.5 and today_chg > 0
        else "温和放量" if vol_today_ratio > 1.2
        else "平稳" if vol_today_ratio > 0.8
        else "缩量"
    )

    bias5 = (latest - ma5) / ma5 * 100 if ma5 > 0 else 0
    result["bias5"] = round(bias5, 2)
    result["bias20"] = round((latest - ma20) / ma20 * 100 if ma20 else 0, 2)

    recent_20_high = high.tail(20).max()
    recent_20_low = low.tail(20).min()
    result["support"] = round(recent_20_low, 2)
    result["resistance"] = round(recent_20_high, 2)
    price_range = recent_20_high - recent_20_low
    result["price_position"] = round(
        (latest - recent_20_low) / price_range if price_range > 0 else 0.5, 2
    )

    if len(close) >= 6:
        result["chg_5d"] = round(chg_5d, 2)
    if len(close) >= 11:
        result["chg_10d"] = round((latest / close_arr[-11] - 1) * 100, 2)

    # RSI
    rsi_val = 50
    if len(close_arr) >= 15:
        deltas = np.diff(close_arr[-15:])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss_val = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.001
        rsi_val = 100 - 100 / (1 + gain / loss_val)
    result["rsi"] = round(rsi_val, 1)

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    macd_bar = 2 * (dif - dea)
    result["macd_dif"] = round(dif.iloc[-1], 3)
    result["macd_dea"] = round(dea.iloc[-1], 3)
    result["macd_bar"] = round(macd_bar.iloc[-1], 3)
    result["macd_signal"] = "多头" if dif.iloc[-1] > dea.iloc[-1] else "空头"

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
    max_price: float = 40.0,
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
      dragon — 龙头打板（涨停/连板/接力/爆量突破）
      trend  — 趋势股（MA多头+缩量回踩）
      breakout — 突破股（放量突破近期高点）
      oversold — 超跌反弹（RSI低+MACD底背离）
      sub_dragon — 副龙头（短期题材+低位+资金流入+即将拉升）
    """
    # 龙头模式自动放宽涨跌幅范围（允许涨停板）
    if mode == "dragon":
        max_change = max(max_change, 20.0)
        min_change = min(min_change, -5.0)
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

        _kline_mode = "dragon" if mode in ("dragon", "sub_dragon") else mode
        ta = analyze_kline(df, mode=_kline_mode)
        if ta.get("score", 0) == 0:
            continue

        # ============ 龙头打板加分（与回测v6同步）============
        bonus = 0
        chg_today = q.get("change_pct", 0)
        sig_type = ta.get("signal_type", "none")

        # 龙头战法：信号类型直接决定加分
        if sig_type == "board_hit":
            bonus += 20     # 涨停打板
        elif sig_type == "relay":
            bonus += 15     # 涨停接力
        elif sig_type == "breakout":
            bonus += 10     # 爆量突破
        elif sig_type == "momentum":
            bonus += 5      # 强势动量

        # 资金流加分（副龙头模式专用）
        if mode == "sub_dragon" and code in fund_flow_stocks:
            net = fund_flow_stocks[code]
            bonus += 15 if net > 5000 else 10 if net > 1000 else 5

        # 连板加分
        consec = ta.get("consec_limit", 0)
        if consec >= 2:
            bonus += 15
        elif consec >= 1:
            bonus += 5

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