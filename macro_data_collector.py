"""
macro_data_collector.py — 海外可用版数据采集
数据源：腾讯行情(qt.gtimg.cn) + Tushare Pro + SearXNG

适用场景：在日本/海外环境无法访问东方财富API时使用
所有函数返回结构化 dict，不消耗 LLM token
"""
import copy
import logging, os, re, time
from datetime import datetime, timedelta
from threading import Lock
from typing import List, Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)
_SEARXNG_CACHE = {}
_SEARXNG_CACHE_LOCK = Lock()
_SEARXNG_CACHE_TTL_SECONDS = 45.0
_SEARXNG_DISABLED_UNTIL = 0.0
_SEARXNG_DISABLED_REASON = ""
_SEARXNG_COOLDOWN_SECONDS = 180.0

# Tushare（可选，有 Token 时启用更多数据）
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
_ts_api = None

def _get_tushare():
    global _ts_api
    if _ts_api is None and TUSHARE_TOKEN:
        try:
            import tushare as ts
            ts.set_token(TUSHARE_TOKEN)
            _ts_api = ts.pro_api()
        except Exception as e:
            logger.warning(f"Tushare 初始化失败: {e}")
    return _ts_api

# SearXNG 地址
SEARXNG_URL = os.getenv("SEARXNG_BASE_URLS", "http://127.0.0.1:8888")


def _get_searxng_cooldown_remaining(now: Optional[float] = None) -> float:
    current = time.time() if now is None else now
    with _SEARXNG_CACHE_LOCK:
        return max(0.0, _SEARXNG_DISABLED_UNTIL - current)


def _mark_searxng_unavailable(reason: str) -> None:
    global _SEARXNG_DISABLED_UNTIL, _SEARXNG_DISABLED_REASON
    with _SEARXNG_CACHE_LOCK:
        already_disabled = _SEARXNG_DISABLED_UNTIL > time.time()
        _SEARXNG_DISABLED_UNTIL = time.time() + _SEARXNG_COOLDOWN_SECONDS
        _SEARXNG_DISABLED_REASON = reason
    if not already_disabled:
        logger.warning(
            "SearXNG 当前不可用，%.0f 秒内跳过后续搜索: %s",
            _SEARXNG_COOLDOWN_SECONDS,
            reason,
        )


def _clear_searxng_unavailable() -> None:
    global _SEARXNG_DISABLED_UNTIL, _SEARXNG_DISABLED_REASON
    with _SEARXNG_CACHE_LOCK:
        _SEARXNG_DISABLED_UNTIL = 0.0
        _SEARXNG_DISABLED_REASON = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 腾讯行情 API 解析工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _fetch_tencent_quote(codes: list, timeout: int = 15) -> dict:
    """
    批量获取腾讯行情数据
    codes: ["sh600519", "sz002506", "sh000001"] 等
    返回: {code: {name, price, change_pct, volume, ...}}
    """
    results = {}
    # 腾讯接口支持批量查询，逗号分隔
    code_str = ",".join(codes)
    url = f"https://qt.gtimg.cn/q={code_str}"
    try:
        r = requests.get(url, timeout=timeout)
        r.encoding = "gbk"
        lines = r.text.strip().split("\n")
        for line in lines:
            if "=" not in line:
                continue
            match = re.match(r'v_(\w+)="(.+)"', line.strip().rstrip(";"))
            if not match:
                continue
            code = match.group(1)
            fields = match.group(2).split("~")
            if len(fields) < 45:
                continue

            def _field_float(idx: int) -> float:
                try:
                    return float(fields[idx]) if fields[idx] else 0.0
                except (ValueError, TypeError, IndexError):
                    return 0.0

            def _field_int(idx: int) -> int:
                try:
                    return int(float(fields[idx])) if fields[idx] else 0
                except (ValueError, TypeError, IndexError):
                    return 0

            quote = {
                "market": fields[0],         # 1=沪 51=深
                "name": fields[1],
                "code": fields[2],
                "price": _field_float(3),
                "prev_close": _field_float(4),
                "open": _field_float(5),
                "volume": _field_int(6),     # 手
                # 历史兼容：原字段名保留，同时补充更明确的 outer/inner 命名
                "buy_volume": _field_int(7),
                "sell_volume": _field_int(8),
                "outer_volume": _field_int(7),
                "inner_volume": _field_int(8),
                "bid1_price": _field_float(9),
                "bid1_volume": _field_int(10),
                "change_amount": _field_float(31),
                "change_pct": _field_float(32),
                "high": _field_float(33),
                "low": _field_float(34),
                "amount": _field_float(37),  # 万元
                "turnover_rate": _field_float(38),
                "pe_ratio": _field_float(39),
                "amplitude": _field_float(43),
                "market_cap": _field_float(44),  # 亿
                "timestamp": fields[30],
            }
            for level in range(1, 6):
                bid_price_idx = 9 + (level - 1) * 2
                bid_volume_idx = bid_price_idx + 1
                ask_price_idx = 19 + (level - 1) * 2
                ask_volume_idx = ask_price_idx + 1
                quote[f"bid{level}_price"] = _field_float(bid_price_idx)
                quote[f"bid{level}_volume"] = _field_int(bid_volume_idx)
                quote[f"ask{level}_price"] = _field_float(ask_price_idx)
                quote[f"ask{level}_volume"] = _field_int(ask_volume_idx)
            results[code] = quote
    except Exception as e:
        logger.error(f"腾讯行情请求失败: {e}")
    return results


def _stock_code_to_tencent(code: str) -> str:
    """股票代码转腾讯格式: 600519 → sh600519"""
    if code.startswith("6") or code.startswith("9"):
        return f"sh{code}"
    else:
        return f"sz{code}"


def _fetch_tencent_kline(code: str, days: int = 60) -> pd.DataFrame:
    """获取腾讯日K线数据"""
    tc = _stock_code_to_tencent(code)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tc},day,,,{days},qfq"
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        # 解析路径: data -> {tc} -> "qfqday" or "day"
        stock_data = data.get("data", {})
        if isinstance(stock_data, dict):
            stock_data = stock_data.get(tc, {})
        if isinstance(stock_data, list):
            # 有时返回 list 格式
            return pd.DataFrame()
        if not isinstance(stock_data, dict):
            return pd.DataFrame()
        kline = stock_data.get("qfqday", stock_data.get("day", []))
        if not kline:
            return pd.DataFrame()
        cols = ["date", "open", "close", "high", "low", "volume"]
        df = pd.DataFrame(kline, columns=cols[:len(kline[0])])
        for col in ["open", "close", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        logger.error(f"腾讯K线 {code} 获取失败: {e}")
        return pd.DataFrame()


def _fetch_tencent_kline_by_tc(tc_code: str, days: int = 60) -> pd.DataFrame:
    """获取腾讯日K线 — 直接传腾讯格式代码（如 sh000001）"""
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tc_code},day,,,{days},qfq"
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        stock_data = data.get("data", {})
        if isinstance(stock_data, dict):
            stock_data = stock_data.get(tc_code, {})
        if not isinstance(stock_data, dict):
            return pd.DataFrame()
        kline = stock_data.get("qfqday", stock_data.get("day", []))
        if not kline:
            return pd.DataFrame()
        cols = ["date", "open", "close", "high", "low", "volume"]
        df = pd.DataFrame(kline, columns=cols[:len(kline[0])])
        for col in ["open", "close", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        logger.debug(f"腾讯K线 {tc_code} 获取失败: {e}")
        return pd.DataFrame()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 三大指数 + 量化信号
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_index_signals() -> dict:
    """三大指数实时行情 + K线量化信号"""
    result = {}
    indices = {
        "上证指数": ("sh000001", "000001"),
        "深证成指": ("sz399001", "399001"),
        "创业板指": ("sz399006", "399006"),
    }

    # 实时行情
    tc_codes = [v[0] for v in indices.values()]
    quotes = _fetch_tencent_quote(tc_codes)

    for name, (tc_code, raw_code) in indices.items():
        try:
            q = quotes.get(tc_code, {})
            # K线计算MA — 直接传腾讯格式代码
            df = _fetch_tencent_kline_by_tc(tc_code, days=60)
            if df.empty or "close" not in df.columns:
                result[name] = {
                    "latest_close": q.get("price", 0),
                    "change_pct": q.get("change_pct", 0),
                    "ma_trend": "数据不足",
                }
                continue

            close = df["close"]
            vol = df["volume"] if "volume" in df.columns else pd.Series([0])

            ma5 = close.rolling(5).mean().iloc[-1]
            ma10 = close.rolling(10).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
            latest = q.get("price", 0) or float(close.iloc[-1])

            if ma5 > ma10 > ma20:
                ma_trend = "多头排列(强势)"
            elif ma5 < ma10 < ma20:
                ma_trend = "空头排列(弱势)"
            else:
                ma_trend = "交叉震荡"

            bias_5 = round((latest - ma5) / ma5 * 100, 2) if ma5 else 0
            bias_20 = round((latest - ma20) / ma20 * 100, 2) if ma20 else 0
            vol_mean_20 = vol.tail(20).mean()
            vol_ratio = round(vol.tail(5).mean() / vol_mean_20, 2) if vol_mean_20 > 0 else 1.0

            result[name] = {
                "latest_close": round(latest, 2),
                "change_pct": q.get("change_pct", 0),
                "ma5": round(ma5, 2),
                "ma10": round(ma10, 2),
                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "ma_trend": ma_trend,
                "bias_5": bias_5,
                "bias_20": bias_20,
                "vol_ratio": vol_ratio,
                "turnover_rate": q.get("turnover_rate", 0),
                "amplitude": q.get("amplitude", 0),
            }
        except Exception as e:
            logger.error(f"计算 {name} 信号失败: {e}")
            result[name] = {"error": str(e)}

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 全市场资金流向（Tushare）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_market_fund_flow(days: int = 10) -> dict:
    """通过 Tushare 获取大盘资金流向"""
    api = _get_tushare()
    if api is None:
        return {"error": "Tushare 未配置，无法获取资金流数据。请配置 TUSHARE_TOKEN"}
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
        df = api.moneyflow_hsgt(start_date=start, end_date=end)
        if df is None or df.empty:
            return {"error": "Tushare 返回空数据"}
        df = df.head(days)
        records = []
        for _, row in df.iterrows():
            records.append({
                "date": str(row.get("trade_date", "")),
                "north_money": float(row.get("north_money", 0)),
                "south_money": float(row.get("south_money", 0)),
            })
        total = sum(r["north_money"] for r in records)
        return {
            "recent_days": len(records),
            "total_north_net": round(total, 2),
            "trend": "北向净流入" if total > 0 else "北向净流出",
            "daily_detail": records[:5],
        }
    except Exception as e:
        logger.error(f"Tushare 资金流获取失败: {e}")
        return {"error": str(e)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 北向资金
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_northbound_flow() -> dict:
    """北向资金 = fetch_market_fund_flow 已含"""
    return fetch_market_fund_flow(days=5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 板块轮动（Tushare 行业数据）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_sector_fund_flow() -> dict:
    """板块涨跌排名（Tushare 或腾讯板块接口）"""
    # 尝试腾讯板块接口
    result = {}
    try:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        # 这个是东方财富的，海外可能不通，先尝试
        # 如果不通就用 Tushare
        pass
    except:
        pass

    # Tushare 方案：获取行业指数日线
    api = _get_tushare()
    if api:
        try:
            today = datetime.now().strftime("%Y%m%d")
            df = api.index_daily(ts_code="", trade_date=today)
            if df is not None and not df.empty:
                # 按涨跌幅排序取 Top
                df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
                top = df.nlargest(10, "pct_chg")[["ts_code", "pct_chg"]].to_dict("records")
                bottom = df.nsmallest(5, "pct_chg")[["ts_code", "pct_chg"]].to_dict("records")
                result["行业_今日"] = {"top_inflow": top, "top_outflow": bottom}
        except Exception as e:
            logger.debug(f"Tushare 板块数据获取失败: {e}")

    if not result:
        result["note"] = "板块数据暂不可用，海外无法访问东方财富板块API"

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 个股资金流向
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_stock_fund_flow(stock_code: str) -> dict:
    """个股资金流 — 优先同花顺爬虫，备用 Tushare，兜底腾讯行情"""

    # 方案1: 同花顺爬虫（最准确，有主力净流入/占比）
    try:
        from ths_scraper import fetch_single_stock_fund_flow as ths_flow
        result = ths_flow(stock_code)
        if result and result.get("main_net") is not None and "error" not in result:
            result["fund_trend"] = (
                "主力流入" if result.get("main_net", 0) > 0
                else "主力流出" if result.get("main_net", 0) < 0
                else "持平"
            )
            logger.debug(f"[个股资金流] {stock_code} 来源: 同花顺")
            return result
    except Exception as e:
        logger.debug(f"同花顺个股资金流 {stock_code} 失败: {e}")

    # 方案2: Tushare
    api = _get_tushare()
    if api:
        try:
            ts_code = f"{stock_code}.SH" if stock_code.startswith("6") else f"{stock_code}.SZ"
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=20)).strftime("%Y%m%d")
            df = api.moneyflow(ts_code=ts_code, start_date=start, end_date=end)
            if df is not None and not df.empty:
                df = df.head(10)
                records = []
                for _, row in df.iterrows():
                    records.append({
                        "date": str(row.get("trade_date", "")),
                        "net_mf_amount": float(row.get("net_mf_amount", 0)),
                    })
                total_10d = sum(r["net_mf_amount"] for r in records)
                total_3d = sum(r["net_mf_amount"] for r in records[:3])
                return {
                    "code": stock_code,
                    "total_main_net_10d": round(total_10d, 0),
                    "total_main_net_3d": round(total_3d, 0),
                    "fund_trend": (
                        "主力持续流入" if total_3d > 0 and total_10d > 0
                        else "主力持续流出" if total_3d < 0 and total_10d < 0
                        else "主力分歧"
                    ),
                    "daily": records[:5],
                    "source": "tushare",
                }
        except Exception as e:
            logger.debug(f"Tushare 个股资金流 {stock_code} 失败: {e}")

    # 方案3: 腾讯行情兜底
    tc = _stock_code_to_tencent(stock_code)
    quotes = _fetch_tencent_quote([tc])
    q = quotes.get(tc, {})
    return {
        "code": stock_code,
        "name": q.get("name", ""),
        "price": q.get("price", 0),
        "change_pct": q.get("change_pct", 0),
        "turnover_rate": q.get("turnover_rate", 0),
        "source": "tencent_basic",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 千股千评（腾讯+Tushare 组合）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_stock_comments(stock_codes: list) -> dict:
    """个股基础面板：腾讯行情实时数据"""
    tc_codes = [_stock_code_to_tencent(c) for c in stock_codes]
    quotes = _fetch_tencent_quote(tc_codes)
    records = {}
    for code in stock_codes:
        tc = _stock_code_to_tencent(code)
        q = quotes.get(tc, {})
        if q:
            records[code] = {
                "name": q.get("name", ""),
                "latest_price": q.get("price", 0),
                "change_pct": q.get("change_pct", 0),
                "turnover_rate": q.get("turnover_rate", 0),
                "pe_ratio": q.get("pe_ratio", 0),
                "market_cap": q.get("market_cap", 0),
                "amplitude": q.get("amplitude", 0),
            }
    return records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 两融余额（Tushare）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_margin_data() -> dict:
    api = _get_tushare()
    if api is None:
        return {"note": "无 Tushare Token，两融数据不可用"}
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=20)).strftime("%Y%m%d")
        df = api.margin(start_date=start, end_date=end)
        if df is None or df.empty:
            return {"error": "无数据"}
        df = df.head(10)
        records = []
        for _, row in df.iterrows():
            records.append({
                "date": str(row.get("trade_date", "")),
                "rzye": float(row.get("rzye", 0)),    # 融资余额
                "rqye": float(row.get("rqye", 0)),    # 融券余额
            })
        if len(records) >= 2:
            trend = "增加" if records[0].get("rzye", 0) > records[-1].get("rzye", 0) else "减少"
        else:
            trend = "未知"
        return {"trend": trend, "recent": records[:3]}
    except Exception as e:
        logger.error(f"两融数据获取失败: {e}")
        return {"error": str(e)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 涨跌家数（腾讯批量行情统计）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_market_breadth() -> dict:
    """用 Tushare 日线统计涨跌家数"""
    api = _get_tushare()
    if api is None:
        return {"note": "无 Tushare Token，涨跌家数不可用"}
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = api.daily(trade_date=today)
        if df is None or df.empty:
            # 可能非交易日，尝试前一天
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            df = api.daily(trade_date=yesterday)
        if df is None or df.empty:
            return {"error": "无数据"}

        total = len(df)
        up = len(df[df["pct_chg"] > 0])
        down = len(df[df["pct_chg"] < 0])
        flat = total - up - down
        limit_up = len(df[df["pct_chg"] >= 9.9])
        limit_down = len(df[df["pct_chg"] <= -9.9])
        up_ratio = round(up / total * 100, 1) if total > 0 else 0

        return {
            "total": total,
            "up": up, "down": down, "flat": flat,
            "limit_up": limit_up, "limit_down": limit_down,
            "up_ratio": up_ratio,
            "market_sentiment": (
                "极度强势" if up_ratio > 75 else "偏强" if up_ratio > 55
                else "分化" if up_ratio > 40 else "偏弱" if up_ratio > 25
                else "极度弱势"
            ),
        }
    except Exception as e:
        logger.error(f"涨跌家数获取失败: {e}")
        return {"error": str(e)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. 新闻 — SearXNG 搜索（可搜 X/Twitter）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENSITIVE_KEYWORDS = [
    "特朗普", "trump", "关税", "tariff", "贸易战", "制裁", "芯片",
    "半导体", "华为", "台湾", "降息", "加息", "央行", "证监会",
]


def _searxng_search(query: str, max_results: int = 10) -> list:
    """通用 SearXNG 搜索"""
    cache_key = (str(query).strip(), int(max_results))
    now = time.time()
    with _SEARXNG_CACHE_LOCK:
        entry = _SEARXNG_CACHE.get(cache_key)
        if entry and (now - entry["ts"]) <= _SEARXNG_CACHE_TTL_SECONDS:
            return copy.deepcopy(entry["value"])
        if _SEARXNG_DISABLED_UNTIL > now:
            return []
    try:
        r = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general"},
            timeout=15,
        )
        if r.ok:
            data = r.json()
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:300],
                    "engine": item.get("engine", ""),
                })
            with _SEARXNG_CACHE_LOCK:
                _SEARXNG_CACHE[cache_key] = {"ts": time.time(), "value": copy.deepcopy(results)}
            _clear_searxng_unavailable()
            return results
        _mark_searxng_unavailable(f"HTTP {r.status_code} @ {SEARXNG_URL}")
    except Exception as e:
        _mark_searxng_unavailable(f"{type(e).__name__}: {e}")
        logger.warning(f"SearXNG 搜索失败: {e}")
    return []


def fetch_trump_news() -> list:
    """
    特朗普最新动态 — 多源搜索
    优先 SearXNG 搜 X/Twitter，备用搜新闻
    """
    results = []

    # 1. SearXNG 搜 X/Twitter
    twitter_results = _searxng_search(
        "site:x.com realDonaldTrump tariff OR china OR trade", max_results=5
    )
    for r in twitter_results:
        r["source"] = "x_twitter"
    results.extend(twitter_results)

    # 2. SearXNG 搜英文财经新闻
    news_results = _searxng_search(
        "Trump tariff China trade latest 2026", max_results=5
    )
    for r in news_results:
        r["source"] = "news"
    results.extend(news_results)

    # 3. SearXNG 搜中文新闻
    cn_results = _searxng_search("特朗普 关税 中国 最新", max_results=5)
    for r in cn_results:
        r["source"] = "cn_news"
    results.extend(cn_results)

    # 标记敏感
    for r in results:
        text = (r.get("title", "") + r.get("content", "")).lower()
        r["is_sensitive"] = any(kw.lower() in text for kw in SENSITIVE_KEYWORDS)

    sensitive_count = sum(1 for r in results if r["is_sensitive"])
    logger.info(f"[特朗普新闻] 共 {len(results)} 条，敏感 {sensitive_count} 条")
    return results


def fetch_financial_news() -> dict:
    """财经快讯 — SearXNG 搜索"""
    results = {"all": [], "sensitive": []}

    # 搜索 A股 相关快讯
    items = _searxng_search("A股 今日 行情 资金", max_results=10)
    items += _searxng_search("中国股市 最新消息", max_results=5)

    for item in items:
        results["all"].append(item)
        text = (item.get("title", "") + item.get("content", "")).lower()
        matched = [kw for kw in SENSITIVE_KEYWORDS if kw.lower() in text]
        if matched:
            item["matched_keywords"] = matched
            results["sensitive"].append(item)

    logger.info(
        f"[财经快讯] {len(results['all'])} 条，"
        f"敏感 {len(results['sensitive'])} 条"
    )
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. 个股新闻 — SearXNG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_stock_news(stock_code: str, stock_name: str = "", limit: int = 5) -> list:
    """用 SearXNG 搜索个股相关新闻"""
    query = f"{stock_name} {stock_code} 股票 最新消息" if stock_name else f"{stock_code} 股票 最新"
    return _searxng_search(query, max_results=limit)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. 基本面深度数据（研究员用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_fundamental_deep(stock_code: str) -> dict:
    """获取单只股票的深度基本面数据，供研究员Agent使用。
    优先链: AkShare基本面 → Tushare估值 → 腾讯行情PE兜底
    """
    result = {
        "pe_current": None, "pb_current": None, "roe_recent": None,
        "revenue_yoy": None, "net_profit_yoy": None, "gross_margin": None,
        "industry_pe_avg": None, "earnings_forecast": "",
        "financial_health": "unknown", "small_cap_caveat": False,
        "source_chain": [],
    }

    # 1) AkShare 基本面适配器
    try:
        from data_provider.fundamental_adapter import AkshareFundamentalAdapter
        adapter = AkshareFundamentalAdapter()
        bundle = adapter.get_fundamental_bundle(stock_code)
        growth = bundle.get("growth", {})
        if growth:
            result["roe_recent"] = growth.get("roe")
            result["revenue_yoy"] = growth.get("revenue_yoy")
            result["net_profit_yoy"] = growth.get("net_profit_yoy")
            result["gross_margin"] = growth.get("gross_margin")
            result["source_chain"].append("akshare_fundamental")

        earnings = bundle.get("earnings", {})
        if earnings:
            result["earnings_forecast"] = earnings.get("summary", "")
    except Exception as e:
        logger.warning(f"AkShare基本面 {stock_code} 失败: {e}")

    # 2) Tushare 估值数据
    api = _get_tushare()
    if api is not None:
        try:
            # 转换为Tushare格式代码
            ts_code = f"{stock_code}.SH" if stock_code.startswith("6") else f"{stock_code}.SZ"
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
            df = api.daily_basic(ts_code=ts_code, start_date=start, end_date=end,
                                 fields="ts_code,trade_date,pe_ttm,pb,total_mv,circ_mv")
            if df is not None and not df.empty:
                latest = df.iloc[0]
                result["pe_current"] = float(latest.get("pe_ttm", 0)) if pd.notna(latest.get("pe_ttm")) else None
                result["pb_current"] = float(latest.get("pb", 0)) if pd.notna(latest.get("pb")) else None
                total_mv = float(latest.get("total_mv", 0)) if pd.notna(latest.get("total_mv")) else 0
                result["total_mv_yi"] = round(total_mv / 10000, 2) if total_mv else None  # 万元→亿元
                result["small_cap_caveat"] = (total_mv > 0 and total_mv / 10000 < 50)
                result["source_chain"].append("tushare_daily_basic")
        except Exception as e:
            logger.warning(f"Tushare估值 {stock_code} 失败: {e}")

    # 3) 财务健康判断
    roe = result.get("roe_recent")
    profit_yoy = result.get("net_profit_yoy")
    pe = result.get("pe_current")
    if roe is not None and profit_yoy is not None:
        if roe < 0 or profit_yoy < -30:
            result["financial_health"] = "danger"
        elif roe < 5 or profit_yoy < -10:
            result["financial_health"] = "warning"
        else:
            result["financial_health"] = "healthy"
    elif pe is not None and pe < 0:
        result["financial_health"] = "danger"

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. 市场情绪指标（分析师用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_sentiment_indicators(margin_data: dict = None,
                                market_breadth: dict = None) -> dict:
    """计算市场情绪复合指标。大部分从已采集数据计算，极少新API调用。

    Args:
        margin_data: 已采集的两融数据 (可选，无则重新采集)
        market_breadth: 已采集的涨跌家数 (可选，无则重新采集)
    """
    result = {
        "fear_greed_score": 50,
        "emotional_cycle": "修复",
        "margin_signal": "中性",
        "breadth_signal": "中性",
        "margin_change_5d_pct": 0,
        "retail_vs_inst_signal": "无明显倾向",
    }

    # 融资余额趋势
    if margin_data is None:
        margin_data = fetch_margin_data()
    recent_margin = margin_data.get("recent", [])
    if len(recent_margin) >= 2:
        latest_rz = recent_margin[0].get("rzye", 0)
        oldest_rz = recent_margin[-1].get("rzye", 0)
        if oldest_rz > 0:
            change_pct = round((latest_rz - oldest_rz) / oldest_rz * 100, 2)
            result["margin_change_5d_pct"] = change_pct
            if change_pct > 2:
                result["margin_signal"] = "杠杆加码（贪婪）"
            elif change_pct < -2:
                result["margin_signal"] = "杠杆收缩（恐慌）"

    # 涨跌家数
    if market_breadth is None:
        market_breadth = fetch_market_breadth()
    up = market_breadth.get("up_count", 0)
    down = market_breadth.get("down_count", 0)
    total = up + down
    if total > 0:
        up_ratio = up / total
        if up_ratio > 0.7:
            result["breadth_signal"] = "普涨（乐观/贪婪）"
        elif up_ratio < 0.3:
            result["breadth_signal"] = "普跌（恐慌）"
        elif up_ratio < 0.4:
            result["breadth_signal"] = "偏弱（犹豫）"
        else:
            result["breadth_signal"] = "分化（正常）"

    # 恐贪复合评分 (0=极度恐慌, 100=极度贪婪)
    score = 50
    # 融资情绪 (±15)
    mc = result["margin_change_5d_pct"]
    score += max(-15, min(15, mc * 5))
    # 涨跌广度 (±20)
    if total > 0:
        score += (up_ratio - 0.5) * 40
    # 融资趋势方向 (±10)
    margin_trend = margin_data.get("trend", "未知")
    if margin_trend == "增加":
        score += 10
    elif margin_trend == "减少":
        score -= 10

    score = max(0, min(100, round(score)))
    result["fear_greed_score"] = score

    # 情绪周期映射
    if score <= 20:
        result["emotional_cycle"] = "恐慌"
    elif score <= 35:
        result["emotional_cycle"] = "犹豫"
    elif score <= 50:
        result["emotional_cycle"] = "修复"
    elif score <= 65:
        result["emotional_cycle"] = "乐观"
    elif score <= 80:
        result["emotional_cycle"] = "贪婪"
    else:
        result["emotional_cycle"] = "疯狂"

    # 散户 vs 机构信号
    if score > 70 and result["breadth_signal"].startswith("普涨"):
        result["retail_vs_inst_signal"] = "散户蜂拥入场，机构可能在派发"
    elif score < 30 and result["breadth_signal"].startswith("普跌"):
        result["retail_vs_inst_signal"] = "散户恐慌出逃，可能接近底部"

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. 历史回溯上下文（回溯验证用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_rebalance_backtest_context() -> dict:
    """从本地DB获取历史准确度数据，零API调用。"""
    result = {
        "scan_accuracy": {},
        "trade_performance": {},
        "winning_patterns": {},
        "recent_rebalance_outcomes": [],
        "note": None,
    }

    # 扫描胜率
    try:
        from data_store import get_backtest_summary
        result["scan_accuracy"] = get_backtest_summary(days=30)
    except Exception as e:
        logger.warning(f"扫描回测数据获取失败: {e}")

    # 交易绩效
    try:
        from trade_journal import get_performance_summary, analyze_winning_patterns
        result["trade_performance"] = get_performance_summary(days=30)
        result["winning_patterns"] = analyze_winning_patterns(days=90)
    except Exception as e:
        logger.warning(f"交易绩效数据获取失败: {e}")

    # 最近蒸馏样本 vs 实际结果
    try:
        from pathlib import Path
        import json as _json
        distill_dir = Path("data/distillation")
        outcomes = []
        if distill_dir.exists():
            files = sorted(distill_dir.glob("distill_*.jsonl"), reverse=True)
            for fpath in files[:2]:  # 最近2个月
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            sample = _json.loads(line)
                            ts = sample.get("timestamp", "")[:10]
                            parsed = sample.get("parsed_json", {})
                            actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
                            for act in actions[:3]:
                                outcomes.append({
                                    "date": ts,
                                    "code": act.get("code", ""),
                                    "action": act.get("action", ""),
                                    "target_sell_price": act.get("target_sell_price"),
                                })
                        except:
                            continue
                if len(outcomes) >= 10:
                    break
        result["recent_rebalance_outcomes"] = outcomes[:10]
    except Exception as e:
        logger.warning(f"蒸馏回溯数据获取失败: {e}")

    # 数据充足性
    perf = result.get("trade_performance", {})
    if perf.get("total_trades", 0) < 10:
        result["note"] = "insufficient_data"

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. 一键采集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def collect_full_macro_data(holding_codes: list) -> dict:
    """一键采集所有数据"""
    total_steps = 10
    logger.info("=" * 50)
    logger.info("开始采集全量数据（海外版）...")
    has_tushare = bool(_get_tushare())
    logger.info(f"  Tushare: {'已配置' if has_tushare else '未配置（部分数据不可用）'}")
    logger.info(f"  SearXNG: {SEARXNG_URL}")
    logger.info(f"  腾讯行情: qt.gtimg.cn")

    data = {}

    # 先获取持仓股名称（后续搜新闻用）
    tc_codes = [_stock_code_to_tencent(c) for c in holding_codes]
    all_quotes = _fetch_tencent_quote(tc_codes)
    code_name_map = {}
    for code in holding_codes:
        tc = _stock_code_to_tencent(code)
        q = all_quotes.get(tc, {})
        code_name_map[code] = q.get("name", "")

    # 1. 大盘信号
    logger.info(f"[1/{total_steps}] 大盘指数信号...")
    data["index_signals"] = fetch_index_signals()

    # 2. 资金流向 / 北向资金
    logger.info(f"[2/{total_steps}] 资金流向...")
    data["market_fund_flow"] = fetch_market_fund_flow(days=10)

    # 3. 北向资金
    logger.info(f"[3/{total_steps}] 北向资金...")
    data["northbound"] = fetch_northbound_flow()

    # 4. 板块轮动
    logger.info(f"[4/{total_steps}] 板块轮动...")
    data["sector_rotation"] = fetch_sector_fund_flow()

    # 5. 个股资金流
    logger.info(f"[5/{total_steps}] 个股资金流向...")
    data["holdings_fund_flow"] = {}
    for code in holding_codes:
        data["holdings_fund_flow"][code] = fetch_stock_fund_flow(code)

    # 6. 个股面板
    logger.info(f"[6/{total_steps}] 个股面板数据...")
    data["stock_comments"] = fetch_stock_comments(holding_codes)

    # 7. 两融
    logger.info(f"[7/{total_steps}] 两融数据...")
    data["margin_data"] = fetch_margin_data()

    # 8. 涨跌家数
    logger.info(f"[8/{total_steps}] 涨跌家数...")
    data["market_breadth"] = fetch_market_breadth()

    # 9. 个股新闻
    logger.info(f"[9/{total_steps}] 个股新闻...")
    data["stock_news"] = {}
    for code in holding_codes:
        name = code_name_map.get(code, "")
        data["stock_news"][code] = fetch_stock_news(code, name, limit=3)

    # 10. 特朗普 + 财经快讯
    logger.info(f"[10/{total_steps}] 特朗普新闻 + 财经快讯...")
    data["trump_news"] = fetch_trump_news()
    data["cls_telegraph"] = fetch_financial_news()

    # 11. 真实换股候选（全市场扫描+K线分析）
    logger.info("[额外] 全市场扫描换股候选...")
    try:
        from market_scanner import scan_market as _scan
        data["hot_candidates"] = _scan(
            max_price=10.0, min_turnover=2.0, max_market_cap=100.0,
            top_n=20, mode="trend",
        )
        logger.info(f"  换股候选: {len(data['hot_candidates'])} 只")
    except Exception as e:
        logger.warning(f"全市场扫描失败，降级到同花顺Top50: {e}")
        try:
            from ths_scraper import fetch_hot_stocks_for_candidate
            data["hot_candidates"] = fetch_hot_stocks_for_candidate(max_price=10.0, top_n=20)
        except:
            data["hot_candidates"] = []

    # 12. 个股基本面深度数据（研究员Agent用）
    logger.info("[额外] 个股基本面深度数据...")
    data["holdings_fundamental"] = {}
    for code in holding_codes:
        try:
            data["holdings_fundamental"][code] = fetch_fundamental_deep(code)
        except Exception as e:
            logger.warning(f"基本面 {code} 跳过: {e}")

    # 13. 市场情绪指标（分析师Agent用）
    logger.info("[额外] 市场情绪指标...")
    try:
        data["sentiment_indicators"] = fetch_sentiment_indicators(
            margin_data=data.get("margin_data"),
            market_breadth=data.get("market_breadth"),
        )
        logger.info(f"  情绪: {data['sentiment_indicators'].get('emotional_cycle')} "
                     f"(恐贪={data['sentiment_indicators'].get('fear_greed_score')})")
    except Exception as e:
        logger.warning(f"情绪指标跳过: {e}")
        data["sentiment_indicators"] = {}

    # 14. 历史回溯上下文（回溯验证Agent用）
    logger.info("[额外] 历史回溯数据...")
    try:
        data["backtest_context"] = fetch_rebalance_backtest_context()
    except Exception as e:
        logger.warning(f"回溯数据跳过: {e}")
        data["backtest_context"] = {}

    # 兼容字段（给 rebalance_engine 用）
    data["northbound_holdings"] = {}

    # 汇总
    news_total = (
        len(data.get("cls_telegraph", {}).get("all", []))
        + sum(len(v) for v in data.get("stock_news", {}).values())
        + len(data.get("trump_news", []))
    )
    logger.info("=" * 50)
    logger.info("数据采集完成！")
    logger.info(f"  指数: {len(data.get('index_signals', {}))} 个")
    logger.info(f"  个股: {len(data.get('holdings_fund_flow', {}))} 只")
    logger.info(f"  新闻: {news_total} 条")
    logger.info(f"  特朗普: {len(data.get('trump_news', []))} 条")
    logger.info("=" * 50)

    return data
