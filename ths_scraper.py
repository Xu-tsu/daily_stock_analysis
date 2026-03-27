# -*- coding: utf-8 -*-
"""
ths_scraper.py — 资金流向数据采集（多数据源自动回退）

数据源优先级：新浪财经 → 东方财富 → 腾讯行情兜底
实测从日本连接：新浪 0.9s（最稳） > 东方财富 1.6s（常抽风） > 腾讯 0.2s（无资金流）

导出函数:
  fetch_stock_fund_flow_rank()       → Top N 个股资金流排行
  fetch_sector_fund_flow_rank(type)  → 板块资金流排行（行业hy/概念gn）
  fetch_single_stock_fund_flow(code) → 单只股票资金流明细
  fetch_hot_stocks_for_candidate()   → 热门换股候选
"""
import json
import logging
import re
import requests
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

_HEADERS_EM = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/",
}
_HEADERS_SINA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://vip.stock.finance.sina.com.cn/",
}
_HEADERS_QQ = {"User-Agent": "Mozilla/5.0"}

# 记录东方财富连续失败次数，避免反复超时浪费时间
_em_fail_count = 0
_EM_SKIP_THRESHOLD = 3  # 连续失败3次后本次运行内跳过东方财富


def _safe_float(val, default=0.0):
    try:
        return float(val) if val not in (None, "", "-") else default
    except (ValueError, TypeError):
        return default


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 东方财富（主数据源）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _em_stock_fund_flow_rank(top_n: int) -> List[dict]:
    """东方财富个股资金流排行"""
    global _em_fail_count
    if _em_fail_count >= _EM_SKIP_THRESHOLD:
        return []

    url = (
        "https://push2.eastmoney.com/api/qt/clist/get?"
        "fid=f62&po=1&pn=1&pz={pz}&np=1&fltt=2&invt=2"
        "&ut=b2884a393a59ad64002292a3e90d46a5"
        "&fields=f12,f14,f2,f3,f62,f184,f66,f72,f8"
        "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
    ).format(pz=min(top_n, 100))

    try:
        r = requests.get(url, headers=_HEADERS_EM, timeout=8)
        data = r.json().get("data")
        if not data or not data.get("diff"):
            return []
        _em_fail_count = 0  # 成功，重置计数
        results = []
        for item in data["diff"][:top_n]:
            results.append({
                "code": str(item.get("f12", "")),
                "name": str(item.get("f14", "")),
                "price": _safe_float(item.get("f2")),
                "change_pct": _safe_float(item.get("f3")),
                "main_net": round(_safe_float(item.get("f62")) / 10000, 2),
                "main_net_pct": _safe_float(item.get("f184")),
                "super_large_net": round(_safe_float(item.get("f66")) / 10000, 2),
                "large_net": round(_safe_float(item.get("f72")) / 10000, 2),
                "turnover_rate": _safe_float(item.get("f8")),
                "source": "eastmoney",
            })
        return results
    except Exception as e:
        _em_fail_count += 1
        logger.debug(f"[资金流] 东方财富个股失败({_em_fail_count}): {e}")
        return []


def _em_sector_fund_flow_rank(sector_type: str, top_n: int) -> List[dict]:
    """东方财富板块资金流排行"""
    global _em_fail_count
    if _em_fail_count >= _EM_SKIP_THRESHOLD:
        return []

    fs = "m:90+t:2" if sector_type == "hy" else "m:90+t:3"
    url = (
        "https://push2.eastmoney.com/api/qt/clist/get?"
        "fid=f62&po=1&pn=1&pz={pz}&np=1&fltt=2&invt=2"
        "&ut=b2884a393a59ad64002292a3e90d46a5"
        "&fields=f12,f14,f3,f62,f184"
        "&fs={fs}"
    ).format(pz=min(top_n, 80), fs=fs)

    try:
        r = requests.get(url, headers=_HEADERS_EM, timeout=8)
        data = r.json().get("data")
        if not data or not data.get("diff"):
            return []
        _em_fail_count = 0
        results = []
        for item in data["diff"][:top_n]:
            results.append({
                "name": str(item.get("f14", "")),
                "change_pct": _safe_float(item.get("f3")),
                "main_net": round(_safe_float(item.get("f62")) / 10000, 2),
                "main_net_pct": _safe_float(item.get("f184")),
            })
        return results
    except Exception as e:
        _em_fail_count += 1
        logger.debug(f"[资金流] 东方财富板块失败({_em_fail_count}): {e}")
        return []


def _em_single_stock_fund_flow(stock_code: str) -> Optional[dict]:
    """东方财富单股资金流"""
    global _em_fail_count
    if _em_fail_count >= _EM_SKIP_THRESHOLD:
        return None

    market = "1" if stock_code.startswith("6") else "0"
    secid = f"{market}.{stock_code}"
    url = (
        f"https://push2.eastmoney.com/api/qt/stock/get?"
        f"secid={secid}&ut=b2884a393a59ad64002292a3e90d46a5"
        f"&fields=f12,f14,f62,f184,f66,f72,f78,f81"
    )
    try:
        r = requests.get(url, headers=_HEADERS_EM, timeout=8)
        data = r.json().get("data")
        if not data or data.get("f62") is None:
            return None
        _em_fail_count = 0
        return {
            "code": stock_code,
            "name": str(data.get("f14", "")),
            "main_net": round(_safe_float(data.get("f62")) / 10000, 2),
            "main_net_pct": _safe_float(data.get("f184")),
            "super_large_net": round(_safe_float(data.get("f66")) / 10000, 2),
            "large_net": round(_safe_float(data.get("f72")) / 10000, 2),
            "medium_net": round(_safe_float(data.get("f78")) / 10000, 2),
            "small_net": round(_safe_float(data.get("f81")) / 10000, 2) if data.get("f81") else 0,
            "source": "eastmoney",
        }
    except Exception as e:
        _em_fail_count += 1
        logger.debug(f"[资金流] 东方财富单股 {stock_code} 失败: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 新浪财经（备选数据源）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _sina_stock_fund_flow_rank(top_n: int) -> List[dict]:
    """新浪财经个股资金流排行（ssl_bkzj_ssggzj 接口，海外稳定快速）"""
    url = (
        "https://vip.stock.finance.sina.com.cn/quotes_service/api/"
        "json_v2.php/MoneyFlow.ssl_bkzj_ssggzj?"
        f"page=1&num={min(top_n, 80)}&sort=netamount&asc=0"
    )
    try:
        r = requests.get(url, headers=_HEADERS_SINA, timeout=8)
        r.encoding = "gbk"
        text = r.text.strip()
        if not text or text == "null":
            return []
        # 新浪返回非标准 JSON（键名无引号）
        text = re.sub(r'(\w+):', r'"\1":', text)
        items = json.loads(text)
        results = []
        for item in items[:top_n]:
            symbol = str(item.get("symbol", ""))
            code = symbol[2:] if len(symbol) > 2 else symbol
            r0_net = _safe_float(item.get("r0_net"))
            r0_in = _safe_float(item.get("r0_in"))
            r0_out = _safe_float(item.get("r0_out"))
            net_amount = _safe_float(item.get("netamount"))
            results.append({
                "code": code,
                "name": str(item.get("name", "")),
                "price": _safe_float(item.get("trade")),
                "change_pct": _safe_float(item.get("changeratio")) * 100,
                "main_net": round(net_amount / 10000, 2),
                "main_net_pct": _safe_float(item.get("ratioamount")) * 100,
                "super_large_net": round(r0_net / 10000, 2),  # 新浪 r0=超大单
                "large_net": round(_safe_float(item.get("r3_net")) / 10000, 2),
                "turnover_rate": _safe_float(item.get("turnover")),
                "source": "sina",
            })
        return results
    except Exception as e:
        logger.debug(f"[资金流] 新浪个股失败: {e}")
        return []


def _sina_sector_fund_flow_rank(sector_type: str, top_n: int) -> List[dict]:
    """新浪财经板块资金流排行"""
    # fenlei: 0=概念, 1=行业
    fenlei = "1" if sector_type == "hy" else "0"
    url = (
        "https://vip.stock.finance.sina.com.cn/quotes_service/api/"
        f"json_v2.php/MoneyFlow.ssl_bkzj_bk?"
        f"page=1&num={min(top_n, 40)}&sort=netamount&asc=0&fenlei={fenlei}"
    )
    try:
        r = requests.get(url, headers=_HEADERS_SINA, timeout=10)
        r.encoding = "gbk"
        text = r.text.strip()
        if not text or text == "null":
            return []
        text = re.sub(r'(\w+):', r'"\1":', text)
        items = json.loads(text)
        results = []
        for item in items[:top_n]:
            net = _safe_float(item.get("netamount"))
            inamount = _safe_float(item.get("inamount"))
            outamount = _safe_float(item.get("outamount"))
            total = inamount + outamount
            results.append({
                "name": str(item.get("name", "")),
                "change_pct": _safe_float(item.get("avg_changeratio")) * 100,
                "main_net": round(net / 10000, 2),
                "main_net_pct": round(net / total * 100, 2) if total > 0 else 0,
            })
        return results
    except Exception as e:
        logger.debug(f"[资金流] 新浪板块失败: {e}")
        return []


def _sina_single_stock_fund_flow(stock_code: str) -> Optional[dict]:
    """新浪财经单股资金流"""
    prefix = "sh" if stock_code.startswith("6") else "sz"
    symbol = f"{prefix}{stock_code}"
    url = (
        "https://vip.stock.finance.sina.com.cn/quotes_service/api/"
        f"json_v2.php/MoneyFlow.ssl_qsfx_lscjfb?"
        f"page=1&num=1&sort=opendate&asc=0&datefrom=&dateto=&symbol={symbol}"
    )
    try:
        r = requests.get(url, headers=_HEADERS_SINA, timeout=8)
        r.encoding = "gbk"
        text = r.text.strip()
        if not text or text == "null":
            return None
        text = re.sub(r'(\w+):', r'"\1":', text)
        items = json.loads(text)
        if not items:
            return None
        item = items[0]
        r0_net = _safe_float(item.get("r0_net"))
        r0_in = _safe_float(item.get("r0_in"))
        r0_out = _safe_float(item.get("r0_out"))
        total = r0_in + r0_out
        return {
            "code": stock_code,
            "name": "",  # 新浪单股接口不返回名称
            "main_net": round(r0_net / 10000, 2),
            "main_net_pct": round(r0_net / total * 100, 2) if total > 0 else 0,
            "super_large_net": round(_safe_float(item.get("r3_net")) / 10000, 2),
            "large_net": round(_safe_float(item.get("r2_net")) / 10000, 2),
            "medium_net": round(_safe_float(item.get("r1_net")) / 10000, 2),
            "small_net": round(_safe_float(item.get("r0x_net")) / 10000, 2),
            "source": "sina",
        }
    except Exception as e:
        logger.debug(f"[资金流] 新浪单股 {stock_code} 失败: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 腾讯行情（兜底——只有基础数据，没有细分资金流）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _tencent_stock_basic_rank(top_n: int) -> List[dict]:
    """腾讯行情获取涨幅+换手率排行作为兜底

    腾讯没有资金流接口，但可以用换手率+涨幅推算活跃度。
    main_net 用 (换手率-均值)*成交额 近似估算。
    """
    # 获取沪深涨幅排行（腾讯 rankcode 接口）
    url = "http://qt.gtimg.cn/q=sh000001"  # 先验证连通性
    try:
        r = requests.get(url, headers=_HEADERS_QQ, timeout=5)
        if '=""' in r.text:
            return []
    except Exception:
        return []

    # 腾讯没有批量资金流排行接口，返回空让上层知道需要降级
    logger.debug("[资金流] 腾讯无资金流排行接口，跳过")
    return []


def _tencent_single_stock_fund_flow(stock_code: str) -> Optional[dict]:
    """腾讯行情单股基础数据（无资金流，只有换手率/成交额）"""
    prefix = "sh" if stock_code.startswith(("6", "9")) else "sz"
    symbol = f"{prefix}{stock_code}"
    try:
        r = requests.get(
            f"http://qt.gtimg.cn/q={symbol}",
            headers=_HEADERS_QQ, timeout=5,
        )
        r.encoding = "gbk"
        m = re.search(r'v_\w+="([^"]+)"', r.text)
        if not m:
            return None
        fields = m.group(1).split("~")
        if len(fields) < 45:
            return None
        amount = _safe_float(fields[37])
        turnover = _safe_float(fields[38])
        return {
            "code": stock_code,
            "name": fields[1] if len(fields) > 1 else "",
            "main_net": round(amount * (turnover - 3) / 100 / 10000, 2) if turnover > 3 else 0,
            "main_net_pct": round(turnover - 3, 2) if turnover > 3 else 0,
            "super_large_net": 0,
            "large_net": 0,
            "medium_net": 0,
            "small_net": 0,
            "source": "tencent_estimate",
        }
    except Exception as e:
        logger.debug(f"[资金流] 腾讯单股 {stock_code} 失败: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对外接口（自动回退）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_stock_fund_flow_rank(top_n: int = 50) -> List[dict]:
    """获取个股主力资金流排行（新浪优先 → 东方财富备选）

    新浪从海外（日本）连接更快更稳（0.9s vs 1.6s），优先使用。
    """
    # 1. 新浪财经（海外更快更稳）
    results = _sina_stock_fund_flow_rank(top_n)
    if results:
        logger.info(f"[资金流] 个股排行 Top{len(results)} (来源: 新浪财经)")
        return results

    # 2. 东方财富（备选）
    results = _em_stock_fund_flow_rank(top_n)
    if results:
        logger.info(f"[资金流] 个股排行 Top{len(results)} (来源: 东方财富)")
        return results

    logger.warning("[资金流] 个股资金流排行所有数据源均失败")
    return []


def fetch_sector_fund_flow_rank(sector_type: str = "hy", top_n: int = 30) -> List[dict]:
    """获取板块资金流排行（新浪优先 → 东方财富备选）"""
    type_name = "行业" if sector_type == "hy" else "概念"

    # 1. 新浪财经
    results = _sina_sector_fund_flow_rank(sector_type, top_n)
    if results:
        logger.info(f"[资金流] {type_name}板块 Top{len(results)} (来源: 新浪财经)")
        return results

    # 2. 东方财富
    results = _em_sector_fund_flow_rank(sector_type, top_n)
    if results:
        logger.info(f"[资金流] {type_name}板块 Top{len(results)} (来源: 东方财富)")
        return results

    logger.warning(f"[资金流] {type_name}板块资金流所有数据源均失败")
    return []


def fetch_single_stock_fund_flow(stock_code: str) -> Optional[dict]:
    """获取单股资金流（新浪 → 东方财富 → 腾讯估算）"""
    # 1. 新浪财经
    result = _sina_single_stock_fund_flow(stock_code)
    if result:
        return result

    # 2. 东方财富
    result = _em_single_stock_fund_flow(stock_code)
    if result:
        return result

    # 3. 腾讯行情估算（兜底）
    result = _tencent_single_stock_fund_flow(stock_code)
    if result:
        logger.debug(f"[资金流] {stock_code} 使用腾讯估算数据")
        return result

    return None


def fetch_hot_stocks_for_candidate(max_price: float = 10.0, top_n: int = 20) -> List[dict]:
    """主力资金流入 + 低价热门股（换股候选）"""
    all_stocks = fetch_stock_fund_flow_rank(top_n=100)
    candidates = []
    for s in all_stocks:
        if s.get("price", 0) > max_price or s.get("price", 0) <= 0:
            continue
        if s.get("main_net", 0) <= 0:
            continue
        candidates.append(s)
        if len(candidates) >= top_n:
            break

    logger.info(f"[换股候选] 筛选 {len(candidates)} 只 (max_price={max_price})")
    return candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    print("=== 个股资金流 Top10 ===")
    stocks = fetch_stock_fund_flow_rank(top_n=10)
    for s in stocks:
        print(f"  [{s.get('source','?')}] {s['code']} {s['name']} "
              f"主力:{s['main_net']:.0f}万 占比:{s['main_net_pct']:.1f}% "
              f"价格:{s['price']:.2f}")

    print("\n=== 行业板块 Top5 ===")
    sectors = fetch_sector_fund_flow_rank("hy", top_n=5)
    for s in sectors:
        print(f"  {s['name']} 涨跌:{s['change_pct']:.1f}% 主力:{s['main_net']:.0f}万")

    print("\n=== 概念板块 Top5 ===")
    sectors = fetch_sector_fund_flow_rank("gn", top_n=5)
    for s in sectors:
        print(f"  {s['name']} 涨跌:{s['change_pct']:.1f}% 主力:{s['main_net']:.0f}万")

    print("\n=== 单股资金流 600519 ===")
    single = fetch_single_stock_fund_flow("600519")
    if single:
        print(f"  [{single.get('source','?')}] {single.get('name','')} "
              f"主力:{single['main_net']:.0f}万 占比:{single['main_net_pct']:.1f}%")

    print("\n=== 换股候选 ===")
    cands = fetch_hot_stocks_for_candidate(max_price=10, top_n=5)
    for s in cands:
        print(f"  [{s.get('source','?')}] {s['code']} {s['name']} "
              f"{s['price']:.2f}元 主力:{s['main_net']:.0f}万")
