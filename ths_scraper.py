# -*- coding: utf-8 -*-
"""
ths_scraper.py — 资金流向数据采集（东方财富接口替代同花顺）

提供个股/板块资金流向排行，供 market_monitor.py 和 macro_data_collector.py 使用。
使用东方财富免费接口（无需登录，无限频限制）。

导出函数:
  fetch_stock_fund_flow_rank()       → Top N 个股资金流排行
  fetch_sector_fund_flow_rank(type)  → 板块资金流排行（行业hy/概念gn）
  fetch_single_stock_fund_flow(code) → 单只股票资金流明细
  fetch_hot_stocks_for_candidate()   → 热门换股候选
"""
import logging
import re
import requests
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/",
}


def _safe_float(val, default=0.0):
    try:
        return float(val) if val not in (None, "", "-") else default
    except (ValueError, TypeError):
        return default


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 个股资金流排行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_stock_fund_flow_rank(top_n: int = 50) -> List[dict]:
    """获取个股主力资金流入排行（东方财富）

    返回格式：[{code, name, main_net, main_net_pct, super_large_net,
               large_net, price, change_pct, turnover_rate, source}, ...]
    """
    # 东方财富个股资金流排行接口
    url = (
        "https://push2.eastmoney.com/api/qt/clist/get?"
        "fid=f62&po=1&pn=1&pz={pz}&np=1&fltt=2&invt=2"
        "&ut=b2884a393a59ad64002292a3e90d46a5"
        "&fields=f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f8"
        "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
    ).format(pz=min(top_n, 100))

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()
        items = data.get("data", {}).get("diff", [])
        if not items:
            logger.warning("[资金流] 东方财富个股资金流返回空数据")
            return []

        results = []
        for item in items[:top_n]:
            results.append({
                "code": str(item.get("f12", "")),
                "name": str(item.get("f14", "")),
                "price": _safe_float(item.get("f2")),
                "change_pct": _safe_float(item.get("f3")),
                "main_net": round(_safe_float(item.get("f62")) / 10000, 2),  # 元→万元
                "main_net_pct": _safe_float(item.get("f184")),
                "super_large_net": round(_safe_float(item.get("f66")) / 10000, 2),
                "large_net": round(_safe_float(item.get("f72")) / 10000, 2),
                "turnover_rate": _safe_float(item.get("f8")),
                "source": "eastmoney",
            })
        logger.info(f"[资金流] 获取个股资金流排行 Top{len(results)}")
        return results

    except Exception as e:
        logger.warning(f"[资金流] 东方财富个股资金流获取失败: {e}")
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 板块资金流排行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_sector_fund_flow_rank(sector_type: str = "hy", top_n: int = 30) -> List[dict]:
    """获取板块资金流排行

    Args:
        sector_type: "hy" 行业板块, "gn" 概念板块
        top_n: 返回条数

    返回格式：[{name, change_pct, main_net, main_net_pct}, ...]
    """
    # 东方财富板块资金流接口
    # hy=行业 m:90+t:2, gn=概念 m:90+t:3
    fs = "m:90+t:2" if sector_type == "hy" else "m:90+t:3"
    url = (
        "https://push2.eastmoney.com/api/qt/clist/get?"
        "fid=f62&po=1&pn=1&pz={pz}&np=1&fltt=2&invt=2"
        "&ut=b2884a393a59ad64002292a3e90d46a5"
        "&fields=f12,f14,f3,f62,f184"
        "&fs={fs}"
    ).format(pz=min(top_n, 80), fs=fs)

    type_name = "行业" if sector_type == "hy" else "概念"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()
        items = data.get("data", {}).get("diff", [])
        if not items:
            logger.warning(f"[资金流] 东方财富{type_name}板块资金流返回空数据")
            return []

        results = []
        for item in items[:top_n]:
            results.append({
                "name": str(item.get("f14", "")),
                "change_pct": _safe_float(item.get("f3")),
                "main_net": round(_safe_float(item.get("f62")) / 10000, 2),  # 元→万元
                "main_net_pct": _safe_float(item.get("f184")),
            })
        logger.info(f"[资金流] 获取{type_name}板块资金流排行 Top{len(results)}")
        return results

    except Exception as e:
        logger.warning(f"[资金流] 东方财富{type_name}板块资金流获取失败: {e}")
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 单只股票资金流
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_single_stock_fund_flow(stock_code: str) -> Optional[dict]:
    """获取单只股票当日资金流明细

    返回格式：{code, name, main_net, main_net_pct, super_large_net, large_net,
              medium_net, small_net, source}
    """
    # 东方财富单股资金流接口
    market = "1" if stock_code.startswith("6") else "0"
    secid = f"{market}.{stock_code}"
    url = (
        "https://push2.eastmoney.com/api/qt/stock/get?"
        f"secid={secid}&ut=b2884a393a59ad64002292a3e90d46a5"
        f"&fields=f12,f14,f62,f184,f66,f69,f72,f75,f78,f81"
    )

    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        data = r.json().get("data", {})
        if not data:
            return None

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
        logger.debug(f"[资金流] 单股 {stock_code} 资金流获取失败: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 热门换股候选
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_hot_stocks_for_candidate(max_price: float = 10.0, top_n: int = 20) -> List[dict]:
    """获取主力资金流入 + 低价热门股作为换股候选

    筛选条件：主力净流入为正 + 价格<=max_price
    """
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

    logger.info(f"[换股候选] 主力流入+低价 筛选出 {len(candidates)} 只 (max_price={max_price})")
    return candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 测试
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=== 个股资金流 Top10 ===")
    stocks = fetch_stock_fund_flow_rank(top_n=10)
    for s in stocks:
        print(f"  {s['code']} {s['name']} 主力净流入:{s['main_net']:.0f}万 "
              f"占比:{s['main_net_pct']:.1f}% 价格:{s['price']:.2f}")

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
        print(f"  {single['name']} 主力:{single['main_net']:.0f}万 占比:{single['main_net_pct']:.1f}%")

    print("\n=== 换股候选 ===")
    cands = fetch_hot_stocks_for_candidate(max_price=10, top_n=5)
    for s in cands:
        print(f"  {s['code']} {s['name']} {s['price']:.2f}元 主力:{s['main_net']:.0f}万")
