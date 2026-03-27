"""
macro_data_collector.py
Lightweight realtime quote helpers used by portfolio/rebalance modules.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0",
}


def _stock_code_to_tencent(code: str) -> str:
    c = str(code or "").strip().lower()
    if c.startswith(("sh", "sz")):
        return c
    if not c.isdigit():
        return c
    if c.startswith(("6", "9", "5")):
        return f"sh{c}"
    return f"sz{c}"


def _parse_tencent_line(line: str) -> Dict:
    # Example: v_sh600000="1~浦发银行~600000~10.23~..."
    parts = line.split("~")
    if len(parts) < 40:
        return {}
    code = parts[2].strip()
    name = parts[1].strip()
    try:
        price = float(parts[3] or 0)
    except ValueError:
        price = 0.0
    try:
        change_pct = float(parts[32] or 0)
    except ValueError:
        change_pct = 0.0
    try:
        turnover = float(parts[38] or 0)
    except ValueError:
        turnover = 0.0
    return {
        "code": code,
        "name": name or code,
        "price": price,
        "change_pct": change_pct,
        "turnover": turnover,
    }


def _fetch_tencent_quote(codes: Iterable[str]) -> Dict[str, Dict]:
    symbols = [str(c).strip() for c in (codes or []) if str(c).strip()]
    if not symbols:
        return {}
    url = f"http://qt.gtimg.cn/q={','.join(symbols)}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    out: Dict[str, Dict] = {}
    for line in resp.text.strip().splitlines():
        if "~" not in line:
            continue
        parsed = _parse_tencent_line(line)
        if not parsed.get("code"):
            continue
        tc = _stock_code_to_tencent(parsed["code"])
        out[tc] = parsed
    return out


def fetch_stock_news(stock_code: str, stock_name: str = "", limit: int = 5) -> List[Dict]:
    """
    Placeholder deterministic news function for local/offline path.
    Returns empty list when no news provider is configured.
    """
    _ = (stock_code, stock_name, limit)
    return []


def normalize_stock_code(text: str) -> str:
    m = re.search(r"\d{6}", str(text or ""))
    return m.group(0) if m else str(text or "").strip()
