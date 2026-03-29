# -*- coding: utf-8 -*-
"""
===================================
交易日历模块 (Issue #373)
===================================

职责：
1. 按市场（A股/港股/美股）判断当日是否为交易日
2. 按市场时区取“今日”日期，避免服务器 UTC 导致日期错误
3. 支持 per-stock 过滤：只分析当日开市市场的股票

依赖：exchange-calendars（可选，不可用时 fail-open）
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Set, Union

logger = logging.getLogger(__name__)

# Exchange-calendars availability
_XCALS_AVAILABLE = False
try:
    import exchange_calendars as xcals
    _XCALS_AVAILABLE = True
except ImportError:
    logger.warning(
        "exchange-calendars not installed; trading day check disabled. "
        "Run: pip install exchange-calendars"
    )

# Market -> exchange code (exchange-calendars)
MARKET_EXCHANGE = {"cn": "XSHG", "hk": "XHKG", "us": "XNYS"}

# Market -> IANA timezone for "today"
MARKET_TIMEZONE = {
    "cn": "Asia/Shanghai",
    "hk": "Asia/Hong_Kong",
    "us": "America/New_York",
}

DateLike = Union[date, datetime, str]


def _weekday_fallback(market: Optional[str], check_date: date) -> bool:
    """Fallback session check when exchange calendars are unavailable/outdated."""
    if market in {"cn", "hk", "us"}:
        return check_date.weekday() < 5
    return True


def get_market_for_stock(code: str) -> Optional[str]:
    """
    Infer market region for a stock code.

    Returns:
        'cn' | 'hk' | 'us' | None (None = unrecognized, fail-open: treat as open)
    """
    if not code or not isinstance(code, str):
        return None
    code = (code or "").strip().upper()

    from data_provider import is_us_stock_code, is_us_index_code, is_hk_stock_code

    if is_us_stock_code(code) or is_us_index_code(code):
        return "us"
    if is_hk_stock_code(code):
        return "hk"
    # A-share: 6-digit numeric
    if code.isdigit() and len(code) == 6:
        return "cn"
    return None


def is_market_open(market: str, check_date: date) -> bool:
    """
    Check if the given market is open on the given date.

    Fail-open: returns True if exchange-calendars unavailable or date out of range.

    Args:
        market: 'cn' | 'hk' | 'us'
        check_date: Date to check

    Returns:
        True if trading day (or fail-open), False otherwise
    """
    if not _XCALS_AVAILABLE:
        return _weekday_fallback(market, check_date)
    ex = MARKET_EXCHANGE.get(market)
    if not ex:
        return True
    try:
        cal = xcals.get_calendar(ex)
        if check_date < cal.first_session.date() or check_date > cal.last_session.date():
            return _weekday_fallback(market, check_date)
        session = datetime(check_date.year, check_date.month, check_date.day)
        return cal.is_session(session)
    except Exception as e:
        logger.warning("trading_calendar.is_market_open fail-open: %s", e)
        return _weekday_fallback(market, check_date)


def _coerce_date(value: DateLike) -> Optional[date]:
    """Convert common date-like inputs to ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def count_trading_days(
    start_date: DateLike,
    end_date: DateLike,
    market: str = "cn",
    *,
    include_start: bool = False,
    include_end: bool = True,
) -> Optional[int]:
    """Count trading sessions between two dates.

    The default holding-period convention is:
    - exclude the buy/open date
    - include the current/sell date when it is a trading session

    This keeps same-day positions at ``0`` while avoiding weekends/holidays.
    """
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if start is None or end is None:
        return None
    if end < start:
        return 0

    current = start
    sessions = 0
    while current <= end:
        if (current != start or include_start) and (current != end or include_end):
            if is_market_open(market, current):
                sessions += 1
        current += timedelta(days=1)
    return sessions


def count_stock_trading_days(
    stock_code: str,
    start_date: DateLike,
    end_date: DateLike,
    *,
    default_market: str = "cn",
    include_start: bool = False,
    include_end: bool = True,
) -> Optional[int]:
    """Count trading days using the stock's market, defaulting to A-shares."""
    market = get_market_for_stock(stock_code) or default_market
    return count_trading_days(
        start_date,
        end_date,
        market=market,
        include_start=include_start,
        include_end=include_end,
    )


def get_open_markets_today() -> Set[str]:
    """
    Get markets that are open today (by each market's local timezone).

    Returns:
        Set of market keys ('cn', 'hk', 'us') that are trading today
    """
    if not _XCALS_AVAILABLE:
        return {"cn", "hk", "us"}
    result: Set[str] = set()
    from zoneinfo import ZoneInfo
    for mkt, tz_name in MARKET_TIMEZONE.items():
        try:
            tz = ZoneInfo(tz_name)
            today = datetime.now(tz).date()
            if is_market_open(mkt, today):
                result.add(mkt)
        except Exception as e:
            logger.warning("get_open_markets_today fail-open for %s: %s", mkt, e)
            result.add(mkt)
    return result


def compute_effective_region(
    config_region: str, open_markets: Set[str]
) -> Optional[str]:
    """
    Compute effective market review region given config and open markets.

    Args:
        config_region: From MARKET_REVIEW_REGION ('cn' | 'us' | 'both')
        open_markets: Markets open today

    Returns:
        None: caller uses config default (check disabled)
        '': all relevant markets closed, skip market review
        'cn' | 'us' | 'both': effective subset for today
    """
    if config_region not in ("cn", "us", "both"):
        config_region = "cn"
    if config_region == "cn":
        return "cn" if "cn" in open_markets else ""
    if config_region == "us":
        return "us" if "us" in open_markets else ""
    # both
    parts = []
    if "cn" in open_markets:
        parts.append("cn")
    if "us" in open_markets:
        parts.append("us")
    if not parts:
        return ""
    return "both" if len(parts) == 2 else parts[0]
