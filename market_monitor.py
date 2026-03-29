"""
market_monitor.py — 盘中全功能监控系统

一天的完整流程:
  09:15 开盘前 — 全市场扫描，输出今日候选股
  09:15-09:25 — 集合竞价资金监控，观察候选股/持仓的竞价强弱
  09:30-11:30 / 13:00-15:00 — 每N分钟监控异动
  15:00-15:30 — 收盘后自动回测+调仓分析
  每天积累 — 资金流/板块数据存入SQLite，用于多日趋势判断

使用:
  python main.py --monitor                  # 启动全功能监控
  python main.py --monitor --interval 5     # 5分钟检查一次
  python market_monitor.py --test           # 测试所有模块
  python market_monitor.py --backtest       # 手动回测
  python market_monitor.py --mainline       # 查看当前主线板块
"""
import argparse, copy, json, logging, os, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
TZ_CN = timezone(timedelta(hours=8))
AUCTION_POLL_SECONDS = 30
AUCTION_WATCHLIST_LIMIT = 15
AUCTION_ALERT_TOP_N = 3
CN_INDEX_CODES = [
    ("sh000001", "上证指数"),
    ("sz399001", "深证成指"),
    ("sz399006", "创业板指"),
]

_RUNTIME_CACHE: Dict[str, dict] = {}
_RUNTIME_CACHE_LOCK = Lock()


def _cache_key(*parts: object) -> str:
    normalized = []
    for part in parts:
        if isinstance(part, (list, tuple, set)):
            normalized.append(tuple(str(item) for item in part))
        else:
            normalized.append(part)
    return repr(tuple(normalized))


def _get_or_load_cached(cache_key: str, ttl_seconds: float, loader):
    now = time.time()
    with _RUNTIME_CACHE_LOCK:
        entry = _RUNTIME_CACHE.get(cache_key)
        if entry and (now - entry["ts"]) <= ttl_seconds:
            return copy.deepcopy(entry["value"])

    value = loader()
    with _RUNTIME_CACHE_LOCK:
        _RUNTIME_CACHE[cache_key] = {"ts": time.time(), "value": copy.deepcopy(value)}
    return value


def _fetch_quotes_cached(tc_codes: list, timeout: int = 8, ttl_seconds: float = 5.0) -> dict:
    from macro_data_collector import _fetch_tencent_quote

    normalized_codes = tuple(sorted(str(code) for code in tc_codes if code))
    if not normalized_codes:
        return {}
    return _get_or_load_cached(
        _cache_key("quotes", normalized_codes, timeout),
        ttl_seconds,
        lambda: _fetch_tencent_quote(list(normalized_codes), timeout=timeout),
    )


def _fetch_sector_flow_cached(
    sector_type: str = "hy",
    top_n: int = 5,
    ttl_seconds: float = 20.0,
) -> list:
    from ths_scraper import fetch_sector_fund_flow_rank

    return _get_or_load_cached(
        _cache_key("sector_flow", sector_type, top_n),
        ttl_seconds,
        lambda: fetch_sector_fund_flow_rank(sector_type, top_n=top_n),
    )


def _fetch_hot_candidates_cached(
    max_price: float = 15.0,
    top_n: int = 5,
    ttl_seconds: float = 20.0,
) -> list:
    from ths_scraper import fetch_hot_stocks_for_candidate

    return _get_or_load_cached(
        _cache_key("hot_candidates", max_price, top_n),
        ttl_seconds,
        lambda: fetch_hot_stocks_for_candidate(max_price=max_price, top_n=top_n),
    )


def _fetch_financial_news_cached(ttl_seconds: float = 45.0) -> dict:
    from macro_data_collector import fetch_financial_news

    return _get_or_load_cached(
        _cache_key("financial_news"),
        ttl_seconds,
        fetch_financial_news,
    )


def _fetch_trump_news_cached(ttl_seconds: float = 45.0) -> list:
    from macro_data_collector import fetch_trump_news

    return _get_or_load_cached(
        _cache_key("trump_news"),
        ttl_seconds,
        fetch_trump_news,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 盘前扫描（09:15 执行）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_pre_market_scan() -> list:
    """盘前全市场扫描 + 存储 + 积累板块/资金数据"""
    from market_scanner import scan_market
    from ths_scraper import fetch_stock_fund_flow_rank, fetch_sector_fund_flow_rank
    from data_store import (
        save_scan_results, save_fund_flow_batch, save_sector_flow_batch,
    )

    today = datetime.now(TZ_CN).strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"盘前扫描 {today}")
    logger.info("=" * 60)

    # 1. 全市场扫描（趋势模式）
    logger.info("[盘前 1/4] 全市场趋势扫描...")
    candidates = scan_market(max_price=10.0, min_turnover=2.0, top_n=30, mode="trend")
    save_scan_results(candidates, scan_date=today, mode="trend")

    # 1.5 副龙头扫描（短期题材+低位+资金流入）
    logger.info("[盘前 1.5/4] 副龙头扫描...")
    try:
        sub_dragons = scan_market(max_price=15.0, min_turnover=3.0, top_n=20, mode="sub_dragon")
        save_scan_results(sub_dragons, scan_date=today, mode="sub_dragon")
        if sub_dragons:
            logger.info(f"  副龙头候选 Top5:")
            for s in sub_dragons[:5]:
                logger.info(
                    f"    {s['code']} {s['name']} {s['price']:.2f}元 "
                    f"主力:{s.get('main_net', 0):.0f}万 得分:{s['tech_score']}"
                )
    except Exception as e:
        logger.warning(f"副龙头扫描失败: {e}")

    # 2. 积累个股资金流（同花顺 Top50）
    logger.info("[盘前 2/3] 积累个股资金流...")
    try:
        fund_stocks = fetch_stock_fund_flow_rank()
        save_fund_flow_batch(fund_stocks, trade_date=today)
    except Exception as e:
        logger.warning(f"个股资金流积累失败: {e}")

    # 3. 积累板块资金流
    logger.info("[盘前 3/3] 积累板块资金流...")
    try:
        hy_sectors = fetch_sector_fund_flow_rank(sector_type="hy")
        save_sector_flow_batch(hy_sectors, sector_type="hy", trade_date=today)
        gn_sectors = fetch_sector_fund_flow_rank(sector_type="gn")
        save_sector_flow_batch(gn_sectors, sector_type="gn", trade_date=today)
    except Exception as e:
        logger.warning(f"板块资金流积累失败: {e}")

    # 输出候选摘要
    if candidates:
        logger.info(f"\n今日候选 Top10:")
        for s in candidates[:10]:
            logger.info(
                f"  {s['code']} {s['name']} {s['price']:.2f}元 "
                f"涨跌:{s['change_pct']:.1f}% MA:{s['ma_trend']} "
                f"MACD:{s['macd_signal']} 得分:{s['tech_score']}"
            )

    return candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 主线板块分析（基于多日积累数据）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_mainline() -> dict:
    """分析当前主线板块（连续多日资金流入的板块）"""
    from data_store import get_sector_mainline, get_consecutive_inflow_stocks

    mainline_sectors = get_sector_mainline(min_days=2)
    continuous_stocks = get_consecutive_inflow_stocks(min_days=2, min_total=300)

    result = {
        "mainline_sectors": mainline_sectors[:5],
        "continuous_inflow_stocks": continuous_stocks[:10],
    }

    if mainline_sectors:
        logger.info("当前主线板块（连续资金流入）:")
        for s in mainline_sectors[:5]:
            logger.info(
                f"  {s['sector_name']} — 连续{s['inflow_days']}天流入 "
                f"累计:{s['total_net']:.0f}万 均涨:{s['avg_chg']:.1f}%"
            )

    if continuous_stocks:
        logger.info("连续主力流入个股:")
        for s in continuous_stocks[:5]:
            logger.info(
                f"  {s['code']} {s['name']} — 连续{s['inflow_days']}天 "
                f"累计:{s['total_net']:.0f}万"
            )

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 盘中异动检测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_market_anomaly(
    index_quotes: Optional[dict] = None,
    holdings: Optional[list] = None,
    trump_news: Optional[list] = None,
) -> dict:
    """检测指数/持仓/新闻异动"""
    anomalies = []

    # 指数异动
    try:
        quotes = index_quotes if isinstance(index_quotes, dict) else _fetch_quotes_cached(
            [code for code, _ in CN_INDEX_CODES],
            timeout=8,
            ttl_seconds=5.0,
        )
        for tc, name in [("sh000001", "上证指数"), ("sz399001", "深证成指"), ("sz399006", "创业板指")]:
            q = quotes.get(tc, {})
            chg = q.get("change_pct", 0)
            if abs(chg) > 2.0:
                anomalies.append({
                    "type": "index_crash" if chg < 0 else "index_surge",
                    "severity": "high",
                    "detail": f"{name} 涨跌幅 {chg}%",
                })
            elif abs(chg) > 1.0:
                anomalies.append({
                    "type": "index_move", "severity": "medium",
                    "detail": f"{name} 涨跌幅 {chg}%",
                })
    except Exception as e:
        logger.warning(f"指数异动检测失败: {e}")

    # 持仓股异动
    try:
        from macro_data_collector import _stock_code_to_tencent
        if holdings is None:
            from portfolio_manager import load_portfolio

            portfolio = load_portfolio()
            holdings = portfolio.get("holdings", [])
        if holdings:
            tc_codes = [_stock_code_to_tencent(h["code"]) for h in holdings if h.get("code")]
            quotes = _fetch_quotes_cached(tc_codes, timeout=8, ttl_seconds=5.0)
            for h in holdings:
                tc = _stock_code_to_tencent(h["code"])
                q = quotes.get(tc, {})
                chg = q.get("change_pct", 0)
                if chg < -5:
                    anomalies.append({
                        "type": "holding_crash", "severity": "critical",
                        "detail": f"持仓 {h.get('name','')}({h['code']}) 跌{chg}%",
                        "code": h["code"],
                    })
                elif chg < -3:
                    anomalies.append({
                        "type": "holding_drop", "severity": "high",
                        "detail": f"持仓 {h.get('name','')}({h['code']}) 跌{chg}%",
                        "code": h["code"],
                    })
    except Exception as e:
        logger.warning(f"持仓异动检测失败: {e}")

    # 特朗普新闻
    try:
        if trump_news is None:
            trump_news = _fetch_trump_news_cached(ttl_seconds=45.0)
    except Exception:
        trump_news = []
    sensitive = [n for n in trump_news if n.get("is_sensitive")]
    if sensitive:
        anomalies.append({
            "type": "trump_sensitive",
            "severity": "high" if len(sensitive) >= 3 else "medium",
            "detail": f"检测到 {len(sensitive)} 条特朗普敏感新闻",
            "news": sensitive[:3],
        })

    return {
        "timestamp": datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S"),
        "anomaly_count": len(anomalies),
        "has_critical": any(a["severity"] == "critical" for a in anomalies),
        "has_high": any(a["severity"] == "high" for a in anomalies),
        "anomalies": anomalies,
        "trump_news": trump_news,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 推送
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _get_current_holdings() -> list:
    """从持仓数据库获取当前持仓（供风控检查用）。"""
    try:
        from src.storage import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT code, name, cost_price, current_price, shares,
                       buy_date, pnl_pct, market_value
                FROM portfolio WHERE shares > 0
            """)).fetchall()
            return [dict(r._mapping) for r in rows]
    except Exception:
        pass
    # 备用：从 trade_journal 推算持仓
    try:
        from trade_journal import _conn
        conn = _conn()
        rows = conn.execute("""
            SELECT code, name, price as cost_price, shares, trade_date as buy_date
            FROM trade_log
            WHERE trade_type = 'buy'
            AND code NOT IN (
                SELECT code FROM trade_log WHERE trade_type = 'sell'
                GROUP BY code HAVING SUM(shares) >= (
                    SELECT SUM(shares) FROM trade_log t2
                    WHERE t2.code = trade_log.code AND t2.trade_type = 'buy'
                )
            )
            ORDER BY trade_date DESC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows] if rows else []
    except Exception:
        return []


def send_alert(message: str):
    try:
        from src.notification import NotificationService
        notifier = NotificationService()
        if notifier.is_available():
            notifier.send(message)
    except Exception as e:
        logger.error(f"推送失败: {e}")


def format_anomaly_alert(result: dict) -> Optional[str]:
    if not result.get("anomalies"):
        return None
    lines = [f"🚨 **盘中异动预警**", f"⏰ {result['timestamp']}", ""]
    for a in result["anomalies"]:
        emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(a["severity"], "⚪")
        lines.append(f"{emoji} {a['detail']}")
        if a.get("type") == "trump_sensitive" and "news" in a:
            for n in a["news"][:2]:
                lines.append(f"  📰 {n.get('title', '')[:60]}")
    lines.extend(["", "💡 建议关注盘面，必要时手动干预"])
    return "\n".join(lines)


def format_scan_alert(candidates: list) -> str:
    lines = [f"📡 **盘前扫描完成** ({len(candidates)} 只候选)", ""]
    for s in candidates[:10]:
        lines.append(
            f"  {s['code']} {s['name']} {s['price']:.2f}元 "
            f"| {s['ma_trend']} | {s['macd_signal']} | 得分:{s['tech_score']}"
        )
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 集合竞价监控
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_opening_auction_watchlist(candidates: list) -> List[dict]:
    """优先观察盘前候选股，同时兼顾当前持仓的竞价承接。"""
    watchlist: List[dict] = []
    seen = set()

    def _add_stock(code: str, name: str, reason: str):
        normalized = str(code or "").strip()
        if not normalized or normalized in seen:
            return
        watchlist.append({
            "code": normalized,
            "name": str(name or "").strip(),
            "reason": reason,
        })
        seen.add(normalized)

    for stock in candidates[:AUCTION_WATCHLIST_LIMIT]:
        _add_stock(stock.get("code"), stock.get("name"), "candidate")

    for holding in _get_current_holdings():
        _add_stock(holding.get("code"), holding.get("name"), "holding")

    return watchlist[:AUCTION_WATCHLIST_LIMIT]


def _extract_opening_auction_snapshot(stock: dict, quote: dict) -> Optional[dict]:
    if not quote:
        return None

    bid_stack = sum(int(quote.get(f"bid{i}_volume", 0) or 0) for i in range(1, 6))
    ask_stack = sum(int(quote.get(f"ask{i}_volume", 0) or 0) for i in range(1, 6))
    auction_amount = float(quote.get("amount", 0) or 0)
    auction_price = (
        float(quote.get("price", 0) or 0)
        or float(quote.get("bid1_price", 0) or 0)
        or float(quote.get("open", 0) or 0)
    )
    prev_close = float(quote.get("prev_close", 0) or 0)
    if auction_price <= 0 and auction_amount <= 0 and bid_stack <= 0 and ask_stack <= 0:
        return None

    total_stack = bid_stack + ask_stack
    imbalance_pct = round((bid_stack - ask_stack) / total_stack * 100, 2) if total_stack > 0 else 0.0
    change_pct = float(quote.get("change_pct", 0) or 0)
    if change_pct == 0 and auction_price > 0 and prev_close > 0:
        change_pct = round((auction_price / prev_close - 1) * 100, 2)

    market_cap = float(quote.get("market_cap", 0) or 0)
    amount_ratio_pct = (
        round(auction_amount / (market_cap * 10000) * 100, 4)
        if market_cap > 0 else 0.0
    )

    return {
        "code": stock.get("code"),
        "name": stock.get("name") or quote.get("name", ""),
        "reason": stock.get("reason", ""),
        "auction_price": round(auction_price, 2) if auction_price else 0.0,
        "change_pct": round(change_pct, 2),
        "auction_amount": round(auction_amount, 2),   # 万元
        "bid_stack": bid_stack,
        "ask_stack": ask_stack,
        "imbalance_pct": imbalance_pct,
        "amount_ratio_pct": amount_ratio_pct,
    }


def collect_opening_auction_sample(watchlist: list) -> Optional[dict]:
    if not watchlist:
        return None

    from macro_data_collector import (
        _stock_code_to_tencent,
    )

    code_map = {
        stock["code"]: _stock_code_to_tencent(stock["code"])
        for stock in watchlist
        if stock.get("code")
    }
    if not code_map:
        return None

    quotes = _fetch_quotes_cached(list(code_map.values()), timeout=8, ttl_seconds=5.0)
    stocks = []
    for stock in watchlist:
        code = stock.get("code")
        tc_code = code_map.get(code)
        snapshot = _extract_opening_auction_snapshot(stock, quotes.get(tc_code, {}))
        if snapshot:
            stocks.append(snapshot)

    if not stocks:
        return None

    stocks.sort(key=lambda item: (item["auction_amount"], abs(item["imbalance_pct"])), reverse=True)
    return {
        "timestamp": _now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": stocks,
    }


def _should_poll_opening_auction(state: dict) -> bool:
    if not state.get("watchlist") or not is_opening_auction_window():
        return False
    last_polled_at = state.get("last_polled_at")
    if last_polled_at is None:
        return True
    return (_now_cn() - last_polled_at).total_seconds() >= AUCTION_POLL_SECONDS


def maybe_record_opening_auction_sample(state: dict) -> Optional[dict]:
    if not _should_poll_opening_auction(state):
        return None

    sample = collect_opening_auction_sample(state.get("watchlist", []))
    state["last_polled_at"] = _now_cn()
    if not sample:
        return None

    samples = state.setdefault("samples", [])
    samples.append(sample)
    if len(samples) > 32:
        del samples[:-32]

    top_stock = sample["stocks"][0]
    logger.info(
        "[竞价] 采样 %s 只，Top: %s %s 竞价额 %.0f万 盘口差 %+0.1f%%",
        len(sample["stocks"]),
        top_stock["code"],
        top_stock["name"],
        top_stock["auction_amount"],
        top_stock["imbalance_pct"],
    )
    return sample


def summarize_opening_auction_state(state: dict) -> dict:
    samples = state.get("samples") or []
    watchlist = state.get("watchlist") or []
    summary = {
        "timestamp": _now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_count": len(samples),
        "watchlist_size": len(watchlist),
        "strong": [],
        "weak": [],
        "stocks": [],
    }
    if not samples:
        return summary

    first_by_code: Dict[str, dict] = {}
    latest_by_code: Dict[str, dict] = {}
    for sample in samples:
        for stock in sample.get("stocks", []):
            code = stock.get("code")
            if not code:
                continue
            first_by_code.setdefault(code, stock)
            latest_by_code[code] = stock

    ranked = []
    for code, latest in latest_by_code.items():
        first = first_by_code.get(code, latest)
        amount_delta = round(float(latest.get("auction_amount", 0) or 0) - float(first.get("auction_amount", 0) or 0), 2)
        imbalance_delta = round(float(latest.get("imbalance_pct", 0) or 0) - float(first.get("imbalance_pct", 0) or 0), 2)
        change_pct = float(latest.get("change_pct", 0) or 0)
        auction_amount = float(latest.get("auction_amount", 0) or 0)
        amount_ratio_pct = float(latest.get("amount_ratio_pct", 0) or 0)
        imbalance_pct = float(latest.get("imbalance_pct", 0) or 0)
        flow_score = round(
            change_pct * 1.2
            + imbalance_pct * 0.10
            + min(amount_ratio_pct * 20, 3)
            + max(min(amount_delta / 1000, 2), -2),
            2,
        )
        is_strong = (
            (change_pct >= 0.5 and imbalance_pct >= 10 and (auction_amount >= 1000 or amount_ratio_pct >= 0.03))
            or (len(samples) > 1 and change_pct >= 0 and imbalance_pct >= 5 and amount_delta >= 500 and imbalance_delta >= 5)
        )
        is_weak = (
            (change_pct <= -0.5 and imbalance_pct <= -10 and (auction_amount >= 1000 or amount_ratio_pct >= 0.03))
            or (len(samples) > 1 and change_pct <= 0 and imbalance_pct <= -5 and amount_delta >= 500 and imbalance_delta <= -5)
        )
        ranked.append({
            **latest,
            "amount_delta": amount_delta,
            "imbalance_delta": imbalance_delta,
            "flow_score": flow_score,
            "is_strong": is_strong,
            "is_weak": is_weak,
        })

    ranked.sort(key=lambda item: item["flow_score"], reverse=True)
    summary["timestamp"] = samples[-1]["timestamp"]
    summary["stocks"] = ranked
    summary["strong"] = sorted(
        [item for item in ranked if item["is_strong"]],
        key=lambda item: (item["flow_score"], item["auction_amount"], item["imbalance_pct"]),
        reverse=True,
    )[:AUCTION_ALERT_TOP_N]
    summary["weak"] = sorted(
        [item for item in ranked if item["is_weak"]],
        key=lambda item: (item["flow_score"], item["change_pct"], item["imbalance_pct"]),
    )[:AUCTION_ALERT_TOP_N]
    return summary


def format_opening_auction_alert(summary: dict) -> Optional[str]:
    strong = summary.get("strong") or []
    weak = summary.get("weak") or []
    if not strong and not weak:
        return None

    lines = [
        "🕘 **集合竞价资金监控**",
        f"⏰ {summary['timestamp']}",
        f"📌 观察池 {summary.get('watchlist_size', 0)} 只 | 采样 {summary.get('sample_count', 0)} 次",
        "",
    ]

    if strong:
        lines.append("📈 竞价偏强:")
        for stock in strong:
            lines.append(
                f"  {stock['code']} {stock['name']} {stock['change_pct']:+.2f}% "
                f"| 竞价额:{stock['auction_amount']:.0f}万 "
                f"| 盘口差:{stock['imbalance_pct']:+.0f}% "
                f"| 增量:{stock['amount_delta']:+.0f}万"
            )

    if weak:
        if strong:
            lines.append("")
        lines.append("📉 竞价偏弱:")
        for stock in weak:
            lines.append(
                f"  {stock['code']} {stock['name']} {stock['change_pct']:+.2f}% "
                f"| 竞价额:{stock['auction_amount']:.0f}万 "
                f"| 盘口差:{stock['imbalance_pct']:+.0f}% "
                f"| 增量:{stock['amount_delta']:+.0f}万"
            )

    lines.extend(["", "💡 竞价信号只用于开盘前过滤，仍需结合开盘承接与板块联动确认。"])
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 日内节点判断
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _get_intraday_checkpoint_meta(checkpoint: str) -> dict:
    mapping = {
        "morning_review": {
            "title": "⏱️ **10:15 早盘复判**",
            "positive": "上午偏强",
            "neutral": "上午震荡",
            "negative": "上午偏弱",
            "advice": "重点看主线是否继续扩散、早盘拉升是否有承接。",
        },
        "afternoon_review": {
            "title": "🧭 **12:30 午后方向判断**",
            "positive": "午后偏强",
            "neutral": "午后震荡",
            "negative": "午后偏弱",
            "advice": "重点看指数共振、资金回流方向，以及午后是否有二次分歧。",
        },
    }
    if checkpoint not in mapping:
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    return mapping[checkpoint]


def build_intraday_checkpoint_summary(
    checkpoint: str,
    index_quotes: dict,
    sectors: list,
    anomaly_result: Optional[dict] = None,
) -> dict:
    meta = _get_intraday_checkpoint_meta(checkpoint)
    anomalies = (anomaly_result or {}).get("anomalies") or []

    index_rows = []
    changes = []
    for tc_code, name in CN_INDEX_CODES:
        quote = index_quotes.get(tc_code, {}) if isinstance(index_quotes, dict) else {}
        change_pct = float(quote.get("change_pct", 0) or 0)
        index_rows.append({
            "code": tc_code,
            "name": name,
            "change_pct": round(change_pct, 2),
        })
        changes.append(change_pct)

    avg_change = round(sum(changes) / len(changes), 2) if changes else 0.0
    positive_indexes = sum(1 for change in changes if change >= 0.6)
    negative_indexes = sum(1 for change in changes if change <= -0.6)

    top_sectors = []
    positive_sectors = 0
    for sector in (sectors or [])[:3]:
        main_net = float(sector.get("main_net", 0) or 0)
        if main_net > 0:
            positive_sectors += 1
        top_sectors.append({
            "name": sector.get("name", ""),
            "change_pct": round(float(sector.get("change_pct", 0) or 0), 2),
            "main_net": round(main_net, 2),
        })

    high_risk_count = sum(
        1 for anomaly in anomalies
        if anomaly.get("severity") in {"critical", "high"}
    )

    if (
        high_risk_count > 0
        or avg_change <= -0.8
        or (negative_indexes >= 2 and positive_sectors == 0)
    ):
        bias = "negative"
    elif (
        avg_change >= 0.8
        or (positive_indexes >= 2 and positive_sectors >= 2)
    ):
        bias = "positive"
    else:
        bias = "neutral"

    return {
        "checkpoint": checkpoint,
        "timestamp": _now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "title": meta["title"],
        "bias": bias,
        "bias_label": meta[bias],
        "advice": meta["advice"],
        "index_rows": index_rows,
        "avg_change_pct": avg_change,
        "top_sectors": top_sectors,
        "high_risk_count": high_risk_count,
        "anomalies": anomalies[:3],
    }


def compose_market_decision_context(
    summary: dict,
    sectors: list,
    anomaly_result: Optional[dict] = None,
    hot_candidates: Optional[list] = None,
    auction_summary: Optional[dict] = None,
    financial_news: Optional[dict] = None,
    stock_code: str = "",
    stock_name: str = "",
    stock_sector: str = "",
) -> dict:
    """Compose a reusable market context for agent prompts and adaptive rules."""

    anomalies = list((anomaly_result or {}).get("anomalies") or summary.get("anomalies") or [])
    trump_news = list((anomaly_result or {}).get("trump_news") or [])
    sensitive_trump_news = [item for item in trump_news if item.get("is_sensitive")]
    sensitive_financial_news = list((financial_news or {}).get("sensitive") or [])

    top_sectors = []
    for rank, sector in enumerate(sectors or [], start=1):
        top_sectors.append({
            "name": str(sector.get("name", "") or "").strip(),
            "rank": rank,
            "change_pct": round(float(sector.get("change_pct", 0) or 0), 2),
            "main_net": round(float(sector.get("main_net", 0) or 0), 2),
        })

    bias = str(summary.get("bias", "neutral") or "neutral").strip().lower()
    avg_change_pct = float(summary.get("avg_change_pct", 0) or 0)
    high_risk_count = int(summary.get("high_risk_count", 0) or 0)
    positive_sector_count = sum(1 for sector in top_sectors if float(sector.get("main_net", 0) or 0) > 0)

    headline_pressure = len(sensitive_trump_news) * 2 + len(sensitive_financial_news)
    if headline_pressure >= 4:
        macro_risk_level = "high"
    elif headline_pressure >= 2:
        macro_risk_level = "medium"
    else:
        macro_risk_level = "low"

    hot_money_targets = []
    for candidate in hot_candidates or []:
        change_pct = float(candidate.get("change_pct", 0) or 0)
        main_net = float(candidate.get("main_net", 0) or 0)
        if main_net <= 0:
            continue
        hot_money_targets.append({
            "code": str(candidate.get("code", "") or "").strip(),
            "name": str(candidate.get("name", "") or "").strip(),
            "change_pct": round(change_pct, 2),
            "main_net": round(main_net, 2),
        })

    active_probe_targets = [
        target
        for target in hot_money_targets
        if 1.0 <= float(target.get("change_pct", 0) or 0) <= 9.5
        and float(target.get("main_net", 0) or 0) >= 2000
    ]
    if len(active_probe_targets) >= 2:
        hot_money_signal = "active"
    elif len(active_probe_targets) == 1:
        hot_money_signal = "constructive"
    else:
        hot_money_signal = "quiet"

    quant_pressure_score = 0
    if bias == "negative":
        quant_pressure_score += 2
    if avg_change_pct <= -0.8:
        quant_pressure_score += 1
    if high_risk_count >= 2:
        quant_pressure_score += 1
    if positive_sector_count > 0 and avg_change_pct < -0.5:
        quant_pressure_score += 1
    if headline_pressure >= 4:
        quant_pressure_score += 1
    quant_signal = "high" if quant_pressure_score >= 4 else "medium" if quant_pressure_score >= 2 else "low"

    market_score = (
        {"positive": 0.9, "neutral": 0.0, "negative": -0.9}.get(bias, 0.0)
        + max(min(avg_change_pct / 1.2, 1.2), -1.2)
        + min(positive_sector_count * 0.15, 0.45)
        - min(high_risk_count * 0.35, 1.05)
        - (0.45 if macro_risk_level == "high" else 0.2 if macro_risk_level == "medium" else 0.0)
        - (0.35 if quant_signal == "high" else 0.12 if quant_signal == "medium" else 0.0)
    )
    market_score = round(max(min(market_score, 2.5), -2.5), 2)

    stock_sector_name = str(stock_sector or "").strip()
    sector_confirmation = {
        "code": stock_code,
        "name": stock_name,
        "sector": stock_sector_name,
        "confirmed": False,
        "strength": "weak",
        "rank": None,
        "change_pct": 0.0,
        "main_net": 0.0,
        "reason": "未找到对应板块确认",
    }
    if stock_sector_name:
        for sector in top_sectors:
            if sector.get("name") != stock_sector_name:
                continue
            confirmed = (
                float(sector.get("main_net", 0) or 0) > 0
                and float(sector.get("change_pct", 0) or 0) >= 0.8
                and bias != "negative"
            )
            strength = (
                "strong"
                if confirmed and int(sector.get("rank") or 99) <= 3 and float(sector.get("main_net", 0) or 0) >= 5000
                else "medium"
                if confirmed
                else "weak"
            )
            reason = (
                f"{stock_sector_name} 排名 {sector.get('rank')}，涨跌 {float(sector.get('change_pct', 0) or 0):+.2f}%，"
                f"主力净流入 {float(sector.get('main_net', 0) or 0):.0f} 万"
            )
            sector_confirmation = {
                **sector_confirmation,
                "confirmed": confirmed,
                "strength": strength,
                "rank": sector.get("rank"),
                "change_pct": sector.get("change_pct"),
                "main_net": sector.get("main_net"),
                "reason": reason,
            }
            break

    opening_auction = {
        "status": "out_of_window",
        "direction": "unavailable",
        "watchlist_size": 0,
        "strong_count": 0,
        "weak_count": 0,
        "focus_stock": None,
    }
    if auction_summary:
        focus_stock = None
        for item in auction_summary.get("stocks") or []:
            if item.get("code") == stock_code:
                focus_stock = {
                    "code": item.get("code"),
                    "name": item.get("name"),
                    "change_pct": round(float(item.get("change_pct", 0) or 0), 2),
                    "auction_amount": round(float(item.get("auction_amount", 0) or 0), 2),
                    "imbalance_pct": round(float(item.get("imbalance_pct", 0) or 0), 2),
                    "flow_score": round(float(item.get("flow_score", 0) or 0), 2),
                }
                break
        strong_count = len(auction_summary.get("strong") or [])
        weak_count = len(auction_summary.get("weak") or [])
        direction = "neutral"
        if focus_stock:
            if focus_stock["change_pct"] >= 0.5 and focus_stock["imbalance_pct"] >= 8:
                direction = "strong"
            elif focus_stock["change_pct"] <= -0.5 and focus_stock["imbalance_pct"] <= -8:
                direction = "weak"
        elif strong_count > weak_count:
            direction = "strong"
        elif weak_count > strong_count:
            direction = "weak"
        opening_auction = {
            "status": "available",
            "direction": direction,
            "watchlist_size": int(auction_summary.get("watchlist_size", 0) or 0),
            "strong_count": strong_count,
            "weak_count": weak_count,
            "focus_stock": focus_stock,
        }

    headline_sentiment = "neutral"
    if macro_risk_level == "high":
        headline_sentiment = "risk_off"
    elif hot_money_signal == "active" and bias == "positive":
        headline_sentiment = "risk_on"

    return {
        "timestamp": summary.get("timestamp") or _now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "bias": bias,
        "bias_label": summary.get("bias_label", ""),
        "avg_change_pct": round(avg_change_pct, 2),
        "market_score": market_score,
        "high_risk_count": high_risk_count,
        "headline_sentiment": headline_sentiment,
        "macro_risk_level": macro_risk_level,
        "decision_basis": [
            "先看特朗普/关税等外部突发是否引发 risk-off",
            "再看集合竞价与指数承接是否同步",
            "再看主线题材资金是否持续回流",
            "再看游资试板是否获得板块跟随",
            "最后看量化砸盘压力与个股盈亏位置",
        ],
        "top_sectors": top_sectors[:5],
        "sector_confirmation": sector_confirmation,
        "opening_auction": opening_auction,
        "hot_money_probe": {
            "signal": hot_money_signal,
            "targets": active_probe_targets[:3],
        },
        "quant_pressure": {
            "signal": quant_signal,
            "score": quant_pressure_score,
        },
        "headline_risk": {
            "trump_sensitive_count": len(sensitive_trump_news),
            "telegraph_sensitive_count": len(sensitive_financial_news),
            "top_trump_news": sensitive_trump_news[:2],
            "top_financial_news": sensitive_financial_news[:2],
        },
        "anomalies": anomalies[:3],
    }


def _select_market_context_checkpoint() -> str:
    current_time = _time_cn()
    if current_time.strftime("%H:%M") < "12:00":
        return "morning_review"
    return "afternoon_review"


def build_agent_market_context(
    stock_code: str = "",
    stock_name: str = "",
    stock_sector: str = "",
) -> dict:
    """Build a live market overview for multi-agent analysis."""

    checkpoint = _select_market_context_checkpoint()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            "quotes": pool.submit(
                _fetch_quotes_cached,
                [code for code, _ in CN_INDEX_CODES],
                8,
                5.0,
            ),
            "sectors": pool.submit(_fetch_sector_flow_cached, "hy", 5, 20.0),
            "hot_candidates": pool.submit(_fetch_hot_candidates_cached, 15.0, 5, 20.0),
            "financial_news": pool.submit(_fetch_financial_news_cached, 45.0),
            "trump_news": pool.submit(_fetch_trump_news_cached, 45.0),
        }

        quotes = futures["quotes"].result()
        sectors = futures["sectors"].result()
        try:
            hot_candidates = futures["hot_candidates"].result()
        except Exception as exc:
            logger.debug("build_agent_market_context hot candidates failed: %s", exc)
            hot_candidates = []
        try:
            financial_news = futures["financial_news"].result()
        except Exception as exc:
            logger.debug("build_agent_market_context financial news failed: %s", exc)
            financial_news = {"all": [], "sensitive": []}
        try:
            trump_news = futures["trump_news"].result()
        except Exception as exc:
            logger.debug("build_agent_market_context trump news failed: %s", exc)
            trump_news = []

    anomaly_result = check_market_anomaly(index_quotes=quotes, trump_news=trump_news)
    summary = build_intraday_checkpoint_summary(
        checkpoint=checkpoint,
        index_quotes=quotes,
        sectors=sectors,
        anomaly_result=anomaly_result,
    )

    auction_summary = None
    if stock_code and is_opening_auction_window():
        watchlist = [{"code": stock_code, "name": stock_name, "reason": "analysis"}]
        sample = collect_opening_auction_sample(watchlist)
        if sample:
            auction_summary = summarize_opening_auction_state({
                "watchlist": watchlist,
                "samples": [sample],
            })

    return compose_market_decision_context(
        summary=summary,
        sectors=sectors,
        anomaly_result=anomaly_result,
        hot_candidates=hot_candidates,
        auction_summary=auction_summary,
        financial_news=financial_news,
        stock_code=stock_code,
        stock_name=stock_name,
        stock_sector=stock_sector,
    )

    try:
        hot_candidates = fetch_hot_stocks_for_candidate(max_price=15.0, top_n=5)
    except Exception as exc:
        logger.debug("构建 agent 市场上下文时获取热门资金失败: %s", exc)
        hot_candidates = []

    try:
        financial_news = fetch_financial_news()
    except Exception as exc:
        logger.debug("构建 agent 市场上下文时获取财经快讯失败: %s", exc)
        financial_news = {"all": [], "sensitive": []}

    return compose_market_decision_context(
        summary=summary,
        sectors=sectors,
        anomaly_result=anomaly_result,
        hot_candidates=hot_candidates,
        auction_summary=auction_summary,
        financial_news=financial_news,
        stock_code=stock_code,
        stock_name=stock_name,
        stock_sector=stock_sector,
    )


def build_intraday_portfolio_advice(
    summary: dict,
    portfolio: dict,
    holding_quotes: dict,
    sectors: list,
    risk_alerts: list,
    rotation_candidates: Optional[list] = None,
) -> dict:
    """Build lightweight portfolio actions from market bias + existing hard rules."""
    from risk_control import (
        MAX_SINGLE_POSITION_PCT,
        TAKE_PROFIT_FULL_PCT,
        TAKE_PROFIT_HALF_PCT,
        check_buy_permission,
        get_position_sizing,
    )

    holdings = portfolio.get("holdings", []) or []
    if not holdings:
        return {
            "has_holdings": False,
            "position_advice": "当前无持仓，暂无盘中调仓动作。",
            "actions": [],
            "rotation_candidates": [],
        }

    bias = summary.get("bias", "neutral")
    range_map = {
        "positive": (0.55, 0.75),
        "neutral": (0.40, 0.60),
        "negative": (0.20, 0.40),
    }
    lower, upper = range_map.get(bias, range_map["neutral"])
    actual_ratio = float(portfolio.get("actual_position_ratio", 0) or 0)
    cash = float(portfolio.get("cash", 0) or 0)
    total_asset = float(portfolio.get("total_asset", 0) or 0)
    market_strength = float(summary.get("avg_change_pct", 0) or 0)
    high_risk_count = int(summary.get("high_risk_count", 0) or 0)
    cash_ratio = (cash / total_asset) if total_asset > 0 else 0.0

    if actual_ratio > upper:
        position_advice = (
            f"当前仓位 {actual_ratio * 100:.1f}% 偏高，高于{summary.get('bias_label', '当前环境')}"
            f"建议区间 {lower * 100:.0f}%~{upper * 100:.0f}%，优先止盈减仓或转仓。"
        )
    elif actual_ratio < lower:
        if bias == "positive":
            position_advice = (
                f"当前仓位 {actual_ratio * 100:.1f}% 偏低，市场偏强，优先寻找主线股的强势回踩加仓；"
                f"只有在主线资金未散、并满足板块回流型或龙头错杀型条件时，"
                f"才考虑小额深跌摊低成本。"
            )
        else:
            position_advice = (
                f"当前仓位 {actual_ratio * 100:.1f}% 偏低，但当前环境不支持主动追仓，"
                f"保持现金等待更清晰信号。"
            )
    else:
        position_advice = (
            f"当前仓位 {actual_ratio * 100:.1f}% 处于建议区间 {lower * 100:.0f}%~{upper * 100:.0f}%，"
            f"以结构优化为主。"
        )

    alert_priority = {"critical": 3, "warning": 2, "info": 1}
    alert_by_code = {}
    for alert in risk_alerts or []:
        code = getattr(alert, "code", "")
        if not code:
            continue
        prev = alert_by_code.get(code)
        if prev is None or alert_priority.get(alert.level, 0) > alert_priority.get(prev.level, 0):
            alert_by_code[code] = alert

    actions = []
    held_codes = {holding.get("code") for holding in holdings if holding.get("code")}
    rotation_pool = [
        candidate for candidate in (rotation_candidates or [])
        if candidate.get("code") not in held_codes
    ][:3]
    has_rotation_pool = bool(rotation_pool)
    sector_snapshot = {}
    for rank, sector in enumerate(sectors or [], start=1):
        sector_name = str(sector.get("name", "") or "").strip()
        if not sector_name:
            continue
        sector_snapshot[sector_name] = {
            "rank": rank,
            "change_pct": float(sector.get("change_pct", 0) or 0),
            "main_net": float(sector.get("main_net", 0) or 0),
        }
    top_sector_names = {
        sector_name
        for sector_name, info in sector_snapshot.items()
        if info["rank"] <= 3 and info["main_net"] > 0
    }
    action_weight = {"sell": 4, "reduce": 3, "buy": 2, "hold": 1}

    for holding in holdings:
        code = holding.get("code", "")
        name = holding.get("name", code)
        quote = holding_quotes.get(code, {})
        pnl_pct = float(holding.get("pnl_pct", 0) or 0)
        change_pct = float(quote.get("change_pct", 0) or 0)
        sellable = int(holding.get("sellable_shares", holding.get("shares", 0)) or 0)
        sector_name = str(holding.get("sector", "") or "").strip()
        holding_value = float(holding.get("market_value", 0) or 0)
        position_pct = (holding_value / total_asset * 100) if total_asset > 0 else 0.0
        alert = alert_by_code.get(code)
        sector_info = sector_snapshot.get(sector_name, {})
        sector_change_pct = float(sector_info.get("change_pct", 0) or 0)
        sector_main_net = float(sector_info.get("main_net", 0) or 0)
        in_hot_sector = sector_name in top_sector_names
        sector_is_lagging = (
            (not in_hot_sector)
            and (
                (not sector_name)
                or sector_main_net <= 0
                or sector_change_pct < 0.8
            )
        )
        market_supports_add = bias == "positive" and high_risk_count <= 1
        strong_pullback_ready = (
            market_supports_add
            and pnl_pct >= 1.5
            and in_hot_sector
            and sector_change_pct >= 0.5
            and -3.5 <= change_pct <= -0.4
            and cash > 0
            and actual_ratio < upper
            and position_pct < (MAX_SINGLE_POSITION_PCT - 3.0)
        )
        sector_reflow_average_down_ready = (
            market_supports_add
            and market_strength >= 0.3
            and actual_ratio < lower
            and cash_ratio >= 0.18
            and -4.8 <= pnl_pct <= -2.0
            and -6.5 <= change_pct <= -3.0
            and in_hot_sector
            and sector_change_pct >= 1.0
            and sector_main_net >= 3000
            and position_pct <= 8.0
            and not alert
        )
        leader_mispricing_average_down_ready = (
            market_supports_add
            and market_strength >= 0.6
            and actual_ratio < upper
            and cash_ratio >= 0.12
            and -2.8 <= pnl_pct <= -0.3
            and -5.5 <= change_pct <= -2.2
            and in_hot_sector
            and sector_change_pct >= 1.6
            and sector_main_net >= 5000
            and position_pct <= 10.0
            and not alert
        )

        action = "hold"
        ratio = "维持"
        reason = "暂无明确调仓触发条件，继续观察盘中承接。"
        strategy = "watch"
        strategy_label = "继续观察"

        if alert and alert.level == "critical" and sellable > 0:
            action = "sell"
            ratio = "清仓"
            reason = alert.message
            strategy = "risk_exit"
            strategy_label = "风控离场"
        elif alert and alert.level == "warning" and sellable > 0:
            if alert.action == "force_sell":
                action = "sell"
                ratio = "清仓"
                strategy = "risk_exit"
                strategy_label = "风控离场"
            else:
                action = "reduce"
                ratio = "减仓50%"
                strategy = "risk_reduce"
                strategy_label = "风险减仓"
            reason = alert.message
        elif bias == "negative":
            if pnl_pct >= TAKE_PROFIT_HALF_PCT and sellable > 0:
                action = "reduce"
                ratio = "减仓50%"
                if sector_is_lagging and has_rotation_pool:
                    strategy = "defensive_lock_profit"
                    strategy_label = "防守锁利"
                    reason = "市场偏弱，先把已有利润收回来，保留现金，等待更稳的切换时点。"
                else:
                    strategy = "defensive_take_profit"
                    strategy_label = "防守止盈"
                    reason = "市场偏弱，优先止盈锁利，避免盈利回撤。"
            elif pnl_pct > 0 and change_pct < -1.5 and sellable > 0:
                action = "reduce"
                ratio = "减仓30%-50%"
                strategy = "defensive_reduce"
                strategy_label = "防守减仓"
                reason = "市场偏弱且个股转弱，先降风险，再看是否转仓。"
            elif pnl_pct < 0:
                strategy = "wait_for_base"
                strategy_label = "等待止跌"
                reason = "市场偏弱且持仓仍亏损，禁止逆势补仓，等待止跌确认。"
        elif bias == "positive":
            if pnl_pct >= TAKE_PROFIT_FULL_PCT and sellable > 0:
                action = "sell"
                ratio = "止盈清仓"
                if sector_is_lagging and has_rotation_pool:
                    strategy = "weak_to_strong_rotation"
                    strategy_label = "弱转强切换"
                    reason = "短线利润已明显超额，且当前板块弱于主线，先兑现利润，再切向更强方向。"
                else:
                    strategy = "take_profit"
                    strategy_label = "分批止盈"
                    reason = "短线利润已明显超额，先兑现，再考虑切换更强主线。"
            elif pnl_pct >= TAKE_PROFIT_HALF_PCT and sellable > 0:
                action = "reduce"
                ratio = "减仓50%"
                if sector_is_lagging and has_rotation_pool:
                    strategy = "weak_to_strong_rotation"
                    strategy_label = "弱转强切换"
                    reason = "已有利润但当前板块开始落后，可先止盈减仓，把仓位从弱线切向更强主线。"
                else:
                    strategy = "take_profit"
                    strategy_label = "分批止盈"
                    reason = "强市中也先锁一部分利润，剩余仓位继续跟随。"
            elif strong_pullback_ready:
                buy_amount = min(
                    cash,
                    get_position_sizing(
                        total_asset=max(total_asset, cash),
                        current_positions=len(holdings),
                        signal_strength="strong",
                    ),
                )
                permission = check_buy_permission(
                    code=code,
                    name=name,
                    holdings=holdings,
                    total_asset=max(total_asset, cash),
                    buy_amount=buy_amount,
                    current_change_pct=change_pct,
                )
                if permission["allowed"] and buy_amount >= 500:
                    action = "buy"
                    ratio = f"加仓约{int(buy_amount)}元"
                    strategy = "strong_pullback_add"
                    strategy_label = "强势回踩加仓"
                    reason = "市场偏强，主线板块资金仍在，个股回踩但趋势未坏，可顺势分批低吸。"
                    if permission["warnings"]:
                        reason = f"{reason} 风险提示: {permission['warnings'][0]}"
                elif permission["reasons"]:
                    strategy = "wait_for_pullback"
                    strategy_label = "等待更好买点"
                    reason = permission["reasons"][0]
                elif permission["warnings"]:
                    strategy = "wait_for_pullback"
                    strategy_label = "等待更好买点"
                    reason = permission["warnings"][0]
            elif leader_mispricing_average_down_ready:
                leader_mispricing_budget = min(
                    cash * 0.40,
                    get_position_sizing(
                        total_asset=max(total_asset, cash),
                        current_positions=len(holdings),
                        signal_strength="strong",
                    ),
                )
                if total_asset > 0:
                    leader_mispricing_budget = min(
                        leader_mispricing_budget,
                        max(total_asset * 0.11 - holding_value, 0),
                    )
                permission = check_buy_permission(
                    code=code,
                    name=name,
                    holdings=holdings,
                    total_asset=max(total_asset, cash),
                    buy_amount=leader_mispricing_budget,
                    current_change_pct=change_pct,
                    allow_averaging_down=True,
                )
                if permission["allowed"] and leader_mispricing_budget >= 500:
                    action = "buy"
                    ratio = f"错杀补仓约{int(leader_mispricing_budget)}元"
                    strategy = "leader_mispricing_average_down"
                    strategy_label = "龙头错杀型补仓"
                    reason = (
                        "指数和主线板块都仍偏强，但个股被异常压低，"
                        "可按龙头错杀处理，小额试探补仓，不宜一次打满。"
                    )
                    if permission["warnings"]:
                        reason = f"{reason} 风险提示: {permission['warnings'][0]}"
                elif permission["reasons"]:
                    strategy = "wait_for_base"
                    strategy_label = "等待止跌"
                    reason = permission["reasons"][0]
                elif permission["warnings"]:
                    strategy = "wait_for_base"
                    strategy_label = "等待止跌"
                    reason = permission["warnings"][0]
            elif sector_reflow_average_down_ready:
                sector_reflow_budget = min(
                    cash * 0.25,
                    get_position_sizing(
                        total_asset=max(total_asset, cash),
                        current_positions=len(holdings),
                        signal_strength="medium",
                    ),
                )
                if total_asset > 0:
                    sector_reflow_budget = min(
                        sector_reflow_budget,
                        max(total_asset * 0.09 - holding_value, 0),
                    )
                permission = check_buy_permission(
                    code=code,
                    name=name,
                    holdings=holdings,
                    total_asset=max(total_asset, cash),
                    buy_amount=sector_reflow_budget,
                    current_change_pct=change_pct,
                    allow_averaging_down=True,
                )
                if permission["allowed"] and sector_reflow_budget >= 500:
                    action = "buy"
                    ratio = f"回流补仓约{int(sector_reflow_budget)}元"
                    strategy = "sector_reflow_average_down"
                    strategy_label = "板块回流型补仓"
                    reason = (
                        "主线板块资金重新回流，个股只是跟随性深跌，"
                        "可更保守地做小额试探补仓。"
                    )
                    if permission["warnings"]:
                        reason = f"{reason} 风险提示: {permission['warnings'][0]}"
                elif permission["reasons"]:
                    strategy = "wait_for_base"
                    strategy_label = "等待止跌"
                    reason = permission["reasons"][0]
                elif permission["warnings"]:
                    strategy = "wait_for_base"
                    strategy_label = "等待止跌"
                    reason = permission["warnings"][0]
            elif pnl_pct < 0:
                strategy = "wait_for_base"
                strategy_label = "等待止跌"
                reason = "虽然市场偏强，但当前持仓仍亏损，若不满足板块回流型或龙头错杀型条件，暂不补仓。"
        else:
            if pnl_pct >= TAKE_PROFIT_HALF_PCT and sellable > 0:
                action = "reduce"
                ratio = "减仓30%-50%"
                if sector_is_lagging and has_rotation_pool:
                    strategy = "defensive_lock_profit"
                    strategy_label = "防守锁利"
                    reason = "震荡市里先把利润锁住，资金先回笼，等更强方向确认后再切换。"
                else:
                    strategy = "take_profit"
                    strategy_label = "分批止盈"
                    reason = "震荡市先落袋一部分利润，保留机动仓位。"
            elif pnl_pct < 0:
                strategy = "wait_for_base"
                strategy_label = "等待止跌"
                reason = "震荡市不适合摊低成本，继续观察，不主动补仓。"

        actions.append({
            "code": code,
            "name": name,
            "action": action,
            "strategy": strategy,
            "strategy_label": strategy_label,
            "ratio": ratio,
            "reason": reason,
            "pnl_pct": round(pnl_pct, 2),
            "change_pct": round(change_pct, 2),
            "sector": sector_name,
            "sellable": sellable,
            "position_pct": round(position_pct, 2),
        })

    actions.sort(
        key=lambda item: (
            action_weight.get(item["action"], 0),
            abs(item["pnl_pct"]),
            abs(item["change_pct"]),
        ),
        reverse=True,
    )

    return {
        "has_holdings": True,
        "position_advice": position_advice,
        "actions": actions,
        "rotation_candidates": rotation_pool,
        "cash": round(cash, 2),
        "total_asset": round(total_asset, 2),
        "actual_position_ratio": round(actual_ratio * 100, 1),
    }


def collect_intraday_portfolio_advice(summary: dict, sectors: list) -> dict:
    """Refresh holdings with realtime quotes, then derive intraday portfolio advice."""
    from macro_data_collector import _stock_code_to_tencent
    from portfolio_manager import (
        load_portfolio, sync_portfolio_from_trades, update_current_prices,
    )
    from risk_control import check_stop_loss

    portfolio = load_portfolio()
    portfolio = sync_portfolio_from_trades(portfolio)
    holding_codes = [holding.get("code") for holding in portfolio.get("holdings", []) if holding.get("code")]
    if not holding_codes:
        return {
            "has_holdings": False,
            "position_advice": "当前无持仓，暂无盘中调仓动作。",
            "actions": [],
            "rotation_candidates": [],
        }

    tc_codes = [_stock_code_to_tencent(code) for code in holding_codes]
    quotes = _fetch_quotes_cached(tc_codes, timeout=8, ttl_seconds=5.0)
    quote_by_code = {}
    price_map = {}
    for code in holding_codes:
        tc_code = _stock_code_to_tencent(code)
        quote = quotes.get(tc_code, {})
        quote_by_code[code] = quote
        price = float(quote.get("price", 0) or 0)
        if price > 0:
            price_map[code] = price

    portfolio = update_current_prices(portfolio, price_map)

    rotation_candidates = []
    if summary.get("bias") != "negative":
        try:
            # Use cached hot-candidate snapshot to avoid duplicate network fetches.

            rotation_candidates = _fetch_hot_candidates_cached(max_price=15.0, top_n=5, ttl_seconds=20.0)
        except Exception as e:
            logger.debug("转仓候选获取失败，跳过: %s", e)

    try:
        financial_news = _fetch_financial_news_cached(ttl_seconds=45.0)
    except Exception as e:
        logger.debug("盘中调仓获取财经快讯失败，跳过: %s", e)
        financial_news = {"all": [], "sensitive": []}

    market_context = compose_market_decision_context(
        summary=summary,
        sectors=sectors,
        anomaly_result={
            "anomalies": summary.get("anomalies") or [],
            "trump_news": [],
        },
        hot_candidates=rotation_candidates,
        financial_news=financial_news,
    )
    risk_alerts = check_stop_loss(portfolio.get("holdings", []), market_context=market_context)

    return build_intraday_portfolio_advice(
        summary=summary,
        portfolio=portfolio,
        holding_quotes=quote_by_code,
        sectors=sectors,
        risk_alerts=risk_alerts,
        rotation_candidates=rotation_candidates,
    )


def format_intraday_checkpoint_alert(summary: dict) -> str:
    lines = [
        summary["title"],
        f"⏰ {summary['timestamp']}",
        f"🎯 结论: {summary['bias_label']} (指数均值 {summary['avg_change_pct']:+.2f}%)",
        "",
    ]

    index_rows = summary.get("index_rows") or []
    if index_rows:
        index_line = " | ".join(
            f"{row['name']} {row['change_pct']:+.2f}%"
            for row in index_rows
        )
        lines.append(f"📊 指数: {index_line}")

    sectors = summary.get("top_sectors") or []
    if sectors:
        sector_line = " | ".join(
            f"{sector['name']} {sector['change_pct']:+.2f}% / {sector['main_net']:.0f}万"
            for sector in sectors
        )
        lines.append(f"🔥 热点: {sector_line}")

    anomalies = summary.get("anomalies") or []
    if anomalies:
        lines.append("⚠️ 风险:")
        for anomaly in anomalies[:2]:
            lines.append(f"  - {anomaly.get('detail', '')}")

    portfolio_advice = summary.get("portfolio_advice") or {}
    if portfolio_advice.get("has_holdings"):
        lines.extend([
            "",
            f"💼 仓位: {portfolio_advice.get('position_advice', '')}",
        ])
        action_items = portfolio_advice.get("actions") or []
        if action_items:
            lines.append("📋 调仓:")
            for item in action_items[:4]:
                emoji = {
                    "buy": "🟢",
                    "hold": "🟡",
                    "reduce": "🟠",
                    "sell": "🔴",
                }.get(item.get("action", "hold"), "⚪")
                strategy_label = item.get("strategy_label", "")
                strategy_suffix = (
                    f" [{strategy_label}]"
                    if strategy_label and strategy_label != "继续观察"
                    else ""
                )
                lines.append(
                    f"  {emoji} {item.get('name','')}({item.get('code','')}) "
                    f"{item.get('ratio','维持')}{strategy_suffix} | 当日{item.get('change_pct', 0):+.2f}% | "
                    f"浮盈亏{item.get('pnl_pct', 0):+.2f}%"
                )
                lines.append(f"    {item.get('reason', '')}")

        rotation_candidates = portfolio_advice.get("rotation_candidates") or []
        if rotation_candidates:
            lines.append("🔄 转仓候选:")
            for candidate in rotation_candidates[:3]:
                lines.append(
                    f"  - {candidate.get('code','')} {candidate.get('name','')} "
                    f"{float(candidate.get('price', 0) or 0):.2f}元 | "
                    f"主力:{float(candidate.get('main_net', 0) or 0):.0f}万"
                )

    lines.extend(["", f"💡 {summary['advice']}"])
    return "\n".join(lines)


def run_intraday_checkpoint(checkpoint: str, send_notification: bool = True) -> Optional[dict]:
    """执行 10:15 / 12:30 的轻量盘中判断。"""
    try:
        from src.core.trading_calendar import get_open_markets_today

        open_markets = {m.lower() for m in get_open_markets_today()}
        if "cn" not in open_markets:
            logger.info("[%s] 今日 A 股非交易日，跳过盘中节点判断", checkpoint)
            return None
    except Exception as e:
        logger.warning("[%s] 交易日判断失败，按 fail-open 继续: %s", checkpoint, e)

    from macro_data_collector import _fetch_tencent_quote
    from ths_scraper import fetch_sector_fund_flow_rank

    quotes = _fetch_tencent_quote([code for code, _ in CN_INDEX_CODES], timeout=8)
    sectors = fetch_sector_fund_flow_rank("hy", top_n=5)
    anomaly_result = check_market_anomaly()
    summary = build_intraday_checkpoint_summary(
        checkpoint=checkpoint,
        index_quotes=quotes,
        sectors=sectors,
        anomaly_result=anomaly_result,
    )
    summary["portfolio_advice"] = collect_intraday_portfolio_advice(summary, sectors)
    alert = format_intraday_checkpoint_alert(summary)
    logger.info("\n%s", alert)
    if send_notification:
        send_alert(alert)
    return summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 时间判断
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _now_cn():
    return datetime.now(TZ_CN)

def _time_cn():
    return _now_cn().time()

def is_pre_market():
    from datetime import time as dt
    return dt(9, 0) <= _time_cn() <= dt(9, 25)

def is_opening_auction_window():
    from datetime import time as dt
    return dt(9, 15) <= _time_cn() <= dt(9, 25)

def is_opening_auction_summary_time():
    from datetime import time as dt
    return dt(9, 24) <= _time_cn() <= dt(9, 25)

def is_trading_time():
    from datetime import time as dt
    t = _time_cn()
    return (dt(9, 30) <= t <= dt(11, 30)) or (dt(13, 0) <= t <= dt(15, 0))

def is_after_close():
    from datetime import time as dt
    return dt(15, 0) <= _time_cn() <= dt(15, 45)

def is_weekday():
    return _now_cn().weekday() < 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 主循环
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_monitor_loop(interval_minutes: int = 10, auto_rebalance: bool = True):
    from src.config import get_config
    from src.logging_config import setup_logging
    config = get_config()
    setup_logging(log_prefix="monitor", debug=False, log_dir=config.log_dir)

    logger.info("=" * 60)
    logger.info("盘中全功能监控系统启动")
    logger.info(f"  检查间隔: {interval_minutes} 分钟")
    logger.info(f"  收盘调仓: {auto_rebalance}")
    logger.info(f"  数据积累: SQLite")
    logger.info("=" * 60)

    pre_market_done = set()   # 记录已执行盘前扫描的日期
    after_close_done = set()  # 记录已执行收盘任务的日期
    auction_state = {"date": None, "watchlist": [], "samples": [], "alert_sent": False, "last_polled_at": None}

    while True:
        now = _now_cn()
        today = now.strftime("%Y-%m-%d")

        if not is_weekday():
            time.sleep(300)
            continue

        if auction_state.get("date") != today:
            auction_state = {
                "date": today,
                "watchlist": [],
                "samples": [],
                "alert_sent": False,
                "last_polled_at": None,
            }

        # ── 盘前扫描 09:00-09:25 ──
        if is_pre_market():
            if today not in pre_market_done:
                logger.info(f"\n[{now.strftime('%H:%M')}] 执行盘前扫描...")
                candidates = []
                try:
                    candidates = run_pre_market_scan()
                    if candidates:
                        alert = format_scan_alert(candidates)
                        send_alert(alert)

                    # 主线板块分析
                    analyze_mainline()
                except Exception as e:
                    logger.error(f"盘前扫描失败: {e}")

                auction_state["watchlist"] = build_opening_auction_watchlist(candidates)
                if auction_state["watchlist"]:
                    logger.info(
                        "[竞价] 已建立观察池 %s 只: %s",
                        len(auction_state["watchlist"]),
                        ",".join(stock["code"] for stock in auction_state["watchlist"][:8]),
                    )
                pre_market_done.add(today)

            if not auction_state.get("watchlist"):
                auction_state["watchlist"] = build_opening_auction_watchlist([])

            if is_opening_auction_window():
                try:
                    maybe_record_opening_auction_sample(auction_state)
                    if is_opening_auction_summary_time() and not auction_state.get("alert_sent"):
                        auction_summary = summarize_opening_auction_state(auction_state)
                        enough_samples = auction_summary.get("sample_count", 0) >= 2
                        reached_auction_end = _time_cn().strftime("%H:%M") >= "09:25"
                        if enough_samples or reached_auction_end:
                            auction_alert = format_opening_auction_alert(auction_summary)
                            if auction_alert:
                                logger.warning(f"竞价信号！\n{auction_alert}")
                                send_alert(auction_alert)
                            else:
                                logger.info("[竞价] 未发现明显强弱分化")
                            auction_state["alert_sent"] = True
                except Exception as e:
                    logger.error(f"竞价监控失败: {e}")

        # ── 盘中监控 09:30-15:00 ──
        elif is_trading_time():
            logger.info(f"[{now.strftime('%H:%M:%S')}] 盘中检查...")
            try:
                result = check_market_anomaly()
                if result["has_critical"] or result["has_high"]:
                    alert = format_anomaly_alert(result)
                    if alert:
                        logger.warning(f"异动！\n{alert}")
                        send_alert(alert)
                else:
                    logger.info(f"  正常 ({result['anomaly_count']} 项检查)")

                trump_count = len(result.get("trump_news", []))
                if trump_count:
                    logger.info(f"  特朗普新闻: {trump_count} 条")
            except Exception as e:
                logger.error(f"盘中检查失败: {e}")

            # ── 风控检查：止损/止盈/持仓天数 ──
            try:
                from risk_control import check_stop_loss, format_risk_alerts
                holdings = _get_current_holdings()
                if holdings:
                    alerts = check_stop_loss(holdings)
                    critical_alerts = [a for a in alerts if a.level == "critical"]
                    if critical_alerts:
                        alert_text = "** 风控紧急预警 **\n" + format_risk_alerts(alerts)
                        logger.warning(f"风控！\n{alert_text}")
                        send_alert(alert_text)
                    elif alerts:
                        logger.info(f"  风控: {len(alerts)} 条提示")
            except Exception as e:
                logger.debug(f"风控检查跳过: {e}")

        # ── 收盘任务 15:00-15:45 ──
        elif is_after_close() and today not in after_close_done:
            logger.info(f"\n[{now.strftime('%H:%M')}] 收盘任务...")

            # 1. 再次积累资金数据（收盘价更准）
            try:
                from ths_scraper import fetch_stock_fund_flow_rank, fetch_sector_fund_flow_rank
                from data_store import save_fund_flow_batch, save_sector_flow_batch

                fund = fetch_stock_fund_flow_rank()
                save_fund_flow_batch(fund, trade_date=today)
                hy = fetch_sector_fund_flow_rank("hy")
                save_sector_flow_batch(hy, "hy", today)
            except Exception as e:
                logger.warning(f"收盘数据积累失败: {e}")

            # 2. 回测历史扫描结果
            try:
                from data_store import run_scan_backtest, get_backtest_summary
                logger.info("执行扫描回测...")
                run_scan_backtest()
                summary = get_backtest_summary(days=30)
                if summary:
                    for mode, stats in summary.items():
                        logger.info(
                            f"  [{mode}] 样本:{stats['total']} "
                            f"3日胜率:{stats['win_rate_3d']}% "
                            f"5日胜率:{stats['win_rate_5d']}% "
                            f"均收益:{stats['avg_return_3d']}%"
                        )
            except Exception as e:
                logger.warning(f"回测失败: {e}")

            # 3. 调仓分析
            if auto_rebalance:
                try:
                    from rebalance_engine import run_rebalance_analysis
                    from portfolio_manager import format_rebalance_report

                    logger.info("执行调仓分析...")
                    result = run_rebalance_analysis(config=config)
                    if "error" not in result:
                        report = format_rebalance_report(result)
                        logger.info(f"\n{report}")
                        send_alert(f"📊 **收盘调仓建议**\n\n{report}")

                        result_dir = Path("data/rebalance_history")
                        result_dir.mkdir(parents=True, exist_ok=True)
                        fname = result_dir / f"rebalance_{now.strftime('%Y%m%d_%H%M%S')}.json"
                        with open(fname, "w", encoding="utf-8") as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"调仓分析失败: {e}")

            after_close_done.add(today)

        sleep_seconds = max(1, interval_minutes) * 60
        if is_opening_auction_window():
            sleep_seconds = min(sleep_seconds, AUCTION_POLL_SECONDS)
        time.sleep(sleep_seconds)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI 入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股盘中全功能监控")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--no-rebalance", action="store_true")
    parser.add_argument("--test", action="store_true", help="测试所有模块")
    parser.add_argument("--backtest", action="store_true", help="手动回测")
    parser.add_argument("--mainline", action="store_true", help="查看主线板块")
    parser.add_argument("--scan", action="store_true", help="立即执行一次扫描")
    args = parser.parse_args()

    from src.config import get_config
    from src.logging_config import setup_logging
    config = get_config()
    setup_logging(log_prefix="monitor", debug=True, log_dir=config.log_dir)

    if args.scan:
        candidates = run_pre_market_scan()
        print(f"\n共 {len(candidates)} 只候选")

    elif args.mainline:
        result = analyze_mainline()
        if not result.get("mainline_sectors"):
            print("数据不足，至少需要积累2天的板块数据才能判断主线")
            print("运行 python market_monitor.py --scan 开始积累")

    elif args.backtest:
        from data_store import run_scan_backtest, get_backtest_summary
        run_scan_backtest()
        summary = get_backtest_summary(30)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    elif args.test:
        print("=== 1. 异动检测 ===")
        result = check_market_anomaly()
        print(f"异动: {result['anomaly_count']} 项, 严重: {result['has_critical']}")
        for a in result["anomalies"][:5]:
            print(f"  {a['detail']}")

        print("\n=== 2. 特朗普新闻 ===")
        from macro_data_collector import fetch_trump_news as _trump
        trump = _trump()
        if trump:
            for n in trump[:3]:
                flag = "🔴" if n.get("is_sensitive") else "⚪"
                print(f"  {flag} [{n.get('source')}] {n.get('title','')[:60]}")
        else:
            print("  无（检查SearXNG）")

        print("\n=== 3. 扫描测试（5只）===")
        from market_scanner import scan_market
        r = scan_market(max_price=10, top_n=5, mode="trend")
        for s in r:
            print(f"  {s['code']} {s['name']} {s['price']} 得分:{s['tech_score']}")

        print("\n=== 4. 主线板块 ===")
        analyze_mainline()

    else:
        run_monitor_loop(
            interval_minutes=args.interval,
            auto_rebalance=not args.no_rebalance,
        )
