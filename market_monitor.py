"""
market_monitor.py — 盘中全功能监控系统

一天的完整流程:
  09:15 开盘前 — 全市场扫描，输出今日候选股
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
import argparse, json, logging, os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
TZ_CN = timezone(timedelta(hours=8))


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

    # 1. 全市场扫描
    logger.info("[盘前 1/3] 全市场扫描...")
    candidates = scan_market(max_price=10.0, min_turnover=2.0, top_n=30, mode="trend")
    save_scan_results(candidates, scan_date=today, mode="trend")

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
def check_market_anomaly() -> dict:
    """检测指数/持仓/新闻异动"""
    anomalies = []

    # 指数异动
    try:
        from macro_data_collector import _fetch_tencent_quote
        quotes = _fetch_tencent_quote(["sh000001", "sz399001", "sz399006"])
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
        from portfolio_manager import load_portfolio
        from macro_data_collector import _fetch_tencent_quote, _stock_code_to_tencent
        portfolio = load_portfolio()
        holdings = portfolio.get("holdings", [])
        if holdings:
            tc_codes = [_stock_code_to_tencent(h["code"]) for h in holdings]
            quotes = _fetch_tencent_quote(tc_codes)
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
        from macro_data_collector import fetch_trump_news as _trump
        trump_news = _trump()
    except:
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
# 5. 时间判断
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _now_cn():
    return datetime.now(TZ_CN)

def _time_cn():
    return _now_cn().time()

def is_pre_market():
    from datetime import time as dt
    return dt(9, 0) <= _time_cn() <= dt(9, 25)

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
# 6. 主循环
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

    while True:
        now = _now_cn()
        today = now.strftime("%Y-%m-%d")

        if not is_weekday():
            time.sleep(300)
            continue

        # ── 盘前扫描 09:00-09:25 ──
        if is_pre_market() and today not in pre_market_done:
            logger.info(f"\n[{now.strftime('%H:%M')}] 执行盘前扫描...")
            try:
                candidates = run_pre_market_scan()
                if candidates:
                    alert = format_scan_alert(candidates)
                    send_alert(alert)

                # 主线板块分析
                analyze_mainline()

            except Exception as e:
                logger.error(f"盘前扫描失败: {e}")
            pre_market_done.add(today)

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

        time.sleep(interval_minutes * 60)


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