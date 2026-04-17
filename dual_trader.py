"""
dual_trader.py — 一键启动：新闻情报 + 龙头打板 + 可转债T+0
==========================================================

全自动流程：
  Phase 0: 新闻扫描 → 识别今日热点概念（特朗普/马斯克/黄仁勋/华为...）
  Phase 1: 龙头选股 → 在热点概念中找涨停/接力/突破信号
  Phase 2: 可转债T+0 → 扫描+日内交易
  Phase 3: 汇总日报

使用:
  python dual_trader.py                    # 全流程模拟
  python dual_trader.py --live             # 实盘（接同花顺）
  python dual_trader.py --live --capital 30000
  python dual_trader.py --cb-only          # 只跑可转债
  python dual_trader.py --stock-only       # 只跑股票
  python dual_trader.py --news-only        # 只看新闻热点
"""

import argparse
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 资金配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL_CAPITAL = 30000           # 总资金
STOCK_RATIO = 0.60              # 股票占比
CB_RATIO = 0.40                 # 可转债占比


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: 新闻情报
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_news_scan() -> List[dict]:
    """扫描新闻热点，返回热点概念列表"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Phase 0: NEWS INTELLIGENCE")
    logger.info("=" * 60)

    try:
        from news_scanner import scan_news, get_concept_stocks

        hot_concepts = scan_news()

        if not hot_concepts:
            logger.info("  [NEWS] No hot concepts detected today")
            return []

        results = []
        for i, concept in enumerate(hot_concepts[:5]):
            name = concept.name
            heat = concept.heat_score
            keywords = concept.keywords_matched[:3]

            # 获取概念成分股（返回DataFrame）
            stock_list = []
            try:
                stocks_df = get_concept_stocks(name)
                if stocks_df is not None and not stocks_df.empty:
                    # DataFrame → list of dict
                    for _, row in stocks_df.head(5).iterrows():
                        code = str(row.get("代码", row.get("code", "")))
                        sname = str(row.get("名称", row.get("name", code)))
                        stock_list.append({"code": code, "name": sname})
            except Exception:
                pass

            results.append({
                "concept": name,
                "heat": heat,
                "keywords": keywords,
                "headlines": concept.sample_headlines[:2],
                "stocks": stock_list,
            })

            kw_str = ", ".join(keywords)
            stock_str = ", ".join([s["name"][:6] for s in stock_list[:3]]) if stock_list else "N/A"
            logger.info(f"  #{i+1} [{name}] heat={heat} kw=[{kw_str}] stocks=[{stock_str}]")

        logger.info(f"  [NEWS] {len(results)} hot concepts found")
        return results

    except Exception as e:
        logger.error(f"  [NEWS] Scan failed: {e}")
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: 龙头打板选股
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_stock_scan(
    hot_concepts: List[dict] = None,
    regime: dict = None,
    event_signals: List[dict] = None,
) -> List[dict]:
    """智能选股（龙头打板主策略 + 新闻/事件加分）"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Phase 1: SMART STOCK SCAN")
    logger.info("=" * 60)

    try:
        from market_scanner import scan_market, detect_market_regime

        # 检测市场环境
        if regime is None:
            regime = detect_market_regime()

        market = regime.get("regime", "sideways")
        logger.info(f"  [REGIME] {market.upper()} (score={regime.get('score', 0)})")

        # Dragon v2：oracle 校准 + 封板强度 + 1→2 / 2→3 / pre_limit / ignition 分级入场
        # 可通过 USE_DRAGON_V2=false 回退到旧 dragon 扫描
        import os
        use_v2 = os.getenv("USE_DRAGON_V2", "true").lower() in ("true", "1", "yes")

        # regime 映射：dual_trader 用小写 bull/sideways/bear/crash，dragon_v2 用大写
        regime_map = {"bull": "BULL", "sideways": "SIDE", "bear": "BEAR", "crash": "CRASH"}
        regime_upper = regime_map.get(market, "SIDE")

        if use_v2:
            logger.info(f"  [STRATEGY] Dragon v2 (regime={regime_upper}) 1→2/2→3/pre_limit/ignition + 封板强度")
            if regime_upper == "CRASH":
                logger.info("  [STRATEGY] CRASH regime → 强制空扫描，保护资金")
                results = []
            else:
                from market_scanner_dragon import scan_market_dragon_v2
                results = scan_market_dragon_v2(
                    regime=regime_upper,
                    top_n=15,
                    use_live_seal=True,
                )
                # 若发现 ALL-IN 信号，日志重点提示
                for r in results[:3]:
                    if r.get("allin_suggest"):
                        logger.info(f"  ★★★ ALL-IN 建议: {r['code']} {r.get('name','')} "
                                    f"封单 {r.get('seal_amount_wan',0):.0f} 万, "
                                    f"续板概率 {r.get('next_prob',0):.2f}")
                # 字段对齐：旧调用方按 tech_score 读，v2 写 dragon_tech_score
                for r in results:
                    if "dragon_tech_score" in r:
                        r["tech_score"] = r["dragon_tech_score"]
        else:
            logger.info(f"  [STRATEGY] Legacy dragon scan (市场={market.upper()})")
            results = scan_market(
                mode="dragon",
                max_price=50.0,
                min_turnover=3.0,
                max_change=20.0,
                min_change=-3.0,
                top_n=15,
            )

        if not results:
            logger.info("  [STOCK] No candidates from market scan")
            return []

        # 用新闻热点给概念内的股票加分
        if hot_concepts:
            hot_stock_codes = set()
            for hc in hot_concepts:
                for s in hc.get("stocks", []):
                    code = str(s.get("code", "")).strip()
                    if code and len(code) == 6:
                        hot_stock_codes.add(code)

            if hot_stock_codes:
                logger.info(f"  [NEWS->STOCK] {len(hot_stock_codes)} stocks in hot concepts")

            for r in results:
                code = str(r.get("code", "")).strip()
                clean_code = code.replace("SH", "").replace("SZ", "").replace("sh", "").replace("sz", "")
                if clean_code in hot_stock_codes:
                    r["tech_score"] = r.get("tech_score", 0) + 20
                    r["hot_concept"] = True
                else:
                    r["hot_concept"] = False

            results.sort(key=lambda x: x.get("tech_score", 0), reverse=True)

        # ── 事件关注池加分: 被事件影响的超跌个股额外加分 ──
        try:
            from event_signal import get_event_watchlist, get_recent_event_watchlist

            event_watchlist = (
                get_event_watchlist(event_signals)
                if event_signals
                else get_recent_event_watchlist(max_age_hours=6.0)
            )
            if event_watchlist:
                boosted = 0
                for r in results:
                    code = str(r.get("code", "")).strip()
                    clean_code = code.replace("SH", "").replace("SZ", "").replace("sh", "").replace("sz", "")
                    bonus = event_watchlist.get(clean_code, 0)
                    if bonus > 0:
                        r["tech_score"] = r.get("tech_score", 0) + bonus
                        r["event_boost"] = True
                        r["event_bonus"] = bonus
                        boosted += 1
                if boosted:
                    logger.info(f"  [EVENT->STOCK] {boosted} stocks boosted by event signals")
                    results.sort(key=lambda x: x.get("tech_score", 0), reverse=True)
        except Exception:
            pass

        # ── 复盘反馈：近期亏损过的股票降权（避免反复踩同一坑）──
        try:
            from trade_journal import get_recent_losers
            recent_losers = get_recent_losers(days=10)  # 近10天亏损卖出的股票
            if recent_losers:
                loser_codes = {l["code"] for l in recent_losers}
                for r in results:
                    if r.get("code") in loser_codes:
                        r["tech_score"] = r.get("tech_score", 0) - 30
                        r["recent_loser"] = True
                        logger.info(f"  [REVIEW] {r.get('name', '')}({r['code']}) 近期亏损-30分")
                results.sort(key=lambda x: x.get("tech_score", 0), reverse=True)
        except Exception:
            pass

        # ── 游资风格评分层（蒸馏 A 股游资打法 + 多周期 + 消息面 + V2 协同）──
        try:
            from youzi_pipeline import enrich_with_youzi, dispatch_report

            # adaptive_coeff 交给 AdaptiveTradeState；这里默认 1.0，main.py 会重写
            enriched, trace, paths = enrich_with_youzi(
                results=results,
                hot_concepts=hot_concepts,
                regime=regime,
                adaptive_coeff=float(os.getenv("YOUZI_ADAPTIVE_COEFF", "1.0") or 1.0),
                event_defense=bool(event_signals),
                save_report=True,
            )
            results = enriched

            # 按环境变量决定是否推送
            if os.getenv("YOUZI_DISPATCH_REPORT", "true").lower() in ("true", "1", "yes"):
                try:
                    dispatch_report(trace, paths.get("md"))
                except Exception as e:
                    logger.debug(f"  [YOUZI] dispatch_report failed: {e}")
        except Exception as e:
            logger.warning(f"  [YOUZI] enrichment skipped: {e}")
            import traceback
            traceback.print_exc()

        # 打印结果
        logger.info(f"  [STOCK] Top candidates:")
        for i, r in enumerate(results[:10]):
            hot_tag = " [HOT]" if r.get("hot_concept") else ""
            votes = r.get("youzi_buy_votes") or []
            votes_tag = f" [YOUZI:{'+'.join(votes)}]" if votes else ""
            sig = r.get("signal_type", r.get("ma_trend", ""))
            logger.info(
                f"    #{i+1} {r.get('name', '?')}({r['code']}) "
                f"price={r.get('price', 0):.2f} chg={r.get('change_pct', 0):+.1f}% "
                f"score={r.get('tech_score', 0)} {sig}{hot_tag}{votes_tag}"
            )

        return results

    except Exception as e:
        logger.error(f"  [STOCK] Scan failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: 可转债 T+0
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_cb_trading(broker=None, capital: float = 12000):
    """可转债T+0日内交易（盘中循环版）

    生命周期：
      1. run_morning_scan()  → 扫描选债
      2. 盘中循环: 获取实时行情 → update_prices → check_signals（每60秒）
      3. 14:50 close_day() 强制清仓
    """
    import time as _time
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  CB T+0 Trading (capital: {capital:.0f})")
    logger.info("=" * 60)

    try:
        from cb_trader import CBDayTrader
        from macro_data_collector import _stock_code_to_tencent, _fetch_tencent_quote

        trader = CBDayTrader(broker=broker, total_capital=capital / 0.4)
        trader.cb_capital = capital
        trader.cash = capital

        # 1. 早盘选债
        watchlist = trader.run_morning_scan()
        if not watchlist:
            logger.info("  [CB] No candidates, skip")
            return None

        # 2. 盘中循环：持续检查信号
        check_interval = int(os.getenv("CB_CHECK_INTERVAL", "60"))  # 默认60秒
        total_signals = 0

        while True:
            try:
                now = datetime.now()
                hhmm = now.strftime("%H:%M")

                # 14:50 之后强制清仓退出
                if hhmm >= "14:50":
                    logger.info("[CB T+0] 14:50 尾盘清仓...")
                    break

                # 非交易时间跳过（09:30之前 or 午休11:30-13:00）
                if hhmm < "09:30":
                    _time.sleep(30)
                    continue
                if "11:30" <= hhmm < "13:00":
                    _time.sleep(30)
                    continue

                # 获取候选转债实时行情
                watch_codes = [w["code"] for w in watchlist]
                # 持仓中的转债也需要实时价格
                for code in trader.positions:
                    if code not in watch_codes:
                        watch_codes.append(code)

                if watch_codes:
                    try:
                        tc_map = {}
                        for code in watch_codes:
                            tc = _stock_code_to_tencent(code)
                            tc_map[tc] = code
                        quotes = _fetch_tencent_quote(list(tc_map.keys()), timeout=8)
                        prices = {}
                        for tc, q in quotes.items():
                            raw_code = tc_map.get(tc, tc.replace("sh", "").replace("sz", ""))
                            if q.get("price", 0) > 0:
                                prices[raw_code] = q["price"]
                        # 更新watchlist价格
                        for w in watchlist:
                            if w["code"] in prices:
                                w["price"] = prices[w["code"]]
                        # 更新持仓价格
                        if prices:
                            trader.update_prices(prices)
                    except Exception as e:
                        logger.debug(f"[CB T+0] 行情获取失败: {e}")

                # 检查交易信号
                signals = trader.check_signals()
                if signals:
                    total_signals += len(signals)
                    for sig in signals:
                        logger.info(f"  [CB T+0] Signal: {sig}")

                _time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("[CB T+0] 用户中断")
                break
            except Exception as e:
                logger.warning(f"[CB T+0] 循环异常: {e}")
                _time.sleep(check_interval)

        # 3. 尾盘清仓
        report = trader.close_day()
        logger.info(f"[CB T+0] 日内完成: {total_signals} 笔信号")
        return report

    except Exception as e:
        logger.error(f"  [CB] Trading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 券商连接
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_broker(live: bool = False):
    """连接同花顺"""
    if not live:
        return None

    # 优先用项目内置的 broker 模块
    try:
        os.environ["BROKER_ENABLED"] = "true"
        from src.broker import get_broker as _get_broker
        broker = _get_broker()
        if broker and broker.is_connected():
            logger.info("[Broker] Connected via src.broker (THS)")
            return broker
    except Exception:
        pass

    # fallback: 直接用 easytrader
    try:
        import easytrader
        exe_path = os.getenv("THS_EXE_PATH", r"G:\同花顺远航版\transaction\xiadan.exe")
        user = easytrader.use("ths")
        user.connect(exe_path)
        user.enable_type_keys_for_editor()  # 键盘输入模式（兼容64位Python）
        logger.info(f"[Broker] Connected via easytrader: {exe_path}")
        return user
    except Exception as e:
        logger.error(f"[Broker] Connection failed: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 一键启动 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_all(live: bool = False, capital: float = 30000):
    """一键启动全部流程"""
    start_time = time.time()
    stock_cap = capital * STOCK_RATIO
    cb_cap = capital * CB_RATIO

    logger.info("")
    logger.info("*" * 60)
    logger.info("*")
    logger.info("*   WOLF OF WALL STREET - DUAL MODE SYSTEM")
    logger.info("*")
    logger.info(f"*   Total Capital : {capital:>10,.0f} RMB")
    logger.info(f"*   Stock (60%)   : {stock_cap:>10,.0f} RMB  [Dragon Head]")
    logger.info(f"*   CB T+0 (40%)  : {cb_cap:>10,.0f} RMB  [Intraday]")
    logger.info(f"*   Mode          : {'>>> LIVE <<<' if live else 'SIMULATION'}")
    logger.info(f"*   Time          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("*")
    logger.info("*" * 60)

    broker = get_broker(live)

    # ─── Phase 0: 新闻情报 + 市场环境 ───
    hot_concepts = run_news_scan()

    from market_scanner import detect_market_regime
    regime = detect_market_regime()

    # ─── Phase 1: 智能选股（环境自适应 + 新闻加分）───
    stock_results = run_stock_scan(hot_concepts=hot_concepts, regime=regime)

    # ─── Phase 2: 可转债 T+0 ───
    cb_report = run_cb_trading(broker=broker, capital=cb_cap)

    # ─── Phase 3: 日报汇总 ───
    elapsed = time.time() - start_time

    logger.info("")
    logger.info("#" * 60)
    logger.info("#")
    logger.info("#   DAILY REPORT")
    logger.info(f"#   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ({elapsed:.0f}s)")
    logger.info("#")
    logger.info("#" * 60)

    # 市场环境
    market = regime.get("regime", "sideways")
    regime_map = {"bull": "🐂 牛市（追涨/连板）", "sideways": "🔄 震荡（均值回归+预埋）", "bear": "🐻 熊市（防守/超跌）"}
    logger.info("")
    logger.info(f"  [REGIME] {regime_map.get(market, market)} score={regime.get('score', 0)}")

    # 新闻
    logger.info("")
    logger.info("  [NEWS] Hot Concepts Today:")
    if hot_concepts:
        for hc in hot_concepts[:3]:
            logger.info(f"    - {hc['concept']} (heat={hc['heat']})")
    else:
        logger.info("    - None detected")

    # 股票
    logger.info("")
    logger.info("  [STOCK] Dragon Head Candidates:")
    if stock_results:
        # 优先展示热点概念内的
        hot_stocks = [r for r in stock_results if r.get("hot_concept")]
        other_stocks = [r for r in stock_results if not r.get("hot_concept")]

        if hot_stocks:
            logger.info("    >>> IN HOT CONCEPT:")
            for r in hot_stocks[:3]:
                logger.info(
                    f"      {r.get('name', '?')}({r['code']}) "
                    f"score={r.get('tech_score', 0)} chg={r.get('change_pct', 0):+.1f}%"
                )

        logger.info("    >>> TOP OVERALL:")
        for r in stock_results[:3]:
            logger.info(
                f"      {r.get('name', '?')}({r['code']}) "
                f"score={r.get('tech_score', 0)} chg={r.get('change_pct', 0):+.1f}%"
            )
    else:
        logger.info("    - No candidates")

    # 可转债
    logger.info("")
    logger.info("  [CB T+0] Convertible Bond:")
    if cb_report:
        logger.info(f"    Trades:   {len(cb_report.trades)}")
        logger.info(f"    Win/Loss: {cb_report.win_count}/{cb_report.loss_count}")
        logger.info(f"    PnL:      {cb_report.total_pnl:+.2f} ({cb_report.total_pnl_pct:+.2f}%)")
    else:
        logger.info("    - No trades")

    # 操作建议
    logger.info("")
    logger.info("  [ACTION] Recommended:")
    if market == "bear":
        logger.info("    ⚠️  熊市环境，建议轻仓或空仓观望")
    if stock_results:
        top = stock_results[0]
        sig = top.get("signal_type", top.get("ma_trend", ""))
        if market == "bull":
            logger.info(f"    STOCK: {top.get('name', '?')}({top['code']}) ALL-IN | {sig} | score={top.get('tech_score', 0)}")
        elif market == "sideways":
            pos_pct = "50%" if sig == "pre_dragon" else "30%"
            logger.info(f"    STOCK: {top.get('name', '?')}({top['code']}) {pos_pct}仓 | {sig} | score={top.get('tech_score', 0)}")
        else:
            logger.info(f"    STOCK: {top.get('name', '?')}({top['code']}) 20%仓试探 | {sig} | score={top.get('tech_score', 0)}")
    if hot_concepts:
        logger.info(f"    SECTOR: Focus on [{hot_concepts[0]['concept']}]")

    logger.info("")
    logger.info("#" * 60)

    return {
        "hot_concepts": hot_concepts,
        "stock_candidates": stock_results,
        "cb_report": cb_report,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 命令行入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Wolf of Wall Street - Dual Mode Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dual_trader.py                   # Full simulation
  python dual_trader.py --live            # Live trading (THS)
  python dual_trader.py --live --capital 50000
  python dual_trader.py --news-only       # Just check news
  python dual_trader.py --stock-only      # Just scan stocks
  python dual_trader.py --cb-only         # Just run CB T+0
        """,
    )
    parser.add_argument("--live", action="store_true", help="Live trading mode (connect THS)")
    parser.add_argument("--capital", type=float, default=TOTAL_CAPITAL, help="Total capital (default 30000)")
    parser.add_argument("--news-only", action="store_true", help="Only run news scan")
    parser.add_argument("--stock-only", action="store_true", help="Only run stock scan")
    parser.add_argument("--cb-only", action="store_true", help="Only run CB T+0")
    args = parser.parse_args()

    capital = args.capital

    if args.news_only:
        run_news_scan()
    elif args.stock_only:
        hot = run_news_scan()
        run_stock_scan(hot_concepts=hot)
    elif args.cb_only:
        run_cb_trading(broker=get_broker(args.live), capital=capital * CB_RATIO)
    else:
        run_all(live=args.live, capital=capital)
