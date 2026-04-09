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

def run_stock_scan(hot_concepts: List[dict] = None, regime: dict = None) -> List[dict]:
    """智能选股（根据市场环境自动切换策略）"""
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

        # 根据市场环境选择策略
        if market == "bull":
            # 牛市：龙头打板，追连板，激进
            logger.info("  [STRATEGY] 牛市策略 → 龙头打板 + 连板接力")
            results = scan_market(
                mode="dragon",
                max_price=50.0,
                min_turnover=3.0,
                max_change=20.0,    # 允许涨停
                min_change=-3.0,
                top_n=15,
            )
        elif market == "bear":
            # 熊市：超跌反弹为主，极少出手
            logger.info("  [STRATEGY] 熊市策略 → 超跌反弹 + 极低仓位")
            results = scan_market(
                mode="oversold",
                max_price=30.0,
                min_turnover=1.5,
                max_change=5.0,     # 不追涨
                min_change=-8.0,    # 买深跌
                top_n=10,
            )
        else:
            # 震荡市：均值回归 + 提前埋伏龙头预期
            logger.info("  [STRATEGY] 震荡策略 → 均值回归 + 龙头预埋")
            results = scan_market(
                mode="trend",
                max_price=30.0,
                min_turnover=2.0,
                max_change=7.0,     # 不追高（7%以上不买）
                min_change=-5.0,
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
                # 去掉可能的前缀 SH/SZ
                clean_code = code.replace("SH", "").replace("SZ", "").replace("sh", "").replace("sz", "")
                if clean_code in hot_stock_codes:
                    r["tech_score"] = r.get("tech_score", 0) + 20
                    r["hot_concept"] = True
                else:
                    r["hot_concept"] = False

            results.sort(key=lambda x: x.get("tech_score", 0), reverse=True)

        # 打印结果
        logger.info(f"  [STOCK] Top candidates:")
        for i, r in enumerate(results[:10]):
            hot_tag = " [HOT]" if r.get("hot_concept") else ""
            sig = r.get("signal_type", r.get("ma_trend", ""))
            logger.info(
                f"    #{i+1} {r.get('name', '?')}({r['code']}) "
                f"price={r.get('price', 0):.2f} chg={r.get('change_pct', 0):+.1f}% "
                f"score={r.get('tech_score', 0)} {sig}{hot_tag}"
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
    """可转债T+0日内交易"""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Phase 2: CB T+0 (capital: {capital:.0f})")
    logger.info("=" * 60)

    try:
        from cb_trader import CBDayTrader

        trader = CBDayTrader(broker=broker, total_capital=capital / 0.4)
        trader.cb_capital = capital
        trader.cash = capital

        # 1. 早盘选债
        watchlist = trader.run_morning_scan()
        if not watchlist:
            logger.info("  [CB] No candidates, skip")
            return None

        # 2. 交易信号
        signals = trader.check_signals()
        logger.info(f"  [CB] Signals triggered: {len(signals)}")

        # 3. 尾盘清仓
        report = trader.close_day()
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
