# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 主调度程序
===================================

职责：
1. 协调各模块完成股票分析流程
2. 实现低并发的线程池调度
3. 全局异常处理，确保单股失败不影响整体
4. 提供命令行入口

使用方式：
    python main.py              # 正常运行
    python main.py --debug      # 调试模式
    python main.py --dry-run    # 仅获取数据不分析

交易理念（已融入分析）：
- 严进策略：不追高，乖离率 > 5% 不买入
- 趋势交易：只做 MA5>MA10>MA20 多头排列
- 效率优先：关注筹码集中度好的股票
- 买点偏好：缩量回踩 MA5/MA10 支撑
"""
import mimetypes
_original_guess = mimetypes.guess_type
def _patched_guess(url, strict=True):
    url_str = str(url)
    if url_str.endswith('.js') or url_str.endswith('.mjs'):
        return ('application/javascript', None)
    if url_str.endswith('.css'):
        return ('text/css', None)
    return _original_guess(url, strict)
mimetypes.guess_type = _patched_guess
import os
from src.config import setup_env
setup_env()

# 代理配置 - 通过 USE_PROXY 环境变量控制，默认关闭
# GitHub Actions 环境自动跳过代理配置
if os.getenv("GITHUB_ACTIONS") != "true" and os.getenv("USE_PROXY", "false").lower() == "true":
    # 本地开发环境，启用代理（可在 .env 中配置 PROXY_HOST 和 PROXY_PORT）
    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
    proxy_port = os.getenv("PROXY_PORT", "10809")
    proxy_url = f"http://{proxy_host}:{proxy_port}"
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

from data_provider.base import canonical_stock_code
from src.core.pipeline import StockAnalysisPipeline
from src.core.market_review import run_market_review
from src.webui_frontend import prepare_webui_frontend_assets
from src.config import get_config, Config
from src.logging_config import setup_logging


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='A股自选股智能分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python main.py                    # 正常运行
  python main.py --debug            # 调试模式
  python main.py --dry-run          # 仅获取数据，不进行 AI 分析
  python main.py --stocks 600519,000001  # 指定分析特定股票
  python main.py --no-notify        # 不发送推送通知
  python main.py --single-notify    # 启用单股推送模式（每分析完一只立即推送）
  python main.py --schedule         # 启用定时任务模式
  python main.py --market-review    # 仅运行大盘复盘
        '''
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式，输出详细日志'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅获取数据，不进行 AI 分析'
    )

    parser.add_argument(
        '--stocks',
        type=str,
        help='指定要分析的股票代码，逗号分隔（覆盖配置文件）'
    )

    parser.add_argument(
        '--no-notify',
        action='store_true',
        help='不发送推送通知'
    )

    parser.add_argument(
        '--single-notify',
        action='store_true',
        help='启用单股推送模式：每分析完一只股票立即推送，而不是汇总推送'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='并发线程数（默认使用配置值）'
    )

    parser.add_argument(
        '--schedule',
        action='store_true',
        help='启用定时任务模式（含日终主任务 + 10:15/12:30 盘中节点）'
    )

    parser.add_argument(
        '--no-run-immediately',
        action='store_true',
        help='定时任务启动时不立即执行一次'
    )

    parser.add_argument(
        '--market-review',
        action='store_true',
        help='仅运行大盘复盘分析'
    )

    parser.add_argument(
        '--no-market-review',
        action='store_true',
        help='跳过大盘复盘分析'
    )

    parser.add_argument(
        '--force-run',
        action='store_true',
        help='跳过交易日检查，强制执行全量分析（Issue #373）'
    )

    parser.add_argument(
        '--webui',
        action='store_true',
        help='启动 Web 管理界面'
    )

    parser.add_argument(
        '--webui-only',
        action='store_true',
        help='仅启动 Web 服务，不执行自动分析'
    )

    parser.add_argument(
        '--serve',
        action='store_true',
        help='启动 FastAPI 后端服务（同时执行分析任务）'
    )

    parser.add_argument(
        '--serve-only',
        action='store_true',
        help='仅启动 FastAPI 后端服务，不自动执行分析'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='FastAPI 服务端口（默认 8000）'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='FastAPI 服务监听地址（默认 0.0.0.0）'
    )

    parser.add_argument(
        '--no-context-snapshot',
        action='store_true',
        help='不保存分析上下文快照'
    )

    # === Backtest ===
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='运行回测（对历史分析结果进行评估）'
    )

    parser.add_argument(
        '--backtest-code',
        type=str,
        default=None,
        help='仅回测指定股票代码'
    )

    parser.add_argument(
        '--backtest-days',
        type=int,
        default=None,
        help='回测评估窗口（交易日数，默认使用配置）'
    )

    parser.add_argument(
        '--backtest-force',
        action='store_true',
        help='强制回测（即使已有回测结果也重新计算）'
    )
    parser.add_argument("--all", action="store_true",
                        help="启动全部功能：Web + 定时分析 + 盘中监控")
    parser.add_argument("--monitor", action="store_true",
                        help="启动盘中实时监控（含集合竞价资金、异动与收盘调仓）")
    parser.add_argument("--interval", type=int, default=1,
                        help="监控检查间隔（分钟，默认1）")
    parser.add_argument("--rebalance", action="store_true",
                       help="执行持仓调仓分析（大盘→板块→个股→调仓建议）")
    parser.add_argument("--portfolio", type=str, default=None,
                       help="持仓文件路径，默认 data/portfolio.json")
    return parser.parse_args()


def _compute_trading_day_filter(
    config: Config,
    args: argparse.Namespace,
    stock_codes: List[str],
) -> Tuple[List[str], Optional[str], bool]:
    """
    Compute filtered stock list and effective market review region (Issue #373).

    Returns:
        (filtered_codes, effective_region, should_skip_all)
        - effective_region None = use config default (check disabled)
        - effective_region '' = all relevant markets closed, skip market review
        - should_skip_all: skip entire run when no stocks and no market review to run
    """
    force_run = getattr(args, 'force_run', False)
    if force_run or not getattr(config, 'trading_day_check_enabled', True):
        return (stock_codes, None, False)

    from src.core.trading_calendar import (
        get_market_for_stock,
        get_open_markets_today,
        compute_effective_region,
    )

    open_markets = get_open_markets_today()
    filtered_codes = []
    for code in stock_codes:
        mkt = get_market_for_stock(code)
        if mkt in open_markets or mkt is None:
            filtered_codes.append(code)

    if config.market_review_enabled and not getattr(args, 'no_market_review', False):
        effective_region = compute_effective_region(
            getattr(config, 'market_review_region', 'cn') or 'cn', open_markets
        )
    else:
        effective_region = None

    should_skip_all = (not filtered_codes) and (effective_region or '') == ''
    return (filtered_codes, effective_region, should_skip_all)


def run_full_analysis(
    config: Config,
    args: argparse.Namespace,
    stock_codes: Optional[List[str]] = None
):
    """执行完整的分析流程（扫描 + 个股 + 大盘复盘）"""
    try:
        if stock_codes is None:
            config.refresh_stock_list()

        # === 新版全市场扫描 ===
        if os.getenv("USE_NEW_SCANNER", "false").lower() == "true":
            try:
                from market_scanner import scan_market as new_scan
                logger.info("正在执行全市场扫描（新版）...")
                candidates = new_scan(max_price=10.0, min_turnover=2.0, top_n=10, mode="trend")
                if candidates:
                    scan_codes = [c["code"] for c in candidates]
                    logger.info(f"扫描到 {len(scan_codes)} 只候选: {scan_codes}")
                    current_list = stock_codes if stock_codes is not None else config.stock_list
                    stock_codes = list(dict.fromkeys(current_list + scan_codes))
                    logger.info(f"合并后分析列表: {len(stock_codes)} 只")
                    try:
                        from data_store import save_scan_results
                        save_scan_results(candidates)
                    except:
                        pass
            except Exception as e:
                logger.warning(f"全市场扫描失败（已跳过）: {e}")
        elif os.getenv("SCHEDULE_RUN_SCANNER", "false").lower() == "true":
            try:
                from scanner import scan_market
                logger.info("正在执行全市场扫描（原版）...")
                candidates = scan_market(max_cap=200, min_turnover=2.0, max_bias=5.0, top_n=10)
                if candidates:
                    scan_codes = [c["代码"] for c in candidates]
                    logger.info(f"扫描到 {len(scan_codes)} 只强势股: {scan_codes}")
                    current_list = stock_codes if stock_codes is not None else config.stock_list
                    stock_codes = list(dict.fromkeys(current_list + scan_codes))
                    logger.info(f"合并后分析列表: {len(stock_codes)} 只")
            except Exception as e:
                logger.warning(f"全市场扫描失败（已跳过）: {e}")

        # === 交易日过滤 ===
        effective_codes = stock_codes if stock_codes is not None else config.stock_list
        filtered_codes, effective_region, should_skip = _compute_trading_day_filter(
            config, args, effective_codes
        )
        if should_skip:
            logger.info("今日所有相关市场均为非交易日，跳过执行。可使用 --force-run 强制执行。")
            return
        if set(filtered_codes) != set(effective_codes):
            skipped = set(effective_codes) - set(filtered_codes)
            logger.info("今日休市股票已跳过: %s", skipped)
        stock_codes = filtered_codes

        if getattr(args, 'single_notify', False):
            config.single_stock_notify = True

        merge_notification = (
            getattr(config, 'merge_email_notification', False)
            and config.market_review_enabled
            and not getattr(args, 'no_market_review', False)
            and not config.single_stock_notify
        )

        # === 创建分析管道 ===
        save_context_snapshot = None
        if getattr(args, 'no_context_snapshot', False):
            save_context_snapshot = False
        query_id = uuid.uuid4().hex
        pipeline = StockAnalysisPipeline(
            config=config,
            max_workers=args.workers,
            query_id=query_id,
            query_source="cli",
            save_context_snapshot=save_context_snapshot
        )

        # === 1. 运行个股分析 ===
        results = pipeline.run(
            stock_codes=stock_codes,
            dry_run=args.dry_run,
            send_notification=not args.no_notify,
            merge_notification=merge_notification
        )

        # 分析间隔
        analysis_delay = getattr(config, 'analysis_delay', 0)
        if (
            analysis_delay > 0
            and config.market_review_enabled
            and not args.no_market_review
            and effective_region != ''
        ):
            logger.info(f"等待 {analysis_delay} 秒后执行大盘复盘...")
            time.sleep(analysis_delay)

        # === 2. 大盘复盘 ===
        market_report = ""
        if (
            config.market_review_enabled
            and not args.no_market_review
            and effective_region != ''
        ):
            review_result = run_market_review(
                notifier=pipeline.notifier,
                analyzer=pipeline.analyzer,
                search_service=pipeline.search_service,
                send_notification=not args.no_notify,
                merge_notification=merge_notification,
                override_region=effective_region,
            )
            if review_result:
                market_report = review_result

        # === 3. 合并推送 ===
        if merge_notification and (results or market_report) and not args.no_notify:
            parts = []
            if market_report:
                parts.append(f"# 📈 大盘复盘\n\n{market_report}")
            if results:
                dashboard_content = pipeline.notifier.generate_aggregate_report(
                    results, getattr(config, 'report_type', 'simple'),
                )
                parts.append(f"# 🚀 个股决策仪表盘\n\n{dashboard_content}")
            if parts:
                combined_content = "\n\n---\n\n".join(parts)
                if pipeline.notifier.is_available():
                    if pipeline.notifier.send(combined_content, email_send_to_all=True):
                        logger.info("已合并推送（个股+大盘复盘）")

        # === 4. 输出摘要 ===
        if results:
            logger.info("\n===== 分析结果摘要 =====")
            for r in sorted(results, key=lambda x: x.sentiment_score, reverse=True):
                emoji = r.get_emoji()
                logger.info(
                    f"{emoji} {r.name}({r.code}): {r.operation_advice} | "
                    f"评分 {r.sentiment_score} | {r.trend_prediction}"
                )

        logger.info("\n任务执行完成")

        # === 5. 飞书云文档 ===
        try:
            from src.feishu_doc import FeishuDocManager
            feishu_doc = FeishuDocManager()
            if feishu_doc.is_configured() and (results or market_report):
                tz_cn = timezone(timedelta(hours=8))
                now = datetime.now(tz_cn)
                doc_title = f"{now.strftime('%Y-%m-%d %H:%M')} 大盘复盘"
                full_content = ""
                if market_report:
                    full_content += f"# 📈 大盘复盘\n\n{market_report}\n\n---\n\n"
                if results:
                    dashboard_content = pipeline.notifier.generate_aggregate_report(
                        results, getattr(config, 'report_type', 'simple'),
                    )
                    full_content += f"# 🚀 个股决策仪表盘\n\n{dashboard_content}"
                doc_url = feishu_doc.create_daily_doc(doc_title, full_content)
                if doc_url:
                    logger.info(f"飞书云文档创建成功: {doc_url}")
                    if not args.no_notify:
                        pipeline.notifier.send(f"[{now.strftime('%Y-%m-%d %H:%M')}] 复盘文档: {doc_url}")
        except Exception as e:
            logger.error(f"飞书文档生成失败: {e}")

        # === 6. 自动回测 ===
        try:
            if getattr(config, 'backtest_enabled', False):
                from src.services.backtest_service import BacktestService
                logger.info("开始自动回测...")
                service = BacktestService()
                stats = service.run_backtest(
                    force=False,
                    eval_window_days=getattr(config, 'backtest_eval_window_days', 10),
                    min_age_days=getattr(config, 'backtest_min_age_days', 14),
                    limit=200,
                )
                logger.info(
                    f"自动回测完成: processed={stats.get('processed')} "
                    f"saved={stats.get('saved')} errors={stats.get('errors')}"
                )
        except Exception as e:
            logger.warning(f"自动回测失败: {e}")

    except Exception as e:
        logger.exception(f"分析流程执行失败: {e}")


def _auto_build_positions(portfolio: dict, config, args, candidates=None):
    """空仓时自动选股+直接在THS买入。

    流程: scan_market → 选3只 → 均分资金 → broker.buy → 更新 portfolio
    如果传入 candidates 则直接使用，不再重新扫描。
    """
    import os
    try:
        if not candidates:
            from market_scanner import scan_market
            candidates = scan_market(max_price=10.0, min_turnover=2.0, top_n=5, mode="trend")
        if not candidates:
            logger.info("[自动建仓] 扫描无合适候选，跳过")
            return

        picks = candidates[:3]
        logger.info(f"[自动建仓] 候选: {[(c['code'], c.get('name','')) for c in picks]}")

        broker_enabled = os.getenv("BROKER_ENABLED", "false").lower() == "true"
        total_cash = portfolio.get("cash", 0) or 200000
        per_stock_cash = total_cash * 0.25  # 每只分配25%资金，保留25%现金

        bought = []
        for c in picks:
            code = c["code"]
            name = c.get("name", code)
            price = c.get("price", 0)
            if price <= 0:
                # 尝试获取实时价
                try:
                    from macro_data_collector import _fetch_tencent_quote, _stock_code_to_tencent
                    tc = _stock_code_to_tencent(code)
                    qt = _fetch_tencent_quote([tc], timeout=5)
                    for _, q in qt.items():
                        if q.get("price", 0) > 0:
                            price = q["price"]
                except Exception:
                    pass
            if price <= 0:
                logger.warning(f"[自动建仓] {name}({code}) 无法获取价格，跳过")
                continue

            # 计算买入股数（向下取整到100的倍数）
            shares = int(per_stock_cash / price / 100) * 100
            if shares < 100:
                logger.warning(f"[自动建仓] {name}({code}) 资金不足买1手 ({price:.2f}元)，跳过")
                continue

            amount = shares * price
            logger.info(f"[自动建仓] 买入 {name}({code}) {shares}股 × {price:.2f}元 = {amount:,.0f}元")

            # 券商下单
            broker_success = False
            if broker_enabled:
                try:
                    from src.broker import get_broker
                    broker = get_broker()
                    if broker and broker.is_connected():
                        result = broker.buy(code, price, shares)
                        logger.info(f"[自动建仓] THS下单: {result.status} {result.message}")
                        if result.is_success:
                            broker_success = True
                            if result.actual_price and result.actual_price > 0:
                                price = result.actual_price
                            if result.actual_shares and result.actual_shares > 0:
                                shares = result.actual_shares
                        else:
                            logger.warning(f"[自动建仓] {name}({code}) 下单失败: {result.message}，跳过")
                            continue  # 下单失败，不记录不更新持仓
                except Exception as e:
                    logger.error(f"[自动建仓] THS下单失败: {e}，跳过 {name}")
                    continue
            else:
                # 非券商模式（模拟），直接视为成功
                broker_success = True

            # 记录到交易日志（复盘用）
            try:
                from trade_journal import record_buy
                record_buy(
                    code=code, name=name, shares=shares, price=price,
                    ma_trend=c.get("ma_trend", ""),
                    macd_signal=c.get("macd_signal", ""),
                    rsi=c.get("rsi", 0),
                    vol_pattern=c.get("vol_pattern", ""),
                    tech_score=c.get("tech_score", 0),
                    sector=c.get("sector", ""),
                    source="auto_build",
                    note=f"自动建仓 score={c.get('tech_score', 0)}",
                )
            except Exception as e:
                logger.warning(f"[自动建仓] 交易日志记录失败: {e}")

            # 更新本地持仓
            # 先移除同code的0股占位
            portfolio["holdings"] = [
                h for h in portfolio.get("holdings", [])
                if h.get("code") != code or h.get("shares", 0) > 0
            ]
            portfolio.setdefault("holdings", []).append({
                "code": code,
                "name": name,
                "shares": shares,
                "cost_price": price,
                "current_price": price,
                "market_value": round(price * shares, 2),
                "sector": c.get("sector", ""),
                "buy_date": datetime.now().strftime("%Y-%m-%d"),
                "strategy_tag": "auto_scan",
            })
            portfolio["cash"] = portfolio.get("cash", 0) - round(price * shares, 2)
            bought.append(f"{name}({code}) {shares}股@{price:.2f}")

        # 清理剩余0股占位
        portfolio["holdings"] = [h for h in portfolio.get("holdings", []) if h.get("shares", 0) > 0]

        from portfolio_manager import save_portfolio
        save_portfolio(portfolio)

        if bought:
            summary = f"🛒 自动建仓完成:\n" + "\n".join(f"  • {b}" for b in bought)
            logger.info(f"[自动建仓] {summary}")
            if not getattr(args, 'no_notify', False):
                try:
                    from src.notification import NotificationService
                    notifier = NotificationService()
                    if notifier.is_available():
                        notifier.send(summary)
                except Exception:
                    pass
        else:
            logger.info("[自动建仓] 未能成功买入任何股票")

    except Exception as e:
        logger.error(f"[自动建仓] 失败: {e}", exc_info=True)


def _run_checkpoint_with_rebalance(checkpoint: str, send_notification: bool):
    """盘中节点：先做市场快照，再跑调仓+自动执行。空仓时自动选股建仓。"""
    from market_monitor import run_intraday_checkpoint
    # 1) 盘中快照（指数/板块/异动）
    run_intraday_checkpoint(checkpoint=checkpoint, send_notification=send_notification)

    # 2) 检查持仓 → 空仓自动建仓 / 有仓正常调仓
    try:
        from portfolio_manager import load_portfolio, format_rebalance_report
        portfolio = load_portfolio()
        real_holdings = [h for h in portfolio.get("holdings", []) if h.get("shares", 0) > 0]

        if not real_holdings:
            logger.info(f"[{checkpoint}] 空仓，启动自动建仓...")
            import argparse as _ap
            _dummy_args = _ap.Namespace(no_notify=not send_notification)
            _auto_build_positions(portfolio, None, _dummy_args)
        else:
            from rebalance_engine import run_rebalance_analysis
            from src.config import get_config
            logger.info(f"[{checkpoint}] 开始调仓分析...")
            result = run_rebalance_analysis(config=get_config())
            if "error" not in result:
                report = format_rebalance_report(result)
                logger.info(f"\n{report}")
                if send_notification:
                    try:
                        from src.notification import NotificationService
                        notifier = NotificationService()
                        if notifier.is_available():
                            notifier.send(report)
                    except Exception:
                        pass
                execution = result.get("_execution", {})
                if execution:
                    logger.info(f"[{checkpoint}] 券商执行: {execution.get('mode', 'N/A')}")
            else:
                logger.warning(f"[{checkpoint}] 调仓失败: {result.get('error')}")
    except Exception as e:
        logger.error(f"[{checkpoint}] 调仓异常: {e}")


def build_intraday_schedule_tasks(args: argparse.Namespace):
    """Build fixed intraday checkpoint tasks for scheduler mode."""
    send_notification = not getattr(args, "no_notify", False)
    return [
        (
            "9:45",
            lambda: _run_checkpoint_with_rebalance(
                checkpoint="morning_review",
                send_notification=send_notification,
            ),
            "morning_review",
        ),
        (
            "13:15",
            lambda: _run_checkpoint_with_rebalance(
                checkpoint="afternoon_review",
                send_notification=send_notification,
            ),
            "afternoon_review",
        ),
    ]


def start_api_server(host: str, port: int, config: Config) -> None:
    """
    在后台线程启动 FastAPI 服务
    
    Args:
        host: 监听地址
        port: 监听端口
        config: 配置对象
    """
    import threading
    import uvicorn

    def run_server():
        level_name = (config.log_level or "INFO").lower()
        uvicorn.run(
            "api.app:app",
            host=host,
            port=port,
            log_level=level_name,
            log_config=None,
        )

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info(f"FastAPI 服务已启动: http://{host}:{port}")


def _is_truthy_env(var_name: str, default: str = "true") -> bool:
    """Parse common truthy / falsy environment values."""
    value = os.getenv(var_name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}

def start_bot_stream_clients(config: Config) -> None:
    """Start bot stream clients when enabled in config."""
    # 启动钉钉 Stream 客户端
    if config.dingtalk_stream_enabled:
        try:
            from bot.platforms import start_dingtalk_stream_background, DINGTALK_STREAM_AVAILABLE
            if DINGTALK_STREAM_AVAILABLE:
                if start_dingtalk_stream_background():
                    logger.info("[Main] Dingtalk Stream client started in background.")
                else:
                    logger.warning("[Main] Dingtalk Stream client failed to start.")
            else:
                logger.warning("[Main] Dingtalk Stream enabled but SDK is missing.")
                logger.warning("[Main] Run: pip install dingtalk-stream")
        except Exception as exc:
            logger.error(f"[Main] Failed to start Dingtalk Stream client: {exc}")

    # 启动飞书 Stream 客户端
    if getattr(config, 'feishu_stream_enabled', False):
        try:
            from bot.platforms import start_feishu_stream_background, FEISHU_SDK_AVAILABLE
            if FEISHU_SDK_AVAILABLE:
                if start_feishu_stream_background():
                    logger.info("[Main] Feishu Stream client started in background.")
                else:
                    logger.warning("[Main] Feishu Stream client failed to start.")
            else:
                logger.warning("[Main] Feishu Stream enabled but SDK is missing.")
                logger.warning("[Main] Run: pip install lark-oapi")
        except Exception as exc:
            logger.error(f"[Main] Failed to start Feishu Stream client: {exc}")


def main() -> int:
    """
    主入口函数

    Returns:
        退出码（0 表示成功）
    """
    # 解析命令行参数
    args = parse_arguments()

    # 加载配置（在设置日志前加载，以获取日志目录）
    config = get_config()

    # 配置日志（输出到控制台和文件）
    setup_logging(log_prefix="stock_analysis", debug=args.debug, log_dir=config.log_dir)

    logger.info("=" * 60)
    logger.info("A股自选股智能分析系统 启动")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 验证配置
    warnings = config.validate()
    for warning in warnings:
        logger.warning(warning)

    # 解析股票列表（统一为大写 Issue #355）
    stock_codes = None
    if args.stocks:
        stock_codes = [canonical_stock_code(c) for c in args.stocks.split(',') if (c or "").strip()]
        logger.info(f"使用命令行指定的股票列表: {stock_codes}")

    # === 处理 --webui / --webui-only 参数，映射到 --serve / --serve-only ===
    if args.webui:
        args.serve = True
    if args.webui_only:
        args.serve_only = True

    # 兼容旧版 WEBUI_ENABLED 环境变量
    if config.webui_enabled and not (args.serve or args.serve_only):
        args.serve = True

    # === 启动 Web 服务 (如果启用) ===
    start_serve = (args.serve or args.serve_only) and os.getenv("GITHUB_ACTIONS") != "true"

    # 兼容旧版 WEBUI_HOST/WEBUI_PORT：如果用户未通过 --host/--port 指定，则使用旧变量
    if start_serve:
        if args.host == '0.0.0.0' and os.getenv('WEBUI_HOST'):
            args.host = os.getenv('WEBUI_HOST')
        if args.port == 8000 and os.getenv('WEBUI_PORT'):
            args.port = int(os.getenv('WEBUI_PORT'))

    bot_clients_started = False
    if start_serve:
        if not prepare_webui_frontend_assets():
            logger.warning("前端静态资源未就绪，继续启动 FastAPI 服务（Web 页面可能不可用）")
        try:
            start_api_server(host=args.host, port=args.port, config=config)
            bot_clients_started = True
        except Exception as e:
            logger.error(f"启动 FastAPI 服务失败: {e}")

    if bot_clients_started:
        start_bot_stream_clients(config)

    # === 仅 Web 服务模式：不自动执行分析 ===
    if args.serve_only:
        logger.info("模式: 仅 Web 服务")
        logger.info(f"Web 服务运行中: http://{args.host}:{args.port}")
        logger.info("通过 /api/v1/analysis/analyze 接口触发分析")
        logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
        logger.info("按 Ctrl+C 退出...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n用户中断，程序退出")
        return 0

    try:
        # ━━━ 模式A: 全功能模式（所有模块一键启动）━━━
        if getattr(args, 'all', False):
            logger.info("=" * 60)
            logger.info("  全功能模式启动")
            logger.info("  新闻情报 + 市场环境 + 智能选股 + 调仓")
            logger.info("  可转债T+0 + 盘中监控 + 技能进化 + 复盘")
            logger.info("=" * 60)

            # 强制跳过交易日检查（监控模块自己判断交易时间）
            args.force_run = True

            import threading

            # ━━━ Phase 0: 市场环境检测 ━━━
            regime = {"regime": "sideways", "score": 0, "detail": ""}
            try:
                from market_scanner import detect_market_regime
                regime = detect_market_regime()
                regime_map = {"bull": "牛市(追涨/连板)", "sideways": "震荡(均值回归+预埋)", "bear": "熊市(防守/超跌)"}
                logger.info(f"[Phase 0] 市场环境: {regime_map.get(regime['regime'], regime['regime'])} score={regime['score']}")
            except Exception as e:
                logger.warning(f"[Phase 0] 市场环境检测失败: {e}")

            # ━━━ Phase 1: 新闻情报扫描 ━━━
            hot_concepts = []
            try:
                from dual_trader import run_news_scan
                hot_concepts = run_news_scan()
                if hot_concepts:
                    logger.info(f"[Phase 1] 热点概念: {', '.join(c['concept'] for c in hot_concepts[:3])}")
                else:
                    logger.info("[Phase 1] 暂无热点概念")
            except Exception as e:
                logger.warning(f"[Phase 1] 新闻扫描失败: {e}")

            # ━━━ Phase 2: 技能引擎加载 ━━━
            try:
                from agent_skill_engine import get_skill_engine
                skill_engine = get_skill_engine()
                logger.info(f"[Phase 2] {skill_engine.get_skill_summary()}")
            except Exception as e:
                logger.warning(f"[Phase 2] 技能引擎加载失败: {e}")

            # ━━━ Phase 3: 智能选股 + 调仓/建仓 ━━━
            skip_startup_analysis = os.getenv("SKIP_STARTUP_ANALYSIS", "false").lower() in ("true", "1", "yes")
            try:
                from portfolio_manager import load_portfolio, save_portfolio, format_rebalance_report
                portfolio = load_portfolio()
                real_holdings = [h for h in portfolio.get("holdings", []) if h.get("shares", 0) > 0]

                if not real_holdings:
                    # ── 空仓：智能选股 → 建仓 ──
                    logger.info("[Phase 3] 空仓状态，智能选股+建仓...")

                    # 用dual_trader的智能选股（结合新闻+市场环境）
                    try:
                        from dual_trader import run_stock_scan
                        smart_candidates = run_stock_scan(hot_concepts=hot_concepts, regime=regime)
                        if smart_candidates:
                            logger.info(f"[Phase 3] 智能选股结果: {len(smart_candidates)} 只候选")
                            # 将智能选股结果传递给建仓（替代默认扫描）
                            _auto_build_positions(portfolio, config, args, candidates=smart_candidates)
                        else:
                            logger.info("[Phase 3] 智能选股无候选，使用默认扫描建仓")
                            _auto_build_positions(portfolio, config, args)
                    except Exception as e:
                        logger.warning(f"[Phase 3] 智能选股失败({e})，使用默认扫描")
                        _auto_build_positions(portfolio, config, args)
                else:
                    # ── 有仓：正常调仓 ──
                    logger.info(f"[Phase 3] 有{len(real_holdings)}只持仓，执行调仓分析...")
                    from rebalance_engine import run_rebalance_analysis
                    result = run_rebalance_analysis(config=config)
                    if "error" not in result:
                        report = format_rebalance_report(result)
                        logger.info(f"\n{report}")
                        if not args.no_notify:
                            try:
                                from src.notification import NotificationService
                                notifier = NotificationService()
                                if notifier.is_available():
                                    notifier.send(report)
                            except:
                                pass
                        execution = result.get("_execution", {})
                        if execution:
                            logger.info(f"券商执行: {execution.get('mode', 'N/A')}")
                    else:
                        logger.warning(f"调仓失败: {result.get('error')}")
            except Exception as e:
                logger.warning(f"[Phase 3] 调仓失败: {e}")

            # ━━━ Phase 4: 可转债T+0（后台线程）━━━
            def _cb_trading():
                try:
                    from dual_trader import run_cb_trading
                    broker = None
                    if os.getenv("BROKER_ENABLED", "false").lower() in ("true", "1", "yes"):
                        from src.broker import get_broker as _get_broker
                        broker = _get_broker()
                    cb_capital = float(os.getenv("CB_CAPITAL", "12000"))
                    logger.info(f"[Phase 4] 可转债T+0启动, 资金={cb_capital:.0f}")
                    report = run_cb_trading(broker=broker, capital=cb_capital)
                    if report:
                        logger.info(f"[Phase 4] 可转债完成: {report}")
                except Exception as e:
                    logger.warning(f"[Phase 4] 可转债跳过: {e}")

            threading.Thread(target=_cb_trading, daemon=True).start()

            # ━━━ Phase 5: 盘中监控（后台线程）━━━
            def _monitor():
                from market_monitor import run_monitor_loop
                run_monitor_loop(
                    interval_minutes=getattr(args, 'interval', 1),
                    auto_rebalance=True,
                )

            threading.Thread(target=_monitor, daemon=True).start()
            logger.info("[Phase 5] 盘中监控已在后台启动")

            # ━━━ Phase 6: 全量分析报告 ━━━
            if not skip_startup_analysis:
                logger.info("[Phase 6] 开始全量分析报告...")
                try:
                    run_full_analysis(config, args, stock_codes)
                except Exception as e:
                    logger.warning(f"[Phase 6] 全量分析失败: {e}")
            else:
                logger.info("[Phase 6] 跳过全量分析（SKIP_STARTUP_ANALYSIS=true）")

            # ━━━ 启动摘要 ━━━
            logger.info("")
            logger.info("=" * 60)
            logger.info("  全模块启动完成")
            logger.info(f"  市场环境: {regime.get('regime', '?').upper()} (score={regime.get('score', 0)})")
            logger.info(f"  热点概念: {len(hot_concepts)} 个")
            logger.info(f"  持仓管理: 调仓/建仓已执行")
            logger.info(f"  可转债T+0: 后台运行中")
            logger.info(f"  盘中监控: 后台运行中 (1分钟间隔)")
            logger.info(f"  收盘任务: 复盘报告+技能进化+涨停分析 (15:00自动)")
            logger.info("=" * 60)

            # ━━━ 定时循环 ━━━
            from src.scheduler import run_with_schedule
            run_with_schedule(
                task=lambda: run_full_analysis(config, args, stock_codes),
                schedule_time=config.schedule_time,
                run_immediately=False,
                extra_daily_tasks=build_intraday_schedule_tasks(args),
            )
            return 0
        # ━━━ 模式M: 盘中实时监控 ━━━
        if getattr(args, 'monitor', False):
            logger.info("模式: 盘中实时监控")
            from market_monitor import run_monitor_loop
            run_monitor_loop(
                interval_minutes=getattr(args, 'interval', 1),
                auto_rebalance=True,
            )
            return 0
        # ━━━ 模式R: 持仓调仓分析（多Agent） ━━━
        if getattr(args, 'rebalance', False):
            logger.info("模式: 持仓调仓分析（多Agent）")

            # 如果指定了持仓文件路径
            if getattr(args, 'portfolio', None):
                os.environ["PORTFOLIO_FILE"] = args.portfolio

            # 交易日检查
            if not getattr(args, 'force_run', False) and getattr(config, 'trading_day_check_enabled', True):
                try:
                    from src.core.trading_calendar import get_open_markets_today
                    open_markets = get_open_markets_today()
                    if 'cn' not in open_markets and 'CN' not in open_markets:
                        logger.info("今日A股非交易日，跳过调仓分析。可使用 --force-run 强制执行。")
                        return 0
                except Exception:
                    pass  # 如果交易日历模块出错就不拦截

            try:
                from rebalance_engine import run_rebalance_analysis
                from portfolio_manager import format_rebalance_report, load_portfolio

                # 检查持仓文件是否存在
                portfolio = load_portfolio()
                if not portfolio.get("holdings"):
                    logger.error("持仓为空！请先编辑 data/portfolio.json 填入你的持仓信息")
                    logger.error("模板参考: https://github.com/你的仓库/data/portfolio.json")
                    return 1

                holding_names = [
                    f"{h.get('name', '')}({h['code']})"
                    for h in portfolio["holdings"]
                ]
                logger.info(f"当前持仓 {len(portfolio['holdings'])} 只: {', '.join(holding_names)}")
                logger.info(f"现金: {portfolio.get('cash', 0)}")

                # 执行多Agent调仓分析
                logger.info("开始执行多Agent调仓分析...")
                rebalance_result = run_rebalance_analysis(config=config)

                if "error" in rebalance_result:
                    logger.error(f"调仓分析失败: {rebalance_result['error']}")
                    return 1

                # 格式化报告
                report = format_rebalance_report(rebalance_result)
                logger.info("\n" + "=" * 60)
                logger.info("调仓建议报告:")
                logger.info("=" * 60)
                logger.info("\n" + report)

                # 推送通知（复用项目已有的通知模块）
                if not args.no_notify:
                    try:
                        from src.notification import NotificationService
                        notifier = NotificationService()
                        if notifier.is_available():
                            notifier.send(report)
                            logger.info("调仓报告已推送")
                        else:
                            logger.warning("未配置通知渠道，跳过推送")
                    except Exception as e:
                        logger.error(f"推送失败: {e}")

                # 保存分析结果到本地
                import json as _json
                from pathlib import Path as _Path
                result_dir = _Path("data/rebalance_history")
                result_dir.mkdir(parents=True, exist_ok=True)
                filename = result_dir / f"rebalance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    _json.dump(rebalance_result, f, ensure_ascii=False, indent=2)
                logger.info(f"调仓结果已保存到 {filename}")

            except ImportError as e:
                logger.error(f"缺少调仓模块: {e}")
                logger.error(
                    "请确认 rebalance_engine.py, macro_data_collector.py, portfolio_manager.py 已放到项目根目录")
                return 1
            except Exception as e:
                logger.exception(f"调仓分析执行失败: {e}")
                return 1

            return 0
        # 模式0: 回测
        if getattr(args, 'backtest', False):
            logger.info("模式: 回测")
            from src.services.backtest_service import BacktestService

            service = BacktestService()
            stats = service.run_backtest(
                code=getattr(args, 'backtest_code', None),
                force=getattr(args, 'backtest_force', False),
                eval_window_days=getattr(args, 'backtest_days', None),
            )
            logger.info(
                f"回测完成: processed={stats.get('processed')} saved={stats.get('saved')} "
                f"completed={stats.get('completed')} insufficient={stats.get('insufficient')} errors={stats.get('errors')}"
            )
            return 0

        # 模式1: 仅大盘复盘
        if args.market_review:
            from src.analyzer import GeminiAnalyzer
            from src.core.market_review import run_market_review
            from src.notification import NotificationService
            from src.search_service import SearchService

            # Issue #373: Trading day check for market-review-only mode.
            # Do NOT use _compute_trading_day_filter here: that helper checks
            # config.market_review_enabled, which would wrongly block an
            # explicit --market-review invocation when the flag is disabled.
            effective_region = None
            if not getattr(args, 'force_run', False) and getattr(config, 'trading_day_check_enabled', True):
                from src.core.trading_calendar import get_open_markets_today, compute_effective_region as _compute_region
                open_markets = get_open_markets_today()
                effective_region = _compute_region(
                    getattr(config, 'market_review_region', 'cn') or 'cn', open_markets
                )
                if effective_region == '':
                    logger.info("今日大盘复盘相关市场均为非交易日，跳过执行。可使用 --force-run 强制执行。")
                    return 0

            logger.info("模式: 仅大盘复盘")
            notifier = NotificationService()

            # 初始化搜索服务和分析器（如果有配置）
            search_service = None
            analyzer = None

            if config.bocha_api_keys or config.tavily_api_keys or config.brave_api_keys or config.serpapi_keys or config.minimax_api_keys or config.searxng_base_urls:
                search_service = SearchService(
                    bocha_keys=config.bocha_api_keys,
                    tavily_keys=config.tavily_api_keys,
                    brave_keys=config.brave_api_keys,
                    serpapi_keys=config.serpapi_keys,
                    minimax_keys=config.minimax_api_keys,
                    searxng_base_urls=config.searxng_base_urls,
                    news_max_age_days=config.news_max_age_days,
                    news_strategy_profile=getattr(config, "news_strategy_profile", "short"),
                )

            if config.gemini_api_key or config.openai_api_key:
                analyzer = GeminiAnalyzer(api_key=config.gemini_api_key)
                if not analyzer.is_available():
                    logger.warning("AI 分析器初始化后不可用，请检查 API Key 配置")
                    analyzer = None
            else:
                logger.warning("未检测到 API Key (Gemini/OpenAI)，将仅使用模板生成报告")

            run_market_review(
                notifier=notifier,
                analyzer=analyzer,
                search_service=search_service,
                send_notification=not args.no_notify,
                override_region=effective_region,
            )
            return 0

        # 模式2: 定时任务模式
        if args.schedule or config.schedule_enabled:
            logger.info("模式: 定时任务")
            logger.info(f"每日执行时间: {config.schedule_time}")
            logger.info("附加盘中节点: 9:45 早盘调仓, 13:15 午后调仓")

            # Determine whether to run immediately:
            # Command line arg --no-run-immediately overrides config if present.
            # Otherwise use config (defaults to True).
            should_run_immediately = config.schedule_run_immediately
            if getattr(args, 'no_run_immediately', False):
                should_run_immediately = False

            logger.info(f"启动时立即执行: {should_run_immediately}")

            from src.scheduler import run_with_schedule

            def scheduled_task():
                run_full_analysis(config, args, stock_codes)

            run_with_schedule(
                task=scheduled_task,
                schedule_time=config.schedule_time,
                run_immediately=should_run_immediately,
                extra_daily_tasks=build_intraday_schedule_tasks(args),
            )
            return 0

        # 模式3: 正常单次运行
        if config.run_immediately:
            run_full_analysis(config, args, stock_codes)
        else:
            logger.info("配置为不立即运行分析 (RUN_IMMEDIATELY=false)")

        logger.info("\n程序执行完成")

        # 如果启用了服务且是非定时任务模式，保持程序运行
        keep_running = start_serve and not (args.schedule or config.schedule_enabled)
        if keep_running:
            logger.info("API 服务运行中 (按 Ctrl+C 退出)...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

        return 0

    except KeyboardInterrupt:
        logger.info("\n用户中断，程序退出")
        return 130

    except Exception as e:
        logger.exception(f"程序执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
