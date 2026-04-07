"""
券商命令节点 — 同步持仓 / 执行质量 / 停止交易 / 恢复交易
"""
import logging
import os

logger = logging.getLogger(__name__)

BROKER_ENABLED = os.getenv("BROKER_ENABLED", "false").lower() == "true"


def broker_sync_node(state: dict) -> dict:
    """从券商同步持仓到本地"""
    if not BROKER_ENABLED:
        return {"response": "⚠️ 券商模块未启用（BROKER_ENABLED=false）"}

    try:
        from src.broker import get_broker
        from src.broker.sync import PositionSyncer
        from portfolio_manager import load_portfolio, save_portfolio

        broker = get_broker()
        if not broker or not broker.is_connected():
            return {"response": "⚠️ 券商未连接，请检查同花顺客户端是否启动"}

        portfolio = load_portfolio()
        syncer = PositionSyncer(broker)
        report = syncer.sync_to_local(portfolio)

        if report.synced:
            save_portfolio(portfolio)

        return {"response": report.format_summary()}

    except Exception as e:
        logger.error(f"[券商同步] 失败: {e}")
        return {"response": f"❌ 同步失败: {e}"}


def broker_quality_node(state: dict) -> dict:
    """显示执行质量报告"""
    try:
        from src.broker.execution_tracker import format_quality_report
        report = format_quality_report(days=30)
        return {"response": report}
    except Exception as e:
        logger.error(f"[执行质量] 查询失败: {e}")
        return {"response": f"❌ 查询失败: {e}"}


def broker_halt_node(state: dict) -> dict:
    """紧急停止所有自动下单"""
    try:
        from src.broker.safety import halt_trading, is_trading_halted
        if is_trading_halted():
            return {"response": "⚠️ 交易已经处于停止状态"}
        halt_trading()
        return {"response": "🛑 已停止所有自动交易\n发送「恢复交易」解除"}
    except Exception as e:
        return {"response": f"❌ 操作失败: {e}"}


def broker_resume_node(state: dict) -> dict:
    """恢复自动交易"""
    try:
        from src.broker.safety import resume_trading, is_trading_halted
        if not is_trading_halted():
            return {"response": "✅ 交易本就处于正常状态"}
        resume_trading()
        return {"response": "✅ 已恢复自动交易"}
    except Exception as e:
        return {"response": f"❌ 操作失败: {e}"}
