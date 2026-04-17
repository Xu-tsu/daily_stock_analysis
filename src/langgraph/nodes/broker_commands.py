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
        from src.broker.safety import resume_trading, get_trading_state
        st = get_trading_state()
        if st["can_buy"] and st["can_sell"]:
            return {"response": "✅ 交易本就处于正常状态（买卖均可）"}
        resume_trading()
        return {"response": "✅ 已恢复自动交易（买入+卖出均已开启）"}
    except Exception as e:
        return {"response": f"❌ 操作失败: {e}"}


def broker_halt_buy_node(state: dict) -> dict:
    """只停自动买入（仍允许止损/减仓/清仓）"""
    try:
        from src.broker.safety import halt_buy, is_buy_halted
        if is_buy_halted():
            return {"response": "⚠️ 自动买入已经处于关闭状态"}
        halt_buy("用户在对话中关闭")
        return {"response": "🛑 已关闭自动买入（卖出仍可执行）\n发送「开启买入」恢复"}
    except Exception as e:
        return {"response": f"❌ 操作失败: {e}"}


def broker_resume_buy_node(state: dict) -> dict:
    """恢复自动买入"""
    try:
        from src.broker.safety import resume_buy, is_buy_halted
        if not is_buy_halted():
            return {"response": "✅ 自动买入本就处于开启状态"}
        resume_buy()
        return {"response": "✅ 已开启自动买入"}
    except Exception as e:
        return {"response": f"❌ 操作失败: {e}"}


def broker_halt_sell_node(state: dict) -> dict:
    """只停自动卖出（极少用，用于锁仓场景）"""
    try:
        from src.broker.safety import halt_sell, is_sell_halted
        if is_sell_halted():
            return {"response": "⚠️ 自动卖出已经处于关闭状态"}
        halt_sell("用户在对话中关闭")
        return {
            "response": "🛑 已关闭自动卖出（注意：止损/减仓也会被阻止！）\n发送「开启卖出」恢复"
        }
    except Exception as e:
        return {"response": f"❌ 操作失败: {e}"}


def broker_resume_sell_node(state: dict) -> dict:
    """恢复自动卖出"""
    try:
        from src.broker.safety import resume_sell, is_sell_halted
        if not is_sell_halted():
            return {"response": "✅ 自动卖出本就处于开启状态"}
        resume_sell()
        return {"response": "✅ 已开启自动卖出"}
    except Exception as e:
        return {"response": f"❌ 操作失败: {e}"}


def broker_status_node(state: dict) -> dict:
    """查询当前交易开关状态"""
    try:
        from src.broker.safety import get_trading_state, get_daily_order_count
        st = get_trading_state()
        daily = get_daily_order_count()
        lines = ["📊 **自动交易开关状态**"]
        lines.append(f"  买入: {'✅ 开' if st['can_buy'] else '🛑 关'}")
        lines.append(f"  卖出: {'✅ 开' if st['can_sell'] else '🛑 关'}")
        if st["halted_all"]:
            lines.append("  状态: 🚨 全局急停中")
        lines.append(f"  今日委托数: {daily}")
        lines.append("")
        lines.append("可用命令:")
        lines.append("  「停止交易」/「急停」 — 全部关闭")
        lines.append("  「恢复交易」 — 全部开启")
        lines.append("  「关闭买入」/「停止买入」 — 只关买入")
        lines.append("  「开启买入」/「恢复买入」 — 开买入")
        lines.append("  「关闭卖出」/「停止卖出」 — 只关卖出")
        lines.append("  「开启卖出」/「恢复卖出」 — 开卖出")
        lines.append("  「交易状态」 — 查看当前状态")
        return {"response": "\n".join(lines)}
    except Exception as e:
        return {"response": f"❌ 状态查询失败: {e}"}
