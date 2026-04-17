"""
交易安全限制 — 每日限额、急停开关、单笔上限
"""
import logging
import os
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

# ── 急停开关（内存级，进程内生效）──
# 三档开关:
#   _kill_switch        → True 则 buy/sell 全停
#   _halt_buy_only      → True 则只停买入（仍允许止损/减仓）
#   _halt_sell_only     → True 则只停卖出（很少用，但留口）
_kill_switch = False
_halt_buy_only = False
_halt_sell_only = False
_kill_switch_lock = threading.Lock()


def is_trading_halted() -> bool:
    """检查是否已触发"完全"急停"""
    return _kill_switch


def is_buy_halted() -> bool:
    """是否禁止买入（完全急停 或 买入单独急停）"""
    return _kill_switch or _halt_buy_only


def is_sell_halted() -> bool:
    """是否禁止卖出（完全急停 或 卖出单独急停）"""
    return _kill_switch or _halt_sell_only


def get_trading_state() -> dict:
    """返回当前交易开关状态，用于 UI 展示。"""
    return {
        "halted_all": _kill_switch,
        "halted_buy": _halt_buy_only,
        "halted_sell": _halt_sell_only,
        "can_buy": not (_kill_switch or _halt_buy_only),
        "can_sell": not (_kill_switch or _halt_sell_only),
    }


def halt_trading(reason: str = "手动停止") -> None:
    """触发急停（买卖全停）"""
    global _kill_switch
    with _kill_switch_lock:
        _kill_switch = True
    logger.warning(f"[安全] 交易已停止: {reason}")


def halt_buy(reason: str = "手动停止买入") -> None:
    """只停买入，保留卖出（用于顶部防御）"""
    global _halt_buy_only
    with _kill_switch_lock:
        _halt_buy_only = True
    logger.warning(f"[安全] 买入已停止: {reason}")


def halt_sell(reason: str = "手动停止卖出") -> None:
    """只停卖出（少用，极端行情持仓锁仓场景）"""
    global _halt_sell_only
    with _kill_switch_lock:
        _halt_sell_only = True
    logger.warning(f"[安全] 卖出已停止: {reason}")


def resume_trading() -> None:
    """解除全部急停（恢复买+卖）"""
    global _kill_switch, _halt_buy_only, _halt_sell_only
    with _kill_switch_lock:
        _kill_switch = False
        _halt_buy_only = False
        _halt_sell_only = False
    logger.info("[安全] 交易已恢复（全部）")


def resume_buy() -> None:
    """只恢复买入"""
    global _halt_buy_only
    with _kill_switch_lock:
        _halt_buy_only = False
    logger.info("[安全] 买入已恢复")


def resume_sell() -> None:
    """只恢复卖出"""
    global _halt_sell_only
    with _kill_switch_lock:
        _halt_sell_only = False
    logger.info("[安全] 卖出已恢复")


# ── 每日委托计数（内存级，每日重置）──
_daily_count = 0
_daily_date = ""
_count_lock = threading.Lock()


def _check_daily_reset():
    """如果日期变了，重置计数"""
    global _daily_count, _daily_date
    today = datetime.now().strftime("%Y-%m-%d")
    if _daily_date != today:
        _daily_count = 0
        _daily_date = today


def increment_order_count() -> None:
    """记录一笔委托"""
    global _daily_count
    with _count_lock:
        _check_daily_reset()
        _daily_count += 1


def get_daily_order_count() -> int:
    with _count_lock:
        _check_daily_reset()
        return _daily_count


def check_order_allowed(
    code: str,
    price: float,
    shares: int,
    total_asset: float,
    direction: str = "",
) -> tuple:
    """检查单笔委托是否允许

    Args:
        direction: "buy"/"sell"，可选。提供则可精细化判定（只关买或只关卖）。

    Returns:
        (allowed: bool, reason: str)
    """
    # 急停检查（方向感知）
    dir_lower = (direction or "").lower()
    if is_trading_halted():
        return False, "交易已被紧急停止，发送「恢复交易」解除"
    if dir_lower in ("buy", "加仓", "建仓") and is_buy_halted():
        return False, "自动买入已关闭，发送「开启买入」或「恢复交易」解除"
    if dir_lower in ("sell", "reduce", "减仓", "清仓") and is_sell_halted():
        return False, "自动卖出已关闭，发送「开启卖出」或「恢复交易」解除"

    # 每日限额
    max_daily = int(os.getenv("BROKER_MAX_DAILY_ORDERS", "20"))
    current = get_daily_order_count()
    if current >= max_daily:
        return False, f"今日委托已达上限({current}/{max_daily})，明日重置"

    # 单笔金额占比
    if total_asset > 0:
        order_amount = price * shares
        max_pct = float(os.getenv("BROKER_MAX_SINGLE_ORDER_PCT", "20"))
        actual_pct = order_amount / total_asset * 100
        if actual_pct > max_pct:
            return (
                False,
                f"单笔金额{order_amount:,.0f}元占总资产{actual_pct:.1f}%，"
                f"超过上限{max_pct:.0f}%",
            )

    return True, ""
