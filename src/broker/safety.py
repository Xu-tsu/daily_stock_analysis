"""
交易安全限制 — 每日限额、急停开关、单笔上限
"""
import logging
import os
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

# ── 急停开关（内存级，进程内生效）──
_kill_switch = False
_kill_switch_lock = threading.Lock()


def is_trading_halted() -> bool:
    """检查是否已触发急停"""
    return _kill_switch


def halt_trading(reason: str = "手动停止") -> None:
    """触发急停"""
    global _kill_switch
    with _kill_switch_lock:
        _kill_switch = True
    logger.warning(f"[安全] 交易已停止: {reason}")


def resume_trading() -> None:
    """解除急停"""
    global _kill_switch
    with _kill_switch_lock:
        _kill_switch = False
    logger.info("[安全] 交易已恢复")


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
) -> tuple:
    """检查单笔委托是否允许

    Returns:
        (allowed: bool, reason: str)
    """
    # 急停检查
    if is_trading_halted():
        return False, "交易已被紧急停止，发送「恢复交易」解除"

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
