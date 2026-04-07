"""
券商模块 — 统一接口操作同花顺/东方财富等券商客户端

用法:
    from src.broker import get_broker
    broker = get_broker()
    if broker and broker.is_connected():
        broker.buy("002506", 4.82, 500)
"""
import logging
import os
from typing import Optional

from src.broker.base import BrokerAdapter

logger = logging.getLogger(__name__)

# 单例：避免重复创建连接
_broker_instance: Optional[BrokerAdapter] = None


def get_broker(force_new: bool = False) -> Optional[BrokerAdapter]:
    """获取券商适配器实例

    Returns:
        BrokerAdapter 实例，如果 BROKER_ENABLED!=true 则返回 None
    """
    global _broker_instance

    if os.getenv("BROKER_ENABLED", "").lower() != "true":
        return None

    if _broker_instance is not None and not force_new:
        return _broker_instance

    broker_type = os.getenv("THS_BROKER_TYPE", "ths").lower()

    if broker_type == "ths":
        from src.broker.ths_adapter import THSBrokerAdapter
        adapter = THSBrokerAdapter()
        if adapter.connect():
            _broker_instance = adapter
            return adapter
        else:
            logger.error("[Broker] 同花顺连接失败")
            return None
    else:
        logger.error(f"[Broker] 不支持的券商类型: {broker_type}")
        return None


def reset_broker() -> None:
    """重置券商连接（用于异常恢复）"""
    global _broker_instance
    if _broker_instance:
        try:
            _broker_instance.disconnect()
        except Exception:
            pass
    _broker_instance = None
