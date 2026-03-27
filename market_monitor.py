"""
market_monitor.py
Minimal intraday monitor loop integrating scanner + alerts.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List

from data_store import save_scan_results

logger = logging.getLogger(__name__)


def _send_alert(text: str) -> None:
    try:
        from src.notification import NotificationService

        notifier = NotificationService()
        if notifier.is_available():
            notifier.send(text)
            return
    except Exception as e:
        logger.debug("notification unavailable: %s", e)
    logger.info("[monitor alert] %s", text)


def run_scan_once(max_price: float = 10.0, top_n: int = 20) -> List[Dict[str, Any]]:
    from market_scanner import scan_market

    logger.info("开始执行盘前/盘中扫描...")
    candidates = scan_market(max_price=max_price, min_turnover=2.0, top_n=top_n, mode="trend")
    if candidates:
        saved = save_scan_results(candidates, strategy="trend")
        logger.info("扫描完成: candidates=%s, saved=%s", len(candidates), saved)
    else:
        logger.info("扫描完成: 未发现候选")
    return candidates or []


def check_market_anomaly() -> Dict[str, Any]:
    now = datetime.now()
    return {"time": now.strftime("%Y-%m-%d %H:%M:%S"), "events": []}


def format_anomaly_alert(result: Dict[str, Any]) -> str:
    events = result.get("events") or []
    if not events:
        return ""
    lines = [f"⚠️ 市场异动 ({result.get('time')})"]
    lines.extend([f"- {e}" for e in events])
    return "\n".join(lines)


def run_monitor_loop(interval_minutes: int = 10, auto_rebalance: bool = True) -> None:
    logger.info("盘中监控启动: interval=%smin, auto_rebalance=%s", interval_minutes, auto_rebalance)

    # 启动即扫描一次，保证立即有输出
    try:
        run_scan_once()
    except Exception as e:
        logger.warning("启动扫描失败: %s", e)

    while True:
        try:
            result = check_market_anomaly()
            alert = format_anomaly_alert(result)
            if alert:
                _send_alert(alert)
        except Exception as e:
            logger.warning("监控轮询失败: %s", e)
        time.sleep(max(1, int(interval_minutes)) * 60)
