"""
price_target_store.py — AI 止盈/止损目标价持久化

调仓分析（rebalance_engine）完成后，从 AI 输出的 actions 中提取
target_sell_price / stop_loss_price，存入 data/price_targets.json。

盘中监控（market_monitor）读取该文件，实时比价，到价自动执行。
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_TARGET_FILE = Path(os.getenv("PRICE_TARGET_FILE", "data/price_targets.json"))


def save_price_targets(actions: List[dict], source: str = "rebalance") -> int:
    """从 AI 调仓结果中提取目标价并持久化。

    Args:
        actions: rebalance 输出的 actions 列表，每项含
                 code, name, target_sell_price, stop_loss_price 等
        source: 来源标识（rebalance / langgraph / manual）

    Returns:
        保存的目标数量
    """
    targets = load_price_targets()
    saved = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for act in actions:
        code = act.get("code", "")
        if not code:
            continue

        target_price = _safe_float(act.get("target_sell_price"))
        stop_price = _safe_float(act.get("stop_loss_price"))

        # 至少要有一个价格才值得保存
        if target_price is None and stop_price is None:
            continue

        action_type = str(act.get("action", "hold")).lower()

        # 对于 sell/reduce 的股票不设目标价（已经建议卖了）
        if action_type in ("sell",):
            # 但保留止损价，万一没卖成
            if stop_price is None:
                continue

        targets[code] = {
            "code": code,
            "name": act.get("name", ""),
            "target_sell_price": target_price,
            "stop_loss_price": stop_price,
            "action": action_type,
            "sell_timing": act.get("sell_timing", ""),
            "reason": str(act.get("reason", ""))[:200],
            "source": source,
            "updated_at": now,
        }
        saved += 1
        logger.info(
            f"[目标价] {act.get('name', code)}({code}): "
            f"止盈={target_price or '-'} 止损={stop_price or '-'} "
            f"({action_type})"
        )

    if saved > 0:
        _TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _TARGET_FILE.write_text(
            json.dumps(targets, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[目标价] 已保存 {saved} 只股票的目标价到 {_TARGET_FILE}")

    return saved


def load_price_targets() -> Dict[str, dict]:
    """加载所有目标价。返回 {code: {...}} 字典。"""
    if not _TARGET_FILE.exists():
        return {}
    try:
        data = json.loads(_TARGET_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"[目标价] 读取失败: {e}")
        return {}


def remove_target(code: str) -> bool:
    """删除某只股票的目标价（已卖出后清理）。"""
    targets = load_price_targets()
    if code not in targets:
        return False
    del targets[code]
    _TARGET_FILE.write_text(
        json.dumps(targets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True


def check_price_targets(positions: list) -> List[dict]:
    """检查当前持仓是否触及 AI 目标价。

    Args:
        positions: broker.get_positions() 返回的持仓列表，
                   每项有 code, current_price, sellable_shares 等

    Returns:
        触发列表: [{"code", "name", "trigger", "target_info", "position"}]
        trigger = "take_profit" | "stop_loss"
    """
    targets = load_price_targets()
    if not targets:
        return []

    triggered = []
    for pos in positions:
        code = getattr(pos, "code", "") or pos.get("code", "")
        if code not in targets:
            continue

        price = float(getattr(pos, "current_price", 0) or pos.get("current_price", 0))
        if price <= 0:
            continue

        t = targets[code]
        target_price = t.get("target_sell_price")
        stop_price = t.get("stop_loss_price")

        # 止盈触发：当前价 >= 目标卖出价
        if target_price and price >= target_price:
            triggered.append({
                "code": code,
                "name": t.get("name", ""),
                "trigger": "take_profit",
                "current_price": price,
                "target_price": target_price,
                "target_info": t,
                "position": pos,
            })
        # 止损触发：当前价 <= 止损价
        elif stop_price and price <= stop_price:
            triggered.append({
                "code": code,
                "name": t.get("name", ""),
                "trigger": "stop_loss",
                "current_price": price,
                "stop_price": stop_price,
                "target_info": t,
                "position": pos,
            })

    return triggered


def _safe_float(val) -> Optional[float]:
    """安全转换为 float，无效值返回 None。"""
    if val is None:
        return None
    try:
        v = float(val)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None
