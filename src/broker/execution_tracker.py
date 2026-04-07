"""
执行质量追踪 — 记录每笔委托的建议价 vs 实际价，计算滑点和准确度
"""
import logging
import sqlite3
from datetime import datetime
from typing import List, Optional

from src.broker.models import OrderResult

logger = logging.getLogger(__name__)

DB_PATH = "data/scanner_history.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS execution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT DEFAULT '',
    direction TEXT NOT NULL,
    recommended_price REAL DEFAULT 0,
    recommended_shares INTEGER DEFAULT 0,
    recommended_action TEXT DEFAULT '',
    actual_price REAL DEFAULT 0,
    actual_shares INTEGER DEFAULT 0,
    order_status TEXT DEFAULT '',
    slippage_pct REAL DEFAULT 0,
    slippage_amount REAL DEFAULT 0,
    session_id TEXT DEFAULT '',
    execution_mode TEXT DEFAULT '',
    source TEXT DEFAULT 'rebalance',
    target_sell_price REAL DEFAULT 0,
    stop_loss_price REAL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now','localtime'))
)
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TABLE)
    return conn


def record_execution(
    order: OrderResult,
    session_id: str = "",
    mode: str = "",
    recommended_action: str = "",
    target_sell_price: float = 0,
    stop_loss_price: float = 0,
) -> None:
    """记录一笔执行结果"""
    try:
        conn = _get_conn()
        slippage_amt = 0.0
        if order.actual_price > 0 and order.requested_price > 0:
            slippage_amt = round(
                (order.actual_price - order.requested_price) * order.actual_shares, 2
            )
        conn.execute("""
            INSERT INTO execution_log (
                trade_date, code, name, direction,
                recommended_price, recommended_shares, recommended_action,
                actual_price, actual_shares, order_status,
                slippage_pct, slippage_amount,
                session_id, execution_mode,
                target_sell_price, stop_loss_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d"),
            order.code, order.name, order.direction,
            order.requested_price, order.requested_shares, recommended_action,
            order.actual_price, order.actual_shares, order.status,
            order.slippage_pct, slippage_amt,
            session_id, mode,
            target_sell_price, stop_loss_price,
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"[执行追踪] 记录失败: {e}")


def get_execution_quality(days: int = 30) -> dict:
    """获取近N天的执行质量统计"""
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM execution_log
            WHERE trade_date >= date('now', '-' || ? || ' days', 'localtime')
            ORDER BY created_at DESC
        """, (days,)).fetchall()
        conn.close()

        if not rows:
            return {"note": "insufficient_data", "total": 0}

        total = len(rows)
        filled = [r for r in rows if r["order_status"] in ("filled", "submitted")]
        rejected = [r for r in rows if r["order_status"] == "rejected"]

        avg_slippage = 0.0
        if filled:
            slippages = [r["slippage_pct"] for r in filled if r["slippage_pct"] != 0]
            if slippages:
                avg_slippage = round(sum(slippages) / len(slippages), 3)

        # 按方向分组
        buy_count = sum(1 for r in rows if r["direction"] == "buy")
        sell_count = sum(1 for r in rows if r["direction"] == "sell")

        return {
            "days": days,
            "total_orders": total,
            "filled_count": len(filled),
            "rejected_count": len(rejected),
            "fill_rate_pct": round(len(filled) / total * 100, 1) if total else 0,
            "avg_slippage_pct": avg_slippage,
            "buy_orders": buy_count,
            "sell_orders": sell_count,
            "rejection_reasons": _top_reasons(rejected),
        }
    except Exception as e:
        logger.error(f"[执行追踪] 查询失败: {e}")
        return {"error": str(e)}


def compare_predicted_vs_actual(days: int = 30) -> dict:
    """对比建议价与实际成交价"""
    try:
        conn = _get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT code, name, direction, recommended_price, actual_price,
                   slippage_pct, target_sell_price, stop_loss_price
            FROM execution_log
            WHERE trade_date >= date('now', '-' || ? || ' days', 'localtime')
              AND order_status IN ('filled', 'submitted')
              AND recommended_price > 0 AND actual_price > 0
        """, (days,)).fetchall()
        conn.close()

        if not rows:
            return {"note": "insufficient_data"}

        results = []
        for r in rows:
            results.append({
                "code": r["code"],
                "name": r["name"],
                "direction": r["direction"],
                "recommended": r["recommended_price"],
                "actual": r["actual_price"],
                "slippage_pct": r["slippage_pct"],
                "had_target": r["target_sell_price"] > 0,
                "had_stop": r["stop_loss_price"] > 0,
            })

        return {
            "days": days,
            "comparisons": len(results),
            "details": results[:20],
        }
    except Exception as e:
        return {"error": str(e)}


def format_quality_report(days: int = 30) -> str:
    """格式化执行质量报告（Feishu用）"""
    q = get_execution_quality(days)
    if q.get("note") == "insufficient_data":
        return f"📊 近{days}天暂无执行记录"

    lines = [f"📊 **执行质量报告** (近{days}天)"]
    lines.append(f"  总委托: {q.get('total_orders', 0)}笔")
    lines.append(f"  成交率: {q.get('fill_rate_pct', 0)}%")
    lines.append(f"  平均滑点: {q.get('avg_slippage_pct', 0):+.3f}%")
    lines.append(f"  买入: {q.get('buy_orders', 0)}笔 | 卖出: {q.get('sell_orders', 0)}笔")

    reasons = q.get("rejection_reasons", [])
    if reasons:
        lines.append("  拒绝原因:")
        for reason, count in reasons[:3]:
            lines.append(f"    • {reason} ({count}次)")

    return "\n".join(lines)


def _top_reasons(rejected_rows: list) -> list:
    """统计拒绝原因 top N"""
    from collections import Counter
    reasons = Counter()
    for r in rejected_rows:
        # 从 OrderResult.message 提取
        msg = r["order_status"] if isinstance(r, dict) else ""
        reasons[msg or "unknown"] += 1
    return reasons.most_common(5)
