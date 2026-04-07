"""
intraday_tracker.py — 盘中分时价格采集 + AI目标价实时监控 + 峰值时段统计

核心功能：
  1. 盘中每分钟采集持仓股+当日交易股的实时价格，存入 SQLite
  2. 对比 AI 目标价（price_targets.json），到价触发自动卖出
  3. 统计 A 股历史峰值时段（哪个时间段最容易出高点/低点），反馈给 AI 优化买卖点
  4. 收盘后生成当日分时回顾（日内最高/最低/成交价对比），计入反馈环

A股 T+1 规则：
  - 当日买入的股票当日不可卖出（sellable_shares = 持仓 - 今日买入）
  - 仅可卖出 sellable_shares > 0 的部分
"""
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("SCANNER_DB_PATH", "data/scanner_history.db")
TZ_CN = timezone(timedelta(hours=8))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CREATE_INTRADAY = """
CREATE TABLE IF NOT EXISTS intraday_ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT DEFAULT '',
    tick_time TEXT NOT NULL,
    price REAL NOT NULL,
    high REAL DEFAULT 0,
    low REAL DEFAULT 0,
    volume INTEGER DEFAULT 0,
    amount REAL DEFAULT 0,
    change_pct REAL DEFAULT 0,
    bid1_price REAL DEFAULT 0,
    ask1_price REAL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(trade_date, code, tick_time)
);
CREATE INDEX IF NOT EXISTS idx_intraday_date_code ON intraday_ticks(trade_date, code);
"""

_CREATE_PEAK_STATS = """
CREATE TABLE IF NOT EXISTS intraday_peak_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT DEFAULT '',
    open_price REAL DEFAULT 0,
    close_price REAL DEFAULT 0,
    day_high REAL DEFAULT 0,
    day_low REAL DEFAULT 0,
    high_time TEXT DEFAULT '',
    low_time TEXT DEFAULT '',
    executed_price REAL DEFAULT 0,
    executed_direction TEXT DEFAULT '',
    executed_time TEXT DEFAULT '',
    price_vs_high_pct REAL DEFAULT 0,
    price_vs_low_pct REAL DEFAULT 0,
    ai_target_price REAL DEFAULT 0,
    ai_stop_price REAL DEFAULT 0,
    target_reached INTEGER DEFAULT 0,
    stop_reached INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(trade_date, code, executed_direction)
);
CREATE INDEX IF NOT EXISTS idx_peak_date ON intraday_peak_stats(trade_date);
"""


def _get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_CREATE_INTRADAY)
    conn.executescript(_CREATE_PEAK_STATS)
    return conn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 分时价格采集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def record_tick(trade_date: str, code: str, name: str, quote: dict) -> None:
    """记录一条分时数据"""
    tick_time = quote.get("timestamp", "")
    if not tick_time:
        tick_time = datetime.now(TZ_CN).strftime("%H:%M:%S")
    # 标准化为 HH:MM 精度（避免秒级重复）
    if len(tick_time) >= 5:
        tick_time = tick_time[:5]

    price = float(quote.get("price", 0) or 0)
    if price <= 0:
        return

    try:
        conn = _get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO intraday_ticks
            (trade_date, code, name, tick_time, price, high, low,
             volume, amount, change_pct, bid1_price, ask1_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_date, code, name, tick_time, price,
            float(quote.get("high", 0) or 0),
            float(quote.get("low", 0) or 0),
            int(float(quote.get("volume", 0) or 0)),
            float(quote.get("amount", 0) or 0),
            float(quote.get("change_pct", 0) or 0),
            float(quote.get("bid1_price", 0) or 0),
            float(quote.get("ask1_price", 0) or 0),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f"[分时] 记录失败 {code}: {e}")


def record_ticks_batch(trade_date: str, quotes: Dict[str, dict],
                       name_map: Dict[str, str] = None) -> int:
    """批量记录分时数据，返回写入条数"""
    if not quotes:
        return 0
    name_map = name_map or {}
    count = 0
    try:
        conn = _get_conn()
        for tc_code, quote in quotes.items():
            code = quote.get("code", tc_code)
            name = name_map.get(code, quote.get("name", ""))
            tick_time = quote.get("timestamp", "")
            if not tick_time:
                tick_time = datetime.now(TZ_CN).strftime("%H:%M:%S")
            if len(tick_time) >= 5:
                tick_time = tick_time[:5]
            price = float(quote.get("price", 0) or 0)
            if price <= 0:
                continue
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO intraday_ticks
                    (trade_date, code, name, tick_time, price, high, low,
                     volume, amount, change_pct, bid1_price, ask1_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_date, code, name, tick_time, price,
                    float(quote.get("high", 0) or 0),
                    float(quote.get("low", 0) or 0),
                    int(float(quote.get("volume", 0) or 0)),
                    float(quote.get("amount", 0) or 0),
                    float(quote.get("change_pct", 0) or 0),
                    float(quote.get("bid1_price", 0) or 0),
                    float(quote.get("ask1_price", 0) or 0),
                ))
                count += 1
            except Exception:
                pass
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"[分时] 批量记录失败: {e}")
    return count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. AI 目标价实时检测（每次采集后调用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_targets_and_execute(
    positions: list,
    today: str,
    executed_set: set,
) -> List[dict]:
    """检测 AI 目标价触发并自动执行卖出。

    严格遵守 T+1：只卖 sellable_shares > 0 的部分。

    Args:
        positions: broker.get_positions() 返回的持仓列表
        today: 日期字符串 YYYY-MM-DD
        executed_set: 已执行集合（防重复），会被修改

    Returns:
        本次执行结果列表
    """
    results = []

    try:
        from src.broker.price_target_store import check_price_targets, remove_target
    except ImportError:
        return results

    triggered = check_price_targets(positions)
    if not triggered:
        return results

    try:
        from src.broker import get_broker
        broker = get_broker()
        if not broker or not broker.is_connected():
            return results
    except Exception:
        return results

    from src.broker.safety import check_order_allowed, increment_order_count

    for hit in triggered:
        code = hit["code"]
        exec_key = f"auto_exec_{today}_{code}"
        if exec_key in executed_set:
            continue

        pos = hit["position"]
        sellable = getattr(pos, "sellable_shares", 0) or 0
        if sellable <= 0:
            logger.info(f"[盘中监控] {hit['name']}({code}) T+1限制，可卖=0，跳过")
            continue

        price = getattr(pos, "current_price", 0) or 0
        if price <= 0:
            continue

        trigger = hit["trigger"]
        target_info = hit["target_info"]

        if trigger == "take_profit":
            label = "AI止盈"
            detail = f"现价{price:.2f} >= 目标{hit['target_price']:.2f}"
        else:
            label = "AI止损"
            detail = f"现价{price:.2f} <= 止损{hit['stop_price']:.2f}"

        # 安全检查
        try:
            balance = broker.get_balance()
            total_asset = balance.total_asset if balance else 0
        except Exception:
            total_asset = 0

        allowed, reason = check_order_allowed(code, price, sellable, total_asset)
        if not allowed:
            logger.warning(f"[{label}] {hit['name']} 安全限制: {reason}")
            continue

        logger.info(
            f"[{label}] {hit['name']}({code}) {sellable}股 @ {price:.2f} | {detail}"
        )

        try:
            # 优先使用智能追单（自动撤单改价直到成交）
            if hasattr(broker, "smart_sell"):
                result = broker.smart_sell(code, round(price, 2), sellable)
            else:
                result = broker.sell(code, round(price, 2), sellable)
            exec_result = {
                "code": code,
                "name": hit["name"],
                "trigger": trigger,
                "price": price,
                "shares": sellable,
                "detail": detail,
                "success": result.is_success,
                "message": result.message,
            }
            results.append(exec_result)

            if result.is_success:
                increment_order_count()
                executed_set.add(exec_key)
                remove_target(code)
                logger.info(f"[{label}] {hit['name']} 执行成功")
            else:
                logger.warning(f"[{label}] {hit['name']} 执行失败: {result.message}")
        except Exception as e:
            logger.error(f"[{label}] {hit['name']} 异常: {e}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 收盘后分时回顾 — 统计日内最高/最低/成交价对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_daily_peak_stats(trade_date: str) -> List[dict]:
    """收盘后分析当日分时数据，生成峰值统计。

    对比实际成交价 vs 日内最优价，衡量执行质量。
    """
    conn = _get_conn()

    # 当日所有采集的股票
    rows = conn.execute("""
        SELECT code, name,
               MIN(price) as day_low, MAX(price) as day_high,
               -- 第一条和最后一条作为开盘/收盘近似
               (SELECT price FROM intraday_ticks t2
                WHERE t2.code = t1.code AND t2.trade_date = t1.trade_date
                ORDER BY tick_time ASC LIMIT 1) as open_price,
               (SELECT price FROM intraday_ticks t2
                WHERE t2.code = t1.code AND t2.trade_date = t1.trade_date
                ORDER BY tick_time DESC LIMIT 1) as close_price,
               (SELECT tick_time FROM intraday_ticks t2
                WHERE t2.code = t1.code AND t2.trade_date = t1.trade_date
                AND t2.price = MAX(t1.price)
                ORDER BY tick_time ASC LIMIT 1) as high_time,
               (SELECT tick_time FROM intraday_ticks t2
                WHERE t2.code = t1.code AND t2.trade_date = t1.trade_date
                AND t2.price = MIN(t1.price)
                ORDER BY tick_time ASC LIMIT 1) as low_time
        FROM intraday_ticks t1
        WHERE trade_date = ?
        GROUP BY code
    """, (trade_date,)).fetchall()

    if not rows:
        conn.close()
        return []

    # 获取当日成交记录
    exec_rows = []
    try:
        exec_rows = conn.execute("""
            SELECT code, direction, actual_price, actual_shares,
                   target_sell_price, stop_loss_price, created_at
            FROM execution_log
            WHERE trade_date = ?
        """, (trade_date,)).fetchall()
    except Exception:
        pass

    exec_map = {}
    for er in exec_rows:
        key = (er["code"], er["direction"])
        exec_map[key] = er

    stats = []
    for row in rows:
        code = row["code"]
        day_high = float(row["day_high"] or 0)
        day_low = float(row["day_low"] or 0)
        high_time = row["high_time"] or ""
        low_time = row["low_time"] or ""

        # 查找该股票的执行记录
        for direction in ("buy", "sell"):
            er = exec_map.get((code, direction))
            exec_price = float(er["actual_price"]) if er else 0
            exec_time = str(er["created_at"]) if er else ""
            ai_target = float(er["target_sell_price"]) if er else 0
            ai_stop = float(er["stop_loss_price"]) if er else 0

            if exec_price <= 0 and not er:
                continue

            # 计算成交价 vs 日内最优价的偏差
            if direction == "buy" and day_low > 0 and exec_price > 0:
                vs_low = (exec_price - day_low) / day_low * 100
            else:
                vs_low = 0
            if direction == "sell" and day_high > 0 and exec_price > 0:
                vs_high = (exec_price - day_high) / day_high * 100
            else:
                vs_high = 0

            stat = {
                "trade_date": trade_date,
                "code": code,
                "name": row["name"] or "",
                "open_price": float(row["open_price"] or 0),
                "close_price": float(row["close_price"] or 0),
                "day_high": day_high,
                "day_low": day_low,
                "high_time": high_time,
                "low_time": low_time,
                "executed_price": exec_price,
                "executed_direction": direction,
                "executed_time": exec_time,
                "price_vs_high_pct": round(vs_high, 2),
                "price_vs_low_pct": round(vs_low, 2),
                "ai_target_price": ai_target,
                "ai_stop_price": ai_stop,
                "target_reached": 1 if ai_target > 0 and day_high >= ai_target else 0,
                "stop_reached": 1 if ai_stop > 0 and day_low <= ai_stop else 0,
            }
            stats.append(stat)

            try:
                conn.execute("""
                    INSERT OR REPLACE INTO intraday_peak_stats
                    (trade_date, code, name, open_price, close_price,
                     day_high, day_low, high_time, low_time,
                     executed_price, executed_direction, executed_time,
                     price_vs_high_pct, price_vs_low_pct,
                     ai_target_price, ai_stop_price,
                     target_reached, stop_reached)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_date, code, stat["name"],
                    stat["open_price"], stat["close_price"],
                    day_high, day_low, high_time, low_time,
                    exec_price, direction, exec_time,
                    stat["price_vs_high_pct"], stat["price_vs_low_pct"],
                    ai_target, ai_stop,
                    stat["target_reached"], stat["stop_reached"],
                ))
            except Exception as e:
                logger.debug(f"[峰值统计] 写入失败 {code}: {e}")

    conn.commit()
    conn.close()
    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 历史峰值时段分析 — 供 AI 优化买卖时机
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_peak_time_patterns(days: int = 30) -> dict:
    """分析近N天 A 股的峰值时段分布。

    返回:
        {
            "high_time_dist": {"09:30-10:00": 12, "10:00-10:30": 8, ...},
            "low_time_dist":  {"13:00-13:30": 15, "09:30-10:00": 10, ...},
            "best_sell_window": "09:45-10:15",
            "best_buy_window": "13:30-14:00",
            "execution_quality": {
                "buy_vs_low_avg_pct": 1.5,   # 买入价平均比日内低点高 1.5%
                "sell_vs_high_avg_pct": -2.3, # 卖出价平均比日内高点低 2.3%
                "target_hit_rate": 0.65,      # AI目标价达到率 65%
                "stop_hit_rate": 0.20,        # AI止损价触发率 20%
            },
            "sample_count": 120,
        }
    """
    conn = _get_conn()

    # 高点时段分布
    high_rows = conn.execute("""
        SELECT high_time FROM intraday_peak_stats
        WHERE trade_date >= date('now', '-' || ? || ' days', 'localtime')
          AND high_time != '' AND day_high > 0
    """, (days,)).fetchall()

    low_rows = conn.execute("""
        SELECT low_time FROM intraday_peak_stats
        WHERE trade_date >= date('now', '-' || ? || ' days', 'localtime')
          AND low_time != '' AND day_low > 0
    """, (days,)).fetchall()

    # 执行质量统计
    exec_rows = conn.execute("""
        SELECT executed_direction, price_vs_high_pct, price_vs_low_pct,
               target_reached, stop_reached
        FROM intraday_peak_stats
        WHERE trade_date >= date('now', '-' || ? || ' days', 'localtime')
          AND executed_price > 0
    """, (days,)).fetchall()

    conn.close()

    # 统计时段分布（30分钟为一个时段）
    high_dist = _time_to_slot_dist([r["high_time"] for r in high_rows])
    low_dist = _time_to_slot_dist([r["low_time"] for r in low_rows])

    # 找最佳窗口
    best_sell = max(high_dist, key=high_dist.get) if high_dist else "10:00-10:30"
    best_buy = max(low_dist, key=low_dist.get) if low_dist else "13:30-14:00"

    # 执行质量
    buy_vs_lows, sell_vs_highs = [], []
    target_hits, stop_hits, total_targets, total_stops = 0, 0, 0, 0
    for er in exec_rows:
        if er["executed_direction"] == "buy" and er["price_vs_low_pct"]:
            buy_vs_lows.append(er["price_vs_low_pct"])
        elif er["executed_direction"] == "sell" and er["price_vs_high_pct"]:
            sell_vs_highs.append(er["price_vs_high_pct"])
        if er["target_reached"] is not None:
            total_targets += 1
            target_hits += er["target_reached"]
        if er["stop_reached"] is not None:
            total_stops += 1
            stop_hits += er["stop_reached"]

    return {
        "high_time_dist": dict(sorted(high_dist.items())),
        "low_time_dist": dict(sorted(low_dist.items())),
        "best_sell_window": best_sell,
        "best_buy_window": best_buy,
        "execution_quality": {
            "buy_vs_low_avg_pct": round(
                sum(buy_vs_lows) / len(buy_vs_lows), 2
            ) if buy_vs_lows else 0,
            "sell_vs_high_avg_pct": round(
                sum(sell_vs_highs) / len(sell_vs_highs), 2
            ) if sell_vs_highs else 0,
            "target_hit_rate": round(
                target_hits / total_targets, 2
            ) if total_targets else 0,
            "stop_hit_rate": round(
                stop_hits / total_stops, 2
            ) if total_stops else 0,
        },
        "sample_count": len(high_rows) + len(low_rows),
    }


def format_peak_patterns_for_prompt(days: int = 30) -> str:
    """生成峰值时段分析文本，注入到 AI prompt 中。"""
    try:
        patterns = analyze_peak_time_patterns(days)
    except Exception as e:
        logger.debug(f"[峰值分析] 生成失败: {e}")
        return ""

    if patterns["sample_count"] < 10:
        return ""  # 数据不够不注入

    eq = patterns["execution_quality"]
    lines = [
        "## 历史日内峰值时段分析（数据驱动，非固定规则）",
        f"样本数: {patterns['sample_count']} (近{days}天)",
        f"A股日内高点集中时段: {patterns['best_sell_window']}",
        f"A股日内低点集中时段: {patterns['best_buy_window']}",
    ]

    if patterns["high_time_dist"]:
        top3_high = sorted(patterns["high_time_dist"].items(), key=lambda x: -x[1])[:3]
        lines.append(f"高点分布TOP3: {', '.join(f'{k}({v}次)' for k, v in top3_high)}")
    if patterns["low_time_dist"]:
        top3_low = sorted(patterns["low_time_dist"].items(), key=lambda x: -x[1])[:3]
        lines.append(f"低点分布TOP3: {', '.join(f'{k}({v}次)' for k, v in top3_low)}")

    if eq["buy_vs_low_avg_pct"] > 0:
        lines.append(f"历史买入价平均比日内低点高 {eq['buy_vs_low_avg_pct']:.1f}%（优化空间）")
    if eq["sell_vs_high_avg_pct"] < 0:
        lines.append(f"历史卖出价平均比日内高点低 {abs(eq['sell_vs_high_avg_pct']):.1f}%（优化空间）")
    if eq["target_hit_rate"] > 0:
        lines.append(f"AI目标价历史达到率: {eq['target_hit_rate']*100:.0f}%")
    if eq["stop_hit_rate"] > 0:
        lines.append(f"AI止损价历史触发率: {eq['stop_hit_rate']*100:.0f}%")

    # 优化建议
    lines.append("")
    lines.append("根据以上数据，请优化你的买卖点位建议：")
    lines.append(f"- 卖出尽量安排在 {patterns['best_sell_window']} 附近（历史高点密集区）")
    lines.append(f"- 买入尽量安排在 {patterns['best_buy_window']} 附近（历史低点密集区）")
    if eq["target_hit_rate"] < 0.5:
        lines.append("- AI目标价达到率偏低，建议适当下调止盈目标使其更现实")
    if eq["stop_hit_rate"] > 0.4:
        lines.append("- AI止损触发率偏高，建议适当放宽止损价或改善选股质量")

    return "\n".join(lines)


def _time_to_slot_dist(times: list) -> dict:
    """将 HH:MM 时间列表按 30 分钟时段归类计数。"""
    slots = {}
    for t in times:
        if not t or len(t) < 5:
            continue
        try:
            h, m = int(t[:2]), int(t[3:5])
        except (ValueError, IndexError):
            continue
        # 归类到 30 分钟时段
        slot_m = (m // 30) * 30
        slot_start = f"{h:02d}:{slot_m:02d}"
        slot_end_m = slot_m + 30
        if slot_end_m >= 60:
            slot_end = f"{h+1:02d}:00"
        else:
            slot_end = f"{h:02d}:{slot_end_m:02d}"
        key = f"{slot_start}-{slot_end}"
        slots[key] = slots.get(key, 0) + 1
    return slots
