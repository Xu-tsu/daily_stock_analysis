"""
data_store.py — 历史数据积累（SQLite）
每天自动存储：扫描结果、资金流向、板块轮动
用于：多日趋势判断、回测验证、策略优化

表结构:
  scan_results   — 每日扫描候选股 + N日后实际涨跌（回测用）
  fund_flow_daily — 个股每日主力资金流（积累多日趋势）
  sector_flow_daily — 板块每日资金流（判断主线）
  scan_backtest  — 扫描结果回测评估
"""
import json, logging, os, sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("SCANNER_DB_PATH", "data/scanner_history.db")


def _get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """初始化所有表"""
    conn = _get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS scan_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_date TEXT NOT NULL,
        scan_mode TEXT DEFAULT 'trend',
        code TEXT NOT NULL,
        name TEXT,
        price REAL,
        change_pct REAL,
        turnover_rate REAL,
        market_cap REAL,
        ma_trend TEXT,
        macd_signal TEXT,
        rsi REAL,
        vol_pattern TEXT,
        tech_score INTEGER,
        -- 回测字段（后续填充）
        price_after_1d REAL,
        price_after_3d REAL,
        price_after_5d REAL,
        return_1d REAL,
        return_3d REAL,
        return_5d REAL,
        backtest_done INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(scan_date, code, scan_mode)
    );

    CREATE TABLE IF NOT EXISTS fund_flow_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT NOT NULL,
        code TEXT NOT NULL,
        name TEXT,
        main_net REAL,
        main_net_pct REAL,
        super_large_net REAL,
        large_net REAL,
        price REAL,
        change_pct REAL,
        turnover_rate REAL,
        source TEXT DEFAULT 'ths',
        created_at TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(trade_date, code)
    );

    CREATE TABLE IF NOT EXISTS sector_flow_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT NOT NULL,
        sector_type TEXT DEFAULT 'hy',
        sector_name TEXT NOT NULL,
        change_pct REAL,
        main_net REAL,
        main_net_pct REAL,
        rank_position INTEGER,
        source TEXT DEFAULT 'ths',
        created_at TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(trade_date, sector_name, sector_type)
    );

    CREATE INDEX IF NOT EXISTS idx_scan_date ON scan_results(scan_date);
    CREATE INDEX IF NOT EXISTS idx_scan_code ON scan_results(code);
    CREATE INDEX IF NOT EXISTS idx_fund_date ON fund_flow_daily(trade_date);
    CREATE INDEX IF NOT EXISTS idx_fund_code ON fund_flow_daily(code);
    CREATE INDEX IF NOT EXISTS idx_sector_date ON sector_flow_daily(trade_date);
    """)
    conn.close()
    logger.info(f"[DataStore] 数据库初始化完成: {DB_PATH}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 写入
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def save_scan_results(results: list, scan_date: str = None, mode: str = "trend"):
    """保存扫描结果"""
    if not results:
        return
    if not scan_date:
        scan_date = datetime.now().strftime("%Y-%m-%d")
    conn = _get_conn()
    saved = 0
    for r in results:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO scan_results
                (scan_date,scan_mode,code,name,price,change_pct,turnover_rate,
                 market_cap,ma_trend,macd_signal,rsi,vol_pattern,tech_score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                scan_date, mode, r.get("code"), r.get("name"),
                r.get("price"), r.get("change_pct"), r.get("turnover_rate"),
                r.get("market_cap"), r.get("ma_trend"), r.get("macd_signal"),
                r.get("rsi"), r.get("vol_pattern"), r.get("tech_score"),
            ))
            saved += 1
        except Exception as e:
            logger.debug(f"保存扫描结果失败 {r.get('code')}: {e}")
    conn.commit()
    conn.close()
    logger.info(f"[DataStore] 保存 {saved} 条扫描结果 ({scan_date}, {mode})")


def save_fund_flow_batch(stocks: list, trade_date: str = None):
    """批量保存个股资金流"""
    if not stocks:
        return
    if not trade_date:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    conn = _get_conn()
    saved = 0
    for s in stocks:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO fund_flow_daily
                (trade_date,code,name,main_net,main_net_pct,super_large_net,
                 large_net,price,change_pct,turnover_rate,source)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                trade_date, s.get("code"), s.get("name"),
                s.get("main_net", 0), s.get("main_net_pct", 0),
                s.get("super_large_net", 0), s.get("large_net", 0),
                s.get("price", 0), s.get("change_pct", 0),
                s.get("turnover_rate", 0), s.get("source", "ths"),
            ))
            saved += 1
        except Exception:
            pass
    conn.commit()
    conn.close()
    logger.info(f"[DataStore] 保存 {saved} 条资金流 ({trade_date})")


def save_sector_flow_batch(sectors: list, sector_type: str = "hy", trade_date: str = None):
    """批量保存板块资金流"""
    if not sectors:
        return
    if not trade_date:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    conn = _get_conn()
    for i, s in enumerate(sectors):
        try:
            conn.execute("""
                INSERT OR REPLACE INTO sector_flow_daily
                (trade_date,sector_type,sector_name,change_pct,main_net,main_net_pct,rank_position,source)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                trade_date, sector_type, s.get("name"),
                s.get("change_pct", 0), s.get("main_net", 0),
                s.get("main_net_pct", 0), i + 1, "ths",
            ))
        except Exception:
            pass
    conn.commit()
    conn.close()
    logger.info(f"[DataStore] 保存 {len(sectors)} 条板块资金流 ({sector_type}, {trade_date})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 查询：多日资金流趋势
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_stock_fund_flow_history(code: str, days: int = 5) -> list:
    """查某只股票近N日的主力资金流（判断连续性）"""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT trade_date, main_net, main_net_pct, price, change_pct
        FROM fund_flow_daily
        WHERE code = ?
        ORDER BY trade_date DESC
        LIMIT ?
    """, (code, days)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_consecutive_inflow_stocks(min_days: int = 3, min_total: float = 500) -> list:
    """找连续N日主力净流入的股票"""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT code, name,
               COUNT(*) as inflow_days,
               SUM(main_net) as total_net,
               AVG(main_net) as avg_net,
               MAX(trade_date) as latest_date
        FROM fund_flow_daily
        WHERE main_net > 0
          AND trade_date >= date('now', '-10 days')
        GROUP BY code
        HAVING COUNT(*) >= ?
           AND SUM(main_net) >= ?
        ORDER BY SUM(main_net) DESC
    """, (min_days, min_total)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_fund_flow_latest(trade_date: str = None, top_n: int = 10) -> dict:
    """获取当日资金流入/流出排行"""
    if not trade_date:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    conn = _get_conn()
    # 最近有数据的日期（可能非交易日）
    actual_date = conn.execute(
        "SELECT MAX(trade_date) FROM fund_flow_daily WHERE trade_date <= ?", (trade_date,)
    ).fetchone()[0]
    if not actual_date:
        conn.close()
        return {"inflow": [], "outflow": []}

    inflow = conn.execute("""
        SELECT code, name, main_net as net, main_net_pct, price, change_pct
        FROM fund_flow_daily WHERE trade_date = ? AND main_net > 0
        ORDER BY main_net DESC LIMIT ?
    """, (actual_date, top_n)).fetchall()
    outflow = conn.execute("""
        SELECT code, name, main_net as net, main_net_pct, price, change_pct
        FROM fund_flow_daily WHERE trade_date = ? AND main_net < 0
        ORDER BY main_net ASC LIMIT ?
    """, (actual_date, top_n)).fetchall()
    conn.close()
    return {"inflow": [dict(r) for r in inflow], "outflow": [dict(r) for r in outflow]}


def get_sector_flow_latest(trade_date: str = None, top_n: int = 10) -> dict:
    """获取当日板块资金流入/流出排行"""
    if not trade_date:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    conn = _get_conn()
    actual_date = conn.execute(
        "SELECT MAX(trade_date) FROM sector_flow_daily WHERE trade_date <= ?", (trade_date,)
    ).fetchone()[0]
    if not actual_date:
        conn.close()
        return {"inflow": [], "outflow": []}

    inflow = conn.execute("""
        SELECT sector_name as name, main_net as net, main_net_pct, change_pct
        FROM sector_flow_daily WHERE trade_date = ? AND main_net > 0 AND sector_type = 'hy'
        ORDER BY main_net DESC LIMIT ?
    """, (actual_date, top_n)).fetchall()
    outflow = conn.execute("""
        SELECT sector_name as name, main_net as net, main_net_pct, change_pct
        FROM sector_flow_daily WHERE trade_date = ? AND main_net < 0 AND sector_type = 'hy'
        ORDER BY main_net ASC LIMIT ?
    """, (actual_date, top_n)).fetchall()
    conn.close()
    return {"inflow": [dict(r) for r in inflow], "outflow": [dict(r) for r in outflow]}


def get_sector_mainline(min_days: int = 3) -> list:
    """找连续N日资金净流入的板块（主线确认）"""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT sector_name,
               COUNT(*) as inflow_days,
               SUM(main_net) as total_net,
               AVG(change_pct) as avg_chg,
               MAX(trade_date) as latest_date
        FROM sector_flow_daily
        WHERE main_net > 0
          AND trade_date >= date('now', '-10 days')
          AND sector_type = 'hy'
        GROUP BY sector_name
        HAVING COUNT(*) >= ?
        ORDER BY SUM(main_net) DESC
        LIMIT 10
    """, (min_days,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 回测：验证扫描准确度
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_scan_backtest():
    """
    对历史扫描结果做回测
    查N天前的扫描记录，获取当前价格，计算实际收益
    """
    from market_scanner import _batch_tencent_quotes

    conn = _get_conn()
    # 找 1/3/5 天前的未回测记录
    for days_ago in [1, 3, 5]:
        target_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        rows = conn.execute("""
            SELECT id, code, price FROM scan_results
            WHERE scan_date = ? AND backtest_done = 0
        """, (target_date,)).fetchall()

        if not rows:
            continue

        codes = [r["code"] for r in rows]
        quotes = _batch_tencent_quotes(codes)

        updated = 0
        for r in rows:
            code = r["code"]
            q = quotes.get(code, {})
            current_price = q.get("price", 0)
            if current_price <= 0 or r["price"] <= 0:
                continue

            ret = round((current_price / r["price"] - 1) * 100, 2)
            col_price = f"price_after_{days_ago}d"
            col_ret = f"return_{days_ago}d"

            conn.execute(f"""
                UPDATE scan_results
                SET {col_price} = ?, {col_ret} = ?,
                    backtest_done = CASE WHEN {days_ago} = 5 THEN 1 ELSE backtest_done END
                WHERE id = ?
            """, (current_price, ret, r["id"]))
            updated += 1

        conn.commit()
        logger.info(f"[回测] {target_date} 更新 {updated} 条 ({days_ago}日收益)")

    conn.close()


def get_backtest_summary(days: int = 30) -> dict:
    """获取回测统计摘要"""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT scan_mode,
               COUNT(*) as total,
               AVG(return_1d) as avg_ret_1d,
               AVG(return_3d) as avg_ret_3d,
               AVG(return_5d) as avg_ret_5d,
               SUM(CASE WHEN return_3d > 0 THEN 1 ELSE 0 END) as win_3d,
               SUM(CASE WHEN return_5d > 0 THEN 1 ELSE 0 END) as win_5d
        FROM scan_results
        WHERE backtest_done = 1
          AND scan_date >= date('now', ? || ' days')
        GROUP BY scan_mode
    """, (f"-{days}",)).fetchall()
    conn.close()

    results = {}
    for r in rows:
        total = r["total"] or 1
        results[r["scan_mode"]] = {
            "total": total,
            "avg_return_1d": round(r["avg_ret_1d"] or 0, 2),
            "avg_return_3d": round(r["avg_ret_3d"] or 0, 2),
            "avg_return_5d": round(r["avg_ret_5d"] or 0, 2),
            "win_rate_3d": round((r["win_3d"] or 0) / total * 100, 1),
            "win_rate_5d": round((r["win_5d"] or 0) / total * 100, 1),
        }
    return results


# 启动时自动建表
init_db()