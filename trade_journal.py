"""
trade_journal.py — 交易日志 + 盈亏分析 + 策略优化

功能:
  1. 自动记录每笔买入/卖出（从 portfolio_bot 的操作同步）
  2. 计算每笔交易的盈亏、持仓天数、买入时技术指标
  3. 分析哪些模式赚钱、哪些亏钱
  4. 输出策略优化建议（什么MA趋势+什么MACD信号+什么板块=高胜率）

飞书指令:
  战绩        → 查看总体盈亏统计
  复盘        → 分析盈利/亏损模式
  交易记录    → 查看最近交易
"""
import json, logging, os, sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("SCANNER_DB_PATH", "data/scanner_history.db")


def _conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_trade_tables():
    """初始化交易记录表"""
    conn = _conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS trade_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT NOT NULL,
        trade_type TEXT NOT NULL,  -- buy / sell
        code TEXT NOT NULL,
        name TEXT,
        shares INTEGER,
        price REAL,
        amount REAL,
        -- 买入时的技术指标（卖出时从买入记录复制）
        ma_trend TEXT,
        macd_signal TEXT,
        rsi REAL,
        vol_pattern TEXT,
        tech_score INTEGER,
        sector TEXT,
        -- 卖出时填充
        buy_price REAL,
        pnl REAL,
        pnl_pct REAL,
        hold_days INTEGER,
        -- 元信息
        source TEXT DEFAULT 'manual',  -- manual / bot / rebalance
        note TEXT,
        created_at TEXT DEFAULT (datetime('now','localtime'))
    );

    CREATE TABLE IF NOT EXISTS trade_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        period TEXT NOT NULL,  -- daily / weekly / monthly
        period_date TEXT NOT NULL,
        total_trades INTEGER,
        win_trades INTEGER,
        lose_trades INTEGER,
        win_rate REAL,
        total_pnl REAL,
        avg_pnl_pct REAL,
        best_trade TEXT,
        worst_trade TEXT,
        avg_hold_days REAL,
        -- 盈利模式
        best_ma_trend TEXT,
        best_macd_signal TEXT,
        best_sector TEXT,
        created_at TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(period, period_date)
    );

    CREATE INDEX IF NOT EXISTS idx_trade_code ON trade_log(code);
    CREATE INDEX IF NOT EXISTS idx_trade_date ON trade_log(trade_date);
    CREATE INDEX IF NOT EXISTS idx_trade_type ON trade_log(trade_type);
    """)
    conn.close()


# 启动时建表
init_trade_tables()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 记录交易
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def record_buy(code: str, name: str, shares: int, price: float,
               sector: str = "", source: str = "bot", note: str = ""):
    """记录买入，同时抓取当时的技术指标"""
    # 获取买入时技术指标
    ta = _get_current_technical(code)

    conn = _conn()
    conn.execute("""
        INSERT INTO trade_log
        (trade_date, trade_type, code, name, shares, price, amount,
         ma_trend, macd_signal, rsi, vol_pattern, tech_score, sector, source, note)
        VALUES (?, 'buy', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d"),
        code, name, shares, price, round(shares * price, 2),
        ta.get("ma_trend", ""), ta.get("macd_signal", ""),
        ta.get("rsi", 0), ta.get("vol_pattern", ""),
        ta.get("score", 0), sector, source, note,
    ))
    conn.commit()
    conn.close()
    logger.info(f"[交易日志] 买入 {name}({code}) {shares}股 {price}元")


def record_sell(code: str, name: str, shares: int, price: float,
                cost_price: float = 0, source: str = "bot", note: str = ""):
    """记录卖出，计算盈亏"""
    # 查找最近的买入记录
    conn = _conn()
    buy_record = conn.execute("""
        SELECT price, trade_date, ma_trend, macd_signal, rsi, vol_pattern,
               tech_score, sector
        FROM trade_log
        WHERE code = ? AND trade_type = 'buy'
        ORDER BY trade_date DESC LIMIT 1
    """, (code,)).fetchone()

    buy_price = cost_price
    hold_days = 0
    ma_trend = ""
    macd_signal = ""
    rsi = 0
    vol_pattern = ""
    tech_score = 0
    sector = ""

    if buy_record:
        if buy_price == 0:
            buy_price = buy_record["price"]
        try:
            buy_date = datetime.strptime(buy_record["trade_date"], "%Y-%m-%d")
            hold_days = (datetime.now() - buy_date).days
        except:
            pass
        ma_trend = buy_record["ma_trend"]
        macd_signal = buy_record["macd_signal"]
        rsi = buy_record["rsi"]
        vol_pattern = buy_record["vol_pattern"]
        tech_score = buy_record["tech_score"]
        sector = buy_record["sector"]

    pnl = round((price - buy_price) * shares, 2) if buy_price > 0 else 0
    pnl_pct = round((price - buy_price) / buy_price * 100, 2) if buy_price > 0 else 0

    conn.execute("""
        INSERT INTO trade_log
        (trade_date, trade_type, code, name, shares, price, amount,
         buy_price, pnl, pnl_pct, hold_days,
         ma_trend, macd_signal, rsi, vol_pattern, tech_score, sector, source, note)
        VALUES (?, 'sell', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d"),
        code, name, shares, price, round(shares * price, 2),
        buy_price, pnl, pnl_pct, hold_days,
        ma_trend, macd_signal, rsi, vol_pattern, tech_score, sector, source, note,
    ))
    conn.commit()
    conn.close()
    logger.info(f"[交易日志] 卖出 {name}({code}) {shares}股 {price}元 盈亏:{pnl}({pnl_pct}%)")


def _get_current_technical(code: str) -> dict:
    """获取当前技术指标"""
    try:
        from market_scanner import _fetch_kline, analyze_kline
        df = _fetch_kline(code, 120)
        if not df.empty:
            return analyze_kline(df)
    except:
        pass
    return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 战绩统计
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_performance_summary(days: int = 30) -> dict:
    """总体盈亏统计"""
    conn = _conn()
    sells = conn.execute("""
        SELECT code, name, pnl, pnl_pct, hold_days, buy_price, price,
               ma_trend, macd_signal, sector, trade_date
        FROM trade_log
        WHERE trade_type = 'sell'
          AND trade_date >= date('now', ? || ' days')
        ORDER BY trade_date DESC
    """, (f"-{days}",)).fetchall()
    conn.close()

    if not sells:
        return {"total_trades": 0, "message": "暂无卖出记录"}

    sells = [dict(s) for s in sells]
    wins = [s for s in sells if s["pnl"] and s["pnl"] > 0]
    losses = [s for s in sells if s["pnl"] and s["pnl"] <= 0]
    total_pnl = sum(s["pnl"] or 0 for s in sells)
    avg_pnl_pct = sum(s["pnl_pct"] or 0 for s in sells) / len(sells) if sells else 0
    avg_hold = sum(s["hold_days"] or 0 for s in sells) / len(sells) if sells else 0

    best = max(sells, key=lambda x: x["pnl"] or 0) if sells else None
    worst = min(sells, key=lambda x: x["pnl"] or 0) if sells else None

    return {
        "period": f"近{days}天",
        "total_trades": len(sells),
        "win_trades": len(wins),
        "lose_trades": len(losses),
        "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_pct": round(avg_pnl_pct, 2),
        "avg_hold_days": round(avg_hold, 1),
        "best_trade": f"{best['name']}({best['code']}) +{best['pnl']}元" if best else "",
        "worst_trade": f"{worst['name']}({worst['code']}) {worst['pnl']}元" if worst else "",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 盈亏模式分析（核心！）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_winning_patterns(days: int = 90) -> dict:
    """
    分析盈利交易的共同特征，找出什么条件下买入最容易赚钱

    输出：
      - 哪种 MA 趋势下买入胜率最高
      - 哪种 MACD 信号下买入胜率最高
      - 哪个板块赚钱最多
      - 最佳持仓天数
      - 综合最优策略
    """
    conn = _conn()
    sells = conn.execute("""
        SELECT code, name, pnl, pnl_pct, hold_days,
               ma_trend, macd_signal, rsi, vol_pattern, tech_score, sector
        FROM trade_log
        WHERE trade_type = 'sell'
          AND trade_date >= date('now', ? || ' days')
          AND pnl IS NOT NULL
    """, (f"-{days}",)).fetchall()
    conn.close()

    if len(sells) < 5:
        return {"message": f"交易记录不足（需至少5笔卖出，当前{len(sells)}笔），继续积累"}

    sells = [dict(s) for s in sells]

    # 按维度分组统计胜率
    result = {"period": f"近{days}天", "sample_size": len(sells)}

    # MA趋势胜率
    result["ma_analysis"] = _group_win_rate(sells, "ma_trend")
    # MACD信号胜率
    result["macd_analysis"] = _group_win_rate(sells, "macd_signal")
    # 板块胜率
    result["sector_analysis"] = _group_win_rate(sells, "sector")
    # 量价模式胜率
    result["vol_analysis"] = _group_win_rate(sells, "vol_pattern")

    # 持仓天数分析
    short = [s for s in sells if (s["hold_days"] or 0) <= 3]
    medium = [s for s in sells if 3 < (s["hold_days"] or 0) <= 7]
    long_ = [s for s in sells if (s["hold_days"] or 0) > 7]
    result["hold_period_analysis"] = {
        "1-3天": _calc_group_stats(short),
        "4-7天": _calc_group_stats(medium),
        "7天+": _calc_group_stats(long_),
    }

    # 综合最优策略
    best_ma = max(result["ma_analysis"].items(), key=lambda x: x[1]["win_rate"], default=("", {}))
    best_macd = max(result["macd_analysis"].items(), key=lambda x: x[1]["win_rate"], default=("", {}))
    best_sector = max(result["sector_analysis"].items(), key=lambda x: x[1]["avg_pnl"], default=("", {}))

    result["optimal_strategy"] = {
        "best_ma_trend": best_ma[0] if best_ma[1] else "",
        "best_macd_signal": best_macd[0] if best_macd[1] else "",
        "best_sector": best_sector[0] if best_sector[1] else "",
        "suggestion": (
            f"最佳买入条件: MA={best_ma[0]}, MACD={best_macd[0]}, "
            f"板块={best_sector[0]}"
        ) if best_ma[0] else "数据不足",
    }

    return result


def _group_win_rate(trades: list, field: str) -> dict:
    """按某个字段分组统计胜率"""
    groups = {}
    for t in trades:
        key = t.get(field, "") or "未知"
        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    result = {}
    for key, group in groups.items():
        if len(group) < 2:
            continue
        result[key] = _calc_group_stats(group)
    return result


def _calc_group_stats(group: list) -> dict:
    """计算一组交易的统计数据"""
    if not group:
        return {"count": 0, "win_rate": 0, "avg_pnl": 0}
    wins = [t for t in group if (t.get("pnl") or 0) > 0]
    total_pnl = sum(t.get("pnl") or 0 for t in group)
    avg_pnl = sum(t.get("pnl_pct") or 0 for t in group) / len(group)
    return {
        "count": len(group),
        "win_rate": round(len(wins) / len(group) * 100, 1),
        "avg_pnl": round(avg_pnl, 2),
        "total_pnl": round(total_pnl, 2),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 最近交易记录
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_recent_trades(limit: int = 20) -> list:
    """获取最近的交易记录"""
    conn = _conn()
    rows = conn.execute("""
        SELECT trade_date, trade_type, code, name, shares, price, amount,
               pnl, pnl_pct, hold_days, ma_trend, macd_signal, sector
        FROM trade_log
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 格式化输出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def format_performance(stats: dict) -> str:
    """格式化战绩统计"""
    if stats.get("total_trades", 0) == 0:
        return "📊 暂无交易记录\n开始交易后会自动记录"

    wr = stats.get("win_rate", 0)
    emoji = "🏆" if wr >= 60 else "📊" if wr >= 40 else "⚠️"

    lines = [
        f"{emoji} **交易战绩** ({stats.get('period', '')})",
        "",
        f"总交易: {stats['total_trades']}笔 | 胜率: {wr}%",
        f"盈利: {stats['win_trades']}笔 | 亏损: {stats['lose_trades']}笔",
        f"总盈亏: {'🟢' if stats['total_pnl']>=0 else '🔴'} {stats['total_pnl']}元",
        f"平均收益: {stats['avg_pnl_pct']}%",
        f"平均持仓: {stats['avg_hold_days']}天",
    ]
    if stats.get("best_trade"):
        lines.append(f"最佳: {stats['best_trade']}")
    if stats.get("worst_trade"):
        lines.append(f"最差: {stats['worst_trade']}")
    return "\n".join(lines)


def format_pattern_analysis(analysis: dict) -> str:
    """格式化盈亏模式分析"""
    if analysis.get("message"):
        return f"📊 {analysis['message']}"

    lines = [
        f"🔍 **盈亏模式分析** ({analysis.get('period', '')})",
        f"样本: {analysis.get('sample_size', 0)}笔交易",
        "",
    ]

    # MA趋势
    ma = analysis.get("ma_analysis", {})
    if ma:
        lines.append("📈 **MA趋势胜率**:")
        for k, v in sorted(ma.items(), key=lambda x: x[1]["win_rate"], reverse=True):
            emoji = "🟢" if v["win_rate"] >= 50 else "🔴"
            lines.append(f"  {emoji} {k}: 胜率{v['win_rate']}% ({v['count']}笔) 均收益{v['avg_pnl']}%")

    # MACD
    macd = analysis.get("macd_analysis", {})
    if macd:
        lines.append("\n📊 **MACD信号胜率**:")
        for k, v in sorted(macd.items(), key=lambda x: x[1]["win_rate"], reverse=True):
            emoji = "🟢" if v["win_rate"] >= 50 else "🔴"
            lines.append(f"  {emoji} {k}: 胜率{v['win_rate']}% ({v['count']}笔)")

    # 板块
    sector = analysis.get("sector_analysis", {})
    if sector:
        lines.append("\n🏭 **板块胜率**:")
        for k, v in sorted(sector.items(), key=lambda x: x[1]["avg_pnl"], reverse=True)[:5]:
            emoji = "🟢" if v["avg_pnl"] >= 0 else "🔴"
            lines.append(f"  {emoji} {k}: 均收益{v['avg_pnl']}% ({v['count']}笔)")

    # 持仓天数
    hold = analysis.get("hold_period_analysis", {})
    if hold:
        lines.append("\n⏱️ **持仓天数**:")
        for k, v in hold.items():
            if v.get("count", 0) > 0:
                emoji = "🟢" if v["win_rate"] >= 50 else "🔴"
                lines.append(f"  {emoji} {k}: 胜率{v['win_rate']}% 均收益{v['avg_pnl']}%")

    # 最优策略
    opt = analysis.get("optimal_strategy", {})
    if opt.get("suggestion"):
        lines.append(f"\n💡 **最优策略**: {opt['suggestion']}")

    return "\n".join(lines)


def format_recent_trades(trades: list) -> str:
    """格式化最近交易"""
    if not trades:
        return "📋 暂无交易记录"

    lines = ["📋 **最近交易记录**", ""]
    for t in trades[:15]:
        if t["trade_type"] == "buy":
            lines.append(
                f"  🟢 {t['trade_date']} 买入 {t['name']}({t['code']}) "
                f"{t['shares']}股 {t['price']}元"
            )
        else:
            pnl = t.get("pnl", 0)
            emoji = "🟢" if pnl and pnl >= 0 else "🔴"
            lines.append(
                f"  {emoji} {t['trade_date']} 卖出 {t['name']}({t['code']}) "
                f"{t['shares']}股 {t['price']}元 "
                f"盈亏:{pnl}元({t.get('pnl_pct',0)}%) 持{t.get('hold_days',0)}天"
            )
    return "\n".join(lines)