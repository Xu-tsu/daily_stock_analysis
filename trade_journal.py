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

from src.core.trading_calendar import count_stock_trading_days

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
        fee REAL DEFAULT 0,
        tax REAL DEFAULT 0,
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

    -- 市场环境表：每笔交易时的大盘/板块/资金状态
    CREATE TABLE IF NOT EXISTS trade_market_context (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_log_id INTEGER NOT NULL,
        -- 大盘环境
        sh_index REAL,           -- 上证指数点位
        sh_change_pct REAL,      -- 上证当日涨跌幅
        sh_position_pct REAL,    -- 上证在近20日的位置(0=最低,100=最高)
        advance_decline REAL,    -- 涨跌家数比(>1多数上涨)
        limit_up_count INTEGER,  -- 涨停数
        limit_down_count INTEGER,-- 跌停数
        -- 个股环境
        stock_position_pct REAL, -- 个股在近20日价格的位置
        vol_ratio REAL,          -- 量比(当日成交/5日均量)
        turnover_rate REAL,      -- 换手率
        main_net_inflow REAL,    -- 当日主力净流入(万元)
        -- 板块环境
        sector_name TEXT,        -- 所属板块
        sector_rank INTEGER,     -- 板块当日涨幅排名
        sector_change_pct REAL,  -- 板块当日涨幅
        -- 元信息
        created_at TEXT DEFAULT (datetime('now','localtime')),
        FOREIGN KEY(trade_log_id) REFERENCES trade_log(id)
    );
    CREATE INDEX IF NOT EXISTS idx_ctx_trade ON trade_market_context(trade_log_id);
    """)
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(trade_log)").fetchall()
    }
    if "fee" not in columns:
        conn.execute("ALTER TABLE trade_log ADD COLUMN fee REAL DEFAULT 0")
    if "tax" not in columns:
        conn.execute("ALTER TABLE trade_log ADD COLUMN tax REAL DEFAULT 0")
    conn.commit()
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
        (trade_date, trade_type, code, name, shares, price, amount, fee, tax,
         ma_trend, macd_signal, rsi, vol_pattern, tech_score, sector, source, note)
        VALUES (?, 'buy', ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d"),
        code, name, shares, price, round(shares * price, 2),
        ta.get("ma_trend", ""), ta.get("macd_signal", ""),
        ta.get("rsi", 0), ta.get("vol_pattern", ""),
        ta.get("score", 0), sector, source, note,
    ))
    trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    # 异步采集市场环境（不阻塞主流程）
    try:
        save_market_context(trade_id, code)
    except Exception:
        pass
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
            hold_days = (
                count_stock_trading_days(
                    code,
                    buy_record["trade_date"],
                    datetime.now(),
                    default_market="cn",
                )
                or 0
            )
        except Exception:
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
        (trade_date, trade_type, code, name, shares, price, amount, fee, tax,
         buy_price, pnl, pnl_pct, hold_days,
         ma_trend, macd_signal, rsi, vol_pattern, tech_score, sector, source, note)
        VALUES (?, 'sell', ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d"),
        code, name, shares, price, round(shares * price, 2),
        buy_price, pnl, pnl_pct, hold_days,
        ma_trend, macd_signal, rsi, vol_pattern, tech_score, sector, source, note,
    ))
    trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    try:
        save_market_context(trade_id, code)
    except Exception:
        pass
    logger.info(f"[交易日志] 卖出 {name}({code}) {shares}股 {price}元 盈亏:{pnl}({pnl_pct}%)")


def _get_current_technical(code: str) -> dict:
    """获取当前技术指标"""
    try:
        from market_scanner import _fetch_kline, analyze_kline
        df = _fetch_kline(code, 120)
        if not df.empty:
            return analyze_kline(df)
    except Exception:
        pass
    return {}


def _collect_market_context(code: str) -> dict:
    """采集当前市场环境数据（买入/卖出时自动调用）。"""
    ctx = {}
    try:
        import requests as _req
        # 1. 上证指数
        try:
            r = _req.get("http://qt.gtimg.cn/q=sh000001", timeout=5,
                         headers={"User-Agent": "Mozilla/5.0"})
            r.encoding = "gbk"
            import re as _re
            m = _re.search(r'v_sh000001="([^"]+)"', r.text)
            if m:
                fields = m.group(1).split("~")
                if len(fields) > 35:
                    sh_price = float(fields[3]) if fields[3] else 0
                    sh_change = float(fields[32]) if fields[32] else 0
                    ctx["sh_index"] = sh_price
                    ctx["sh_change_pct"] = sh_change
        except Exception:
            pass

        # 2. 个股量比/换手率
        try:
            prefix = "sh" if code.startswith(("6", "9")) else "sz"
            r = _req.get(f"http://qt.gtimg.cn/q={prefix}{code}", timeout=5,
                         headers={"User-Agent": "Mozilla/5.0"})
            r.encoding = "gbk"
            m = _re.search(r'v_\w+="([^"]+)"', r.text)
            if m:
                fields = m.group(1).split("~")
                if len(fields) > 45:
                    ctx["turnover_rate"] = float(fields[38]) if fields[38] else 0
                    ctx["vol_ratio"] = float(fields[49]) if fields[49] else 0
        except Exception:
            pass

        # 3. 个股20日位置
        try:
            from market_scanner import _fetch_kline
            df = _fetch_kline(code, 20)
            if df is not None and len(df) >= 5:
                high_20 = df["high"].max()
                low_20 = df["low"].min()
                current = df["close"].iloc[-1]
                if high_20 > low_20:
                    ctx["stock_position_pct"] = round(
                        (current - low_20) / (high_20 - low_20) * 100, 1)
        except Exception:
            pass

        # 4. 主力资金流向
        try:
            from ths_scraper import fetch_single_stock_fund_flow
            flow = fetch_single_stock_fund_flow(code)
            if flow:
                ctx["main_net_inflow"] = flow.get("main_net", 0)
        except Exception:
            pass

    except Exception as e:
        logger.debug(f"采集市场环境失败: {e}")

    return ctx


def save_market_context(trade_log_id: int, code: str, ctx: Optional[dict] = None):
    """保存交易时的市场环境到数据库。"""
    if ctx is None:
        ctx = _collect_market_context(code)
    if not ctx:
        return

    conn = _conn()
    conn.execute("""
        INSERT OR IGNORE INTO trade_market_context
        (trade_log_id, sh_index, sh_change_pct, sh_position_pct,
         stock_position_pct, vol_ratio, turnover_rate, main_net_inflow)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_log_id,
        ctx.get("sh_index"),
        ctx.get("sh_change_pct"),
        ctx.get("sh_position_pct"),
        ctx.get("stock_position_pct"),
        ctx.get("vol_ratio"),
        ctx.get("turnover_rate"),
        ctx.get("main_net_inflow"),
    ))
    conn.commit()
    conn.close()


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
    # JOIN 市场环境数据
    sells = conn.execute("""
        SELECT t.id, t.code, t.name, t.pnl, t.pnl_pct, t.hold_days,
               t.ma_trend, t.macd_signal, t.rsi, t.vol_pattern, t.tech_score, t.sector,
               c.sh_change_pct, c.stock_position_pct, c.vol_ratio,
               c.turnover_rate, c.main_net_inflow
        FROM trade_log t
        LEFT JOIN trade_market_context c ON c.trade_log_id = t.id
        WHERE t.trade_type = 'sell'
          AND t.trade_date >= date('now', ? || ' days')
          AND t.pnl IS NOT NULL
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

    # 市场环境分析
    ctx_sells = [s for s in sells if s.get("sh_change_pct") is not None]
    if ctx_sells:
        # 大盘涨跌对胜率的影响
        up_market = [s for s in ctx_sells if (s["sh_change_pct"] or 0) > 0]
        down_market = [s for s in ctx_sells if (s["sh_change_pct"] or 0) <= 0]
        result["market_env_analysis"] = {
            "大盘上涨日买入": _calc_group_stats(up_market),
            "大盘下跌日买入": _calc_group_stats(down_market),
        }
        # 个股位置对胜率的影响
        low_pos = [s for s in ctx_sells if (s.get("stock_position_pct") or 50) < 30]
        mid_pos = [s for s in ctx_sells if 30 <= (s.get("stock_position_pct") or 50) < 70]
        high_pos = [s for s in ctx_sells if (s.get("stock_position_pct") or 50) >= 70]
        result["position_analysis"] = {
            "低位(0-30%)": _calc_group_stats(low_pos),
            "中位(30-70%)": _calc_group_stats(mid_pos),
            "高位(70-100%)": _calc_group_stats(high_pos),
        }
        # 主力资金对胜率的影响
        inflow = [s for s in ctx_sells if (s.get("main_net_inflow") or 0) > 0]
        outflow = [s for s in ctx_sells if (s.get("main_net_inflow") or 0) <= 0]
        result["fund_flow_analysis"] = {
            "主力净流入时买入": _calc_group_stats(inflow),
            "主力净流出时买入": _calc_group_stats(outflow),
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
        f"平均持仓: {stats['avg_hold_days']}个交易日",
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 批量导入交易数据（CSV / Excel）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 支持的列名映射（兼容各券商导出格式）
_COL_ALIASES = {
    "date": ["成交日期", "交易日期", "日期", "trade_date", "date", "委托日期"],
    "direction": ["操作", "买卖方向", "交易类型", "方向", "trade_type", "direction", "type", "买卖标志"],
    "code": ["证券代码", "股票代码", "代码", "code", "stock_code", "symbol"],
    "name": ["证券名称", "股票名称", "名称", "name", "stock_name"],
    "price": ["成交均价", "成交价格", "价格", "price", "成交价"],
    "shares": ["成交数量", "成交股数", "数量", "shares", "volume", "qty"],
    "amount": ["成交金额", "发生金额", "金额", "amount"],
    "fee": ["手续费", "佣金", "交易费", "规费", "过户费", "fee", "commission"],
    "tax": ["印花税", "税费", "tax", "stamp_tax"],
}

_BUY_KEYWORDS = {"买入", "买", "buy", "b", "证券买入", "担保品买入", "融资买入"}
_SELL_KEYWORDS = {"卖出", "卖", "sell", "s", "证券卖出", "担保品卖出", "融券卖出"}


def _find_column(df_columns, field: str) -> Optional[str]:
    """在 DataFrame 列名中查找匹配的列"""
    aliases = _COL_ALIASES.get(field, [])
    for alias in aliases:
        for col in df_columns:
            if alias == col.strip():
                return col
    # 模糊匹配
    for alias in aliases:
        for col in df_columns:
            if alias in col.strip():
                return col
    return None


def _normalize_code(raw_code) -> str:
    """标准化股票代码为6位数字"""
    s = str(raw_code).strip()
    # 去掉 SH/SZ/sh/sz 前缀和 .SH/.SZ 后缀
    for prefix in ("SH", "SZ", "sh", "sz"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in (".SH", ".SZ", ".sh", ".sz"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    s = s.strip().lstrip("0") if len(s.strip()) > 6 else s.strip()
    return s.zfill(6)


def _normalize_date(raw_date) -> str:
    """标准化日期为 YYYY-MM-DD"""
    s = str(raw_date).strip()
    # 处理 pandas Timestamp
    if hasattr(raw_date, 'strftime'):
        return raw_date.strftime("%Y-%m-%d")
    # 常见格式
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y.%m.%d",
                "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s[:10], fmt).strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            continue
    return s[:10]


def import_trades_from_file(file_path: str, source: str = "import") -> dict:
    """
    从 CSV 或 Excel 文件批量导入交易记录。

    支持格式:
      - CSV (.csv) — 自动检测编码 (UTF-8 / GBK)
      - Excel (.xlsx / .xls)

    列名自动识别，兼容各主流券商导出格式:
      必须列: 日期、买卖方向、证券代码、成交价格、成交数量
      可选列: 证券名称、成交金额

    Returns:
        {"imported": 导入条数, "skipped": 跳过条数, "errors": [错误信息]}
    """
    import pandas as pd

    file_path = str(file_path).strip()
    ext = os.path.splitext(file_path)[1].lower()

    # 读取文件（同花顺导出的 .xls 实际是 GBK 编码的 TSV 文本文件）
    df = None
    if ext in (".xls", ".csv", ".txt"):
        # 先尝试当 TSV/CSV 文本读取（同花顺格式）
        for enc in ("gbk", "gb2312", "utf-8", "utf-8-sig"):
            for sep in ("\t", ","):
                try:
                    _df = pd.read_csv(file_path, encoding=enc, sep=sep, dtype=str)
                    if len(_df.columns) >= 5:
                        df = _df
                        break
                except Exception:
                    continue
            if df is not None:
                break
        # 文本方式失败，尝试 Excel 引擎
        if df is None and ext in (".xls", ".xlsx"):
            try:
                df = pd.read_excel(file_path, engine="xlrd", dtype=str)
            except Exception:
                try:
                    df = pd.read_excel(file_path, engine="openpyxl", dtype=str)
                except Exception:
                    pass
    elif ext == ".xlsx":
        df = pd.read_excel(file_path, dtype=str)
    else:
        return {"imported": 0, "skipped": 0, "errors": [f"不支持的文件格式: {ext}，请用 .csv / .xlsx / .xls"]}

    if df is None or df.empty:
        return {"imported": 0, "skipped": 0, "errors": ["无法读取文件或文件为空"]}

    # 识别列
    col_date = _find_column(df.columns, "date")
    col_dir = _find_column(df.columns, "direction")
    col_code = _find_column(df.columns, "code")
    col_price = _find_column(df.columns, "price")
    col_shares = _find_column(df.columns, "shares")
    col_name = _find_column(df.columns, "name")
    col_amount = _find_column(df.columns, "amount")
    col_fee = _find_column(df.columns, "fee")
    col_tax = _find_column(df.columns, "tax")

    missing = []
    if not col_date: missing.append("日期")
    if not col_dir: missing.append("买卖方向")
    if not col_code: missing.append("证券代码")
    if not col_price: missing.append("成交价格")
    if not col_shares: missing.append("成交数量")
    if missing:
        return {
            "imported": 0, "skipped": 0,
            "errors": [f"缺少必须列: {', '.join(missing)}。检测到的列: {list(df.columns)}"],
        }

    logger.info(f"[导入] 列映射: date={col_date}, dir={col_dir}, code={col_code}, "
                f"price={col_price}, shares={col_shares}, name={col_name}")

    # 逐行导入
    imported = 0
    skipped = 0
    errors = []
    conn = _conn()

    for idx, row in df.iterrows():
        try:
            raw_dir = str(row[col_dir]).strip()
            if raw_dir.lower() in _BUY_KEYWORDS:
                trade_type = "buy"
            elif raw_dir.lower() in _SELL_KEYWORDS:
                trade_type = "sell"
            else:
                skipped += 1
                continue

            code = _normalize_code(row[col_code])
            trade_date = _normalize_date(row[col_date])
            price = float(str(row[col_price]).replace(",", ""))
            shares = int(float(str(row[col_shares]).replace(",", "")))
            # 同花顺卖出数量为负数（如 -1100），取绝对值
            shares = abs(shares)
            name = str(row[col_name]).strip() if col_name and pd.notna(row.get(col_name)) else ""
            amount = float(str(row[col_amount]).replace(",", "")) if col_amount and pd.notna(row.get(col_amount)) else round(price * shares, 2)
            amount = abs(amount)
            fee = (
                abs(float(str(row[col_fee]).replace(",", "")))
                if col_fee and pd.notna(row.get(col_fee))
                else 0.0
            )
            tax = (
                abs(float(str(row[col_tax]).replace(",", "")))
                if col_tax and pd.notna(row.get(col_tax))
                else 0.0
            )

            if trade_type == "buy":
                conn.execute("""
                    INSERT INTO trade_log
                    (trade_date, trade_type, code, name, shares, price, amount, fee, tax, source)
                    VALUES (?, 'buy', ?, ?, ?, ?, ?, ?, ?, ?)
                """, (trade_date, code, name, shares, price, amount, fee, tax, source))
            else:
                # 卖出：查找对应买入记录算盈亏
                buy_rec = conn.execute("""
                    SELECT price, trade_date FROM trade_log
                    WHERE code = ? AND trade_type = 'buy'
                    ORDER BY trade_date DESC LIMIT 1
                """, (code,)).fetchone()

                buy_price = buy_rec["price"] if buy_rec else 0
                hold_days = 0
                if buy_rec:
                    try:
                        hold_days = (
                            count_stock_trading_days(
                                code,
                                buy_rec["trade_date"],
                                trade_date,
                                default_market="cn",
                            )
                            or 0
                        )
                    except Exception:
                        pass

                pnl = round((price - buy_price) * shares, 2) if buy_price > 0 else 0
                pnl_pct = round((price - buy_price) / buy_price * 100, 2) if buy_price > 0 else 0

                conn.execute("""
                    INSERT INTO trade_log
                    (trade_date, trade_type, code, name, shares, price, amount, fee, tax,
                     buy_price, pnl, pnl_pct, hold_days, source)
                    VALUES (?, 'sell', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (trade_date, code, name, shares, price, amount,
                      fee, tax, buy_price, pnl, pnl_pct, hold_days, source))

            imported += 1
        except Exception as e:
            errors.append(f"第{idx+2}行: {e}")
            if len(errors) > 20:
                errors.append("...错误过多，已截断")
                break

    conn.commit()
    conn.close()

    logger.info(f"[导入] 完成: 导入{imported}条, 跳过{skipped}条, 错误{len(errors)}条")
    return {"imported": imported, "skipped": skipped, "errors": errors}


def import_trades_cli():
    """命令行导入入口: python trade_journal.py import <文件路径>"""
    import sys
    if len(sys.argv) < 3:
        print("用法: python trade_journal.py import <CSV或Excel文件路径>")
        print("")
        print("支持格式: .csv / .xlsx / .xls")
        print("必须列: 成交日期, 买卖方向, 证券代码, 成交价格, 成交数量")
        print("可选列: 证券名称, 成交金额")
        print("")
        print("示例:")
        print("  python trade_journal.py import 我的交易记录.csv")
        print("  python trade_journal.py import D:\\下载\\成交明细.xlsx")
        return

    file_path = sys.argv[2]
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    print(f"正在导入: {file_path}")
    result = import_trades_from_file(file_path)
    print(f"\n导入完成:")
    print(f"  成功: {result['imported']} 条")
    print(f"  跳过: {result['skipped']} 条（非买卖记录）")
    if result['errors']:
        print(f"  错误: {len(result['errors'])} 条")
        for e in result['errors'][:10]:
            print(f"    - {e}")

    # 导入后显示统计
    if result['imported'] > 0:
        print("\n--- 导入后战绩统计 ---")
        stats = get_performance_summary(days=365)
        # 移除 emoji 避免终端编码问题
        text = format_performance(stats)
        import re
        text = re.sub(r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\ufe00-\ufe0f]|[\U0001f000-\U0001f9ff]', '', text)
        print(text)


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
                f"盈亏:{pnl}元({t.get('pnl_pct',0)}%) 持{t.get('hold_days',0)}个交易日"
            )
    return "\n".join(lines)


def backfill_technical_indicators():
    """回填历史买入记录的技术指标（导入数据后运行一次）"""
    conn = _conn()
    buys = conn.execute("""
        SELECT id, code, trade_date FROM trade_log
        WHERE trade_type = 'buy' AND (ma_trend IS NULL OR ma_trend = '')
        ORDER BY trade_date
    """).fetchall()

    if not buys:
        print("没有需要回填的记录")
        return

    print(f"需要回填 {len(buys)} 条买入记录的技术指标...")
    updated = 0
    for i, row in enumerate(buys):
        code = row["code"]
        try:
            ta = _get_current_technical(code)
            if ta:
                conn.execute("""
                    UPDATE trade_log SET
                        ma_trend = ?, macd_signal = ?, rsi = ?,
                        vol_pattern = ?, tech_score = ?
                    WHERE id = ?
                """, (
                    ta.get("ma_trend", ""),
                    ta.get("macd_signal", ""),
                    ta.get("rsi", 0),
                    ta.get("vol_pattern", ""),
                    ta.get("score", 0),
                    row["id"],
                ))
                updated += 1
                if (i + 1) % 10 == 0:
                    conn.commit()
                    print(f"  进度: {i+1}/{len(buys)} (已更新 {updated})")
        except Exception as e:
            logger.debug(f"回填 {code} 失败: {e}")

    conn.commit()
    conn.close()
    print(f"回填完成: {updated}/{len(buys)} 条记录已更新技术指标")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "import":
        import_trades_cli()
    elif len(sys.argv) >= 2 and sys.argv[1] == "backfill":
        backfill_technical_indicators()
    elif len(sys.argv) >= 2 and sys.argv[1] == "stats":
        days = int(sys.argv[2]) if len(sys.argv) >= 3 else 90
        print(format_performance(get_performance_summary(days)))
    elif len(sys.argv) >= 2 and sys.argv[1] == "analyze":
        days = int(sys.argv[2]) if len(sys.argv) >= 3 else 90
        print(format_pattern_analysis(analyze_winning_patterns(days)))
    else:
        print("用法:")
        print("  python trade_journal.py import <文件路径>   — 导入交易记录(CSV/Excel)")
        print("  python trade_journal.py backfill            — 回填历史技术指标")
        print("  python trade_journal.py stats [天数]        — 查看战绩统计")
        print("  python trade_journal.py analyze [天数]      — 分析盈亏模式")
