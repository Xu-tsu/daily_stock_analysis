"""
data_store.py
SQLite persistence helpers for scan/history data.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DB_PATH = Path("data/scanner_history.db")


def _ensure_parent() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn():
    _ensure_parent()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                name TEXT NOT NULL,
                price REAL,
                change_pct REAL,
                turnover REAL,
                tech_score REAL,
                strategy TEXT,
                scan_date TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_results_scan_date ON scan_results(scan_date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_results_code ON scan_results(code)"
        )
        conn.commit()


def save_scan_results(results: Iterable[Dict], *, strategy: Optional[str] = None) -> int:
    init_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for r in results or []:
        rows.append(
            (
                str(r.get("code", "")),
                str(r.get("name", "")),
                float(r.get("price", 0) or 0),
                float(r.get("change_pct", 0) or 0),
                float(r.get("turnover", 0) or 0),
                float(r.get("tech_score", 0) or 0),
                strategy or str(r.get("strategy", "")),
                now,
            )
        )
    if not rows:
        return 0

    with _conn() as conn:
        conn.executemany(
            """
            INSERT INTO scan_results
            (code, name, price, change_pct, turnover, tech_score, strategy, scan_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def get_recent_scan_results(limit: int = 20) -> List[Dict]:
    init_db()
    with _conn() as conn:
        cur = conn.execute(
            """
            SELECT code, name, price, change_pct, turnover, tech_score, strategy, scan_date
            FROM scan_results
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        out = []
        for row in cur.fetchall():
            out.append(
                {
                    "code": row[0],
                    "name": row[1],
                    "price": row[2],
                    "change_pct": row[3],
                    "turnover": row[4],
                    "tech_score": row[5],
                    "strategy": row[6],
                    "scan_date": row[7],
                }
            )
        return out
