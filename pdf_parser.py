# -*- coding: utf-8 -*-
"""
pdf_parser.py — PDF 解析器：交割单/对账单/持仓明细/历史成交

支持:
  1. 券商交割单 PDF → 逐笔交易记录
  2. 对账单 PDF → 每日资产快照
  3. 持仓明细 PDF → 当前持仓
  4. 通用文本 PDF → 切片供向量存储

用法:
  python pdf_parser.py parse <文件路径>           — 自动识别类型并解析
  python pdf_parser.py import <文件路径>          — 解析并导入到交易数据库
  python pdf_parser.py chunks <文件路径>          — 切片输出（供向量存储）
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.trading_calendar import count_stock_trading_days

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PDF 文本提取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """提取 PDF 每页的文本内容。

    Returns:
        [{"page": 1, "text": "...", "tables": [...]}, ...]
    """
    import fitz  # pymupdf

    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        # 尝试提取表格（pymupdf 内置表格检测）
        tables = []
        try:
            page_tables = page.find_tables()
            for table in page_tables:
                rows = table.extract()
                if rows:
                    tables.append(rows)
        except Exception:
            pass
        pages.append({
            "page": i + 1,
            "text": text,
            "tables": tables,
        })
    doc.close()
    return pages


def extract_all_tables(file_path: str) -> List[List[List[str]]]:
    """提取 PDF 中所有表格数据。"""
    pages = extract_text_from_pdf(file_path)
    all_tables = []
    for page in pages:
        all_tables.extend(page.get("tables", []))
    return all_tables


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 文档类型自动识别
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_DOC_TYPE_KEYWORDS = {
    "delivery": ["交割单", "交割明细", "交割确认", "settlement"],
    "statement": ["对账单", "资产负债", "客户账单", "资产明细", "account statement"],
    "position": ["持仓明细", "股份余额", "证券余额", "持仓汇总", "position"],
    "trade_history": ["成交明细", "历史成交", "成交记录", "成交汇总", "trade history"],
}


def detect_document_type(pages: List[Dict[str, Any]]) -> str:
    """根据文本内容自动识别文档类型。"""
    # 只检查前3页
    sample_text = " ".join(p["text"] for p in pages[:3]).lower()

    for doc_type, keywords in _DOC_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in sample_text:
                return doc_type
    return "unknown"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 交割单 / 成交明细解析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 券商交割单/成交明细的列名映射
_TRADE_COL_PATTERNS = {
    "date": re.compile(r"(成交日期|交割日期|交易日期|日期)"),
    "code": re.compile(r"(证券代码|股票代码|代码)"),
    "name": re.compile(r"(证券名称|股票名称|名称)"),
    "direction": re.compile(r"(买卖方向|操作|摘要|交易类型|买卖标志)"),
    "price": re.compile(r"(成交均价|成交价格|成交价)"),
    "shares": re.compile(r"(成交数量|成交股数|数量)"),
    "amount": re.compile(r"(成交金额|发生金额|金额)"),
    "fee": re.compile(r"(手续费|佣金|总费用)"),
    "stamp_tax": re.compile(r"(印花税)"),
    "transfer_fee": re.compile(r"(过户费)"),
    "net_amount": re.compile(r"(实付金额|实收金额|净金额|发生金额)"),
}

_BUY_PATTERNS = re.compile(r"(买入|证券买入|担保品买入|融资买入|买)", re.IGNORECASE)
_SELL_PATTERNS = re.compile(r"(卖出|证券卖出|担保品卖出|融券卖出|卖)", re.IGNORECASE)


def _find_col_index(header: List[str], field: str) -> int:
    """在表头中查找匹配的列索引。"""
    pattern = _TRADE_COL_PATTERNS.get(field)
    if not pattern:
        return -1
    for i, col in enumerate(header):
        if col and pattern.search(str(col).strip()):
            return i
    return -1


def _normalize_code(raw: str) -> str:
    """标准化股票代码为6位。"""
    s = str(raw).strip()
    for prefix in ("SH", "SZ", "sh", "sz"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in (".SH", ".SZ", ".sh", ".sz"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    s = re.sub(r"[^\d]", "", s)
    return s.zfill(6) if len(s) <= 6 else s[-6:]


def _normalize_date(raw: str) -> str:
    """标准化日期为 YYYY-MM-DD。"""
    s = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y.%m.%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:10], fmt).strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            continue
    # 尝试从文本中提取日期
    m = re.search(r"(\d{4})[/-.]?(\d{2})[/-.]?(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return s[:10]


def parse_trade_tables(tables: List[List[List[str]]]) -> List[Dict[str, Any]]:
    """从表格数据中解析交易记录。

    Returns:
        [{"date": "2026-03-27", "code": "002310", "name": "东方新能",
          "direction": "sell", "price": 3.752, "shares": 1100,
          "amount": 4127.0, "fee": 2.06, "stamp_tax": 2.06, ...}, ...]
    """
    trades = []
    for table in tables:
        if not table or len(table) < 2:
            continue

        # 第一行作为表头
        header = [str(c).strip() if c else "" for c in table[0]]

        # 检查必须列
        col_date = _find_col_index(header, "date")
        col_code = _find_col_index(header, "code")
        col_dir = _find_col_index(header, "direction")
        col_price = _find_col_index(header, "price")
        col_shares = _find_col_index(header, "shares")

        if col_date < 0 or col_code < 0:
            # 可能表头不在第一行，试第二行
            if len(table) >= 3:
                header = [str(c).strip() if c else "" for c in table[1]]
                col_date = _find_col_index(header, "date")
                col_code = _find_col_index(header, "code")
                col_dir = _find_col_index(header, "direction")
                col_price = _find_col_index(header, "price")
                col_shares = _find_col_index(header, "shares")
                start_row = 2
            else:
                continue
        else:
            start_row = 1

        if col_date < 0 or col_code < 0:
            continue

        col_amount = _find_col_index(header, "amount")
        col_name = _find_col_index(header, "name")
        col_fee = _find_col_index(header, "fee")
        col_stamp = _find_col_index(header, "stamp_tax")
        col_transfer = _find_col_index(header, "transfer_fee")
        col_net = _find_col_index(header, "net_amount")

        for row in table[start_row:]:
            try:
                if len(row) <= max(col_date, col_code):
                    continue

                raw_code = str(row[col_code]).strip() if row[col_code] else ""
                if not raw_code or not re.search(r"\d{5,6}", raw_code):
                    continue

                code = _normalize_code(raw_code)
                date = _normalize_date(str(row[col_date]))

                # 方向
                direction = "unknown"
                if col_dir >= 0 and row[col_dir]:
                    raw_dir = str(row[col_dir]).strip()
                    if _BUY_PATTERNS.search(raw_dir):
                        direction = "buy"
                    elif _SELL_PATTERNS.search(raw_dir):
                        direction = "sell"

                # 数值
                def _float(idx):
                    if idx < 0 or idx >= len(row) or not row[idx]:
                        return 0.0
                    return float(str(row[idx]).replace(",", "").strip() or "0")

                price = _float(col_price)
                shares = abs(int(_float(col_shares)))
                amount = abs(_float(col_amount))
                name = str(row[col_name]).strip() if col_name >= 0 and col_name < len(row) and row[col_name] else ""

                # 如果方向未识别，通过数量正负判断
                if direction == "unknown" and col_shares >= 0:
                    raw_shares = _float(col_shares)
                    direction = "buy" if raw_shares > 0 else "sell" if raw_shares < 0 else "unknown"

                if direction == "unknown":
                    continue

                trade = {
                    "date": date,
                    "code": code,
                    "name": name,
                    "direction": direction,
                    "price": price,
                    "shares": shares,
                    "amount": amount if amount > 0 else round(price * shares, 2),
                    "fee": _float(col_fee),
                    "stamp_tax": _float(col_stamp),
                    "transfer_fee": _float(col_transfer),
                    "net_amount": abs(_float(col_net)),
                }
                trades.append(trade)
            except Exception as e:
                logger.debug(f"解析行失败: {e}, row={row}")
                continue

    return trades


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 对账单解析（每日资产快照）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_statement_tables(tables: List[List[List[str]]]) -> List[Dict[str, Any]]:
    """解析对账单表格，提取每日资产数据。"""
    records = []
    for table in tables:
        if not table or len(table) < 2:
            continue
        header = [str(c).strip() if c else "" for c in table[0]]

        # 查找日期和资产相关列
        date_idx = -1
        asset_idx = -1
        market_idx = -1
        cash_idx = -1

        for i, h in enumerate(header):
            h_lower = h.lower()
            if any(k in h_lower for k in ("日期", "date")):
                date_idx = i
            if any(k in h_lower for k in ("总资产", "资产总值", "total")):
                asset_idx = i
            if any(k in h_lower for k in ("市值", "证券市值", "stock")):
                market_idx = i
            if any(k in h_lower for k in ("资金余额", "可用资金", "现金", "cash")):
                cash_idx = i

        if date_idx < 0:
            continue

        for row in table[1:]:
            try:
                if len(row) <= date_idx or not row[date_idx]:
                    continue
                date = _normalize_date(str(row[date_idx]))

                def _val(idx):
                    if idx < 0 or idx >= len(row) or not row[idx]:
                        return None
                    try:
                        return float(str(row[idx]).replace(",", "").strip())
                    except ValueError:
                        return None

                record = {
                    "date": date,
                    "total_asset": _val(asset_idx),
                    "market_value": _val(market_idx),
                    "cash": _val(cash_idx),
                }
                if record["total_asset"] is not None:
                    records.append(record)
            except Exception:
                continue
    return records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 持仓明细解析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_position_tables(tables: List[List[List[str]]]) -> List[Dict[str, Any]]:
    """解析持仓明细表格。"""
    positions = []
    for table in tables:
        if not table or len(table) < 2:
            continue
        header = [str(c).strip() if c else "" for c in table[0]]

        code_idx = -1
        name_idx = -1
        shares_idx = -1
        cost_idx = -1
        current_idx = -1
        pnl_idx = -1

        for i, h in enumerate(header):
            if re.search(r"(证券代码|代码)", h): code_idx = i
            if re.search(r"(证券名称|名称)", h): name_idx = i
            if re.search(r"(持仓数量|股份余额|数量)", h): shares_idx = i
            if re.search(r"(成本价|成本)", h): cost_idx = i
            if re.search(r"(现价|最新价|市价)", h): current_idx = i
            if re.search(r"(浮动盈亏|盈亏)", h): pnl_idx = i

        if code_idx < 0:
            continue

        for row in table[1:]:
            try:
                if len(row) <= code_idx or not row[code_idx]:
                    continue
                raw_code = str(row[code_idx]).strip()
                if not re.search(r"\d{5,6}", raw_code):
                    continue

                def _val(idx):
                    if idx < 0 or idx >= len(row) or not row[idx]:
                        return 0.0
                    try:
                        return float(str(row[idx]).replace(",", "").strip())
                    except ValueError:
                        return 0.0

                positions.append({
                    "code": _normalize_code(raw_code),
                    "name": str(row[name_idx]).strip() if name_idx >= 0 and name_idx < len(row) and row[name_idx] else "",
                    "shares": int(_val(shares_idx)),
                    "cost_price": _val(cost_idx),
                    "current_price": _val(current_idx),
                    "pnl": _val(pnl_idx),
                })
            except Exception:
                continue
    return positions


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 文本切片（供向量存储）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """将文本按指定大小切片，带重叠。"""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def chunk_pdf(file_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """将 PDF 切片，带页码元数据。

    Returns:
        [{"text": "...", "page": 1, "chunk_idx": 0, "source": "file.pdf"}, ...]
    """
    pages = extract_text_from_pdf(file_path)
    filename = os.path.basename(file_path)
    all_chunks = []
    chunk_idx = 0

    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]

        # 表格也转为文本
        for table in page_info.get("tables", []):
            for row in table:
                text += "\n" + " | ".join(str(c) for c in row if c)

        page_chunks = chunk_text(text, chunk_size, overlap)
        for chunk in page_chunks:
            all_chunks.append({
                "text": chunk,
                "page": page_num,
                "chunk_idx": chunk_idx,
                "source": filename,
            })
            chunk_idx += 1

    return all_chunks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 统一解析入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_pdf(file_path: str) -> Dict[str, Any]:
    """统一解析入口：自动识别类型并解析。

    Returns:
        {
            "file": "xxx.pdf",
            "doc_type": "delivery|statement|position|trade_history|unknown",
            "trades": [...],       # 交割单/成交明细
            "statements": [...],   # 对账单
            "positions": [...],    # 持仓明细
            "chunks": [...],       # 文本切片
            "raw_text": "...",     # 原始文本
        }
    """
    pages = extract_text_from_pdf(file_path)
    doc_type = detect_document_type(pages)
    tables = []
    for p in pages:
        tables.extend(p.get("tables", []))

    full_text = "\n".join(p["text"] for p in pages)

    result = {
        "file": os.path.basename(file_path),
        "doc_type": doc_type,
        "page_count": len(pages),
        "trades": [],
        "statements": [],
        "positions": [],
        "chunks": chunk_pdf(file_path),
        "raw_text": full_text,
    }

    if doc_type in ("delivery", "trade_history"):
        result["trades"] = parse_trade_tables(tables)
    elif doc_type == "statement":
        result["statements"] = parse_statement_tables(tables)
    elif doc_type == "position":
        result["positions"] = parse_position_tables(tables)
    else:
        # 未知类型：尝试所有解析器
        result["trades"] = parse_trade_tables(tables)
        if not result["trades"]:
            result["statements"] = parse_statement_tables(tables)
        if not result["trades"] and not result["statements"]:
            result["positions"] = parse_position_tables(tables)

    # 如果表格解析失败，尝试正则从文本中提取交易
    if not result["trades"] and doc_type in ("delivery", "trade_history", "unknown"):
        result["trades"] = _extract_trades_from_text(full_text)

    return result


def _extract_trades_from_text(text: str) -> List[Dict[str, Any]]:
    """从纯文本中用正则提取交易记录（表格解析失败时的后备方案）。"""
    trades = []
    # 匹配模式: 日期 代码 名称 买入/卖出 数量 价格 金额
    pattern = re.compile(
        r"(\d{4}[/-]?\d{2}[/-]?\d{2})\s+"   # 日期
        r"(\d{6})\s+"                         # 代码
        r"(\S+)\s+"                           # 名称
        r"(证券[买卖]入|买入|卖出)\s+"          # 方向
        r"[-]?(\d+)\s+"                       # 数量
        r"([\d.]+)\s+"                        # 价格
        r"([\d.]+)"                           # 金额
    )
    for m in pattern.finditer(text):
        direction = "buy" if "买" in m.group(4) else "sell"
        trades.append({
            "date": _normalize_date(m.group(1)),
            "code": _normalize_code(m.group(2)),
            "name": m.group(3),
            "direction": direction,
            "shares": abs(int(m.group(5))),
            "price": float(m.group(6)),
            "amount": float(m.group(7)),
            "fee": 0, "stamp_tax": 0, "transfer_fee": 0, "net_amount": 0,
        })
    return trades


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 导入到 trade_journal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def import_pdf_to_trade_journal(file_path: str) -> Dict[str, Any]:
    """解析 PDF 并导入交易记录到数据库。"""
    from trade_journal import _conn, init_trade_tables

    init_trade_tables()
    parsed = parse_pdf(file_path)

    if not parsed["trades"]:
        return {
            "imported": 0,
            "doc_type": parsed["doc_type"],
            "message": f"未从 PDF 中解析到交易记录（类型: {parsed['doc_type']}，共 {parsed['page_count']} 页）",
            "chunks": len(parsed["chunks"]),
        }

    conn = _conn()
    imported = 0
    errors = []

    for t in parsed["trades"]:
        try:
            fee = round(float(t.get("fee", 0) or 0) + float(t.get("transfer_fee", 0) or 0), 2)
            tax = round(float(t.get("stamp_tax", 0) or 0), 2)
            if t["direction"] == "buy":
                conn.execute("""
                    INSERT INTO trade_log
                    (trade_date, trade_type, code, name, shares, price, amount, fee, tax, source)
                    VALUES (?, 'buy', ?, ?, ?, ?, ?, ?, ?, 'pdf')
                """, (t["date"], t["code"], t["name"], t["shares"],
                      t["price"], t["amount"], fee, tax))
            else:
                # 查找买入记录算盈亏
                buy = conn.execute("""
                    SELECT price, trade_date FROM trade_log
                    WHERE code = ? AND trade_type = 'buy'
                    ORDER BY trade_date DESC LIMIT 1
                """, (t["code"],)).fetchone()

                buy_price = buy["price"] if buy else 0
                hold_days = 0
                if buy:
                    try:
                        hold_days = (
                            count_stock_trading_days(
                                t["code"],
                                buy["trade_date"],
                                t["date"],
                                default_market="cn",
                            )
                            or 0
                        )
                    except Exception:
                        pass

                pnl = round((t["price"] - buy_price) * t["shares"], 2) if buy_price > 0 else 0
                pnl_pct = round((t["price"] - buy_price) / buy_price * 100, 2) if buy_price > 0 else 0

                conn.execute("""
                    INSERT INTO trade_log
                    (trade_date, trade_type, code, name, shares, price, amount, fee, tax,
                     buy_price, pnl, pnl_pct, hold_days, source)
                    VALUES (?, 'sell', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pdf')
                """, (t["date"], t["code"], t["name"], t["shares"],
                      t["price"], t["amount"], fee, tax, buy_price, pnl, pnl_pct, hold_days))

            imported += 1
        except Exception as e:
            errors.append(str(e))

    conn.commit()
    conn.close()

    return {
        "imported": imported,
        "doc_type": parsed["doc_type"],
        "total_parsed": len(parsed["trades"]),
        "errors": errors[:10],
        "chunks": len(parsed["chunks"]),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("用法:")
        print("  python pdf_parser.py parse <PDF文件>    — 解析并显示内容")
        print("  python pdf_parser.py import <PDF文件>   — 解析并导入交易数据库")
        print("  python pdf_parser.py chunks <PDF文件>   — 输出文本切片")
        sys.exit(0)

    cmd = sys.argv[1]
    file_path = sys.argv[2]

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        sys.exit(1)

    if cmd == "parse":
        result = parse_pdf(file_path)
        print(f"文件: {result['file']}")
        print(f"类型: {result['doc_type']}")
        print(f"页数: {result['page_count']}")
        print(f"交易记录: {len(result['trades'])} 条")
        print(f"对账单记录: {len(result['statements'])} 条")
        print(f"持仓记录: {len(result['positions'])} 条")
        print(f"文本切片: {len(result['chunks'])} 块")
        if result['trades']:
            print("\n前5条交易:")
            for t in result['trades'][:5]:
                dir_cn = "买入" if t["direction"] == "buy" else "卖出"
                print(f"  {t['date']} {dir_cn} {t['name']}({t['code']}) "
                      f"{t['shares']}股 {t['price']}元")
        if result['statements']:
            print("\n对账单摘要:")
            for s in result['statements'][:5]:
                print(f"  {s['date']} 总资产: {s['total_asset']}")
        if result['positions']:
            print("\n持仓:")
            for p in result['positions']:
                print(f"  {p['name']}({p['code']}) {p['shares']}股 "
                      f"成本:{p['cost_price']} 现价:{p['current_price']}")

    elif cmd == "import":
        result = import_pdf_to_trade_journal(file_path)
        print(f"文档类型: {result['doc_type']}")
        print(f"解析到: {result.get('total_parsed', 0)} 条交易")
        print(f"导入成功: {result['imported']} 条")
        if result.get('errors'):
            print(f"错误: {len(result['errors'])} 条")
        print(f"文本切片: {result['chunks']} 块（可用于向量存储）")

    elif cmd == "chunks":
        chunks = chunk_pdf(file_path)
        print(f"共 {len(chunks)} 个切片:\n")
        for c in chunks[:10]:
            print(f"--- 页{c['page']} / 切片{c['chunk_idx']} ---")
            print(c['text'][:200])
            print()
