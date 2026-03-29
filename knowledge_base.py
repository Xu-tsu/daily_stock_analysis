# -*- coding: utf-8 -*-
"""
knowledge_base.py — 向量知识库：PDF/交易数据存储 + 本地 Ollama Embedding + ChromaDB

功能:
  1. 将 PDF 切片存入 ChromaDB 向量数据库
  2. 将交易记录+市场环境向量化存储
  3. 查询时检索相似的历史交易场景
  4. 供 LLM 做买卖点推理的上下文

用法:
  python knowledge_base.py ingest <PDF文件>        — 导入 PDF 到知识库
  python knowledge_base.py query "低位放量金叉"     — 检索相似场景
  python knowledge_base.py sync                     — 同步交易记录到向量库
  python knowledge_base.py stats                    — 查看知识库统计
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

# 知识库存储路径
KB_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# embedding 模型（Ollama 支持的 embedding 模型）
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Ollama Embedding Function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OllamaEmbeddingFunction:
    """使用本地 Ollama 生成 embedding，不消耗云端额度。"""

    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                # 检查 embedding 模型是否已拉取
                base_names = [m.split(":")[0] for m in models]
                self._available = self.model.split(":")[0] in base_names
                if not self._available:
                    logger.info(
                        f"[知识库] Ollama embedding 模型 {self.model} 未找到。"
                        f"请运行: ollama pull {self.model}")
            else:
                self._available = False
        except Exception:
            self._available = False
        return self._available

    def __call__(self, input: List[str]) -> List[List[float]]:
        """生成 embedding 向量。"""
        import requests
        embeddings = []
        for text in input:
            try:
                r = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                    timeout=30,
                )
                if r.status_code == 200:
                    data = r.json()
                    # Ollama /api/embed 返回 {"embeddings": [[...]]}
                    emb = data.get("embeddings", [[]])[0]
                    if emb:
                        embeddings.append(emb)
                        continue
            except Exception as e:
                logger.debug(f"Ollama embedding 失败: {e}")
            # 失败时返回零向量（chromadb 需要相同维度）
            embeddings.append([0.0] * 768)
        return embeddings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. ChromaDB 初始化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_chroma_client() -> chromadb.ClientAPI:
    """获取持久化 ChromaDB 客户端。"""
    Path(KB_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=KB_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_embed_fn() -> Optional[OllamaEmbeddingFunction]:
    """获取 embedding 函数，不可用时返回 None。"""
    fn = OllamaEmbeddingFunction()
    if fn.is_available():
        return fn
    return None


def _get_collection(name: str = "trade_knowledge"):
    """获取或创建 ChromaDB 集合。"""
    client = _get_chroma_client()
    embed_fn = _get_embed_fn()

    kwargs = {"name": name}
    if embed_fn:
        kwargs["embedding_function"] = embed_fn

    return client.get_or_create_collection(**kwargs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. PDF 导入知识库
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ingest_pdf(file_path: str, collection_name: str = "documents") -> Dict[str, Any]:
    """将 PDF 切片并存入向量数据库。

    Returns:
        {"file": "xxx.pdf", "chunks": 42, "collection": "documents"}
    """
    from pdf_parser import chunk_pdf

    chunks = chunk_pdf(file_path, chunk_size=500, overlap=50)
    if not chunks:
        return {"file": file_path, "chunks": 0, "error": "无法从 PDF 提取文本"}

    collection = _get_collection(collection_name)
    filename = os.path.basename(file_path)

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = f"{filename}__p{chunk['page']}_c{chunk['chunk_idx']}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "source": filename,
            "page": chunk["page"],
            "chunk_idx": chunk["chunk_idx"],
            "type": "pdf",
        })

    # ChromaDB upsert（去重）
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    logger.info(f"[知识库] PDF 导入完成: {filename}, {len(chunks)} 个切片")
    return {"file": filename, "chunks": len(chunks), "collection": collection_name}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 交易记录同步到向量库
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sync_trades_to_knowledge_base() -> Dict[str, Any]:
    """将交易记录+市场环境向量化存入知识库。

    每笔卖出交易生成一段描述文本，包含：
      - 买入条件（MA趋势、MACD、RSI、量价）
      - 市场环境（大盘涨跌、个股位置、主力资金）
      - 结果（盈亏、持仓天数）
    LLM 检索时就能找到 "类似场景下历史上是赚还是亏"。
    """
    from trade_journal import _conn

    conn = _conn()
    sells = conn.execute("""
        SELECT t.id, t.trade_date, t.code, t.name, t.price, t.shares,
               t.buy_price, t.pnl, t.pnl_pct, t.hold_days,
               t.ma_trend, t.macd_signal, t.rsi, t.vol_pattern, t.tech_score, t.sector,
               c.sh_change_pct, c.stock_position_pct, c.vol_ratio,
               c.turnover_rate, c.main_net_inflow
        FROM trade_log t
        LEFT JOIN trade_market_context c ON c.trade_log_id = t.id
        WHERE t.trade_type = 'sell' AND t.pnl IS NOT NULL
        ORDER BY t.trade_date
    """).fetchall()
    conn.close()

    if not sells:
        return {"synced": 0, "message": "无卖出记录"}

    collection = _get_collection("trade_knowledge")

    ids = []
    documents = []
    metadatas = []

    for s in sells:
        s = dict(s)
        trade_id = f"trade_{s['id']}"
        result_label = "盈利" if (s["pnl"] or 0) > 0 else "亏损"

        # 构建自然语言描述（LLM 可以理解的格式）
        text_parts = [
            f"交易记录: {s['trade_date']} {result_label}",
            f"股票: {s['name']}({s['code']})",
            f"买入价: {s['buy_price']}, 卖出价: {s['price']}",
            f"盈亏: {s['pnl']}元({s['pnl_pct']}%), 持仓{s['hold_days']}个交易日",
        ]

        if s.get("ma_trend"):
            text_parts.append(f"买入时MA趋势: {s['ma_trend']}")
        if s.get("macd_signal"):
            text_parts.append(f"买入时MACD信号: {s['macd_signal']}")
        if s.get("rsi"):
            text_parts.append(f"买入时RSI: {s['rsi']}")
        if s.get("vol_pattern"):
            text_parts.append(f"量价模式: {s['vol_pattern']}")
        if s.get("sh_change_pct") is not None:
            text_parts.append(f"大盘当日涨跌: {s['sh_change_pct']}%")
        if s.get("stock_position_pct") is not None:
            text_parts.append(f"个股20日位置: {s['stock_position_pct']}%")
        if s.get("vol_ratio"):
            text_parts.append(f"量比: {s['vol_ratio']}")
        if s.get("main_net_inflow") is not None:
            flow_dir = "流入" if s["main_net_inflow"] > 0 else "流出"
            text_parts.append(f"主力资金{flow_dir}: {abs(s['main_net_inflow'])}万")

        text = "\n".join(text_parts)

        ids.append(trade_id)
        documents.append(text)
        metadatas.append({
            "type": "trade",
            "code": s["code"],
            "name": s["name"] or "",
            "date": s["trade_date"],
            "pnl": s["pnl"] or 0,
            "result": result_label,
            "ma_trend": s.get("ma_trend") or "",
            "macd_signal": s.get("macd_signal") or "",
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    logger.info(f"[知识库] 交易记录同步完成: {len(sells)} 笔")
    return {"synced": len(sells), "collection": "trade_knowledge"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 相似场景检索
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def query_similar_trades(
    query: str,
    n_results: int = 5,
    collection_name: str = "trade_knowledge",
) -> List[Dict[str, Any]]:
    """检索与查询最相似的历史交易。

    Args:
        query: 自然语言描述，如 "空头排列 MACD底部收敛 主力资金流入 低位"
        n_results: 返回条数

    Returns:
        [{"text": "...", "metadata": {...}, "distance": 0.xx}, ...]
    """
    collection = _get_collection(collection_name)

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )
    except Exception as e:
        logger.warning(f"[知识库] 检索失败: {e}")
        return []

    items = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            items.append({
                "text": doc,
                "metadata": meta,
                "distance": round(dist, 4),
            })

    return items


def build_trade_context_for_llm(
    code: str,
    ma_trend: str = "",
    macd_signal: str = "",
    stock_position_pct: float = 50,
    main_net_inflow: float = 0,
) -> str:
    """为 LLM 构建交易决策的历史参考上下文。

    根据当前股票的技术状态，检索历史上类似场景的交易结果。
    """
    # 构建查询
    query_parts = []
    if ma_trend:
        query_parts.append(f"MA趋势: {ma_trend}")
    if macd_signal:
        query_parts.append(f"MACD信号: {macd_signal}")
    if stock_position_pct < 30:
        query_parts.append("低位")
    elif stock_position_pct > 70:
        query_parts.append("高位")
    if main_net_inflow > 0:
        query_parts.append("主力资金流入")
    elif main_net_inflow < 0:
        query_parts.append("主力资金流出")

    if not query_parts:
        query_parts = ["交易记录"]

    query = " ".join(query_parts)
    similar = query_similar_trades(query, n_results=5)

    if not similar:
        return ""

    lines = ["[历史相似交易参考]"]
    wins = sum(1 for s in similar if s["metadata"].get("result") == "盈利")
    total = len(similar)
    lines.append(f"找到 {total} 条相似场景，历史胜率: {wins}/{total} ({wins/total*100:.0f}%)")
    lines.append("")

    for i, s in enumerate(similar, 1):
        lines.append(f"案例{i}: {s['text'][:200]}")
        lines.append("")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 知识库统计
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_kb_stats() -> Dict[str, Any]:
    """获取知识库统计信息。"""
    client = _get_chroma_client()
    collections = client.list_collections()

    stats = {"collections": {}}
    for col in collections:
        name = col.name if hasattr(col, 'name') else str(col)
        try:
            c = client.get_collection(name)
            count = c.count()
            stats["collections"][name] = count
        except Exception:
            stats["collections"][name] = "error"

    embed_fn = OllamaEmbeddingFunction()
    stats["ollama_embedding"] = embed_fn.is_available()
    stats["embed_model"] = EMBED_MODEL
    stats["storage_path"] = KB_DIR

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python knowledge_base.py ingest <PDF文件>    — 导入 PDF 到知识库")
        print("  python knowledge_base.py query \"查询文本\"    — 检索相似场景")
        print("  python knowledge_base.py sync                — 同步交易记录到向量库")
        print("  python knowledge_base.py stats               — 查看知识库统计")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "ingest":
        if len(sys.argv) < 3:
            print("请指定 PDF 文件路径")
            sys.exit(1)
        result = ingest_pdf(sys.argv[2])
        print(f"导入完成: {result['file']}, {result['chunks']} 个切片")

    elif cmd == "query":
        if len(sys.argv) < 3:
            print("请指定查询文本")
            sys.exit(1)
        query = sys.argv[2]
        results = query_similar_trades(query)
        if not results:
            print("未找到相似记录（知识库可能为空，请先 sync）")
        else:
            for i, r in enumerate(results, 1):
                print(f"\n--- 相似场景 {i} (距离: {r['distance']}) ---")
                print(r["text"])

    elif cmd == "sync":
        # 先检查 Ollama embedding
        embed_fn = OllamaEmbeddingFunction()
        if not embed_fn.is_available():
            print(f"Ollama embedding 模型 {EMBED_MODEL} 不可用。")
            print(f"请运行: ollama pull {EMBED_MODEL}")
            print("安装后重新运行 sync")
            sys.exit(1)
        result = sync_trades_to_knowledge_base()
        print(f"同步完成: {result['synced']} 笔交易记录")

    elif cmd == "stats":
        stats = get_kb_stats()
        print(f"存储路径: {stats['storage_path']}")
        print(f"Embedding 模型: {stats['embed_model']}")
        print(f"Ollama 可用: {stats['ollama_embedding']}")
        print(f"集合:")
        for name, count in stats.get("collections", {}).items():
            print(f"  {name}: {count} 条记录")
        if not stats.get("collections"):
            print("  (空)")

    else:
        print(f"未知命令: {cmd}")
