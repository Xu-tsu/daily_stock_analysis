# -*- coding: utf-8 -*-
"""
news_scanner.py -- A股新闻舆情扫描器
====================================

核心功能：
  - 抓取东方财富/同花顺等财经新闻标题
  - 关键词→概念板块映射（特朗普→关税概念、马斯克→新能源车...）
  - 按提及频率对热点概念排序
  - 获取概念板块成分股

数据来源：akshare（东方财富、同花顺）

使用:
  from news_scanner import scan_news, get_concept_stocks
  hot = scan_news()
  stocks = get_concept_stocks("AI概念")
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 关键词 → 概念板块映射表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 格式: { "关键词": ["概念1", "概念2", ...] }
# 一个关键词可映射到多个概念，一个概念也可被多个关键词触发

KEYWORD_CONCEPT_MAP: Dict[str, List[str]] = {
    # ---- 特朗普/关税 ----
    "特朗普":     ["关税概念", "中美贸易"],
    "Trump":      ["关税概念", "中美贸易"],
    "trump":      ["关税概念", "中美贸易"],
    "关税":       ["关税概念", "中美贸易"],
    "贸易战":     ["关税概念", "中美贸易"],
    "贸易摩擦":   ["关税概念", "中美贸易"],
    "中美":       ["中美贸易"],

    # ---- 马斯克/特斯拉 ----
    "马斯克":     ["新能源车", "自动驾驶", "机器人概念"],
    "Musk":       ["新能源车", "自动驾驶", "机器人概念"],
    "musk":       ["新能源车", "自动驾驶", "机器人概念"],
    "特斯拉":     ["新能源车", "自动驾驶"],
    "Tesla":      ["新能源车", "自动驾驶"],

    # ---- 黄仁勋/英伟达 ----
    "黄仁勋":     ["AI芯片", "算力", "GPU"],
    "英伟达":     ["AI芯片", "算力", "GPU"],
    "NVIDIA":     ["AI芯片", "算力", "GPU"],
    "nvidia":     ["AI芯片", "算力", "GPU"],
    "N卡":        ["AI芯片", "GPU"],

    # ---- 华为生态 ----
    "华为":       ["国产替代", "芯片", "鸿蒙"],
    "鸿蒙":       ["鸿蒙", "国产替代"],
    "昇腾":       ["AI芯片", "算力", "国产替代"],
    "麒麟":       ["芯片", "国产替代"],

    # ---- 机器人 ----
    "机器人":     ["机器人概念"],
    "人形机器人": ["机器人概念"],
    "具身智能":   ["机器人概念", "AI概念"],
    "Figure":     ["机器人概念"],
    "Optimus":    ["机器人概念"],

    # ---- 低空经济 ----
    "低空经济":   ["低空经济"],
    "低空":       ["低空经济"],
    "eVTOL":      ["低空经济"],
    "飞行汽车":   ["低空经济"],
    "无人机":     ["低空经济"],

    # ---- AI / 大模型 ----
    "AI":         ["AI概念"],
    "人工智能":   ["AI概念"],
    "大模型":     ["AI概念"],
    "DeepSeek":   ["AI概念"],
    "deepseek":   ["AI概念"],
    "GPT":        ["AI概念"],
    "Sora":       ["AI概念"],
    "智能体":     ["AI概念"],
    "AIGC":       ["AI概念"],
    "算力":       ["算力"],
    "液冷":       ["算力"],
    "光模块":     ["算力"],

    # ---- 量子计算 ----
    "量子计算":   ["量子概念"],
    "量子":       ["量子概念"],
    "量子通信":   ["量子概念"],

    # ---- 军工/国防 ----
    "军工":       ["军工概念"],
    "国防":       ["军工概念"],
    "军事":       ["军工概念"],
    "导弹":       ["军工概念"],
    "战斗机":     ["军工概念"],
    "航母":       ["军工概念"],

    # ---- 其他热点 ----
    "半导体":     ["芯片"],
    "芯片":       ["芯片"],
    "光刻机":     ["芯片", "国产替代"],
    "新能源":     ["新能源车"],
    "锂电":       ["新能源车"],
    "固态电池":   ["新能源车"],
    "光伏":       ["光伏概念"],
    "储能":       ["储能概念"],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ConceptHeat:
    """概念板块热度"""
    name: str                                # 概念名称
    heat_score: int = 0                      # 热度分（=提及次数加权）
    keywords_matched: List[str] = field(default_factory=list)   # 命中的关键词
    sample_headlines: List[str] = field(default_factory=list)   # 样本标题（最多5条）

    def add_match(self, keyword: str, headline: str):
        """记录一次命中"""
        self.heat_score += 1
        if keyword not in self.keywords_matched:
            self.keywords_matched.append(keyword)
        if len(self.sample_headlines) < 5 and headline not in self.sample_headlines:
            self.sample_headlines.append(headline)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 新闻数据获取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fetch_news_eastmoney() -> List[str]:
    """从东方财富获取财经新闻标题

    尝试多个 akshare 接口，按优先级依次回退:
      1. stock_news_em()           — 个股新闻汇总
      2. news_cctv()               — 央视新闻（偏宏观）
      3. stock_zh_a_alerts_cls()   — 财联社快讯
    """
    import akshare as ak

    headlines: List[str] = []

    # 方案1: 东方财富股票新闻
    try:
        logger.info("尝试获取东方财富新闻 (stock_news_em)...")
        df = ak.stock_news_em(symbol="300059")  # 需要传股票代码，用东方财富自身
        if df is not None and len(df) > 0:
            # 标题列名可能是 "新闻标题" 或 "title"
            title_col = None
            for col in ["新闻标题", "title", "标题"]:
                if col in df.columns:
                    title_col = col
                    break
            if title_col is None and len(df.columns) > 0:
                # 取第一个看起来像标题的列（最长文本列）
                title_col = df.columns[0]
            if title_col:
                headlines.extend(df[title_col].astype(str).tolist())
                logger.info(f"stock_news_em 获取 {len(headlines)} 条标题")
    except Exception as e:
        logger.debug(f"stock_news_em 失败: {e}")

    # 方案2: 央视新闻（宏观政策面）
    try:
        logger.info("尝试获取央视新闻 (news_cctv)...")
        today = datetime.now().strftime("%Y%m%d")
        df = ak.news_cctv(date=today)
        if df is not None and len(df) > 0:
            title_col = None
            for col in ["title", "标题"]:
                if col in df.columns:
                    title_col = col
                    break
            if title_col is None and len(df.columns) > 0:
                title_col = df.columns[0]
            if title_col:
                cctv_titles = df[title_col].astype(str).tolist()
                headlines.extend(cctv_titles)
                logger.info(f"news_cctv 获取 {len(cctv_titles)} 条标题")
    except Exception as e:
        logger.debug(f"news_cctv 失败: {e}")

    # 方案3: 财联社电报/快讯
    try:
        logger.info("尝试获取财联社快讯 (stock_zh_a_alerts_cls)...")
        df = ak.stock_zh_a_alerts_cls()
        if df is not None and len(df) > 0:
            title_col = None
            for col in ["title", "标题", "内容"]:
                if col in df.columns:
                    title_col = col
                    break
            if title_col is None and len(df.columns) > 0:
                title_col = df.columns[0]
            if title_col:
                cls_titles = df[title_col].astype(str).tolist()
                headlines.extend(cls_titles)
                logger.info(f"stock_zh_a_alerts_cls 获取 {len(cls_titles)} 条标题")
    except Exception as e:
        logger.debug(f"stock_zh_a_alerts_cls 失败: {e}")

    return headlines


def _fetch_news_ths_hot() -> List[str]:
    """从同花顺获取概念板块热度/热门资讯

    尝试:
      1. stock_board_concept_name_em()  — 同花顺概念板块列表（板块名本身就是热点信号）
      2. stock_hot_rank_em()            — 东方财富个股人气榜
    """
    import akshare as ak

    headlines: List[str] = []

    # 方案1: 概念板块名称列表 — 板块名本身就是关键词
    try:
        logger.info("尝试获取概念板块列表 (stock_board_concept_name_em)...")
        df = ak.stock_board_concept_name_em()
        if df is not None and len(df) > 0:
            name_col = None
            for col in ["板块名称", "name", "概念名称"]:
                if col in df.columns:
                    name_col = col
                    break
            if name_col:
                # 取涨幅最高的前30个板块名作为"热点标题"
                if "涨跌幅" in df.columns:
                    df = df.sort_values("涨跌幅", ascending=False)
                concept_names = df[name_col].astype(str).head(30).tolist()
                headlines.extend(concept_names)
                logger.info(f"概念板块 获取 {len(concept_names)} 个板块名")
    except Exception as e:
        logger.debug(f"stock_board_concept_name_em 失败: {e}")

    # 方案2: 东方财富人气榜
    try:
        logger.info("尝试获取人气排名 (stock_hot_rank_em)...")
        df = ak.stock_hot_rank_em()
        if df is not None and len(df) > 0:
            name_col = None
            for col in ["股票名称", "name"]:
                if col in df.columns:
                    name_col = col
                    break
            if name_col:
                hot_names = df[name_col].astype(str).head(20).tolist()
                headlines.extend(hot_names)
                logger.info(f"人气榜 获取 {len(hot_names)} 个热股名")
    except Exception as e:
        logger.debug(f"stock_hot_rank_em 失败: {e}")

    return headlines


def fetch_all_headlines() -> List[str]:
    """汇总所有来源的新闻/热点标题

    合并去重后返回，保持顺序（最新的在前）。
    """
    all_headlines: List[str] = []

    # 来源1: 东方财富新闻
    try:
        em_news = _fetch_news_eastmoney()
        all_headlines.extend(em_news)
    except Exception as e:
        logger.error(f"东方财富新闻获取异常: {e}")

    # 来源2: 同花顺热度
    try:
        ths_hot = _fetch_news_ths_hot()
        all_headlines.extend(ths_hot)
    except Exception as e:
        logger.error(f"同花顺热度获取异常: {e}")

    # 去重保序
    seen = set()
    unique = []
    for h in all_headlines:
        h = h.strip()
        if h and h not in seen:
            seen.add(h)
            unique.append(h)

    logger.info(f"共获取 {len(unique)} 条不重复标题/热点")
    return unique


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 核心：新闻扫描 → 概念热度
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scan_news() -> List[ConceptHeat]:
    """扫描新闻标题，返回按热度排序的概念板块列表

    流程:
      1. 获取所有新闻标题
      2. 对每条标题做关键词匹配
      3. 累计每个概念的热度分
      4. 按热度降序排列返回

    Returns:
        List[ConceptHeat]: 热度从高到低排列的概念列表
    """
    logger.info("=" * 50)
    logger.info("开始新闻舆情扫描...")
    logger.info("=" * 50)

    # 1. 获取标题
    headlines = fetch_all_headlines()
    if not headlines:
        logger.warning("未获取到任何新闻标题，扫描终止")
        return []

    # 2. 关键词匹配 → 概念热度
    concept_map: Dict[str, ConceptHeat] = {}

    for headline in headlines:
        for keyword, concepts in KEYWORD_CONCEPT_MAP.items():
            if keyword in headline:
                for concept_name in concepts:
                    if concept_name not in concept_map:
                        concept_map[concept_name] = ConceptHeat(name=concept_name)
                    concept_map[concept_name].add_match(keyword, headline)

    # 3. 排序
    result = sorted(concept_map.values(), key=lambda c: c.heat_score, reverse=True)

    logger.info(f"扫描完毕，发现 {len(result)} 个热点概念")
    for i, c in enumerate(result[:10]):
        logger.info(f"  #{i+1} {c.name}: 热度={c.heat_score}, "
                     f"关键词={c.keywords_matched}")

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 概念板块 → 成分股
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_concept_stocks(concept_name: str) -> pd.DataFrame:
    """获取指定概念板块的成分股列表

    尝试多个 akshare 接口:
      1. stock_board_concept_cons_em()  — 东方财富概念板块成分
      2. stock_board_concept_name_em()  — 先查板块代码再查成分

    Args:
        concept_name: 概念名称，如 "AI概念"、"机器人概念"

    Returns:
        DataFrame，包含成分股代码、名称等信息；失败返回空DataFrame
    """
    import akshare as ak

    # 方案1: 直接用概念名查成分股
    try:
        logger.info(f"查询概念板块成分股: {concept_name}")
        df = ak.stock_board_concept_cons_em(symbol=concept_name)
        if df is not None and len(df) > 0:
            logger.info(f"概念 [{concept_name}] 包含 {len(df)} 只成分股")
            return df
    except Exception as e:
        logger.debug(f"stock_board_concept_cons_em('{concept_name}') 失败: {e}")

    # 方案2: 先获取板块列表，模糊匹配板块名，再查成分
    try:
        logger.info(f"模糊匹配概念板块: {concept_name}")
        boards = ak.stock_board_concept_name_em()
        if boards is not None and len(boards) > 0:
            name_col = None
            for col in ["板块名称", "name", "概念名称"]:
                if col in boards.columns:
                    name_col = col
                    break
            if name_col:
                # 模糊匹配: 概念名包含在板块名中，或板块名包含在概念名中
                matched = boards[
                    boards[name_col].str.contains(concept_name, na=False) |
                    boards[name_col].apply(lambda x: concept_name in str(x) or str(x) in concept_name)
                ]
                if len(matched) > 0:
                    board_name = matched.iloc[0][name_col]
                    logger.info(f"匹配到板块: {board_name}")
                    df = ak.stock_board_concept_cons_em(symbol=board_name)
                    if df is not None and len(df) > 0:
                        logger.info(f"概念 [{board_name}] 包含 {len(df)} 只成分股")
                        return df
    except Exception as e:
        logger.debug(f"模糊匹配概念板块失败: {e}")

    logger.warning(f"未找到概念 [{concept_name}] 的成分股")
    return pd.DataFrame()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_top_concepts(n: int = 5) -> List[Dict]:
    """快捷接口: 获取Top N热点概念及其代表股票

    Returns:
        [{"concept": "AI概念", "heat": 15, "keywords": [...],
          "top_stocks": [{"code": "xxx", "name": "xxx"}, ...]}, ...]
    """
    concepts = scan_news()
    results = []

    for c in concepts[:n]:
        item = {
            "concept": c.name,
            "heat": c.heat_score,
            "keywords": c.keywords_matched,
            "sample_headlines": c.sample_headlines[:3],
            "top_stocks": [],
        }

        # 查成分股，取涨幅前5
        try:
            stocks_df = get_concept_stocks(c.name)
            if not stocks_df.empty:
                # 尝试按涨跌幅排序
                sort_col = None
                for col in ["涨跌幅", "change_pct", "pct_change"]:
                    if col in stocks_df.columns:
                        sort_col = col
                        break
                if sort_col:
                    stocks_df = stocks_df.sort_values(sort_col, ascending=False)

                # 提取代码和名称
                code_col = None
                name_col = None
                for col in ["代码", "code", "股票代码"]:
                    if col in stocks_df.columns:
                        code_col = col
                        break
                for col in ["名称", "name", "股票名称"]:
                    if col in stocks_df.columns:
                        name_col = col
                        break

                if code_col and name_col:
                    for _, row in stocks_df.head(5).iterrows():
                        item["top_stocks"].append({
                            "code": str(row[code_col]),
                            "name": str(row[name_col]),
                        })
        except Exception as e:
            logger.debug(f"获取概念 {c.name} 成分股失败: {e}")

        results.append(item)
        time.sleep(0.3)  # 避免请求过快

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # 配置日志到控制台
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  A-Share News Scanner - Hot Concept Detector")
    print("=" * 60)
    print()

    # 1. 扫描新闻
    hot_concepts = scan_news()

    if not hot_concepts:
        print("[!] No hot concepts found. Check network or akshare version.")
    else:
        print(f"\n>>> Found {len(hot_concepts)} hot concepts:\n")
        for i, c in enumerate(hot_concepts[:15]):
            print(f"  #{i+1:2d}  {c.name}")
            print(f"       Heat: {c.heat_score}  |  Keywords: {', '.join(c.keywords_matched)}")
            if c.sample_headlines:
                # ASCII-safe: 只打印前80字符避免GBK问题
                for h in c.sample_headlines[:2]:
                    try:
                        print(f"       -> {h[:80]}")
                    except UnicodeEncodeError:
                        print(f"       -> (headline contains special chars)")
            print()

    # 2. 查Top 3概念的成分股
    print("-" * 60)
    print("  Top 3 Concept Stocks:")
    print("-" * 60)

    for c in hot_concepts[:3]:
        print(f"\n  [{c.name}]")
        try:
            stocks_df = get_concept_stocks(c.name)
            if not stocks_df.empty:
                # 打印前5只
                code_col, name_col = None, None
                for col in ["代码", "code", "股票代码"]:
                    if col in stocks_df.columns:
                        code_col = col
                        break
                for col in ["名称", "name", "股票名称"]:
                    if col in stocks_df.columns:
                        name_col = col
                        break
                if code_col and name_col:
                    for _, row in stocks_df.head(5).iterrows():
                        try:
                            print(f"    {row[code_col]}  {row[name_col]}")
                        except UnicodeEncodeError:
                            print(f"    {row[code_col]}  (name encoding issue)")
                else:
                    print(f"    (got {len(stocks_df)} stocks, columns: {list(stocks_df.columns)[:5]})")
            else:
                print("    (no stocks found)")
        except Exception as e:
            print(f"    (error: {e})")
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("  Scan complete.")
    print("=" * 60)
