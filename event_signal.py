"""
事件驱动信号模块
================

统一封装:
1. 新闻热点 -> 受影响板块 -> 事件信号
2. 事件信号 -> 个股关注池 / 风险动作
3. 自适应仓位状态的持久化与更新
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


NEWS_SECTOR_MAP = {
    "特朗普": ["贸易", "石油", "黄金", "军工", "稀土"],
    "trump": ["贸易", "石油", "黄金", "军工", "稀土"],
    "关税": ["贸易", "出口", "电子", "纺织"],
    "tariff": ["贸易", "出口", "电子"],
    "贸易战": ["贸易", "出口", "半导体"],
    "制裁": ["半导体", "军工", "通信"],
    "石油": ["石油", "化工", "航空"],
    "原油": ["石油", "化工"],
    "黄金": ["黄金", "有色金属"],
    "白银": ["黄金", "有色金属"],
    "煤炭": ["煤炭", "电力"],
    "天然气": ["石油", "化工"],
    "芯片": ["半导体", "电子"],
    "半导体": ["半导体", "电子"],
    "AI": ["AI", "半导体", "通信"],
    "人工智能": ["AI", "半导体", "软件"],
    "大模型": ["AI", "软件", "算力"],
    "华为": ["华为", "半导体", "通信"],
    "台湾": ["军工", "半导体"],
    "军事": ["军工", "航天"],
    "导弹": ["军工"],
    "南海": ["军工"],
    "降息": ["银行", "地产", "券商"],
    "加息": ["银行", "黄金"],
    "央行": ["银行", "券商"],
    "LPR": ["银行", "地产"],
    "新能源": ["新能源", "锂电池", "光伏"],
    "光伏": ["光伏", "新能源"],
    "锂电": ["锂电池", "新能源"],
    "储能": ["储能", "电力设备"],
    "集采": ["医药", "医疗器械"],
    "疫苗": ["医药"],
    "医保": ["医药", "医疗器械"],
    "消费": ["消费", "白酒", "食品"],
    "内需": ["消费", "零售"],
}


SECTOR_INDUSTRY_MAP = {
    "石油": ["石油", "油服", "能源", "燃气"],
    "化工": ["化学", "化工", "化纤"],
    "黄金": ["贵金属", "有色", "金"],
    "有色金属": ["有色", "金属", "稀土", "钴", "锂"],
    "军工": ["军工", "航天", "航空", "兵器", "船舶"],
    "半导体": ["半导体", "芯片", "集成电路"],
    "电子": ["电子", "元件", "光电"],
    "通信": ["通信", "通讯"],
    "AI": ["软件", "IT", "计算机", "互联网"],
    "软件": ["软件", "IT", "计算机"],
    "银行": ["银行"],
    "券商": ["证券"],
    "地产": ["房地产", "地产"],
    "医药": ["医药", "药", "生物"],
    "医疗器械": ["医疗器械"],
    "新能源": ["新能源", "光伏", "风电"],
    "锂电池": ["锂电", "电池"],
    "消费": ["消费", "食品", "饮料", "零售"],
    "白酒": ["白酒", "酒"],
    "贸易": ["贸易", "进出口", "外贸"],
    "出口": ["贸易", "纺织", "家电"],
    "煤炭": ["煤炭", "煤"],
    "电力": ["电力", "电网", "发电"],
    "光伏": ["光伏", "太阳能"],
    "稀土": ["稀土"],
    "纺织": ["纺织", "服装"],
}


NEGATIVE_WORDS = [
    "暴跌",
    "崩",
    "制裁",
    "关税",
    "封锁",
    "禁止",
    "下跌",
    "危机",
    "熔断",
    "跌停",
    "利空",
    "大跌",
]


POSITIVE_WORDS = [
    "暴涨",
    "利好",
    "突破",
    "新高",
    "政策支持",
    "扶持",
    "涨停",
    "大涨",
    "反弹",
    "协议",
    "谈判",
]


_STATE_FILE = Path("data/adaptive_trade_state.json")
_EVENT_LOG_FILE = Path("data/event_signals.json")
_INDUSTRY_MAP_FILE = Path("data/stock_industry_map.json")

_industry_map: Optional[Dict[str, str]] = None
_sector_code_map: Optional[Dict[str, List[str]]] = None


@dataclass
class EventSignal:
    event_type: str
    trigger: str
    affected_sectors: List[str]
    affected_codes: List[str]
    severity: int
    action: str
    timestamp: str = ""
    bonus_score: int = 0


@dataclass
class AdaptiveTradeState:
    """
    根据近期真实交易结果动态调整仓位。
    """

    trade_results: List[float] = field(default_factory=list)
    max_equity: float = 0
    current_equity: float = 0
    _cooldown_until: Optional[str] = None

    def record_trade(self, pnl_pct: float) -> None:
        self.trade_results.append(float(pnl_pct))
        if len(self.trade_results) > 20:
            self.trade_results.pop(0)

    def update_equity(self, equity: float) -> None:
        self.current_equity = float(equity or 0)
        if self.current_equity > self.max_equity:
            self.max_equity = self.current_equity

    @property
    def recent_win_rate(self) -> float:
        recent = self.trade_results[-7:]
        if len(recent) < 2:
            return 0.5
        return sum(1 for result in recent if result > 0) / len(recent)

    @property
    def consecutive_losses(self) -> int:
        count = 0
        for result in reversed(self.trade_results):
            if result < 0:
                count += 1
            else:
                break
        return count

    @property
    def drawdown_pct(self) -> float:
        if self.max_equity <= 0:
            return 0.0
        return (self.max_equity - self.current_equity) / self.max_equity * 100

    @property
    def position_coeff(self) -> float:
        win_rate = self.recent_win_rate
        if win_rate >= 0.7:
            base = 0.80
        elif win_rate >= 0.5:
            base = 0.60
        elif win_rate >= 0.3:
            base = 0.40
        else:
            base = 0.20

        losses = self.consecutive_losses
        if losses >= 4:
            base *= 0.3
        elif losses >= 3:
            base *= 0.5
        elif losses >= 2:
            base *= 0.7

        drawdown = self.drawdown_pct
        if drawdown > 10:
            base *= 0.3
        elif drawdown > 7:
            base *= 0.5
        elif drawdown > 5:
            base *= 0.7

        return max(0.10, min(1.0, base))

    @property
    def suggested_position_pct(self) -> float:
        return round(self.position_coeff * 100, 0)

    def to_dict(self) -> dict:
        return {
            "position_coeff": round(self.position_coeff, 3),
            "position_pct": self.suggested_position_pct,
            "win_rate": round(self.recent_win_rate, 2),
            "consecutive_losses": self.consecutive_losses,
            "drawdown_pct": round(self.drawdown_pct, 2),
            "recent_trades": len(self.trade_results),
        }


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _normalize_code(code: str) -> str:
    value = str(code or "").strip()
    return (
        value.replace("SH", "")
        .replace("SZ", "")
        .replace("sh", "")
        .replace("sz", "")
    )


def _load_industry_map() -> Dict[str, str]:
    global _industry_map, _sector_code_map
    if _industry_map is not None:
        return _industry_map

    if _INDUSTRY_MAP_FILE.exists():
        with open(_INDUSTRY_MAP_FILE, "r", encoding="utf-8") as f:
            _industry_map = json.load(f)
    else:
        _industry_map = {}
        logger.warning("[事件信号] 行业映射文件不存在: %s", _INDUSTRY_MAP_FILE)

    _sector_code_map = defaultdict(list)
    for raw_code, industry in _industry_map.items():
        code = _normalize_code(raw_code)
        for sector, keywords in SECTOR_INDUSTRY_MAP.items():
            if any(keyword in industry for keyword in keywords):
                _sector_code_map[sector].append(code)

    logger.info(
        "[事件信号] 加载行业映射: %s只股票, %s个板块",
        len(_industry_map),
        len(_sector_code_map),
    )
    return _industry_map


def get_sector_codes(sector: str) -> List[str]:
    _load_industry_map()
    return list(_sector_code_map.get(sector, []))


def get_stock_sector(code: str) -> Optional[str]:
    _load_industry_map()
    industry = _industry_map.get(_normalize_code(code), "") or _industry_map.get(str(code), "")
    for sector, keywords in SECTOR_INDUSTRY_MAP.items():
        if any(keyword in industry for keyword in keywords):
            return sector
    return None


def _coerce_hot_concept(raw) -> dict:
    if raw is None:
        return {}

    if isinstance(raw, dict):
        concept = raw.get("concept") or raw.get("name") or ""
        heat = raw.get("heat_score", raw.get("heat", 0))
        keywords = raw.get("keywords_matched", raw.get("keywords", [])) or []
        headlines = raw.get("sample_headlines", raw.get("headlines", [])) or []
        stocks = raw.get("stocks", []) or []
    else:
        concept = getattr(raw, "concept", "") or getattr(raw, "name", "") or ""
        heat = getattr(raw, "heat_score", 0) or getattr(raw, "heat", 0)
        keywords = getattr(raw, "keywords_matched", []) or getattr(raw, "keywords", []) or []
        headlines = getattr(raw, "sample_headlines", []) or getattr(raw, "headlines", []) or []
        stocks = getattr(raw, "stocks", []) or []

    return {
        "concept": str(concept).strip(),
        "heat": int(heat or 0),
        "keywords": [str(item).strip() for item in keywords if str(item).strip()],
        "headlines": [str(item).strip() for item in headlines if str(item).strip()],
        "stocks": stocks,
    }


def _collect_affected_sectors(concept: dict) -> List[str]:
    haystack = " ".join(
        [concept.get("concept", ""), *concept.get("keywords", []), *concept.get("headlines", [])]
    ).lower()
    matched: List[str] = []
    for keyword, sectors in NEWS_SECTOR_MAP.items():
        if keyword.lower() in haystack:
            matched.extend(sectors)
    return _dedupe_preserve(matched)


def _collect_sector_codes(sectors: Iterable[str], limit: int = 50) -> List[str]:
    codes: List[str] = []
    for sector in sectors:
        codes.extend(get_sector_codes(sector))
    return _dedupe_preserve(_normalize_code(code) for code in codes)[:limit]


def analyze_news_events(
    hot_concepts: List[dict] = None,
    trump_news: List[dict] = None,
    market_changes: Dict[str, float] = None,
) -> List[EventSignal]:
    """
    分析新闻和行情，生成可被选股/建仓/调仓复用的事件信号。
    """

    signals: List[EventSignal] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    for raw_concept in hot_concepts or []:
        concept = _coerce_hot_concept(raw_concept)
        if concept.get("heat", 0) < 30:
            continue

        affected_sectors = _collect_affected_sectors(concept)
        if not affected_sectors:
            continue

        text_blob = " ".join(
            [concept.get("concept", ""), *concept.get("keywords", []), *concept.get("headlines", [])]
        )
        neg_count = sum(1 for word in NEGATIVE_WORDS if word in text_blob)
        pos_count = sum(1 for word in POSITIVE_WORDS if word in text_blob)

        if neg_count > pos_count:
            severity = min(5, 2 + neg_count)
            action = "defense" if severity >= 4 else "watch"
            bonus = 0
        elif pos_count > neg_count:
            severity = min(3, 1 + pos_count)
            action = "buy_dip"
            bonus = min(20, max(8, concept["heat"] // 5))
        else:
            severity = 2
            action = "watch"
            bonus = 0

        signals.append(
            EventSignal(
                event_type="news_event",
                trigger=f"{concept['concept']}(heat={concept['heat']})",
                affected_sectors=affected_sectors,
                affected_codes=_collect_sector_codes(affected_sectors),
                severity=severity,
                action=action,
                timestamp=now,
                bonus_score=bonus,
            )
        )

    if trump_news:
        sensitive_items = []
        for item in trump_news:
            if not isinstance(item, dict):
                continue
            if item.get("is_sensitive"):
                sensitive_items.append(item)
                continue
            merged = " ".join(str(item.get(key, "")) for key in ("title", "summary", "content"))
            if any(keyword in merged.lower() for keyword in ("trump", "tariff", "china", "trade")):
                sensitive_items.append(item)

        if len(sensitive_items) >= 2:
            sectors = ["贸易", "石油", "黄金", "半导体", "军工"]
            signals.append(
                EventSignal(
                    event_type="trump_alert",
                    trigger=f"Trump敏感新闻x{len(sensitive_items)}",
                    affected_sectors=sectors,
                    affected_codes=_collect_sector_codes(sectors),
                    severity=4,
                    action="defense",
                    timestamp=now,
                )
            )

    if market_changes:
        _load_industry_map()
        sector_changes: Dict[str, List[float]] = defaultdict(list)
        normalized_changes = {_normalize_code(code): float(change) for code, change in market_changes.items()}

        for code, change in normalized_changes.items():
            sector = get_stock_sector(code)
            if sector:
                sector_changes[sector].append(change)

        for sector, changes in sector_changes.items():
            if len(changes) < 3:
                continue
            avg_change = sum(changes) / len(changes)
            down_count = sum(1 for value in changes if value < -3)
            down_ratio = down_count / len(changes)

            if avg_change < -3 and down_ratio > 0.5:
                crashed_codes = [
                    code
                    for code, change in normalized_changes.items()
                    if get_stock_sector(code) == sector and change < -3
                ]
                signals.append(
                    EventSignal(
                        event_type="sector_crash",
                        trigger=f"{sector}板块暴跌{avg_change:.1f}%",
                        affected_sectors=[sector],
                        affected_codes=crashed_codes,
                        severity=min(5, int(abs(avg_change))),
                        action="watch",
                        timestamp=now,
                    )
                )
            elif avg_change > 3:
                hot_codes = [
                    code
                    for code, change in normalized_changes.items()
                    if get_stock_sector(code) == sector and change > 3
                ]
                signals.append(
                    EventSignal(
                        event_type="sector_surge",
                        trigger=f"{sector}板块大涨{avg_change:.1f}%",
                        affected_sectors=[sector],
                        affected_codes=hot_codes,
                        severity=2,
                        action="watch",
                        timestamp=now,
                        bonus_score=10,
                    )
                )

    _save_event_log(signals)
    return signals


def get_event_watchlist(signals: List[EventSignal]) -> Dict[str, int]:
    watchlist: Dict[str, int] = {}
    for signal in signals or []:
        if signal.action not in {"buy_dip", "watch"}:
            continue
        bonus = int(signal.bonus_score or 0)
        if bonus <= 0 and signal.action == "watch":
            continue
        for code in signal.affected_codes:
            normalized = _normalize_code(code)
            watchlist[normalized] = max(watchlist.get(normalized, 0), bonus)
    return watchlist


def get_event_action(signals: List[EventSignal]) -> str:
    if not signals:
        return "normal"

    max_severity = max(signal.severity for signal in signals)
    defense_count = sum(1 for signal in signals if signal.action == "defense")
    buy_count = sum(1 for signal in signals if signal.action == "buy_dip")

    if max_severity >= 4 or defense_count >= 2:
        return "defense"
    if buy_count >= 2 and max_severity <= 2:
        return "aggressive"
    return "normal"


def summarize_event_signals(signals: List[EventSignal], max_items: int = 3) -> str:
    if not signals:
        return "无事件信号"

    parts = []
    for signal in signals[:max_items]:
        parts.append(
            f"{signal.event_type}:{signal.trigger} -> {signal.action}"
            f"(severity={signal.severity}, sectors={','.join(signal.affected_sectors[:3]) or '无'})"
        )
    return "；".join(parts)


def load_recent_event_entries(
    max_age_hours: Optional[float] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    if not _EVENT_LOG_FILE.exists():
        return []

    try:
        with open(_EVENT_LOG_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as exc:
        logger.warning("[事件信号] 读取事件日志失败: %s", exc)
        return []

    filtered = []
    cutoff = datetime.now() - timedelta(hours=max_age_hours) if max_age_hours else None
    for entry in entries:
        if cutoff is not None:
            try:
                ts = datetime.strptime(entry.get("timestamp", ""), "%Y-%m-%d %H:%M")
            except Exception:
                continue
            if ts < cutoff:
                continue
        filtered.append(entry)

    if limit is not None:
        filtered = filtered[-limit:]
    return filtered


def get_recent_event_action(max_age_hours: float = 1.0) -> str:
    entries = load_recent_event_entries(max_age_hours=max_age_hours, limit=20)
    if not entries:
        return "normal"
    if any(entry.get("action") == "defense" and int(entry.get("severity", 0)) >= 4 for entry in entries):
        return "defense"
    if sum(1 for entry in entries if entry.get("action") == "buy_dip") >= 2:
        return "aggressive"
    return "normal"


def get_recent_event_watchlist(max_age_hours: float = 6.0) -> Dict[str, int]:
    watchlist: Dict[str, int] = {}
    for entry in load_recent_event_entries(max_age_hours=max_age_hours, limit=50):
        if entry.get("action") != "buy_dip":
            continue
        bonus = int(entry.get("bonus_score", 0) or 0)
        codes = entry.get("codes", []) or []
        if not codes:
            for sector in entry.get("sectors", []) or []:
                codes.extend(get_sector_codes(sector))
        for code in codes:
            normalized = _normalize_code(code)
            watchlist[normalized] = max(watchlist.get(normalized, 0), bonus)
    return watchlist


def load_adaptive_state() -> AdaptiveTradeState:
    state = AdaptiveTradeState()
    if not _STATE_FILE.exists():
        return state

    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        state.trade_results = [float(item) for item in data.get("trade_results", [])[-20:]]
        state.max_equity = float(data.get("max_equity", 0) or 0)
        state.current_equity = float(data.get("current_equity", 0) or 0)
        state._cooldown_until = data.get("cooldown_until")
    except Exception as exc:
        logger.warning("[自适应] 加载状态失败: %s", exc)
    return state


def save_adaptive_state(state: AdaptiveTradeState) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "trade_results": state.trade_results[-20:],
        "max_equity": state.max_equity,
        "current_equity": state.current_equity,
        "cooldown_until": state._cooldown_until,
        "updated": datetime.now().isoformat(),
    }
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_adaptive_trade_result(pnl_pct: float, equity: Optional[float] = None) -> AdaptiveTradeState:
    state = load_adaptive_state()
    state.record_trade(pnl_pct)
    if equity is not None:
        state.update_equity(equity)
    save_adaptive_state(state)
    logger.info(
        "[自适应] 记录交易结果: pnl=%.2f%%, win_rate=%.0f%%, 连亏=%s, 回撤=%.1f%%, 建议仓位=%.0f%%",
        pnl_pct,
        state.recent_win_rate * 100,
        state.consecutive_losses,
        state.drawdown_pct,
        state.suggested_position_pct,
    )
    return state


def _save_event_log(signals: List[EventSignal]) -> None:
    if not signals:
        return

    _EVENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = load_recent_event_entries(limit=200)
    for signal in signals:
        payload = asdict(signal)
        payload["codes"] = payload.pop("affected_codes")
        payload["sectors"] = payload.pop("affected_sectors")
        existing.append(payload)

    existing = existing[-100:]
    with open(_EVENT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
