"""Theme rotation helpers for intraday and rebalance decisions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        if isinstance(value, str):
            value = value.replace("%", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_sector_rows(sectors: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, sector in enumerate(sectors or [], start=1):
        name = str(sector.get("name", "") or "").strip()
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "rank": int(sector.get("rank") or index),
                "change_pct": round(_to_float(sector.get("change_pct")), 2),
                "main_net": round(_to_float(sector.get("main_net")), 2),
                "sector_type": str(sector.get("sector_type", "") or "").strip() or "hy",
            }
        )
    return rows


def _build_buy_range(price: float, support: float, bias5: float, entry_state: str) -> str:
    if support > 0:
        lower = support * 0.995
        if entry_state == "pullback_ready":
            upper = min(price, support * 1.015)
        elif entry_state == "secondary_relay":
            upper = min(price, support * 1.03)
        else:
            upper = support * 1.015
    else:
        base = price * (0.985 if bias5 >= 0 else 0.975)
        lower = base * 0.995
        upper = max(base * 1.01, lower)
    if upper < lower:
        upper = lower
    return f"{lower:.2f}-{upper:.2f}"


def annotate_rotation_candidates(
    candidates: Iterable[Dict[str, Any]],
    dominant_themes: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Add entry-timing / relay labels to candidate stocks and sort them."""

    dominant_names = {str(item or "").strip() for item in (dominant_themes or []) if str(item or "").strip()}
    annotated: List[Dict[str, Any]] = []
    for raw in candidates or []:
        row = dict(raw)
        change_pct = _to_float(row.get("change_pct"))
        main_net = _to_float(row.get("main_net"))
        price = _to_float(row.get("price"))
        support = _to_float(row.get("support"))
        resistance = _to_float(row.get("resistance"))
        bias5 = _to_float(row.get("bias5"))
        tech_score = _to_float(row.get("tech_score"))
        source = str(row.get("candidate_source", row.get("source", "hot_flow")) or "hot_flow").strip()
        ma_trend = str(row.get("ma_trend", "") or "").strip()
        vol_pattern = str(row.get("vol_pattern", "") or "").strip()
        sector = str(row.get("sector", "") or "").strip()

        score = 0.0
        relay_role = "普通候选"
        entry_state = "watch"
        timing_note = "先观察分时承接，等待更明确的进场信号。"

        if source == "sub_dragon":
            relay_role = "副龙/补涨候选"
            score += 14
        elif change_pct >= 8.0 and main_net >= 2000:
            relay_role = "板块领涨龙头"
            score += 6
        elif 1.5 <= change_pct <= 6.5 and main_net >= 1200:
            relay_role = "主线扩散候选"
            score += 8

        if vol_pattern == "缩量回踩":
            score += 12
        elif vol_pattern == "温和放量":
            score += 4

        if ma_trend in {"多头排列", "短期多头"}:
            score += 8
        if -1.8 <= bias5 <= 1.2:
            score += 7
        if main_net >= 5000:
            score += 8
        elif main_net >= 2000:
            score += 5
        elif main_net > 0:
            score += 2

        if dominant_names and sector and sector in dominant_names:
            score += 10

        if change_pct >= 8.5:
            entry_state = "leader_locked"
            timing_note = "已明显加速甚至接近封板，更适合作为龙头观察，不适合此时追高。"
            score -= 20
        elif change_pct >= 5.0:
            entry_state = "overextended"
            timing_note = "涨幅已明显扩张，原低吸位大概率已被市场消化，除非再回踩否则不追。"
            score -= 12
        elif -2.5 <= change_pct <= 1.8 and main_net > 0 and ma_trend in {"多头排列", "短期多头"}:
            entry_state = "pullback_ready"
            timing_note = "更接近分歧低吸窗口，优先等回踩支撑附近承接，而不是追高。"
            score += 18
        elif 1.8 < change_pct < 5.0 and main_net > 0:
            entry_state = "secondary_relay"
            timing_note = "资金有扩散到副龙/补涨的迹象，但当前更适合等二次回踩，不宜追价。"
            score += 6
        elif change_pct <= -3.0 and main_net <= 0:
            entry_state = "falling_knife"
            timing_note = "回落幅度偏大且资金承接不够，暂不按低吸处理。"
            score -= 10

        preferred_buy_range = _build_buy_range(price, support, bias5, entry_state) if price > 0 else ""
        missed_entry = False
        if preferred_buy_range and price > 0:
            try:
                _, upper_text = preferred_buy_range.split("-", 1)
                upper = _to_float(upper_text)
                missed_entry = upper > 0 and price > upper * 1.015 and entry_state in {"secondary_relay", "overextended", "leader_locked"}
            except ValueError:
                missed_entry = False

        if missed_entry:
            timing_note = f"{timing_note} 当前价格已明显脱离原低吸窗口，若盘中未成交，现阶段不追高，等回踩 {preferred_buy_range} 再看。"
            score -= 8

        row["sector"] = sector
        row["candidate_source"] = source
        row["relay_role"] = relay_role
        row["entry_state"] = entry_state
        row["timing_note"] = timing_note
        row["preferred_buy_range"] = preferred_buy_range
        row["missed_entry"] = missed_entry
        row["rotation_score"] = round(score + tech_score / 10.0, 2)
        annotated.append(row)

    priority = {
        "pullback_ready": 0,
        "secondary_relay": 1,
        "watch": 2,
        "overextended": 3,
        "leader_locked": 4,
        "falling_knife": 5,
    }
    annotated.sort(
        key=lambda item: (
            priority.get(str(item.get("entry_state", "watch")), 9),
            -_to_float(item.get("rotation_score")),
            -_to_float(item.get("main_net")),
            abs(_to_float(item.get("change_pct"))),
        )
    )
    if isinstance(limit, int) and limit > 0:
        return annotated[:limit]
    return annotated


def analyze_theme_rotation(
    sectors: Iterable[Dict[str, Any]],
    hot_candidates: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Infer current theme rotation / relay intensity from sector flow + candidate flow."""

    sector_rows = _normalize_sector_rows(sectors)
    positive_sectors = [row for row in sector_rows if row["main_net"] > 0]
    dominant_themes = [row["name"] for row in positive_sectors[:3]]
    annotated_candidates = annotate_rotation_candidates(hot_candidates, dominant_themes=dominant_themes, limit=6)
    leader = next(
        (candidate for candidate in annotated_candidates if candidate.get("entry_state") in {"leader_locked", "overextended"}),
        annotated_candidates[0] if annotated_candidates else None,
    )
    relay_candidates = [
        candidate for candidate in annotated_candidates
        if candidate.get("entry_state") in {"pullback_ready", "secondary_relay"}
    ][:3]

    switch_signal = "quiet"
    if dominant_themes and leader and relay_candidates:
        switch_signal = "active"
    elif dominant_themes and (leader or relay_candidates):
        switch_signal = "watch"

    summary = "题材轮动暂不明显，继续观察。"
    if switch_signal == "active" and leader:
        relay_names = "、".join(item.get("name", "") for item in relay_candidates if item.get("name"))
        summary = (
            f"{dominant_themes[0]}有转主线迹象，{leader.get('name', '')}负责领涨，"
            f"{relay_names or '其余补涨股'}更适合承接扩散。"
        )
    elif switch_signal == "watch" and dominant_themes:
        summary = f"{dominant_themes[0]}正在升温，但是否成为新主线仍需看领涨股连板后的资金扩散。"

    return {
        "dominant_themes": dominant_themes,
        "switch_signal": switch_signal,
        "leader": leader,
        "relay_candidates": relay_candidates,
        "summary": summary,
    }
