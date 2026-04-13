"""
Local-model agentic backtest for the blind-agent execution layer.

This variant keeps the strict "no future data" rule:
1. Each trading day, the local model only sees that day's close snapshot.
2. The model decides next-day actions: buy / sell / hold / empty.
3. Orders still execute on the next trading day's open.
4. Every daily decision is persisted for audit in thought logs.

Usage:
    python backtest_blind_agent_agentic.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import backtest_blind_agent as blind
import backtest_blind_agent_v4b as v4b
import train_blind_agent as trainer
from backtest_2025 import fetch_all_a_share_daily
from src.agent.llm_adapter import LLMToolAdapter
from src.config import detect_local_ollama_models, get_config


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

OUT_DIR = Path("data/backtest_blind_agent_agentic")
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_VALID_ACTIONS = {"buy", "sell", "hold", "empty"}
_VALID_CRITIQUE_VERDICTS = {"approve", "caution", "reject", "modify"}

SYSTEM_PROMPT = """你是 A 股短线/波段回测代理。

你当前处于某个交易日收盘后的复盘时刻，只能使用用户给出的当日及历史数据，绝对不能假设任何未来新闻、未来价格、未来财报或未来涨跌。

你的任务是为“下一交易日开盘”给出一个明确计划：
- buy: 保留当前持仓，并从候选池里挑选可以新开仓的股票
- sell: 卖出部分或全部现有持仓；如果需要换仓，也可以同时给出 buy_codes
- hold: 维持现有持仓，不新增买入
- empty: 下一交易日尽量清空全部可卖持仓，不新增买入

硬约束：
- sell_codes 只能来自“当前持仓”
- buy_codes 只能来自“候选池”
- 只能返回一个 JSON 对象，不要 markdown，不要解释，不要代码块
- 如果机会一般，优先 hold 或 empty，不要为了交易而交易

你必须严格输出这个 JSON schema：
{
  "action": "buy|sell|hold|empty",
  "market_view": "一句话描述你对当前环境的判断",
  "strategy": "oversold_rebound|breakout|pullback|rotation|defensive|mixed",
  "sell_codes": ["000001"],
  "buy_codes": ["000002", "600000"],
  "rationale": "一句话解释为什么这样做",
  "risk_note": "一句话说明主要风险"
}
"""

TEAM_PROPOSAL_SYSTEM_PROMPT = """你是 A 股短线/波段团队里的进攻派交易员。
你当前处于某个交易日收盘后的复盘时刻，只能使用输入中给出的当日及历史数据，绝不能假设任何未来新闻、未来价格、未来财报或未来涨跌。

你的任务是为“下一交易日开盘”提出一个明确计划：
- buy: 保留当前持仓，并从候选池里挑选可以新开仓的股票
- sell: 卖出部分或全部现有持仓；如果需要换仓，也可以同时给出 buy_codes
- hold: 维持现有持仓，不新增买入
- empty: 下一交易日尽量清空全部可卖持仓，不新增买入

硬约束：
- sell_codes 只能来自“当前持仓”
- buy_codes 只能来自“候选池”
- 如果机会一般，优先 hold 或 empty，不要为了交易而交易
- 只返回一个 JSON 对象，不要 markdown，不要解释，不要代码块

你必须严格输出这个 JSON schema：
{
  "action": "buy|sell|hold|empty",
  "market_view": "一句话描述你对当前环境的判断",
  "strategy": "oversold_rebound|breakout|pullback|rotation|defensive|mixed",
  "sell_codes": ["000001"],
  "buy_codes": ["000002", "600000"],
  "rationale": "一句话解释为什么这么做",
  "risk_note": "一句话说明主要风险"
}
"""

TEAM_CRITIQUE_SYSTEM_PROMPT = """你是 A 股短线/波段团队里的保守派风控员。
你不负责重新发明一套方案，而是严格审查进攻派方案，指出其中的追高、风险失控、该卖不卖、该空仓不空仓的问题。

硬约束：
- 只能根据输入给出的当日及历史数据审查，绝不能使用未来信息
- sell_codes 只能来自当前持仓
- block_buy_codes 和 prefer_buy_codes 只能来自候选池
- 只返回一个 JSON 对象，不要 markdown，不要解释，不要代码块

请严格输出：
{
  "verdict": "approve|caution|reject|modify",
  "action_bias": "buy|sell|hold|empty",
  "sell_codes": ["000001"],
  "block_buy_codes": ["000002"],
  "prefer_buy_codes": ["600000"],
  "critique": "一句话说明你对进攻派方案的主要质疑",
  "risk_note": "一句话说明你看到的核心风险"
}
"""

TEAM_ARBITER_SYSTEM_PROMPT = """你是 A 股短线/波段团队的最终仲裁者。
你需要综合进攻派交易员的 proposal 与风控员的 critique，为下一交易日开盘做最终拍板。

优先原则：
1. 只用输入里的当日及历史数据，绝不能使用未来信息
2. 风险优先，但也不能因为过度保守而错过明显机会
3. sell_codes 只能来自当前持仓
4. buy_codes 只能来自候选池
5. 如果风控指出明显追高或大环境偏弱，优先降低进攻性
6. 只返回一个 JSON 对象，不要 markdown，不要解释，不要代码块

你必须严格输出这个 JSON schema：
{
  "action": "buy|sell|hold|empty",
  "market_view": "一句话描述你对当前环境的最终判断",
  "strategy": "oversold_rebound|breakout|pullback|rotation|defensive|mixed",
  "sell_codes": ["000001"],
  "buy_codes": ["000002", "600000"],
  "rationale": "一句话解释为什么做出这个最终裁决",
  "risk_note": "一句话说明最终方案的主要风险"
}
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-model agentic blind-agent backtest")
    parser.add_argument("--start", default="2026-01-05", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-04-10", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--fetch-start", default="", help="Optional raw data fetch start date (YYYYMMDD)")
    parser.add_argument("--fetch-end", default="", help="Optional raw data fetch end date (YYYYMMDD)")
    parser.add_argument("--cache-name", default="backtest_cache_2026ytd.pkl", help="Cache file under data/")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    parser.add_argument("--model", default="", help="Override local model (defaults to REBALANCE_LOCAL_MODEL / Ollama)")
    parser.add_argument("--candidate-limit", type=int, default=8, help="How many candidates to expose to the LLM each day")
    parser.add_argument("--min-score", type=int, default=45, help="Minimum unified_score to enter the LLM candidate pool")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    return parser.parse_args()


def resolve_local_backtest_model(explicit: str = "") -> str:
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit.strip())

    env_model = os.getenv("REBALANCE_LOCAL_MODEL", "").strip()
    if env_model:
        candidates.append(env_model)

    cfg = get_config()
    primary = str(getattr(cfg, "litellm_model", "") or "").strip()
    if primary.startswith("ollama/"):
        candidates.append(primary)

    for fallback in list(getattr(cfg, "litellm_fallback_models", []) or []):
        fallback = str(fallback or "").strip()
        if fallback.startswith("ollama/"):
            candidates.append(fallback)

    for detected in detect_local_ollama_models():
        if detected:
            candidates.append(detected)

    seen = set()
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            return model
    return ""


def resolve_team_models(explicit_proposal: str = "") -> Dict[str, str]:
    proposal_model = resolve_local_backtest_model(explicit_proposal)

    critique_model = os.getenv("REBALANCE_DEBATE_MODEL", "").strip()
    if not critique_model:
        critique_model = proposal_model

    arbiter_model = (
        os.getenv("REBALANCE_CLOUD_MODEL", "").strip()
        or os.getenv("REBALANCE_CLOUD_FALLBACK", "").strip()
        or os.getenv("LITELLM_MODEL", "").strip()
        or critique_model
        or proposal_model
    )

    return {
        "proposal": proposal_model,
        "critique": critique_model,
        "arbiter": arbiter_model,
    }


def infer_fetch_start(start_date: str) -> str:
    return (pd.Timestamp(start_date) - timedelta(days=120)).strftime("%Y%m%d")


def strip_json_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = _JSON_FENCE_RE.sub("", cleaned).strip()
    return cleaned


def parse_json_object(text: str) -> Optional[dict]:
    cleaned = strip_json_fence(text)
    if not cleaned:
        return None

    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        try:
            from json_repair import repair_json

            repaired = repair_json(candidate, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            continue

    return None


class AgenticLocalBlindBacktest(blind.BlindAgentBacktest):
    def __init__(
        self,
        all_data: Dict[str, pd.DataFrame],
        *,
        team_models: Dict[str, str],
        candidate_limit: int,
        min_candidate_score: int,
        llm_temperature: float,
    ):
        super().__init__(all_data)
        self.team_models = {
            "proposal": str((team_models or {}).get("proposal", "") or "").strip(),
            "critique": str((team_models or {}).get("critique", "") or "").strip(),
            "arbiter": str((team_models or {}).get("arbiter", "") or "").strip(),
        }
        self.candidate_limit = candidate_limit
        self.min_candidate_score = min_candidate_score
        self.llm_temperature = llm_temperature
        self.market_contexts = v4b.build_market_contexts(all_data)
        self.adapter = LLMToolAdapter()

        self.thought_logs: List[Dict[str, Any]] = []
        self._staged_agent_buys: List[dict] = []
        self._rule_buy_fallback_today = False
        self._last_decision_date = ""
        self._candidate_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.stats = {
            "llm_calls": 0,
            "llm_parse_failures": 0,
            "proposal_calls": 0,
            "critique_calls": 0,
            "arbiter_calls": 0,
            "proposal_parse_failures": 0,
            "critique_parse_failures": 0,
            "arbiter_parse_failures": 0,
            "team_merge_fallback_days": 0,
            "rule_fallback_days": 0,
            "auto_empty_days": 0,
        }

    def _candidate_setup(self, ind: Dict[str, Any]) -> str:
        tags: List[str] = []
        if ind["price_pos"] < 0.12 and ind["bias20"] < -5:
            tags.append("oversold")
        if ind["vol_breakout"] and ind["ma5"] > ind["ma10"]:
            tags.append("breakout")
        if ind["vol_pullback"] and ind["ma5"] > ind["ma10"]:
            tags.append("pullback")
        if ind["yest_limit"] and ind["consec_limit"] <= 2:
            tags.append("relay")
        return "+".join(tags) if tags else "mixed"

    def _build_candidate_pool(self, date: str) -> List[Dict[str, Any]]:
        cached = self._candidate_cache.get(date)
        if cached is not None:
            return cached

        pmin, pmax = blind.RISK_PARAMS["price_range"]
        min_turnover = blind.RISK_PARAMS["min_turnover"]
        candidates: List[Dict[str, Any]] = []
        for code, df in self.all_data.items():
            rows = df.index[df["date"] == date]
            if len(rows) == 0:
                continue
            idx = int(rows[0])
            row = df.iloc[idx]
            price = float(row["close"])
            turnover = float(row.get("turnover_rate", 0) or 0)
            if price < pmin or price > pmax or turnover < min_turnover:
                continue

            ind = blind.calc_indicators(df, idx)
            if ind is None:
                continue

            score = blind.unified_score(ind)
            candidates.append(
                {
                    "code": code,
                    "name": str(row.get("stock_name", code)),
                    "signal_close": round(price, 3),
                    "score": int(score),
                    "setup": self._candidate_setup(ind),
                    "today_chg": round(float(ind["today_chg"]), 2),
                    "rsi": round(float(ind["rsi"]), 1),
                    "bias20": round(float(ind["bias20"]), 2),
                    "price_pos": round(float(ind["price_pos"]), 3),
                    "turnover": round(turnover, 2),
                    "vol_ratio": round(float(ind["vol_ratio"]), 2),
                    "ma_trend": (
                        "ma5>ma10>ma20"
                        if ind["ma5"] > ind["ma10"] > ind["ma20"]
                        else "ma5>ma10"
                        if ind["ma5"] > ind["ma10"]
                        else "mixed"
                    ),
                    "macd_cross": bool(ind["macd_cross"]),
                }
            )

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["turnover"],
                item["today_chg"],
            ),
            reverse=True,
        )
        shortlisted = [item for item in candidates if item["score"] >= self.min_candidate_score][: self.candidate_limit]
        if not shortlisted:
            shortlisted = candidates[: min(3, len(candidates))]
        self._candidate_cache[date] = shortlisted
        return shortlisted

    def _recent_trade_payload(self, limit: int = 6) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for trade in self.trades[-limit:]:
            rows.append(
                {
                    "date": trade.date,
                    "code": trade.code,
                    "dir": trade.direction,
                    "price": round(trade.price, 3),
                    "shares": trade.shares,
                    "reason": trade.reason,
                }
            )
        return rows

    def _holdings_payload(self, day_idx: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for code, holding in self.holdings.items():
            peak_pnl = (
                (holding.peak_price - holding.cost_price) / holding.cost_price * 100
                if holding.cost_price > 0
                else 0.0
            )
            peak_drawdown = (
                (holding.peak_price - holding.current_price) / holding.peak_price * 100
                if holding.peak_price > 0
                else 0.0
            )
            rows.append(
                {
                    "code": code,
                    "name": holding.name,
                    "shares": holding.shares,
                    "days_held": int(day_idx - holding.buy_day_idx),
                    "cost_price": round(holding.cost_price, 3),
                    "close_price": round(holding.current_price, 3),
                    "pnl_pct": round(holding.pnl_pct, 2),
                    "peak_pnl_pct": round(peak_pnl, 2),
                    "pullback_from_peak_pct": round(peak_drawdown, 2),
                    "sellable_next_open": bool(holding.buy_day_idx < day_idx),
                }
            )
        rows.sort(key=lambda item: item["pnl_pct"])
        return rows

    def _default_decision(self, action: str, reason: str) -> Dict[str, Any]:
        normalized_action = action if action in _VALID_ACTIONS else "hold"
        return {
            "action": normalized_action,
            "market_view": reason,
            "strategy": "defensive" if normalized_action in {"hold", "empty"} else "mixed",
            "sell_codes": [],
            "buy_codes": [],
            "rationale": reason,
            "risk_note": reason,
        }

    def _call_json_model(
        self,
        *,
        role_name: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 700,
        timeout: int = 180,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not model:
            return "MODEL_MISSING", None

        self.stats["llm_calls"] += 1
        self.stats[f"{role_name}_calls"] += 1
        try:
            prompt = f"{system_prompt}\n\n{user_prompt}"
            if role_name == "critique":
                from rebalance_engine import _call_debate_llm

                raw_response = str(_call_debate_llm(prompt, agent_name="agentic_critique") or "").strip()
            elif role_name == "arbiter":
                from rebalance_engine import _call_cloud_llm

                raw_response = str(_call_cloud_llm(prompt, agent_name="agentic_arbiter") or "").strip()
            else:
                response = self.adapter.call_text(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.llm_temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    model_override=model,
                    fallback_models_override=[],
                )
                raw_response = str(response.content or "").strip()
        except Exception as exc:
            return f"LLM_ERROR: {exc}", None

        parsed = parse_json_object(raw_response)
        if parsed is None:
            self.stats["llm_parse_failures"] += 1
            self.stats[f"{role_name}_parse_failures"] += 1
        return raw_response, parsed

    def _normalize_critique(
        self,
        parsed: Dict[str, Any],
        holdings_map: Dict[str, Dict[str, Any]],
        candidate_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _extract_codes(value: Any) -> List[str]:
            codes: List[str] = []
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        code = str(item.get("code", "") or "").strip()
                    else:
                        code = str(item or "").strip()
                    if code:
                        codes.append(code)
            elif isinstance(value, str):
                codes.extend(part.strip() for part in value.split(",") if part.strip())
            return codes

        verdict = str(parsed.get("verdict", "caution") or "caution").strip().lower()
        if verdict not in _VALID_CRITIQUE_VERDICTS:
            verdict = "caution"

        action_bias = str(parsed.get("action_bias", "hold") or "hold").strip().lower()
        if action_bias not in _VALID_ACTIONS:
            action_bias = "hold"

        sell_codes = [
            code
            for code in _extract_codes(parsed.get("sell_codes") or parsed.get("risk_sell_codes"))
            if code in holdings_map
        ]
        block_buy_codes = [
            code
            for code in _extract_codes(parsed.get("block_buy_codes") or parsed.get("reject_buy_codes"))
            if code in candidate_map
        ]
        prefer_buy_codes = [
            code
            for code in _extract_codes(parsed.get("prefer_buy_codes") or parsed.get("safer_buy_codes"))
            if code in candidate_map
        ]
        return {
            "verdict": verdict,
            "action_bias": action_bias,
            "sell_codes": list(dict.fromkeys(sell_codes)),
            "block_buy_codes": list(dict.fromkeys(block_buy_codes)),
            "prefer_buy_codes": list(dict.fromkeys(prefer_buy_codes)),
            "critique": str(parsed.get("critique", "") or "").strip()[:400],
            "risk_note": str(parsed.get("risk_note", "") or "").strip()[:200],
        }

    def _merge_team_decisions(
        self,
        proposal: Dict[str, Any],
        critique: Dict[str, Any],
        candidate_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        final = {
            "action": proposal["action"],
            "market_view": proposal["market_view"],
            "strategy": proposal["strategy"],
            "sell_codes": list(proposal["sell_codes"]),
            "buy_codes": list(proposal["buy_codes"]),
            "rationale": proposal["rationale"],
            "risk_note": proposal["risk_note"],
        }

        blocked = set(critique["block_buy_codes"])
        preferred = [code for code in critique["prefer_buy_codes"] if code in candidate_map and code not in blocked]
        final["buy_codes"] = [code for code in final["buy_codes"] if code not in blocked]
        for code in reversed(preferred):
            if code not in final["buy_codes"]:
                final["buy_codes"].insert(0, code)

        if critique["sell_codes"]:
            final["sell_codes"] = list(dict.fromkeys(final["sell_codes"] + critique["sell_codes"]))

        verdict = critique["verdict"]
        action_bias = critique["action_bias"]
        if verdict in {"reject", "modify"}:
            if action_bias == "empty":
                final["action"] = "empty"
                final["buy_codes"] = []
            elif action_bias == "sell":
                final["action"] = "sell"
                final["buy_codes"] = []
            elif action_bias == "hold" and final["action"] == "buy":
                final["action"] = "hold"
                final["buy_codes"] = []

        if final["action"] == "buy" and not final["buy_codes"]:
            final["action"] = "sell" if final["sell_codes"] else "hold"
        if final["action"] == "sell" and not final["sell_codes"]:
            final["action"] = "hold"
        if final["action"] == "empty":
            final["buy_codes"] = []

        critique_text = critique.get("critique", "")
        risk_note = critique.get("risk_note", "")
        if critique_text:
            final["rationale"] = f"{final['rationale']} | 风控: {critique_text}"[:400]
        if risk_note:
            final["risk_note"] = risk_note[:200]
        return final

    def _normalize_decision(
        self,
        parsed: Dict[str, Any],
        holdings_map: Dict[str, Dict[str, Any]],
        candidate_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        action = str(parsed.get("action", "hold") or "hold").strip().lower()
        if action not in _VALID_ACTIONS:
            action = "hold"

        def _extract_codes(value: Any) -> List[str]:
            codes: List[str] = []
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        code = str(item.get("code", "") or "").strip()
                    else:
                        code = str(item or "").strip()
                    if code:
                        codes.append(code)
            elif isinstance(value, str):
                codes.extend(part.strip() for part in value.split(",") if part.strip())
            return codes

        sell_codes = [code for code in _extract_codes(parsed.get("sell_codes") or parsed.get("sells")) if code in holdings_map]
        buy_codes = [code for code in _extract_codes(parsed.get("buy_codes") or parsed.get("buys")) if code in candidate_map]

        primary_code = str(parsed.get("primary_code", "") or "").strip()
        if primary_code and primary_code in candidate_map and primary_code not in buy_codes:
            buy_codes.insert(0, primary_code)

        if action == "empty":
            sell_codes = list(holdings_map.keys())
            buy_codes = []
        elif action == "hold":
            buy_codes = []

        dedup_sell = list(dict.fromkeys(sell_codes))
        dedup_buy = list(dict.fromkeys(buy_codes))
        return {
            "action": action,
            "market_view": str(parsed.get("market_view", "") or "").strip()[:200],
            "strategy": str(parsed.get("strategy", "mixed") or "mixed").strip()[:64],
            "sell_codes": dedup_sell,
            "buy_codes": dedup_buy,
            "rationale": str(parsed.get("rationale", "") or "").strip()[:400],
            "risk_note": str(parsed.get("risk_note", "") or "").strip()[:200],
        }

    def _build_user_prompt(
        self,
        date: str,
        holdings_payload: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
    ) -> str:
        market_ctx = self.market_contexts.get(date)
        market_view = {}
        if market_ctx:
            market_view = {
                "cnt50": market_ctx.cnt50,
                "cnt60": market_ctx.cnt60,
                "top3_avg": market_ctx.top3_avg,
                "strong_count": market_ctx.strong_count,
                "weak_count": market_ctx.weak_count,
                "weak_ratio": market_ctx.weak_ratio,
                "breakout_count": market_ctx.breakout_count,
                "pullback_count": market_ctx.pullback_count,
            }

        state_view = {
            "cash": round(self.cash, 0),
            "holdings_count": len(self.holdings),
            "position_coeff": round(self._state.position_coeff, 2),
            "buy_threshold": self._state.buy_threshold,
            "recent_win_rate": round(self._state.recent_win_rate, 2),
            "recent_avg_pnl": round(self._state.recent_avg_pnl, 2),
            "consecutive_losses": self._state.consecutive_losses,
            "drawdown_pct": round(self._state.drawdown_pct, 2),
            "max_positions": blind.RISK_PARAMS["max_positions"],
        }

        return (
            f"当前复盘日期: {date}\n"
            "你只能根据今天收盘后已知的数据，决定下一交易日开盘的计划。\n"
            "A股约束: 今天新买的股票明天之前不能卖；买入必须来自候选池，卖出必须来自当前持仓。\n\n"
            f"账户状态:\n{json.dumps(state_view, ensure_ascii=False)}\n\n"
            f"市场横截面摘要:\n{json.dumps(market_view, ensure_ascii=False)}\n\n"
            f"当前持仓:\n{json.dumps(holdings_payload, ensure_ascii=False)}\n\n"
            f"候选池(已按规则预筛选后的可交易股票):\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
            f"最近交易记录:\n{json.dumps(self._recent_trade_payload(), ensure_ascii=False)}\n\n"
            "请现在返回唯一 JSON。"
        )

    def _build_day_payload(
        self,
        date: str,
        holdings_payload: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        market_ctx = self.market_contexts.get(date)
        market_view = {}
        if market_ctx:
            market_view = {
                "cnt50": market_ctx.cnt50,
                "cnt60": market_ctx.cnt60,
                "top3_avg": market_ctx.top3_avg,
                "strong_count": market_ctx.strong_count,
                "weak_count": market_ctx.weak_count,
                "weak_ratio": market_ctx.weak_ratio,
                "breakout_count": market_ctx.breakout_count,
                "pullback_count": market_ctx.pullback_count,
            }

        state_view = {
            "cash": round(self.cash, 0),
            "holdings_count": len(self.holdings),
            "position_coeff": round(self._state.position_coeff, 2),
            "buy_threshold": self._state.buy_threshold,
            "recent_win_rate": round(self._state.recent_win_rate, 2),
            "recent_avg_pnl": round(self._state.recent_avg_pnl, 2),
            "consecutive_losses": self._state.consecutive_losses,
            "drawdown_pct": round(self._state.drawdown_pct, 2),
            "max_positions": blind.RISK_PARAMS["max_positions"],
        }

        return {
            "date": date,
            "state": state_view,
            "market_view": market_view,
            "holdings": holdings_payload,
            "candidates": candidates,
            "recent_trades": self._recent_trade_payload(),
            "constraints": {
                "t_plus_one": "今天新买的股票下一交易日前不能卖",
                "sell_codes_from_holdings_only": True,
                "buy_codes_from_candidates_only": True,
            },
        }

    def _build_proposal_prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "请基于下面这份严格无未来数据的日终快照，给出进攻派下一交易日开盘计划。\n"
            "如果机会一般，优先 hold 或 empty。\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def _build_critique_prompt(self, payload: Dict[str, Any], proposal: Dict[str, Any]) -> str:
        return (
            "请审查下面这份进攻派 proposal，专门挑出追高、风控不足、该卖不卖的问题。\n\n"
            f"日终快照:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            f"进攻派 proposal:\n{json.dumps(proposal, ensure_ascii=False)}"
        )

    def _build_arbiter_prompt(
        self,
        payload: Dict[str, Any],
        proposal: Dict[str, Any],
        critique: Dict[str, Any],
    ) -> str:
        return (
            "请综合 proposal 和 critique，给出最终 next-open 交易决策。\n"
            "不要照抄任一方，优先输出风险收益比更合理的最终方案。\n\n"
            f"日终快照:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            f"进攻派 proposal:\n{json.dumps(proposal, ensure_ascii=False)}\n\n"
            f"风控 critique:\n{json.dumps(critique, ensure_ascii=False)}"
        )

    def _append_thought_log(
        self,
        date: str,
        source: str,
        raw_response: str,
        parsed_decision: Dict[str, Any],
        holdings_payload: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        team_trace: Optional[Dict[str, Any]] = None,
    ) -> None:
        market_ctx = self.market_contexts.get(date)
        self.thought_logs.append(
            {
                "date": date,
                "source": source,
                "models": self.team_models,
                "raw_response": raw_response,
                "decision": parsed_decision,
                "team_trace": team_trace or {},
                "state": {
                    "cash": round(self.cash, 0),
                    "position_coeff": round(self._state.position_coeff, 3),
                    "buy_threshold": self._state.buy_threshold,
                    "recent_win_rate": round(self._state.recent_win_rate, 3),
                    "recent_avg_pnl": round(self._state.recent_avg_pnl, 3),
                    "consecutive_losses": self._state.consecutive_losses,
                    "drawdown_pct": round(self._state.drawdown_pct, 3),
                },
                "market_context": {
                    "cnt50": getattr(market_ctx, "cnt50", 0),
                    "cnt60": getattr(market_ctx, "cnt60", 0),
                    "top3_avg": getattr(market_ctx, "top3_avg", 0.0),
                    "strong_count": getattr(market_ctx, "strong_count", 0),
                    "weak_count": getattr(market_ctx, "weak_count", 0),
                    "weak_ratio": getattr(market_ctx, "weak_ratio", 0.0),
                },
                "holdings": holdings_payload,
                "candidates": candidates,
            }
        )

    def _stage_agent_decision(self, date: str, day_idx: int) -> None:
        if self._last_decision_date == date:
            return

        self._last_decision_date = date
        self._staged_agent_buys = []
        self._rule_buy_fallback_today = False

        holdings_payload = self._holdings_payload(day_idx)
        holdings_map = {item["code"]: item for item in holdings_payload}
        candidates = self._build_candidate_pool(date)
        candidate_map = {item["code"]: item for item in candidates}

        if not holdings_payload and not candidates:
            self.stats["auto_empty_days"] += 1
            decision = self._default_decision("empty", "no_holdings_and_no_candidates")
            self._append_thought_log(date, "auto_empty", "", decision, holdings_payload, candidates)
            return

        payload = self._build_day_payload(date, holdings_payload, candidates)
        team_trace: Dict[str, Any] = {}
        source = "team_llm"

        proposal_raw, proposal_parsed = self._call_json_model(
            role_name="proposal",
            model=self.team_models["proposal"],
            system_prompt=TEAM_PROPOSAL_SYSTEM_PROMPT,
            user_prompt=self._build_proposal_prompt(payload),
            max_tokens=500,
            timeout=180,
        )
        team_trace["proposal"] = {
            "model": self.team_models["proposal"],
            "raw_response": proposal_raw,
            "parsed": proposal_parsed,
        }
        if proposal_parsed is None:
            source = "rule_fallback"
            raw_response = proposal_raw
        else:
            proposal = self._normalize_decision(proposal_parsed, holdings_map, candidate_map)

            critique_raw, critique_parsed = self._call_json_model(
                role_name="critique",
                model=self.team_models["critique"],
                system_prompt=TEAM_CRITIQUE_SYSTEM_PROMPT,
                user_prompt=self._build_critique_prompt(payload, proposal),
                max_tokens=450,
                timeout=240,
            )
            team_trace["critique"] = {
                "model": self.team_models["critique"],
                "raw_response": critique_raw,
                "parsed": critique_parsed,
            }

            critique = None
            if critique_parsed is not None:
                critique = self._normalize_critique(critique_parsed, holdings_map, candidate_map)

            decision = proposal
            raw_response = proposal_raw
            if critique is not None:
                arbiter_raw, arbiter_parsed = self._call_json_model(
                    role_name="arbiter",
                    model=self.team_models["arbiter"],
                    system_prompt=TEAM_ARBITER_SYSTEM_PROMPT,
                    user_prompt=self._build_arbiter_prompt(payload, proposal, critique),
                    max_tokens=550,
                    timeout=300,
                )
                team_trace["arbiter"] = {
                    "model": self.team_models["arbiter"],
                    "raw_response": arbiter_raw,
                    "parsed": arbiter_parsed,
                }

                if arbiter_parsed is not None:
                    decision = self._normalize_decision(arbiter_parsed, holdings_map, candidate_map)
                    raw_response = arbiter_raw
                else:
                    source = "team_merge_fallback"
                    self.stats["team_merge_fallback_days"] += 1
                    decision = self._merge_team_decisions(proposal, critique, candidate_map)
                    raw_response = json.dumps(
                        {
                            "proposal": proposal,
                            "critique": critique,
                            "fallback_merge": decision,
                        },
                        ensure_ascii=False,
                    )
            else:
                source = "team_partial"

            if decision["action"] == "empty":
                for code, holding in list(self.holdings.items()):
                    if holding.buy_day_idx != day_idx:
                        self._pending_sells.append((code, holding.shares, "TEAM_EMPTY"))
            else:
                for code in decision["sell_codes"]:
                    holding = self.holdings.get(code)
                    if holding and holding.buy_day_idx != day_idx:
                        self._pending_sells.append((code, holding.shares, "TEAM_SELL"))
                self._staged_agent_buys = [candidate_map[code] for code in decision["buy_codes"] if code in candidate_map]
            self._append_thought_log(date, source, raw_response, decision, holdings_payload, candidates, team_trace=team_trace)
            return

        self.stats["rule_fallback_days"] += 1
        self._rule_buy_fallback_today = True
        super()._review_sells(date, day_idx)
        fallback_decision = self._default_decision("hold", "llm_unavailable_or_invalid_json")
        self._append_thought_log(date, source, raw_response, fallback_decision, holdings_payload, candidates, team_trace=team_trace)

    def _review_sells(self, date, day_idx):
        self._stage_agent_decision(date, day_idx)

    def _scan_buys(self, date, day_idx):
        if self._last_decision_date != date:
            self._stage_agent_decision(date, day_idx)

        if self._rule_buy_fallback_today:
            super()._scan_buys(date, day_idx)
            return

        if len(self.holdings) >= blind.RISK_PARAMS["max_positions"]:
            return
        if self._pending_buys:
            return
        if self._state.in_cooldown:
            return
        if self._state.consecutive_losses >= 4 and not self._state._cooldown_consumed:
            self._state.trigger_cooldown()
            return

        slots = blind.RISK_PARAMS["max_positions"] - len(self.holdings)
        if slots <= 0:
            return

        for candidate in self._staged_agent_buys[:slots]:
            if candidate["code"] in self.holdings:
                continue
            self._pending_buys.append(candidate)


class SilentAgenticLocalBlindBacktest(AgenticLocalBlindBacktest):
    def _print_report(self):
        return


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    *,
    team_models: Dict[str, str],
    candidate_limit: int,
    min_candidate_score: int,
    llm_temperature: float,
    silent: bool = True,
) -> Tuple[AgenticLocalBlindBacktest, trainer.Metrics]:
    previous_level = blind.logger.level
    blind.logger.setLevel(logging.WARNING)
    try:
        engine_cls = SilentAgenticLocalBlindBacktest if silent else AgenticLocalBlindBacktest
        engine = engine_cls(
            all_data,
            team_models=team_models,
            candidate_limit=candidate_limit,
            min_candidate_score=min_candidate_score,
            llm_temperature=llm_temperature,
        )
        engine.run(start_date, end_date)
        metrics = trainer._summarize_engine(engine)
        return engine, metrics
    finally:
        blind.logger.setLevel(previous_level)


def save_outputs(
    engine: AgenticLocalBlindBacktest,
    metrics: trainer.Metrics,
    out_dir: Path,
    *,
    team_models: Dict[str, str],
    candidate_limit: int,
    min_candidate_score: int,
    llm_temperature: float,
) -> None:
    trainer.save_backtest_outputs(
        engine,
        metrics,
        out_dir,
        extra={
            "strategy": "blind_agent_agentic_team",
            "team_models": team_models,
            "candidate_limit": candidate_limit,
            "min_candidate_score": min_candidate_score,
            "llm_temperature": llm_temperature,
            **engine.stats,
        },
    )

    with open(out_dir / "thought_logs.jsonl", "w", encoding="utf-8") as file:
        for row in engine.thought_logs:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = [
        {
            "date": row["date"],
            "source": row["source"],
            "action": (row.get("decision") or {}).get("action", ""),
            "strategy": (row.get("decision") or {}).get("strategy", ""),
            "proposal_action": (((row.get("team_trace") or {}).get("proposal") or {}).get("parsed") or {}).get("action", ""),
            "critique_verdict": (((row.get("team_trace") or {}).get("critique") or {}).get("parsed") or {}).get("verdict", ""),
            "arbiter_action": (((row.get("team_trace") or {}).get("arbiter") or {}).get("parsed") or {}).get("action", ""),
            "sell_codes": ",".join((row.get("decision") or {}).get("sell_codes", [])),
            "buy_codes": ",".join((row.get("decision") or {}).get("buy_codes", [])),
            "market_view": (row.get("decision") or {}).get("market_view", ""),
            "rationale": (row.get("decision") or {}).get("rationale", ""),
            "risk_note": (row.get("decision") or {}).get("risk_note", ""),
        }
        for row in engine.thought_logs
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "thought_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    team_models = resolve_team_models(args.model)
    if not team_models["proposal"]:
        raise RuntimeError(
            "No local model available for agentic backtest. "
            "Please set REBALANCE_LOCAL_MODEL to an Ollama model first."
        )

    fetch_start = args.fetch_start or infer_fetch_start(args.start)
    fetch_end = args.fetch_end or args.end.replace("-", "")

    logger.info("Loading data cache...")
    all_data = fetch_all_a_share_daily(fetch_start, fetch_end, cache_name=args.cache_name)
    logger.info(
        "Agentic team models | proposal=%s critique=%s arbiter=%s",
        team_models["proposal"] or "N/A",
        team_models["critique"] or "N/A",
        team_models["arbiter"] or "N/A",
    )
    engine, metrics = run_backtest(
        all_data,
        args.start,
        args.end,
        team_models=team_models,
        candidate_limit=args.candidate_limit,
        min_candidate_score=args.min_score,
        llm_temperature=args.temperature,
        silent=False,
    )

    out_dir = Path(args.out_dir)
    save_outputs(
        engine,
        metrics,
        out_dir,
        team_models=team_models,
        candidate_limit=args.candidate_limit,
        min_candidate_score=args.min_score,
        llm_temperature=args.temperature,
    )
    logger.info(
        "Agentic result | return=%+.2f%% max_dd=%.2f%% sharpe=%.2f avg_month=%+.2f%%",
        metrics.ret,
        metrics.max_dd,
        metrics.sharpe,
        metrics.avg_month,
    )
    logger.info("Thought logs saved to %s", out_dir / "thought_logs.jsonl")


if __name__ == "__main__":
    main()
