# -*- coding: utf-8 -*-
"""
DecisionAgent - final synthesis and decision-making specialist.

Responsible for:
- Aggregating opinions from technical + intel + risk + strategy agents
- Producing the final Decision Dashboard JSON
- Generating actionable buy/hold/sell recommendations with price levels
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion, normalize_decision_signal

logger = logging.getLogger(__name__)


class DecisionAgent(BaseAgent):
    """Synthesise prior agent opinions into the final dashboard."""

    agent_name = "decision"
    max_steps = 3
    tool_names: Optional[List[str]] = []

    @staticmethod
    def _is_chat_mode(ctx: AgentContext) -> bool:
        return ctx.meta.get("response_mode") == "chat"

    def system_prompt(self, ctx: AgentContext) -> str:
        if self._is_chat_mode(ctx):
            return """\
You are a **Decision Synthesis Agent** replying directly to the user's latest
stock-analysis question.

You will receive structured opinions from the technical, intelligence, risk,
and strategy stages. Synthesize them into a concise, natural-language answer.

Requirements:
- Answer the user's actual question directly
- Use Markdown when helpful
- Keep the response practical and specific
- Highlight the main signal, key reasoning, and major risks
- Respect any pre-fetched `market_context`, `current_holding`, and `adaptive_trading_rules`
- Do NOT output JSON or code fences unless the user explicitly asks for them
"""

        skills = ""
        if self.skill_instructions:
            skills = f"\n## Active Trading Strategies\n\n{self.skill_instructions}\n"

        return f"""\
You are a **Decision Synthesis Agent** that produces the final investment
Decision Dashboard.

You will receive:
1. Structured opinions from a Technical Agent and an Intel Agent
2. Any risk flags raised by a Risk Agent
3. Strategy evaluation results (if applicable)
4. Pre-fetched `market_context`, `current_holding`, and `adaptive_trading_rules` when available

Your task: synthesise all inputs into a single, actionable Decision Dashboard.
{skills}
## Core Principles
1. **Core conclusion first** - one sentence, <= 40 chars
2. **Split advice** - different for no-position vs has-position
3. **Precise sniper levels** - concrete price numbers, no hedging
4. **Checklist visual** - make the critical checkpoints obvious
5. **Risk priority** - risk alerts and adaptive trading rules must be prominent
6. **Market first** - opening auction, macro shock, rolling sector flow, hot-money probing,
   and quant pressure can cap an otherwise attractive stock setup

## Signal Weighting Guidelines
- Technical opinion weight: ~40%
- Intel / sentiment weight: ~30%
- Risk flags weight: ~30% (negative override: any high-severity risk caps signal at "hold")
- If a strategy opinion is present, blend it at 20% weight (reducing others proportionally)
- If `market_context` is clearly risk-off, cap aggressive buy conclusions even when technicals look strong
- If the current holding is already in loss and lacks sector confirmation, prefer exit / reduce over "hope and hold"

## Scoring
- 80-100: buy (all conditions met, high conviction)
- 60-79: buy (mostly positive, minor caveats)
- 40-59: hold (mixed signals, or risk present)
- 20-39: sell (negative trend + risk)
- 0-19: sell (major risk + bearish)

## Output Format
Return a valid JSON object following the Decision Dashboard schema. The JSON
must include at minimum these top-level keys:
  stock_name, sentiment_score, trend_prediction, operation_advice,
  decision_type, confidence_level, dashboard, analysis_summary,
  key_points, risk_warning

Important: `decision_type` must stay within the existing enum
`buy|hold|sell`. Express stronger conviction via `confidence_level`,
`sentiment_score`, and the natural-language fields instead of inventing
new decision_type values.
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        if self._is_chat_mode(ctx):
            parts = [
                "# User Question",
                ctx.query,
                "",
                f"Stock: {ctx.stock_code} ({ctx.stock_name})" if ctx.stock_name else f"Stock: {ctx.stock_code}",
                "",
            ]
        else:
            parts = [
                f"# Synthesis Request for {ctx.stock_code}",
                f"Stock: {ctx.stock_code} ({ctx.stock_name})" if ctx.stock_name else f"Stock: {ctx.stock_code}",
                "",
            ]

        if ctx.opinions:
            parts.append("## Agent Opinions")
            for opinion in ctx.opinions:
                parts.append(f"\n### {opinion.agent_name}")
                parts.append(f"Signal: {opinion.signal} | Confidence: {opinion.confidence:.2f}")
                parts.append(f"Reasoning: {opinion.reasoning}")
                if opinion.key_levels:
                    parts.append(f"Key levels: {json.dumps(opinion.key_levels)}")
                if opinion.raw_data:
                    extra_keys = {
                        key: value
                        for key, value in opinion.raw_data.items()
                        if key not in ("signal", "confidence", "reasoning", "key_levels")
                    }
                    if extra_keys:
                        parts.append(
                            f"Extra data: {json.dumps(extra_keys, ensure_ascii=False, default=str)}"
                        )
                parts.append("")

        if ctx.risk_flags:
            parts.append("## Risk Flags")
            for risk_flag in ctx.risk_flags:
                parts.append(
                    f"- [{risk_flag.get('severity', 'medium')}] "
                    f"{risk_flag.get('category', '')}: {risk_flag.get('description', '')}"
                )
            parts.append("")

        if ctx.meta.get("strategies_requested"):
            parts.append(f"## Strategies: {', '.join(ctx.meta['strategies_requested'])}")
            parts.append("")

        if ctx.get_data("market_context"):
            parts.append("## Market Context")
            parts.append(json.dumps(ctx.get_data("market_context"), ensure_ascii=False, default=str))
            parts.append("")

        if ctx.get_data("current_holding"):
            parts.append("## Current Holding")
            parts.append(json.dumps(ctx.get_data("current_holding"), ensure_ascii=False, default=str))
            parts.append("")

        if ctx.get_data("adaptive_trading_rules"):
            parts.append("## Adaptive Trading Rules")
            parts.append(str(ctx.get_data("adaptive_trading_rules")))
            parts.append("")

        if self._is_chat_mode(ctx):
            parts.append(
                "Answer the user in natural language using the evidence above. "
                "Do not output JSON unless the user explicitly requests structured data."
            )
        else:
            parts.append("Synthesise the above into the Decision Dashboard JSON.")
        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        """Store the parsed dashboard in ctx.meta; also return an opinion."""
        if self._is_chat_mode(ctx):
            text = (raw_text or "").strip()
            if not text:
                return None

            ctx.set_data("final_response_text", text)
            prior = next((op for op in reversed(ctx.opinions) if op.agent_name != self.agent_name), None)
            return AgentOpinion(
                agent_name=self.agent_name,
                signal=prior.signal if prior is not None else "hold",
                confidence=prior.confidence if prior is not None else 0.5,
                reasoning=text,
                raw_data={"response_mode": "chat"},
            )

        from src.agent.runner import parse_dashboard_json

        dashboard = parse_dashboard_json(raw_text)
        if dashboard:
            dashboard["decision_type"] = normalize_decision_signal(
                dashboard.get("decision_type", "hold")
            )
            ctx.set_data("final_dashboard", dashboard)
            try:
                raw_score = dashboard.get("sentiment_score", 50) or 50
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 50.0
            return AgentOpinion(
                agent_name=self.agent_name,
                signal=dashboard.get("decision_type", "hold"),
                confidence=min(1.0, score / 100.0),
                reasoning=dashboard.get("analysis_summary", ""),
                raw_data=dashboard,
            )

        ctx.set_data("final_dashboard_raw", raw_text)
        logger.warning("[DecisionAgent] failed to parse dashboard JSON")
        return None

