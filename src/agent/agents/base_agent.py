# -*- coding: utf-8 -*-
"""
BaseAgent — abstract base for all specialised agents.

Every agent in the multi-agent pipeline inherits from this class and
implements :meth:`run`.  The base class provides shared utilities:
tool-subset selection, prompt assembly, LLM invocation via the shared
runner, and structured opinion output.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.llm_adapter import LLMToolAdapter
from src.agent.memory import AgentMemory
from src.agent.protocols import AgentContext, AgentOpinion, StageResult, StageStatus
from src.agent.runner import RunLoopResult, run_agent_loop
from src.agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all specialised agents.

    Subclasses **must** implement:
    - :pyattr:`agent_name` — unique agent identifier
    - :meth:`system_prompt` — return the LLM system prompt
    - :meth:`build_user_message` — construct the user message for the LLM

    Subclasses **may** override:
    - :pyattr:`tool_names` — restrict which tools the agent can access
    - :pyattr:`max_steps` — per-agent step limit  (default 6)
    - :meth:`post_process` — transform the raw LLM text into an :class:`AgentOpinion`
    """

    # Subclass overrides
    agent_name: str = "base"
    tool_names: Optional[List[str]] = None  # None → all tools available
    max_steps: int = 6

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm_adapter: LLMToolAdapter,
        skill_instructions: str = "",
    ):
        self.tool_registry = tool_registry
        self.llm_adapter = llm_adapter
        self.skill_instructions = skill_instructions
        self.memory = AgentMemory.from_config()

    # -----------------------------------------------------------------
    # Abstract interface
    # -----------------------------------------------------------------

    @abstractmethod
    def system_prompt(self, ctx: AgentContext) -> str:
        """Build the system prompt for this agent."""

    @abstractmethod
    def build_user_message(self, ctx: AgentContext) -> str:
        """Build the user message sent to the LLM."""

    # -----------------------------------------------------------------
    # Default hook for structured output
    # -----------------------------------------------------------------

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        """Extract a structured :class:`AgentOpinion` from the raw LLM text.

        Default: returns ``None`` (the raw text is still stored in
        ``StageResult.meta["raw_text"]``).  Subclasses that produce
        analysis opinions should override this.
        """
        return None

    def resolve_model_route(self, ctx: AgentContext) -> Tuple[Optional[str], List[str]]:
        """Resolve primary/fallback models for this stage.

        Default behaviour keeps backward compatibility:
        - Technical / Intel / Strategy stages prefer the local Ollama model when configured
        - Risk prefers the local debate / critique model when configured
        - Decision prefers the cloud arbiter model when configured
        - Global ``LITELLM_MODEL`` remains the generic fallback chain
        """
        del ctx  # reserved for future context-aware routing

        config = getattr(self.llm_adapter, "_config", None)
        default_model = (getattr(config, "litellm_model", "") or "").strip()
        default_fallbacks = [
            str(model).strip()
            for model in (getattr(config, "litellm_fallback_models", None) or [])
            if str(model).strip()
        ]

        local_model = os.getenv("REBALANCE_LOCAL_MODEL", "").strip()
        debate_model = os.getenv("REBALANCE_DEBATE_MODEL", "").strip()
        cloud_model = os.getenv("REBALANCE_CLOUD_MODEL", "").strip()
        cloud_fallback = os.getenv("REBALANCE_CLOUD_FALLBACK", "").strip()

        def _dedupe(models: List[str], primary: str = "") -> List[str]:
            seen = {primary} if primary else set()
            ordered: List[str] = []
            for model in models:
                candidate = str(model or "").strip()
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                ordered.append(candidate)
            return ordered

        stage_name = self.agent_name or ""
        if stage_name == "decision":
            primary = cloud_model or default_model
            fallbacks = _dedupe(
                [cloud_fallback, default_model, *default_fallbacks, debate_model, local_model],
                primary=primary,
            )
            return (primary or None), fallbacks

        if stage_name == "risk":
            primary = debate_model or local_model or default_model
            fallbacks = _dedupe(
                [local_model, default_model, *default_fallbacks, cloud_model, cloud_fallback],
                primary=primary,
            )
            return (primary or None), fallbacks

        primary = local_model or default_model
        fallbacks = _dedupe(
            [debate_model, default_model, *default_fallbacks, cloud_model, cloud_fallback],
            primary=primary,
        )
        return (primary or None), fallbacks

    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------

    def run(
        self,
        ctx: AgentContext,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> StageResult:
        """Execute this agent and return a :class:`StageResult`.

        Steps:
        1. Build system + user messages.
        2. Optionally inject pre-fetched data from ``ctx.data``.
        3. Delegate to :func:`run_agent_loop`.
        4. Call :meth:`post_process` to produce an opinion.
        5. Append the opinion to ``ctx.opinions``.
        """
        t0 = time.time()
        result = StageResult(stage_name=self.agent_name, status=StageStatus.RUNNING)

        try:
            messages = self._build_messages(ctx)
            model_override, fallback_models_override = self.resolve_model_route(ctx)
            if model_override:
                logger.info(
                    "[%s] model route -> primary=%s fallbacks=%s",
                    self.agent_name,
                    model_override,
                    fallback_models_override,
                )

            # Restrict tools if the agent declares a subset
            registry = self._filtered_registry()

            loop_result: RunLoopResult = run_agent_loop(
                messages=messages,
                tool_registry=registry,
                llm_adapter=self.llm_adapter,
                max_steps=self.max_steps,
                progress_callback=progress_callback,
                model_override=model_override,
                fallback_models_override=fallback_models_override,
            )

            result.tokens_used = loop_result.total_tokens
            result.tool_calls_count = len(loop_result.tool_calls_log)
            result.meta["raw_text"] = loop_result.content
            result.meta["models_used"] = loop_result.models_used
            result.meta["tool_calls_log"] = loop_result.tool_calls_log
            result.meta["model_route"] = {
                "primary": model_override,
                "fallbacks": fallback_models_override,
            }

            if not loop_result.success:
                result.status = StageStatus.FAILED
                result.error = loop_result.error or "Agent loop did not produce a final answer"
                return result

            # Post-process into structured opinion
            opinion = self.post_process(ctx, loop_result.content)
            if opinion is not None:
                opinion.agent_name = self.agent_name
                self._apply_memory_calibration(ctx, opinion, result)
                ctx.add_opinion(opinion)
                result.opinion = opinion

            result.status = StageStatus.COMPLETED

        except Exception as exc:
            logger.error("[%s] execution failed: %s", self.agent_name, exc, exc_info=True)
            result.status = StageStatus.FAILED
            result.error = str(exc)
        finally:
            result.duration_s = round(time.time() - t0, 2)

        return result

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _build_messages(self, ctx: AgentContext) -> List[Dict[str, Any]]:
        """Assemble the initial messages list for the LLM."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt(ctx)},
        ]

        history = ctx.meta.get("conversation_history")
        if isinstance(history, list):
            for message in history:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = message.get("content")
                if role in {"user", "assistant", "system"} and isinstance(content, str) and content:
                    messages.append({"role": role, "content": content})

        # Inject pre-fetched data as a synthetic assistant context
        cached_data = self._inject_cached_data(ctx)
        if cached_data:
            messages.append({"role": "user", "content": cached_data})
            messages.append({"role": "assistant", "content": "Understood, I have the pre-fetched data. Proceeding with analysis."})

        messages.append({"role": "user", "content": self.build_user_message(ctx)})
        return messages

    def _inject_cached_data(self, ctx: AgentContext) -> str:
        """Build a context string from already-fetched data in ``ctx.data``.

        This avoids redundant tool calls when earlier stages have already
        fetched the data this agent needs.
        """
        import json
        parts: List[str] = []
        for key, value in ctx.data.items():
            if value is not None:
                try:
                    serialised = json.dumps(value, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    serialised = str(value)
                # Cap per-field size to avoid overwhelming the context window
                if len(serialised) > 8000:
                    serialised = serialised[:8000] + "...(truncated)"
                parts.append(f"[Pre-fetched: {key}]\n{serialised}")
        memory_context = self._build_memory_context(ctx)
        if memory_context:
            parts.append(memory_context)
        return "\n\n".join(parts) if parts else ""

    def _filtered_registry(self) -> ToolRegistry:
        """Return a ToolRegistry restricted to ``self.tool_names``.

        If ``tool_names`` is None (default), the full registry is returned.
        """
        if self.tool_names is None:
            return self.tool_registry

        from src.agent.tools.registry import ToolRegistry as TR
        filtered = TR()
        for name in self.tool_names:
            tool_def = self.tool_registry.get(name)
            if tool_def:
                filtered.register(tool_def)
            else:
                logger.warning("[%s] requested tool '%s' not found in registry", self.agent_name, name)
        return filtered

    def _build_memory_context(self, ctx: AgentContext) -> str:
        """Summarise recent analysis history for prompt injection."""
        if not self.memory.enabled or not ctx.stock_code:
            return ""

        entries = self.memory.get_stock_history(ctx.stock_code, limit=3)
        if not entries:
            return ""

        lines = ["[Memory: recent analysis history]"]
        for entry in entries:
            parts = [
                entry.date or "unknown_date",
                f"signal={entry.signal or 'unknown'}",
                f"sentiment={entry.sentiment_score}",
            ]
            if entry.price_at_analysis:
                parts.append(f"price={entry.price_at_analysis}")
            if entry.outcome_5d is not None:
                parts.append(f"outcome_5d={entry.outcome_5d}")
            if entry.outcome_20d is not None:
                parts.append(f"outcome_20d={entry.outcome_20d}")
            if entry.was_correct is not None:
                parts.append(f"was_correct={entry.was_correct}")
            lines.append("- " + ", ".join(parts))
        lines.append("Use this memory as context only; do not copy it verbatim into the final answer.")
        return "\n".join(lines)

    def _apply_memory_calibration(self, ctx: AgentContext, opinion: AgentOpinion, result: StageResult) -> None:
        """Adjust confidence using historical calibration when enabled."""
        if not self.memory.enabled:
            return

        strategy_id = self.agent_name.replace("strategy_", "") if self.agent_name.startswith("strategy_") else None
        calibration = self.memory.get_calibration(
            agent_name=self.agent_name,
            stock_code=ctx.stock_code or None,
            strategy_id=strategy_id,
        )
        if not calibration.calibrated:
            return

        raw_confidence = opinion.confidence
        opinion.confidence = max(0.0, min(1.0, raw_confidence * calibration.calibration_factor))
        result.meta["memory_calibration"] = {
            "raw_confidence": raw_confidence,
            "calibrated_confidence": opinion.confidence,
            "factor": calibration.calibration_factor,
            "samples": calibration.total_samples,
        }
