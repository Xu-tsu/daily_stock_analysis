"""
rebalance_engine.py — 多 Agent 调仓决策引擎
放在项目根目录，与 main.py / analyzer_service.py 同级

架构（辩论模式）:
  Agent 1-3:  本地 Ollama Qwen（数据分析苦力）
  Agent 4a:   本地 Ollama Qwen（激进派 — 提出调仓方案）
  Agent 4b:   本地 Ollama DeepSeek-R1（保守派 — 质疑挑刺）
  Agent 4c:   云端 Gemini（仲裁者 — 综合双方意见做最终决策）
  蒸馏采集:   每次调用的 prompt+response 自动存为训练样本

LLM 调用方式:
  直接使用 litellm.completion()，这是项目底层实际使用的库。
  根据 .env 中的 LITELLM_MODEL / LITELLM_FALLBACK_MODELS 路由。
"""
import json, logging, os, time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── 导入项目已有模块 ──
from src.config import Config, get_config
from src.core.trading_calendar import count_stock_trading_days
from macro_data_collector import collect_full_macro_data
from portfolio_manager import (
    load_portfolio, update_current_prices, format_rebalance_report,
)
from src.services.trade_sizing_service import annotate_a_share_trade_suggestions

# 蒸馏数据保存目录
DISTILL_DIR = Path("data/distillation")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM 调用封装（区分本地 / 云端）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_local_llm(prompt: str, agent_name: str = "", return_model: bool = False):
    """
    调用本地 Ollama 模型（Agent 1-3 用）
    使用 REBALANCE_LOCAL_MODEL 环境变量，不碰主项目的 LITELLM_MODEL
    """
    import litellm
    model = os.getenv("REBALANCE_LOCAL_MODEL", "ollama/qwen2.5:14b-instruct-q4_K_M")
    try:
        logger.debug(f"[{agent_name}] 调用本地模型: {model}")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=180,
            temperature=0.3,  # 低温度，结果更稳定
        )
        result = response.choices[0].message.content
        logger.debug(f"[{agent_name}] 本地模型返回 {len(result)} 字符")
        if return_model:
            return result, model
        return result
    except Exception as e:
        logger.error(f"[{agent_name}] 本地 LLM 调用失败: {e}")
        if return_model:
            return "{}", model
        return "{}"


def _call_debate_llm(prompt: str, agent_name: str = "", return_model: bool = False):
    """
    调用本地第二模型（DeepSeek-R1，辩论用）
    使用 REBALANCE_DEBATE_MODEL 环境变量
    """
    import litellm
    model = os.getenv("REBALANCE_DEBATE_MODEL", "ollama/deepseek-r1:14b")
    try:
        logger.info(f"[{agent_name}] 调用辩论模型: {model}")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=300,
            temperature=0.4,
        )
        result = response.choices[0].message.content
        # DeepSeek-R1 会输出 <think>...</think> 思考过程，提取最终回答
        if "<think>" in result and "</think>" in result:
            think_part = result[result.index("<think>"):result.index("</think>") + len("</think>")]
            final_part = result[result.index("</think>") + len("</think>"):].strip()
            logger.info(f"[{agent_name}] DeepSeek思考链: {len(think_part)}字, 最终回答: {len(final_part)}字")
            if final_part:
                result = final_part
        logger.info(f"[{agent_name}] 辩论模型返回 {len(result)} 字符")
        if return_model:
            return result, model
        return result
    except Exception as e:
        logger.error(f"[{agent_name}] 辩论模型调用失败: {e}")
        if return_model:
            return "{}", model
        return "{}"


def _call_cloud_llm(prompt: str, agent_name: str = "", return_model: bool = False):
    """
    调用云端强模型（Agent 4c 仲裁用）
    降级链：REBALANCE_CLOUD_MODEL → REBALANCE_CLOUD_FALLBACK → LITELLM_MODEL → 本地DeepSeek
    """
    import litellm

    # 云端模型候选列表（按优先级）
    candidates = []
    primary = os.getenv("REBALANCE_CLOUD_MODEL")
    if primary:
        candidates.append(primary)
    fallback = os.getenv("REBALANCE_CLOUD_FALLBACK")
    if fallback:
        candidates.append(fallback)
    default = os.getenv("LITELLM_MODEL")
    if default and default not in candidates:
        candidates.append(default)

    if not candidates:
        logger.warning(f"[{agent_name}] 未配置任何云端模型，回退到本地辩论模型")
        return _call_debate_llm(
            prompt,
            f"{agent_name}_本地回退",
            return_model=return_model,
        )

    # 逐个尝试云端模型
    last_error = None
    for i, model in enumerate(candidates):
        try:
            tag = "主力" if i == 0 else f"备用{i}"
            logger.info(f"[{agent_name}] 调用云端模型({tag}): {model}")
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=300,
                temperature=0.3,
                num_retries=1,
            )
            result = response.choices[0].message.content
            logger.info(f"[{agent_name}] 云端模型({tag})返回 {len(result)} 字符")
            if return_model:
                return result, model
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"[{agent_name}] 云端{tag} {model} 失败: {e}")
            continue

    # 所有云端都挂了 → 本地 DeepSeek 兜底
    logger.error(
        f"[{agent_name}] 所有云端模型({len(candidates)}个)均失败，"
        f"最后错误: {last_error}，回退到本地辩论模型"
    )
    return _call_debate_llm(
        prompt,
        f"{agent_name}_本地回退",
        return_model=return_model,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 蒸馏数据采集 — 自动积累微调训练样本
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _save_distillation_sample(
    agent_name: str,
    prompt: str,
    response: str,
    parsed_json: dict,
    metadata: dict = None,
):
    """
    保存云端模型的 prompt+response 作为蒸馏训练样本

    每条样本保存为一个 JSONL 行，格式兼容常见微调框架:
    {
      "instruction": "...(system prompt 可选)",
      "input": "...(用户 prompt)",
      "output": "...(模型 response)",
      "agent": "agent4_rebalance",
      "timestamp": "2026-03-28 18:00:00",
      "quality_score": null  (后续可手动标注)
    }

    积累到 500-1000 条后，可用 LLaMA-Factory / Unsloth 等工具
    对本地 Qwen2.5-14B 做 QLoRA 微调。
    """
    try:
        DISTILL_DIR.mkdir(parents=True, exist_ok=True)

        # 按月份分文件，方便管理
        month_str = datetime.now().strftime("%Y%m")
        filepath = DISTILL_DIR / f"distill_{month_str}.jsonl"

        sample = {
            "instruction": "你是一位专业的A股投资组合管理人。请根据提供的分析数据给出调仓建议。",
            "input": prompt,
            "output": response,
            "agent": agent_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cloud_model": os.getenv("REBALANCE_CLOUD_MODEL")
                          or os.getenv("LITELLM_MODEL")
                          or "unknown",
            "parsed_success": bool(parsed_json),
            "quality_score": None,  # 后续手动标注：对比实际走势评分
        }
        if metadata:
            sample["metadata"] = metadata

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # 统计当前样本数
        total = sum(
            1 for p in DISTILL_DIR.glob("distill_*.jsonl")
            for _ in open(p, encoding="utf-8")
        )
        logger.info(f"[蒸馏] 样本已保存到 {filepath.name}，累计 {total} 条")
        if total >= 500:
            logger.info("[蒸馏] 已积累 500+ 条样本，可以考虑开始微调了！")

    except Exception as e:
        logger.warning(f"[蒸馏] 保存样本失败（不影响主流程）: {e}")


def _save_agent_local_sample(
    agent_name: str,
    prompt: str,
    response: str,
    parsed_json: dict,
):
    """
    也保存本地 Agent 的数据（用于分析本地模型的表现，以及对比蒸馏效果）
    保存到单独的文件，不混入云端蒸馏样本
    """
    try:
        DISTILL_DIR.mkdir(parents=True, exist_ok=True)
        month_str = datetime.now().strftime("%Y%m")
        filepath = DISTILL_DIR / f"local_{month_str}.jsonl"

        sample = {
            "input": prompt,
            "output": response,
            "agent": agent_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "local_model": os.getenv("REBALANCE_LOCAL_MODEL", "unknown"),
            "parsed_success": bool(parsed_json),
        }

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 本地样本保存失败不重要


def _parse_llm_json(response: str) -> dict:
    """安全解析 LLM 返回的 JSON"""
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except:
                pass
        logger.error(f"JSON 解析失败，原始响应前300字: {text[:300]}...")
        return {}


def _truncate_discussion_text(value, limit: int = 140) -> str:
    """压缩日志/报告里的讨论文本，避免单行过长。"""
    text = str(value or "").replace("\n", " ").strip()
    text = " ".join(text.split())
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _action_to_cn(action: str) -> str:
    return {
        "buy": "加仓",
        "hold": "持有",
        "reduce": "减仓",
        "sell": "清仓",
    }.get(action or "", action or "待定")


def _safe_numeric(value, default: float = 999.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summarize_actions(actions: list, limit: int = 4) -> str:
    if not actions:
        return "未生成个股操作建议"

    counts = {"buy": 0, "hold": 0, "reduce": 0, "sell": 0}
    for item in actions:
        action = item.get("action", "")
        if action in counts:
            counts[action] += 1

    parts = []
    if counts["sell"]:
        parts.append(f"清仓{counts['sell']}只")
    if counts["reduce"]:
        parts.append(f"减仓{counts['reduce']}只")
    if counts["hold"]:
        parts.append(f"持有{counts['hold']}只")
    if counts["buy"]:
        parts.append(f"加仓{counts['buy']}只")
    if not parts:
        parts.append(f"共{len(actions)}个动作")

    focus = []
    for item in actions[:limit]:
        name = item.get("name") or item.get("code") or "未知标的"
        focus.append(f"{name}{_action_to_cn(item.get('action', ''))}")

    if focus:
        parts.append("重点: " + "、".join(focus))
    return "；".join(parts)


def _summarize_holdings_ratings(holdings_ratings: list, limit: int = 4) -> str:
    if not holdings_ratings:
        return "持仓扫描未返回有效评级"

    counts = {}
    for item in holdings_ratings:
        rating = item.get("rating", "待定")
        counts[rating] = counts.get(rating, 0) + 1

    count_text = "，".join(f"{rating}{count}只" for rating, count in counts.items())
    priority = {"清仓": 0, "减仓": 1, "持有": 2, "加仓": 3}
    sorted_items = sorted(
        holdings_ratings,
        key=lambda item: (
            priority.get(item.get("rating", ""), 9),
            _safe_numeric(item.get("score", 999)),
        ),
    )

    focus = []
    for item in sorted_items[:limit]:
        name = item.get("name") or item.get("code") or "未知标的"
        rating = item.get("rating", "待定")
        score = item.get("score")
        score_text = f"({score}分)" if score not in (None, "") else ""
        focus.append(f"{name}{rating}{score_text}")

    if focus:
        return f"{count_text}；重点: {'、'.join(focus)}"
    return count_text


def _extract_disagreements(critique: dict, limit: int = 4) -> list:
    if not critique:
        return []

    disagreements = []
    for item in critique.get("position_disagreements", [])[:limit]:
        name = item.get("name") or item.get("code") or "未知标的"
        original = item.get("original_action") or "原方案"
        suggestion = item.get("my_suggestion") or "调整建议"
        reason = _truncate_discussion_text(item.get("reason", ""), 70)
        text = f"{name}: {original} → {suggestion}"
        if reason:
            text += f"（{reason}）"
        disagreements.append(text)

    if disagreements:
        return disagreements

    for issue in critique.get("critical_issues", [])[:limit]:
        text = _truncate_discussion_text(issue, 90)
        if text:
            disagreements.append(text)
    return disagreements


def _build_rebalance_discussion(
    *,
    market_judge: dict,
    sector_judge: dict,
    holdings_ratings: list,
    proposal: dict,
    critique: dict,
    rebalance: dict,
    debate_mode: str,
    market_model: str,
    sector_model: str,
    holdings_model: str,
    proposal_model: str,
    critique_model: str,
    arbiter_model: str,
) -> dict:
    """把调仓链路中的多模型观点整理成可展示的讨论轨迹。"""
    rounds = []

    market_signal = market_judge.get("market_stage") or market_judge.get("position_advice") or "待确认"
    if market_judge:
        if market_judge.get("position_advice"):
            market_signal = f"{market_signal} | 仓位{market_judge.get('position_advice')}"
        rounds.append(
            {
                "agent_label": "Agent1 大盘研判",
                "role_label": "宏观/指数",
                "model": market_model or "unknown",
                "signal_label": _truncate_discussion_text(market_signal, 80),
                "reasoning": _truncate_discussion_text(
                    market_judge.get("summary")
                    or "；".join(market_judge.get("key_signals", [])[:3]),
                    160,
                ),
            }
        )

    if sector_judge:
        hot_sectors = "、".join(sector_judge.get("hot_sectors", [])[:3])
        sector_signal = hot_sectors and f"热点偏向 {hot_sectors}" or sector_judge.get("rotation_direction", "")
        rounds.append(
            {
                "agent_label": "Agent2 板块轮动",
                "role_label": "题材/资金",
                "model": sector_model or "unknown",
                "signal_label": _truncate_discussion_text(sector_signal or "板块轮动待确认", 80),
                "reasoning": _truncate_discussion_text(
                    sector_judge.get("summary")
                    or "；".join(
                        f"{item.get('sector', '未知')}{item.get('status', '')}"
                        for item in sector_judge.get("holding_sector_assessment", [])[:3]
                    ),
                    160,
                ),
            }
        )

    if holdings_ratings:
        rounds.append(
            {
                "agent_label": "Agent3 持仓扫描",
                "role_label": "个股扫描",
                "model": holdings_model or "unknown",
                "signal_label": f"完成{len(holdings_ratings)}只持仓评级",
                "reasoning": _truncate_discussion_text(
                    _summarize_holdings_ratings(holdings_ratings),
                    180,
                ),
            }
        )

    if proposal:
        proposal_reason = []
        if proposal.get("market_assessment"):
            proposal_reason.append(proposal.get("market_assessment"))
        if proposal.get("sector_assessment"):
            proposal_reason.append(proposal.get("sector_assessment"))
        proposal_reason.append(_summarize_actions(proposal.get("actions", [])))
        rounds.append(
            {
                "agent_label": "Agent4a 激进派提案",
                "role_label": "进攻方案",
                "model": proposal_model or "unknown",
                "signal_label": _truncate_discussion_text(
                    proposal.get("overall_position_advice") or "已给出提案",
                    80,
                ),
                "reasoning": _truncate_discussion_text("；".join(filter(None, proposal_reason)), 180),
            }
        )

    disagreements = _extract_disagreements(critique)
    if critique:
        critique_reason = []
        if critique.get("critical_issues"):
            critique_reason.append(
                "关键问题: " + "；".join(
                    _truncate_discussion_text(item, 60)
                    for item in critique.get("critical_issues", [])[:2]
                )
            )
        if critique.get("warnings"):
            critique_reason.append(
                "警告: " + "；".join(
                    _truncate_discussion_text(item, 60)
                    for item in critique.get("warnings", [])[:2]
                )
            )
        if disagreements:
            critique_reason.append("分歧: " + "；".join(disagreements[:2]))
        rounds.append(
            {
                "agent_label": "Agent4b 保守派质疑",
                "role_label": "风控审查",
                "model": critique_model or "unknown",
                "signal_label": f"方案评分 {critique.get('overall_assessment', 'N/A')}/10",
                "reasoning": _truncate_discussion_text("；".join(filter(None, critique_reason)), 180),
            }
        )

    final_label = {
        "full_debate": "Agent4c 云端仲裁",
        "local_merge": "本地合并裁决",
        "proposal_only": "硬规则过滤",
        "single_fallback": "云端单模型裁决",
        "rules_only": "规则引擎兜底",
    }.get(debate_mode, "最终调仓结论")
    final_reason = []
    if rebalance.get("market_assessment"):
        final_reason.append(rebalance.get("market_assessment"))
    if rebalance.get("sector_assessment"):
        final_reason.append(rebalance.get("sector_assessment"))
    if rebalance.get("debate_summary"):
        final_reason.append(rebalance.get("debate_summary"))
    final_reason.append(_summarize_actions(rebalance.get("actions", [])))
    rounds.append(
        {
            "agent_label": final_label,
            "role_label": "最终裁决",
            "model": arbiter_model or ("rules_only" if debate_mode == "rules_only" else "local"),
            "signal_label": _truncate_discussion_text(
                rebalance.get("overall_position_advice") or "已生成最终调仓建议",
                80,
            ),
            "reasoning": _truncate_discussion_text("；".join(filter(None, final_reason)), 200),
        }
    )

    summary = rebalance.get("debate_summary") or ""
    if not summary:
        if disagreements:
            summary = f"激进派先给出调仓草案，保守派提出{len(disagreements)}处关键分歧，最终按{final_label}收敛。"
        elif proposal:
            summary = f"多模型已完成调仓讨论，最终由{final_label}输出结论。"
        else:
            summary = "本次调仓建议由降级链路生成，缺少完整辩论记录。"

    return {
        "summary": _truncate_discussion_text(summary, 180),
        "debate_mode": debate_mode,
        "rounds": rounds,
        "disagreements": disagreements,
    }


def _log_rebalance_discussion(discussion: dict) -> None:
    """把多模型讨论轨迹打进日志，方便盘中回看。"""
    if not discussion:
        return

    summary = discussion.get("summary")
    if summary:
        logger.info("[Discussion] %s", summary)

    rounds = discussion.get("rounds", [])
    total = len(rounds)
    for idx, item in enumerate(rounds, start=1):
        logger.info(
            "[Discussion][%s/%s] %s | role=%s | model=%s | signal=%s",
            idx,
            total,
            item.get("agent_label", "Agent"),
            item.get("role_label", "未知"),
            item.get("model", "unknown"),
            item.get("signal_label", "N/A"),
        )
        reasoning = item.get("reasoning")
        if reasoning:
            logger.info("[Discussion][%s/%s] reasoning: %s", idx, total, reasoning)

    for item in discussion.get("disagreements", [])[:5]:
        logger.info("[Discussion][分歧] %s", item)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 降级处理函数（LLM全挂时的兜底策略）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _merge_proposal_and_critique(proposal: dict, critique: dict) -> dict:
    """云端仲裁失败时，本地合并激进派方案和保守派质疑。
    策略：采纳激进派方案，但接受保守派的所有 critical_issues 修正。
    """
    result = dict(proposal)
    result["debate_summary"] = "云端仲裁不可用，本地自动合并：采纳激进派方案，保守派关键修正已注入"

    # 如果保守派对某只股有不同意见，优先采纳保守派
    disagreements = {
        d["code"]: d for d in critique.get("position_disagreements", [])
        if d.get("code")
    }
    if disagreements:
        for action in result.get("actions", []):
            code = action.get("code", "")
            if code in disagreements:
                d = disagreements[code]
                # 保守派建议更谨慎的操作（如 hold→reduce, hold→sell）
                action["reason"] = (
                    f"[激进派] {action.get('reason', '')} "
                    f"[保守派修正] {d.get('reason', '')}"
                )
                # 如果保守派说要卖而激进派说持有，偏向保守
                conservative_actions = {"sell": 4, "reduce": 3, "hold": 2, "buy": 1}
                orig_weight = conservative_actions.get(action.get("action", "hold"), 2)
                # 从 my_suggestion 提取动作
                suggestion = d.get("my_suggestion", "")
                if "清仓" in suggestion or "卖出" in suggestion:
                    new_weight = 4
                elif "减仓" in suggestion:
                    new_weight = 3
                elif "持有" in suggestion:
                    new_weight = 2
                else:
                    new_weight = orig_weight
                if new_weight > orig_weight:
                    action["action"] = {4: "sell", 3: "reduce", 2: "hold", 1: "buy"}[new_weight]
                    action["detail"] = f"保守派修正: {suggestion}"

    return result


def _apply_hard_rules(proposal: dict, portfolio: dict) -> dict:
    """只有激进派方案、保守派挂了时，用硬规则过滤明显违规的建议。"""
    result = dict(proposal)
    result["debate_summary"] = "保守派审查不可用，已用硬规则自动风控过滤"

    for action in result.get("actions", []):
        code = action.get("code", "")
        # 从portfolio找到对应持仓
        holding = None
        for h in portfolio.get("holdings", []):
            if h["code"] == code:
                holding = h
                break
        if not holding:
            continue

        pnl = holding.get("pnl_pct", 0)
        sellable = holding.get("sellable_shares", holding.get("shares", 0))

        # 硬规则1：亏损>5%必须清仓
        if pnl <= -8.0 and action.get("action") not in ("sell",):
            action["action"] = "sell"
            action["detail"] = f"硬规则风控: 亏损{pnl}%超5%止损线，强制清仓"
            action["reason"] = "止损5%硬规则触发"

        # 硬规则2：卖出数不能超过可卖余额
            action["detail"] = f"自适应风控: 亏损{pnl}%已跌破-8%强制退出线，执行清仓"
            action["reason"] = "深度亏损超出容错区间，优先保护本金"
        elif pnl <= -5.0 and action.get("action") in ("buy", "hold"):
            action["action"] = "reduce"
            action["detail"] = f"自适应风控: 亏损{pnl}%已到风险复核线，先降到观察仓"
            action["reason"] = "5%亏损先降风险，再看是否有板块回流确认"

            action["detail"] = f"自适应风控: 亏损{pnl}%已到-5%风险复核线，先降到观察仓"
            action["reason"] = "先降风险，再观察是否出现板块回流和资金承接"
        if action.get("action") in ("sell", "reduce") and sellable == 0:
            action["action"] = "hold"
            action["detail"] = f"T+1约束: 今天无可卖余额（全部为今日买入），只能明天操作"

    return result


def _generate_rules_only_advice(portfolio: dict) -> dict:
    """所有模型全挂了，纯规则引擎兜底——只做止损和超期清仓，不做新买入。"""
    from datetime import datetime
    today = datetime.now()
    actions = []

    for h in portfolio.get("holdings", []):
        pnl = h.get("pnl_pct", 0)
        code = h["code"]
        name = h.get("name", code)
        sellable = h.get("sellable_shares", h.get("shares", 0))

        # 计算持仓天数
        hold_days = (
            count_stock_trading_days(
                h.get("code", ""),
                h.get("buy_date", ""),
                today,
                default_market="cn",
            )
            if h.get("buy_date")
            else 0
        ) or 0

        action_item = {
            "code": code, "name": name,
            "target_sell_price": None, "stop_loss_price": None,
            "sell_timing": "模型不可用，仅硬规则判断",
        }

        if pnl <= -8.0 and sellable > 0:
            action_item.update({
                "action": "sell", "ratio": "清仓",
                "detail": f"硬止损触发: 亏损{pnl:.1f}%超5%，可卖{sellable}股",
                "reason": "所有模型不可用，纯规则引擎：止损5%强制清仓",
            })
        elif pnl <= -5.0 and sellable > 0:
            action_item.update({
                "action": "reduce", "ratio": "减仓50%",
                "detail": f"风险复核线触发: 亏损{pnl:.1f}%已到5%附近，可卖{sellable}股",
                "reason": "所有模型不可用，纯规则引擎：先降到观察仓，不再机械一刀切",
            })
        elif hold_days >= 7 and pnl < 5.0 and sellable > 0:
            action_item.update({
                "action": "sell", "ratio": "清仓",
                "detail": f"超期清仓: 持仓{hold_days}个交易日，盈利{pnl:.1f}%不足5%，可卖{sellable}股",
                "reason": "所有模型不可用，纯规则引擎：超7天+盈利不足清仓",
            })
        else:
            action_item.update({
                "action": "hold", "ratio": "维持",
                "detail": f"持仓{hold_days}个交易日，盈亏{pnl:.1f}%，暂无触发条件",
                "reason": "所有模型不可用，无触发止损/超期规则，默认持有",
            })
        actions.append(action_item)

    return {
        "overall_position_advice": "模型不可用，维持现有仓位，仅执行止损和超期清仓",
        "market_assessment": "模型不可用，无法判断大盘",
        "sector_assessment": "模型不可用，无法判断板块",
        "debate_summary": "⚠️ 所有AI模型均不可用，本报告由纯规则引擎生成，仅包含止损和超期清仓建议，不包含新买入建议",
        "actions": actions,
        "new_candidates": [],
        "risk_warning": "所有AI模型不可用！本报告仅基于硬规则（5%止损+7天超期），不包含趋势分析和换股建议，请谨慎参考。",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt 模板
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT_MARKET_JUDGE = """你是一位专业的A股宏观策略分析师。

## 当前大盘数据
{index_signals}

## 全市场资金流向（近10日）
{market_fund_flow}

## 北向资金
{northbound}

## 两融余额（杠杆资金情绪）
{margin_data}

## 涨跌家数（市场广度）
{market_breadth}

## 重要快讯（含特朗普言论、政策动向）
{sensitive_news}

## 特朗普最新动态
{trump_news}

## 你的任务
根据以上数据，判断当前A股大盘所处阶段，并给出仓位建议。
特别注意：如果特朗普有涉及关税、中国、贸易战的最新言论，需重点评估其对A股的冲击。

请严格按以下 JSON 格式回复（不要加 markdown 代码块）：
{{
  "market_stage": "趋势上涨 / 高位震荡 / 震荡筑底 / 趋势下跌 / 恐慌出清",
  "confidence": "高/中/低",
  "position_advice": "建议仓位比例，如 0.7 表示 70%",
  "key_signals": ["信号1", "信号2", "信号3"],
  "risk_factors": ["风险1", "风险2"],
  "trump_impact": "特朗普言论对A股的影响评估（无/轻微/中等/重大）",
  "summary": "一句话总结当前大盘状态和操作建议"
}}"""

PROMPT_SECTOR_ROTATION = """你是一位专业的A股行业研究分析师。

## 大盘研判结论（上一步得出）
{market_judge}

## 行业板块资金流向
{sector_data}

## 热门概念板块
{concept_data}

## 我的持仓所在板块
{holding_sectors}

## 你的任务
分析板块轮动方向，判断我持仓所在板块的强弱。

请严格按以下 JSON 格式回复：
{{
  "hot_sectors": ["正在被资金追捧的板块1", "板块2", "板块3"],
  "cold_sectors": ["资金正在撤退的板块1", "板块2"],
  "rotation_direction": "资金轮动方向的一句话描述",
  "holding_sector_assessment": [
    {{"sector": "板块名", "status": "强势/中性/弱势", "reason": "原因"}}
  ],
  "summary": "一句话总结板块轮动态势"
}}"""

PROMPT_HOLDING_SCAN = """你是一位专业的A股短线交易顾问，专注低价小盘题材股。

## 基于真实交易数据的评级标准

### 核心风控规则（必须严格执行）
1. 亏损超5%：必须"清仓"，无例外（用户历史数据：不止损→平均亏12.67%）
2. 持仓超3个交易日且盈利不足5%：建议"减仓"或"清仓"（历史：超3天胜率骤降至56%）
3. 盈利超8%：建议"减仓"一半锁定利润（用户平均盈利仅5.12%，8%已是超额）
4. 禁止对亏损中的股票建议"加仓"（历史：补仓亏损股胜率仅17%）

### T+1风险评估
- 如果该股今日已大涨(>3%)，不建议加仓（明天大概率回调，T+1无法当天止损）
- 评估该股是否处于"追高买入"状态：远高于MA5 = 高风险

### 评级标准
- 加仓条件：板块资金持续流入 + 个股缩量回踩MA5支撑 + 乖离率<2% + 当日涨幅<3%
- 持有条件：趋势未破 + 板块未转弱 + 亏损在3%以内 + 持仓不超过3天
- 减仓条件：板块资金流出 + 放量下跌破MA5 或 持仓超3天盈利不足 或 盈利超8%
- 清仓条件：亏损超5% 或 连续3日主力净流出 或 板块崩塌 或 持仓超7天

## 大盘研判
{market_judge}

## 板块轮动
{sector_judge}

## 该股基本分析（来自系统已有分析）
{stock_analysis}

## 该股资金流向（近10日，含超大单/大单/中单/小单明细）
{fund_flow}

## 该股北向资金持仓变化
{northbound_holding}

## 千股千评
{comment}

## 该股最新新闻
{stock_news}

## 你的任务
综合以上所有信息，对 {name}({code}) 给出操作评级。

请严格按以下 JSON 格式回复：
{{
  "code": "{code}",
  "name": "{name}",
  "rating": "加仓 / 持有 / 减仓 / 清仓",
  "score": 0-100,
  "reasons": ["原因1", "原因2", "原因3"],
  "risk_level": "低/中/高",
  "key_price_levels": {{
    "support": "支撑位",
    "resistance": "压力位",
    "stop_loss": "止损位"
  }}
}}"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 辩论 Prompt 模板
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT_DEBATE_CRITIQUE = """你是一位严谨保守的A股风控专家，你的工作是审查另一位交易员的调仓方案，找出其中的漏洞和风险。

## 交易员的调仓方案
{proposal}

## 当前持仓明细（注意：sellable_shares 是T+1可卖余额，今天买入的股不能卖）
{portfolio}

## 市场数据摘要
- 大盘研判: {market_summary}
- 板块轮动: {sector_summary}

## 你的任务：挑刺和质疑
请从以下角度严格审查这个方案：

1. **T+1风险**：有没有追高买入的建议？买入价是否远离MA5支撑？
2. **止损执行**：亏损超5%的是否建议了清仓？有没有心存侥幸？
3. **持仓天数**：是否死板套用天数规则？应该看趋势——
   - 如果一只股虽然持仓久但趋势在好转（亏损收窄、板块走强），不应无脑清仓
   - 如果一只股持仓短但趋势恶化，应该提前跑
4. **可卖余额**：建议卖出的股数是否超过了 sellable_shares？
5. **补仓禁忌**：有没有建议对亏损股加仓？
6. **换股质量**：推荐的新股是否真的符合低价小盘+缩量回踩+板块资金流入？
7. **卖点合理性**：目标卖出价和止损价是否合理？止盈太贪或止损太松都不行

请严格按以下 JSON 格式回复：
{{
  "overall_assessment": "该方案整体质量评分 1-10",
  "critical_issues": ["必须修正的严重问题1", "问题2"],
  "warnings": ["需要注意但不致命的问题1", "问题2"],
  "suggestions": ["改进建议1", "建议2"],
  "position_disagreements": [
    {{
      "code": "股票代码",
      "name": "股票名称",
      "original_action": "原方案建议",
      "my_suggestion": "我认为应该...",
      "reason": "原因"
    }}
  ]
}}"""

PROMPT_DEBATE_ARBITRATE = """你是最终仲裁者，需要综合激进派交易员和保守派风控专家的意见，做出最终调仓决策。

## 激进派交易员的方案
{proposal}

## 保守派风控专家的质疑
{critique}

## 当前持仓明细（注意：sellable_shares 是T+1可卖余额）
{portfolio}

## 今日主力净流入的低价热门股（换股必须从这里选）
{hot_picks}

## 仲裁规则
1. 如果双方对某只股意见一致 → 直接采纳
2. 如果有分歧 → 偏向保守派（风控优先），但如果保守派的理由是纯粹死套规则而忽略趋势，则偏向激进派
3. 止损5%红线不可商量 → 如果激进派想保留亏损超5%的股票，必须否决
4. 卖出股数不能超过 sellable_shares（T+1约束）
5. 每只股必须给出 target_sell_price 和 stop_loss_price
6. 换股只能从热门股列表中选，不能自己编造

请严格按以下 JSON 格式回复：
{{
  "overall_position_advice": "当前仓位X%，建议调整至Y%",
  "market_assessment": "一句话大盘判断",
  "sector_assessment": "一句话板块判断",
  "debate_summary": "一句话总结辩论过程中的关键分歧和最终裁决理由",
  "actions": [
    {{
      "code": "600519",
      "name": "贵州茅台",
      "action": "hold/buy/reduce/sell",
      "ratio": "维持当前仓位 / 加仓X元 / 减仓50% / 清仓",
      "detail": "具体操作说明（包含可卖股数约束）",
      "reason": "综合理由（引用激进派和保守派的观点）",
      "target_sell_price": 10.5,
      "stop_loss_price": 9.0,
      "sell_timing": "建议在什么条件下卖出"
    }}
  ],
  "new_candidates": [
    {{
      "code": "代码",
      "name": "名称",
      "sector": "所属板块",
      "reason": "推荐理由",
      "target_sell_price": "目标卖出价",
      "stop_loss_price": "止损价",
      "buy_price_range": "建议买入价格区间"
    }}
  ],
  "risk_warning": "整体风险提示"
}}"""

PROMPT_REBALANCE_FINAL = """你是一位经验丰富的A股短线交易员，擅长低价小盘题材股的板块轮动策略。

## 我的交易风格（基于233笔真实交易数据优化，必须严格遵守）

### 核心策略：快进快出，小赚即走，严格止损
- 操作风格：超短线趋势交易，持股周期1-3天（历史数据：1天内胜率93%，超3天暴降至56%）
- 选股偏好：10元以下低价股、流通市值50亿以下小盘股、有热门题材概念的
- 买入条件：缩量回踩MA5支撑 + 板块资金持续流入 + 乖离率<2% + 当日涨幅<3%
- 卖出条件：盈利5-8%止盈 / 亏损5%止损 / 持仓超3天 / 板块转弱
- 绝对禁止：不买大盘蓝筹白马股，不做价值投资，不补仓亏损股

### A股T+1追高禁令（最重要的规则！）
- 绝对禁止买入当日涨幅超5%的股票
- 当日涨幅超3%的股票需要降低仓位50%
- 理由：A股T+1，追高买入后当天无法卖出，次日砸盘会导致巨额亏损
- 我的血泪教训：追高买入的交易中，雪人集团-28%、招金黄金-28%、中国卫通-19%
- 正确做法：买在回调支撑位（贴近MA5），而非追涨途中

### T+1 可卖余额约束
- 持仓明细中 sellable_shares 表示今天能实际卖出的股数（今天买入的不能卖）
- 建议卖出时不能超过 sellable_shares，否则操作无法执行
- 如果 sellable_shares < shares，说明有今天刚买入的部分，这部分只能明天操作

### 风控硬规则
1. 止损5%：亏损超5%的持仓必须建议清仓，无任何例外
2. 止盈8%：盈利超8%建议减仓一半锁利（我的平均盈利仅5.12%）
3. 持仓天数规则（不是死板的，要结合趋势判断）：
   - 超3天且亏损：建议减仓或清仓
   - 超3天但处于缓慢上涨趋势（沿MA5稳步上攻）：可以继续持有，设移动止损
   - 超7天且盈利不足5%：建议清仓
   - 注意：持仓天数必须从 buy_date 字段准确计算，不要瞎猜
4. 禁止补仓亏损股：浮亏中的股票绝对不能加仓（历史补仓胜率仅17%）
5. 单只仓位不超15%，最多同时持5只
6. 换股方向：必须从当前资金流入的热门题材板块中选回调到位的低价小盘股

## 大盘研判
{market_judge}

## 板块轮动分析
{sector_judge}

## 各持仓股评级
{holdings_ratings}

## 当前持仓明细（注意：sellable_shares 是T+1可卖余额，今天买入的股不能卖）
{portfolio}

## 今日主力净流入的低价热门股（真实数据，换股必须从这里选）
{hot_picks}

## 你的任务
综合所有分析，给出具体的调仓指令。要求：
1. 判断是否需要调整总仓位（当前实际仓位 vs 建议仓位）
2. 对每只持仓股给出 buy/hold/reduce/sell 建议和具体比例
3. 如建议换股，**必须且只能**从上面"低价热门股"列表中选择，禁止自己编造股票代码和名称
4. 给出具体比例（如 "减仓50%" 而非 "适当减仓"）
5. **每只股票必须给出 target_sell_price（目标卖出价）和 stop_loss_price（止损价）**
6. **每只股票必须给出 sell_timing（什么条件下卖出）**，如"盈利5%或跌破MA5卖出"
7. 标注风险等级
8. 持仓天数必须严格按照 buy_date 字段计算到今天的交易日数，不要编造
9. 如果大盘极弱，可以建议空仓等待，但一旦有板块异动要给出抄底候选

请严格按以下 JSON 格式回复：
{{
  "overall_position_advice": "当前仓位X%，建议调整至Y%",
  "market_assessment": "一句话大盘判断",
  "sector_assessment": "一句话板块判断",
  "actions": [
    {{
      "code": "600519",
      "name": "贵州茅台",
      "action": "hold/buy/reduce/sell",
      "ratio": "维持当前仓位 / 加仓X元 / 减仓50% / 清仓",
      "detail": "具体操作说明",
      "reason": "综合理由",
      "target_sell_price": 10.5,
      "stop_loss_price": 9.0,
      "sell_timing": "建议在什么条件下卖出（如：盈利5%或跌破MA5时卖出）"
    }}
  ],
  "new_candidates": [
    {{
      "code": "代码",
      "name": "名称",
      "sector": "所属板块",
      "reason": "推荐理由（必须是低价小盘题材股）",
      "target_sell_price": "目标卖出价",
      "stop_loss_price": "止损价",
      "buy_price_range": "建议买入价格区间"
    }}
  ],
  "risk_warning": "整体风险提示"
}}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_rebalance_analysis(config: Config = None) -> dict:
    """
    执行完整的多 Agent 调仓分析流程
    返回调仓建议 dict
    """
    if config is None:
        config = get_config()

    start_time = time.time()
    local_model = os.getenv("REBALANCE_LOCAL_MODEL", "unknown")
    debate_model = os.getenv("REBALANCE_DEBATE_MODEL", "unknown")
    logger.info("=" * 60)
    logger.info("开始执行多Agent调仓分析...")
    logger.info(f"本地模型: {local_model}")
    logger.info(f"辩论模型: {debate_model}")
    cloud_model = os.getenv("REBALANCE_CLOUD_MODEL") or os.getenv("LITELLM_MODEL") or "未配置"
    logger.info(f"云端模型: {cloud_model}")
    logger.info("=" * 60)

    # ── Step 0: 加载持仓 + 从trade_log同步校准buy_date ──
    portfolio = load_portfolio()
    try:
        from portfolio_manager import sync_portfolio_from_trades
        portfolio = sync_portfolio_from_trades(portfolio)
    except Exception as e:
        logger.warning(f"持仓同步失败（不影响主流程）: {e}")
    holding_codes = [h["code"] for h in portfolio.get("holdings", [])]
    holding_sectors = list(set(
        h.get("sector", "未知") for h in portfolio.get("holdings", [])
    ))

    if not holding_codes:
        return {"error": "持仓为空"}

    logger.info(f"当前持仓: {holding_codes}")

    # ── Step 1: 数据采集（纯Python，零token消耗）──
    logger.info("\n[Step 1/5] 采集宏观数据...")
    macro_data = collect_full_macro_data(holding_codes)

    # 更新持仓实时价格
    price_map = {}
    for code, ff in macro_data.get("holdings_fund_flow", {}).items():
        # 优先从 daily 取收盘价
        daily = ff.get("daily", [])
        if daily:
            price_map[code] = daily[-1].get("close", 0)
        # 降级：直接取 price 字段（腾讯行情返回的）
        if code not in price_map or price_map[code] == 0:
            if ff.get("price", 0) > 0:
                price_map[code] = ff["price"]
    # 再用 stock_comments（腾讯行情面板）补充
    for code, comment in macro_data.get("stock_comments", {}).items():
        if code not in price_map or price_map[code] == 0:
            if comment.get("latest_price", 0) > 0:
                price_map[code] = comment["latest_price"]
    portfolio = update_current_prices(portfolio, price_map)

    # ── Step 2: Agent 1 — 大盘研判（本地LLM）──
    logger.info("\n[Step 2/5] Agent 1: 大盘研判（本地模型）...")
    # 提取敏感快讯（只取前5条给LLM，控制token）
    sensitive_news = macro_data.get("cls_telegraph", {}).get("sensitive", [])[:5]
    trump_news = macro_data.get("trump_news", [])[:3]

    prompt_market = PROMPT_MARKET_JUDGE.format(
        index_signals=json.dumps(
            macro_data["index_signals"], ensure_ascii=False, indent=2
        ),
        market_fund_flow=json.dumps(
            macro_data["market_fund_flow"], ensure_ascii=False, indent=2
        ),
        northbound=json.dumps(
            macro_data["northbound"], ensure_ascii=False, indent=2
        ),
        margin_data=json.dumps(
            macro_data.get("margin_data", {}), ensure_ascii=False, indent=2
        ),
        market_breadth=json.dumps(
            macro_data.get("market_breadth", {}), ensure_ascii=False, indent=2
        ),
        sensitive_news=json.dumps(sensitive_news, ensure_ascii=False, indent=2),
        trump_news=json.dumps(trump_news, ensure_ascii=False, indent=2),
    )
    market_judge_raw, market_model_used = _call_local_llm(
        prompt_market,
        "Agent1_大盘",
        return_model=True,
    )
    market_judge = _parse_llm_json(market_judge_raw)
    _save_agent_local_sample("agent1_market", prompt_market, market_judge_raw, market_judge)
    logger.info(
        f"  大盘判断: {market_judge.get('market_stage', 'N/A')} "
        f"| 建议仓位: {market_judge.get('position_advice', 'N/A')}"
    )

    # ── Step 3: Agent 2 — 板块轮动（本地LLM）──
    logger.info("\n[Step 3/5] Agent 2: 板块轮动分析（本地模型）...")
    sector_data = macro_data.get("sector_rotation", {})
    prompt_sector = PROMPT_SECTOR_ROTATION.format(
        market_judge=json.dumps(market_judge, ensure_ascii=False, indent=2),
        sector_data=json.dumps(
            {k: v for k, v in sector_data.items() if "行业" in k},
            ensure_ascii=False, indent=2,
        ),
        concept_data=json.dumps(
            sector_data.get("概念_今日_top10", []),
            ensure_ascii=False, indent=2,
        ),
        holding_sectors=json.dumps(holding_sectors, ensure_ascii=False),
    )
    sector_judge_raw, sector_model_used = _call_local_llm(
        prompt_sector,
        "Agent2_板块",
        return_model=True,
    )
    sector_judge = _parse_llm_json(sector_judge_raw)
    _save_agent_local_sample("agent2_sector", prompt_sector, sector_judge_raw, sector_judge)
    logger.info(f"  热门板块: {sector_judge.get('hot_sectors', [])}")

    # ── Step 4: Agent 3 — 逐只持仓扫描（本地LLM）──
    logger.info("\n[Step 4/5] Agent 3: 持仓个股扫描（本地模型）...")
    holdings_ratings = []

    for h in portfolio.get("holdings", []):
        code = h["code"]
        name = h.get("name", code)
        logger.info(f"  分析 {name}({code})...")

        # 用腾讯行情+同花顺数据替代原项目的 analyze_stock（避免额外LLM调用）
        stock_analysis_text = "暂无已有分析数据"
        comment = macro_data.get("stock_comments", {}).get(code, {})
        if comment:
            stock_analysis_text = (
                f"当前价:{comment.get('latest_price', 0)}, "
                f"涨跌幅:{comment.get('change_pct', 0)}%, "
                f"换手率:{comment.get('turnover_rate', 0)}%, "
                f"PE:{comment.get('pe_ratio', 0)}, "
                f"振幅:{comment.get('amplitude', 0)}%"
            )

        fund_flow = macro_data.get("holdings_fund_flow", {}).get(code, {})
        comment = macro_data.get("stock_comments", {}).get(code, {})
        nb_holding = macro_data.get("northbound_holdings", {}).get(code, {})
        s_news = macro_data.get("stock_news", {}).get(code, [])[:3]

        prompt_holding = PROMPT_HOLDING_SCAN.format(
            market_judge=json.dumps(market_judge, ensure_ascii=False),
            sector_judge=json.dumps(sector_judge, ensure_ascii=False),
            stock_analysis=stock_analysis_text,
            fund_flow=json.dumps(fund_flow, ensure_ascii=False, indent=2),
            northbound_holding=json.dumps(nb_holding, ensure_ascii=False),
            comment=json.dumps(comment, ensure_ascii=False),
            stock_news=json.dumps(s_news, ensure_ascii=False, indent=2),
            code=code, name=name,
        )
        rating_raw, _ = _call_local_llm(
            prompt_holding,
            f"Agent3_{name}",
            return_model=True,
        )
        rating = _parse_llm_json(rating_raw)
        _save_agent_local_sample(f"agent3_{code}", prompt_holding, rating_raw, rating)

        if rating:
            holdings_ratings.append(rating)
            logger.info(
                f"  → {name}: {rating.get('rating', 'N/A')} "
                f"(得分: {rating.get('score', 'N/A')})"
            )
        else:
            logger.warning(f"  → {name}: 分析结果解析失败")

    # ── Step 5: 多模型辩论调仓（激进派 vs 保守派 → 仲裁）──
    logger.info("\n[Step 5/7] 多模型辩论调仓决策...")

    # 真实换股候选（来自同花顺爬虫，不让LLM编造）
    hot_picks = macro_data.get("hot_candidates", [])
    if not hot_picks:
        # 兜底：从板块数据中提取
        for sector_info in sector_data.values():
            if isinstance(sector_info, dict):
                for item in sector_info.get("top_inflow", [])[:3]:
                    hot_picks.append(item)

    # 风控检查：为每只持仓标注风控状态
    risk_alerts_text = ""
    try:
        from risk_control import check_stop_loss, format_risk_alerts, TRADING_RULES_FOR_LLM
        _risk_holdings = [
            {
                "code": hh["code"], "name": hh.get("name", ""),
                "cost_price": hh.get("cost_price", 0),
                "current_price": hh.get("current_price", 0),
                "pnl_pct": hh.get("pnl_pct", 0),
                "buy_date": hh.get("buy_date", ""),
                "shares": hh.get("shares", 0),
            }
            for hh in portfolio.get("holdings", [])
        ]
        alerts = check_stop_loss(_risk_holdings)
        if alerts:
            risk_alerts_text = "\n## 风控预警（必须优先处理）\n" + format_risk_alerts(alerts)
    except Exception as e:
        logger.debug(f"风控检查跳过: {e}")

    # 过滤热门候选：去除当日涨幅>5%的追高股
    filtered_hot = []
    for pick in hot_picks[:15]:
        chg = pick.get("change_pct", pick.get("涨跌幅", 0))
        if isinstance(chg, str):
            try:
                chg = float(chg.replace("%", ""))
            except (ValueError, TypeError):
                chg = 0
        if chg < 5.0:  # 涨幅<5%才推荐（T+1安全）
            filtered_hot.append(pick)
    hot_picks = filtered_hot[:10]

    # 构建持仓JSON（三步辩论共用）
    portfolio_holdings = []
    now = datetime.now()
    for hh in portfolio.get("holdings", []):
        hold_days = None
        if hh.get("buy_date"):
            hold_days = count_stock_trading_days(
                hh.get("code", ""),
                hh.get("buy_date", ""),
                now,
                default_market="cn",
            )
        portfolio_holdings.append(
            {
                "code": hh["code"],
                "name": hh.get("name", ""),
                "shares": hh.get("shares", 0),
                "sellable_shares": hh.get("sellable_shares", hh.get("shares", 0)),
                "cost_price": hh.get("cost_price", 0),
                "current_price": hh.get("current_price", 0),
                "pnl_pct": hh.get("pnl_pct", 0),
                "sector": hh.get("sector", ""),
                "buy_date": hh.get("buy_date", ""),
                "hold_days": hold_days if hold_days is not None else "未知",
            }
        )

    portfolio_json = json.dumps(
        {
            "cash": portfolio.get("cash", 0),
            "total_asset": portfolio.get("total_asset", 0),
            "actual_position_ratio": portfolio.get("actual_position_ratio", 0),
            "today": now.strftime("%Y-%m-%d"),
            "holdings": portfolio_holdings,
        },
        ensure_ascii=False,
        indent=2,
    )
    hot_picks_json = json.dumps(hot_picks[:10], ensure_ascii=False, indent=2)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 5a: 激进派（Qwen）— 提出调仓方案
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("\n[Step 5a/7] 激进派（Qwen）提出调仓方案...")

    prompt_proposal = PROMPT_REBALANCE_FINAL.format(
        market_judge=json.dumps(market_judge, ensure_ascii=False, indent=2),
        sector_judge=json.dumps(sector_judge, ensure_ascii=False, indent=2),
        holdings_ratings=json.dumps(
            holdings_ratings, ensure_ascii=False, indent=2
        ),
        portfolio=portfolio_json,
        hot_picks=hot_picks_json,
    )
    if risk_alerts_text:
        prompt_proposal = prompt_proposal + risk_alerts_text

    proposal_raw, proposal_model_used = _call_local_llm(
        prompt_proposal,
        "Agent4a_激进派",
        return_model=True,
    )
    proposal = _parse_llm_json(proposal_raw)
    if proposal:
        logger.info(f"  激进派方案: {proposal.get('overall_position_advice', 'N/A')}")
        for a in proposal.get("actions", []):
            logger.info(f"    {a.get('name','?')}: {a.get('action','?')} - {a.get('reason','')[:50]}")
    else:
        logger.warning("  激进派方案解析失败，将使用云端直接决策")

    _save_agent_local_sample("agent4a_proposal", prompt_proposal, proposal_raw, proposal)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 5b: 保守派（DeepSeek-R1）— 质疑和挑刺
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("\n[Step 6/7] 保守派（DeepSeek-R1）审查质疑...")

    critique = {}
    critique_raw = "{}"
    critique_model_used = ""
    if proposal:
        prompt_critique = PROMPT_DEBATE_CRITIQUE.format(
            proposal=json.dumps(proposal, ensure_ascii=False, indent=2),
            portfolio=portfolio_json,
            market_summary=market_judge.get("summary", ""),
            sector_summary=sector_judge.get("summary", ""),
        )
        critique_raw, critique_model_used = _call_debate_llm(
            prompt_critique,
            "Agent4b_保守派",
            return_model=True,
        )
        critique = _parse_llm_json(critique_raw)
        if critique:
            score = critique.get("overall_assessment", "?")
            issues = critique.get("critical_issues", [])
            logger.info(f"  保守派评分: {score}/10")
            for issue in issues:
                logger.info(f"    ❌ {issue}")
            for warn in critique.get("warnings", [])[:3]:
                logger.info(f"    ⚠️ {warn}")
            for disagree in critique.get("position_disagreements", []):
                logger.info(
                    f"    🔄 {disagree.get('name','?')}: "
                    f"{disagree.get('original_action','')} → {disagree.get('my_suggestion','')}"
                )
        else:
            logger.warning("  保守派审查解析失败")

    _save_agent_local_sample("agent4b_critique", prompt_critique if proposal else "", critique_raw, critique)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 5c: 仲裁者（Gemini云端）— 最终决策
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("\n[Step 7/7] 仲裁者综合双方意见，最终裁决...")

    debate_mode = "none"
    prompt_used = ""
    arbiter_model_used = ""

    if proposal and critique:
        # ✅ 完整辩论 → 仲裁模式
        debate_mode = "full_debate"
        prompt_arbitrate = PROMPT_DEBATE_ARBITRATE.format(
            proposal=json.dumps(proposal, ensure_ascii=False, indent=2),
            critique=json.dumps(critique, ensure_ascii=False, indent=2),
            portfolio=portfolio_json,
            hot_picks=hot_picks_json,
        )
        if risk_alerts_text:
            prompt_arbitrate = prompt_arbitrate + "\n" + risk_alerts_text
        prompt_used = prompt_arbitrate

        # 仲裁模型降级链：云端 → DeepSeek → Qwen
        rebalance_raw, arbiter_model_used = _call_cloud_llm(
            prompt_arbitrate,
            "Agent4c_仲裁",
            return_model=True,
        )
        rebalance = _parse_llm_json(rebalance_raw)

        if not rebalance or not rebalance.get("actions"):
            logger.warning("[仲裁] 云端仲裁失败或返回空，尝试本地投票合并...")
            debate_mode = "local_merge"
            rebalance = _merge_proposal_and_critique(proposal, critique)
            arbiter_model_used = "local_merge"

    elif proposal:
        # ⚠️ 只有激进派方案，保守派挂了 → 直接用方案但加风控过滤
        debate_mode = "proposal_only"
        logger.warning("  保守派审查失败，使用激进派方案 + 硬规则风控过滤...")
        rebalance = _apply_hard_rules(proposal, portfolio)
        rebalance_raw = json.dumps(rebalance, ensure_ascii=False)
        arbiter_model_used = "hard_rules"

    else:
        # ❌ 全挂了 → 单模型直接决策
        debate_mode = "single_fallback"
        logger.warning("  辩论未完成，退回单模型直接决策...")
        prompt_fallback = PROMPT_REBALANCE_FINAL.format(
            market_judge=json.dumps(market_judge, ensure_ascii=False, indent=2),
            sector_judge=json.dumps(sector_judge, ensure_ascii=False, indent=2),
            holdings_ratings=json.dumps(holdings_ratings, ensure_ascii=False, indent=2),
            portfolio=portfolio_json,
            hot_picks=hot_picks_json,
        )
        if risk_alerts_text:
            prompt_fallback = prompt_fallback + risk_alerts_text
        prompt_used = prompt_fallback
        rebalance_raw, arbiter_model_used = _call_cloud_llm(
            prompt_fallback,
            "Agent4_仲裁_fallback",
            return_model=True,
        )
        rebalance = _parse_llm_json(rebalance_raw)

    # 最终兜底：如果解析全失败，至少给出风控硬规则的建议
    if not rebalance or not rebalance.get("actions"):
        logger.error("[仲裁] 所有模型均失败，启用纯规则引擎兜底...")
        debate_mode = "rules_only"
        rebalance = _generate_rules_only_advice(portfolio)
        rebalance_raw = json.dumps(rebalance, ensure_ascii=False)
        arbiter_model_used = "rules_only"

    logger.info(f"[仲裁] 决策模式: {debate_mode}, 返回长度: {len(rebalance_raw) if isinstance(rebalance_raw, str) else 'N/A'}")

    # 保存蒸馏样本（仅在有实际LLM调用时）
    if isinstance(rebalance_raw, str) and len(rebalance_raw) > 10:
        _save_distillation_sample(
            agent_name="agent4c_arbitrate",
            prompt=prompt_used or "(local merge/rules only)",
            response=rebalance_raw,
            parsed_json=rebalance,
            metadata={
                "holdings_count": len(holding_codes),
                "holding_codes": holding_codes,
                "market_stage": market_judge.get("market_stage", ""),
                "hot_sectors": sector_judge.get("hot_sectors", []),
                "debate_mode": debate_mode,
                "debate_score": critique.get("overall_assessment", "N/A") if critique else "N/A",
            },
        )

    elapsed = round(time.time() - start_time, 1)
    logger.info(f"\n调仓分析完成！耗时 {elapsed} 秒")
    logger.info(
        f"总仓位建议: {rebalance.get('overall_position_advice', 'N/A')}"
    )

    try:
        from risk_control import MAX_SINGLE_POSITION_PCT

        annotated_actions, annotated_candidates, execution_profile = annotate_a_share_trade_suggestions(
            actions=rebalance.get("actions", []),
            holdings=portfolio.get("holdings", []),
            cash=portfolio.get("cash", 0),
            total_asset=portfolio.get("total_asset", 0),
            candidates=rebalance.get("new_candidates", []),
            max_single_position_pct=MAX_SINGLE_POSITION_PCT,
        )
        rebalance["actions"] = annotated_actions
        rebalance["new_candidates"] = annotated_candidates
        rebalance.setdefault("_meta", {})
        rebalance["_meta"]["execution_profile_source"] = execution_profile.source
        rebalance["_meta"]["execution_profile_samples"] = execution_profile.sample_size
    except Exception as e:
        logger.warning("[执行数量规划] 生成整手数量失败，保留原始建议: %s", e)

    discussion = _build_rebalance_discussion(
        market_judge=market_judge,
        sector_judge=sector_judge,
        holdings_ratings=holdings_ratings,
        proposal=proposal,
        critique=critique,
        rebalance=rebalance,
        debate_mode=debate_mode,
        market_model=market_model_used,
        sector_model=sector_model_used,
        holdings_model=local_model,
        proposal_model=proposal_model_used,
        critique_model=critique_model_used or debate_model,
        arbiter_model=arbiter_model_used,
    )
    rebalance["agent_discussion"] = discussion
    _log_rebalance_discussion(discussion)

    rebalance["_meta"] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "holdings_count": len(holding_codes),
        "agents_used": debate_mode,
        "local_model": local_model,
        "debate_model": debate_model,
        "cloud_model": cloud_model,
        "market_model_used": market_model_used,
        "sector_model_used": sector_model_used,
        "proposal_model_used": proposal_model_used,
        "critique_model_used": critique_model_used or debate_model,
        "arbiter_model_used": arbiter_model_used,
        **rebalance.get("_meta", {}),
    }

    return rebalance
