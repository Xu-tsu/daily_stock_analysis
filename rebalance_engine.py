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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 导入项目已有模块 ──
from src.config import Config, get_config
from src.core.trading_calendar import count_stock_trading_days
from macro_data_collector import (
    collect_full_macro_data, _fetch_tencent_quote,
    _stock_code_to_tencent, _fetch_tencent_kline,
)
from portfolio_manager import (
    load_portfolio, update_current_prices, format_rebalance_report,
)
from src.stock_analyzer import StockTrendAnalyzer
from src.services.trade_feedback_service import format_feedback_for_prompt
from src.services.trade_sizing_service import annotate_a_share_trade_suggestions

# 蒸馏数据保存目录
DISTILL_DIR = Path("data/distillation")

# ── 趋势分析器（复用单例）──
_trend_analyzer = StockTrendAnalyzer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 降级追踪 — 记录每次降级的原因和修复建议
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _classify_error(e: Exception) -> tuple:
    """根据异常类型返回 (error_type, chinese_reason, fix_suggestion)"""
    err_str = str(e).lower()
    err_type = type(e).__name__

    # Timeout
    if "timeout" in err_str or "timed out" in err_str or "Timeout" in err_type:
        return (
            "timeout",
            f"模型响应超时（{err_type}）",
            "1) 检查Ollama是否正常运行: ollama ps\n"
            "2) 检查GPU显存: nvidia-smi\n"
            "3) 考虑换用更小模型: set REBALANCE_LOCAL_MODEL=ollama/qwen2.5:7b-instruct\n"
            "4) 增大超时: 当前prompt可能过长，考虑精简输入数据",
        )

    # Connection refused / not running
    if "connection" in err_str or "refused" in err_str or "connect" in err_str:
        return (
            "connection",
            f"无法连接模型服务（{err_type}）",
            "1) 启动Ollama: ollama serve\n"
            "2) 确认端口: 默认 http://localhost:11434\n"
            "3) 检查防火墙是否拦截本地端口",
        )

    # API key / auth
    if "auth" in err_str or "api_key" in err_str or "401" in err_str or "403" in err_str:
        return (
            "auth",
            f"API认证失败（{err_type}）",
            "1) 检查 .env 中的 API Key 是否正确\n"
            "2) REBALANCE_CLOUD_MODEL 对应的 key: GEMINI_API_KEY / OPENAI_API_KEY\n"
            "3) 确认 key 未过期或超出配额",
        )

    # Rate limit
    if "rate" in err_str or "429" in err_str or "quota" in err_str:
        return (
            "rate_limit",
            f"API调用频率/配额超限（{err_type}）",
            "1) 稍后重试（等待1-2分钟）\n"
            "2) 检查API配额剩余量\n"
            "3) 考虑切换到备用模型: set REBALANCE_CLOUD_FALLBACK=...",
        )

    # JSON parse
    if "json" in err_str or "decode" in err_str or "parse" in err_str:
        return (
            "parse_error",
            f"模型返回内容解析失败（{err_type}）",
            "1) 模型可能返回了非JSON文本，检查日志中的原始返回\n"
            "2) 考虑降低temperature提高输出稳定性\n"
            "3) 如果频繁出现，可能需要调整prompt模板",
        )

    # Generic
    return (
        "unknown",
        f"{err_type}: {str(e)[:120]}",
        "1) 查看完整日志定位具体错误\n"
        "2) 检查网络连接和模型服务状态\n"
        "3) 尝试重新运行调仓分析",
    )


def _make_degradation_entry(
    step: str,
    agent: str,
    error: Exception = None,
    reason: str = "",
    fix: str = "",
    severity: str = "warning",
) -> dict:
    """构建标准化降级记录条目"""
    if error and not reason:
        _, reason, fix = _classify_error(error)
    return {
        "step": step,
        "agent": agent,
        "severity": severity,        # warning / error / critical
        "reason": reason,
        "fix_suggestion": fix,
        "error_type": type(error).__name__ if error else "",
        "error_detail": str(error)[:200] if error else "",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


def _fetch_holding_technical(code: str, cost_price: float = 0) -> dict:
    """
    获取单只持仓的K线趋势 + 筹码分布，用于增强持仓扫描。
    返回紧凑字典给 LLM prompt。
    """
    result = {}

    # ---- K线趋势分析 ----
    try:
        kdf = _fetch_tencent_kline(code, days=90)
        if kdf is not None and len(kdf) >= 20:
            tr = _trend_analyzer.analyze(kdf, code)
            result["trend"] = {
                "trend_status": tr.trend_status.value,
                "ma_alignment": tr.ma_alignment,
                "trend_strength": round(tr.trend_strength, 1),
                "ma5": round(tr.ma5, 3),
                "ma10": round(tr.ma10, 3),
                "ma20": round(tr.ma20, 3),
                "bias_ma5": round(tr.bias_ma5, 2),
                "macd_signal": tr.macd_signal,
                "rsi_6": round(tr.rsi_6, 1),
                "rsi_signal": tr.rsi_signal,
                "volume_status": tr.volume_status.value,
                "volume_ratio_5d": round(tr.volume_ratio_5d, 2),
                "support_levels": [round(s, 3) for s in tr.support_levels[:3]],
                "resistance_levels": [round(r, 3) for r in tr.resistance_levels[:3]],
                "buy_signal": tr.buy_signal.value,
                "signal_score": tr.signal_score,
            }
            # ---- 盈亏转正预测 ----
            if cost_price > 0 and tr.current_price > 0 and tr.current_price < cost_price:
                gap_pct = round((cost_price - tr.current_price) / tr.current_price * 100, 2)
                # 用近20日平均日涨幅估算回本天数
                if len(kdf) >= 20:
                    recent = kdf.tail(20)
                    daily_returns = recent["close"].pct_change().dropna()
                    avg_daily = daily_returns.mean()
                    if avg_daily > 0:
                        est_days = int(gap_pct / (avg_daily * 100)) + 1
                        result["profitability_forecast"] = {
                            "current_loss_pct": round(-gap_pct, 2),
                            "avg_daily_return_pct": round(avg_daily * 100, 3),
                            "estimated_days_to_breakeven": est_days,
                            "confidence": "低" if est_days > 15 else ("中" if est_days > 5 else "高"),
                            "note": f"按近20日均涨幅{avg_daily*100:.3f}%估算，约需{est_days}个交易日回本",
                        }
                    else:
                        result["profitability_forecast"] = {
                            "current_loss_pct": round(-gap_pct, 2),
                            "avg_daily_return_pct": round(avg_daily * 100, 3),
                            "estimated_days_to_breakeven": "无法估计（近期均涨幅≤0）",
                            "note": "近20日平均日收益率为负，短期回本概率低",
                        }
    except Exception as e:
        logger.warning(f"K线趋势分析 {code} 失败: {e}")

    # ---- 筹码分布 ----
    try:
        from data_provider.akshare_fetcher import AkshareFetcher
        fetcher = AkshareFetcher()
        chip = fetcher.get_chip_distribution(code)
        if chip:
            result["chip"] = {
                "date": chip.date,
                "profit_ratio": round(chip.profit_ratio * 100, 1),
                "avg_cost": chip.avg_cost,
                "cost_90_range": f"{chip.cost_90_low}-{chip.cost_90_high}",
                "concentration_90": round(chip.concentration_90 * 100, 2),
                "cost_70_range": f"{chip.cost_70_low}-{chip.cost_70_high}",
                "concentration_70": round(chip.concentration_70 * 100, 2),
            }
            # 筹码峰与当前价的关系
            if chip.avg_cost > 0:
                price_vs_avg = round(
                    (tr.current_price - chip.avg_cost) / chip.avg_cost * 100, 2
                ) if 'tr' in dir() and hasattr(tr, 'current_price') else None
                if price_vs_avg is not None:
                    result["chip"]["price_vs_avg_cost_pct"] = price_vs_avg
    except Exception as e:
        logger.warning(f"筹码分布 {code} 获取失败: {e}")

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM 调用封装（区分本地 / 云端）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_OLLAMA_ALIVE = None   # None=未检测, True=在线, False=离线


def _check_ollama_alive() -> bool:
    """快速探测 Ollama 是否在运行（2秒超时），结果缓存60秒"""
    global _OLLAMA_ALIVE
    import time
    cache_key = "_ollama_alive_ts"
    now = time.time()
    # 缓存60秒
    if _OLLAMA_ALIVE is not None and hasattr(_check_ollama_alive, cache_key):
        if now - getattr(_check_ollama_alive, cache_key) < 60:
            return _OLLAMA_ALIVE
    try:
        import requests
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        r = requests.get(f"{base}/api/tags", timeout=2)
        _OLLAMA_ALIVE = r.status_code == 200
    except Exception:
        _OLLAMA_ALIVE = False
    setattr(_check_ollama_alive, cache_key, now)
    if not _OLLAMA_ALIVE:
        logger.warning("[LLM] Ollama 未启动，本地模型调用将直接跳过")
    return _OLLAMA_ALIVE


def _call_local_llm(
    prompt: str,
    agent_name: str = "",
    return_model: bool = False,
    timeout: int = 0,
):
    """
    调用本地 Ollama 模型（Agent 1-3 用）
    使用 REBALANCE_LOCAL_MODEL 环境变量，不碰主项目的 LITELLM_MODEL
    timeout: 覆盖默认超时（秒）。0 表示使用环境变量 REBALANCE_LOCAL_TIMEOUT，默认 180。
    """
    # 快速检测：Ollama 没启动就直接跳过，不等180秒超时
    if not _check_ollama_alive():
        logger.warning(f"[{agent_name}] Ollama 离线，跳过本地LLM，走云端/规则兜底")
        if return_model:
            return "{}", "ollama_offline"
        return "{}"

    import litellm
    model = os.getenv("REBALANCE_LOCAL_MODEL", "ollama/qwen2.5:14b-instruct-q4_K_M")
    if timeout <= 0:
        timeout = int(os.getenv("REBALANCE_LOCAL_TIMEOUT", "180"))
    # num_ctx: 控制 Ollama KV-cache 大小，直接影响 GPU 显存占用
    # 默认32768太大（14B模型需18GB），8192够用且能全部装进12GB显存
    num_ctx = int(os.getenv("REBALANCE_LOCAL_NUM_CTX", "8192"))
    prompt_len = len(prompt)
    try:
        logger.debug(f"[{agent_name}] 调用本地模型: {model} (timeout={timeout}s, num_ctx={num_ctx}, prompt={prompt_len}字符)")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
            temperature=0.3,  # 低温度，结果更稳定
            num_ctx=num_ctx,
        )
        result = response.choices[0].message.content
        logger.debug(f"[{agent_name}] 本地模型返回 {len(result)} 字符")
        if return_model:
            return result, model
        return result
    except Exception as e:
        logger.error(f"[{agent_name}] 本地 LLM 调用失败 (timeout={timeout}s, prompt={prompt_len}字符): {e}")
        if return_model:
            return "{}", model
        return "{}"


def _call_debate_llm(prompt: str, agent_name: str = "", return_model: bool = False):
    """
    调用本地第二模型（DeepSeek-R1，辩论用）
    使用 REBALANCE_DEBATE_MODEL 环境变量
    """
    # 快速检测 Ollama
    if not _check_ollama_alive():
        logger.warning(f"[{agent_name}] Ollama 离线，跳过辩论模型")
        if return_model:
            return "{}", "ollama_offline"
        return "{}"

    import litellm
    model = os.getenv("REBALANCE_DEBATE_MODEL", "ollama/deepseek-r1:14b")
    num_ctx = int(os.getenv("REBALANCE_DEBATE_NUM_CTX", "8192"))
    try:
        logger.info(f"[{agent_name}] 调用辩论模型: {model} (num_ctx={num_ctx})")
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=300,
            temperature=0.4,
            num_ctx=num_ctx,
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


def _call_scan_llm(prompt: str, agent_name: str = "", return_model: bool = False):
    """
    持仓扫描用 LLM — 根据 REBALANCE_SCAN_USE_CLOUD 决定走云端还是本地。
    云端（Gemini）: 5-15s/股，支持并发，适合多只持仓并行分析。
    本地（Ollama）: 100-260s/股，单GPU串行，适合省钱但慢。
    """
    use_cloud = os.getenv("REBALANCE_SCAN_USE_CLOUD", "true").lower() in ("true", "1", "yes")
    if use_cloud:
        return _call_cloud_llm(prompt, agent_name, return_model=return_model)
    return _call_local_llm(prompt, agent_name, return_model=return_model)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 即时卖出 — 分析一只、执行一只
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _should_immediate_sell(rating: dict, holding: dict) -> bool:
    """判断是否应该立即卖出（不等辩论）

    核心原则：
    - 不设硬止损！量化可能砸到地板再拉涨停（地天板），硬止损会在最低点交出筹码
    - AI判断 + 实际亏损双重确认，防止模型幻觉误杀
    - 大跌时优先判断是否有地天板机会（量化砸盘→反拉）

    触发条件（必须同时满足）：
    - AI: rating.action="sell" 且 score <= IMMEDIATE_SELL_SCORE
    - 实际: 确实在亏钱（pnl_pct < 0）
    - 排除: 盘中急跌可能是量化砸盘（不在急跌时卖出）
    """
    # 可配置阈值
    sell_score = float(os.getenv("IMMEDIATE_SELL_SCORE", "30"))
    sell_loss_pct = float(os.getenv("IMMEDIATE_SELL_LOSS_PCT", "-5"))

    pnl_pct = float(holding.get("pnl_pct", 0) or 0)

    action = str(rating.get("action", "")).lower()
    score = rating.get("score", 50)
    try:
        score = float(score)
    except (ValueError, TypeError):
        score = 50
    risk = str(rating.get("risk_level", "")).lower()
    reason = str(rating.get("reason", ""))

    # ── 地天板保护：盘中急跌不卖 ──
    # 如果今日跌幅很大（接近跌停），可能是量化砸盘，不卖！
    # 等收盘后辩论决定，给地天板反转留机会
    today_chg = float(rating.get("today_change_pct", 0) or 0)
    if today_chg <= -7:
        logger.info(f"[地天板保护] 今日跌{today_chg:.1f}%，疑似量化砸盘，不即时卖出（等辩论）")
        return False

    # ── AI + 亏损双重确认 ──
    # AI说卖(score≤30) + 确实在亏钱 → 卖
    if action == "sell" and score <= sell_score and pnl_pct < 0:
        return True

    # AI说高风险 + 亏损超阈值 → 卖
    if risk == "high" and pnl_pct < sell_loss_pct:
        return True

    # AI说止损 + 确实在亏钱 → 卖（但不在急跌时）
    if "止损" in reason and pnl_pct < -2:
        return True

    return False


def _immediate_sell(holding, rating, broker, total_asset, results_list):
    """立即执行卖出操作（不等辩论完成）"""
    code = holding["code"]
    name = holding.get("name", code)
    shares = int(holding.get("sellable_shares", holding.get("shares", 0)) or 0)
    price = float(holding.get("current_price", 0) or 0)

    if not broker or shares <= 0 or price <= 0:
        logger.warning(f"[即时卖出] {name}({code}) 跳过: broker={bool(broker)} shares={shares} price={price}")
        return

    from src.broker.safety import check_order_allowed, increment_order_count
    allowed, reason = check_order_allowed(code, price, shares, total_asset)
    if not allowed:
        logger.warning(f"[即时卖出] {name} 安全限制: {reason}")
        return

    score = rating.get("score", "?")
    sell_reason = str(rating.get("reason", ""))[:50]
    logger.info(
        f"[即时卖出] {name}({code}) {shares}股 @ {price:.2f} "
        f"(评分:{score}, 原因:{sell_reason})"
    )

    result = broker.sell(code, round(price, 2), shares)
    result.name = name
    results_list.append(result)

    if result.is_success:
        increment_order_count()
        logger.info(f"[即时卖出] {name} 成功: {result.message}")
        # 记录到交易日志（复盘用）
        try:
            from trade_journal import record_sell
            actual_price = result.actual_price if result.actual_price and result.actual_price > 0 else price
            record_sell(
                code=code, name=name, shares=shares, price=actual_price,
                buy_price=float(holding.get("cost_price", 0) or 0),
                ma_trend=str(rating.get("trend_assessment", ""))[:20],
                macd_signal="",
                rsi=0,
                vol_pattern="",
                tech_score=rating.get("score", 0),
                sector="",
                source="immediate_sell",
                note=f"即时卖出 score={score} {sell_reason}",
            )
        except Exception as e:
            logger.warning(f"[即时卖出] 交易日志记录失败: {e}")
    else:
        logger.warning(f"[即时卖出] {name} 失败: {result.message}")


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


def _build_last_analysis_context() -> str:
    """加载上一次调仓分析结果，供 AI 保持决策连贯性。

    解决问题：每次分析独立运行，AI 不知道上一次建议了什么，
    导致"昨天说持有今天说清仓"这类前后矛盾。
    """
    history_dir = Path("data/rebalance_history")
    if not history_dir.exists():
        return ""

    # 找最新的历史文件
    files = sorted(history_dir.glob("rebalance_*.json"), reverse=True)
    if not files:
        return ""

    # 跳过太旧的（超过3天不注入，避免误导）
    latest = files[0]
    try:
        ts_str = latest.stem.replace("rebalance_", "")  # 20260407_150051
        file_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        if (datetime.now() - file_dt).days > 3:
            return ""
    except Exception:
        pass

    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return ""

    # 提取关键信息
    lines = [
        f"\n\n## 上次调仓分析结果（{file_dt.strftime('%Y-%m-%d %H:%M')}，必须参考保持连贯）",
        f"整体仓位建议: {data.get('overall_position_advice', 'N/A')}",
        f"大盘判断: {data.get('market_assessment', 'N/A')}",
        f"板块判断: {data.get('sector_assessment', 'N/A')}",
    ]

    actions = data.get("actions", [])
    if actions:
        lines.append("各股票操作建议:")
        for a in actions:
            target = a.get("target_sell_price", "无")
            stop = a.get("stop_loss_price", "无")
            lines.append(
                f"  - {a.get('name','')}({a.get('code','')}): "
                f"{a.get('action','?')} {a.get('ratio','')} | "
                f"止盈目标={target} 止损={stop} | "
                f"理由: {str(a.get('reason',''))[:80]}"
            )

    debate = data.get("debate_summary", "")
    if debate:
        lines.append(f"辩论总结: {debate[:150]}")

    lines.append(
        "\n**重要**: 请对照上次分析结果，如果本次建议与上次不同，"
        "必须在 reason 中明确说明变化原因（如：市场环境变化、价格已到目标、"
        "新信号出现等）。不要无理由地翻转上次的结论。"
    )

    return "\n".join(lines)


def _build_recent_feedback_prompt_block(limit: int = 6) -> str:
    """Return a compact manual-feedback block for future rebalance prompts."""
    try:
        feedback_text = format_feedback_for_prompt(limit=limit)
    except Exception as exc:
        logger.debug("加载人工反馈纠偏样本失败: %s", exc)
        return ""
    if not feedback_text:
        return ""
    return (
        "\n\n## 最近的实盘反馈纠偏（来自飞书人工回灌，优先用于修正执行偏差）\n"
        f"{feedback_text}\n"
        "请把这些反馈当作真实交易后的纠偏样本，在不破坏核心风控的前提下，优先修正容易卖飞、追高或左侧接早的问题。"
    )


def _load_scan_mode_candidates(scan_mode: str, limit: int = 12) -> List[dict]:
    try:
        from data_store import _get_conn
    except Exception as exc:
        logger.debug("加载 %s 扫描结果失败: %s", scan_mode, exc)
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT code, name, price, change_pct, turnover_rate, market_cap,
                   ma_trend, macd_signal, rsi, vol_pattern, tech_score
            FROM scan_results
            WHERE scan_date = ? AND scan_mode = ?
            ORDER BY tech_score DESC, created_at DESC
            LIMIT ?
            """,
            (today, scan_mode, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.debug("读取 %s 扫描结果失败: %s", scan_mode, exc)
        return []
    finally:
        conn.close()


def _summarize_fundamentals(fundamental_ratings: list) -> str:
    """将研究员评级列表压缩为紧凑文本，供辩论 prompt 使用。"""
    if not fundamental_ratings:
        return "无基本面数据"
    lines = []
    for fr in fundamental_ratings:
        lines.append(
            f"- {fr.get('name','?')}({fr.get('code','?')}): "
            f"评级{fr.get('fundamental_grade','?')} "
            f"风险{fr.get('financial_risk','?')} "
            f"增长{fr.get('growth_trend','?')} "
            f"| {fr.get('key_finding','')}"
        )
    return "\n".join(lines)


def _build_relay_candidate_pool(
    hot_candidates: list,
    dominant_themes: Optional[List[str]] = None,
    limit: int = 10,
) -> List[dict]:
    """Merge trend candidates with sub-dragon scans and annotate entry timing."""
    from src.services.theme_rotation_service import annotate_rotation_candidates

    merged: Dict[str, dict] = {}
    for item in hot_candidates or []:
        code = str(item.get("code", "") or "").strip()
        if not code:
            continue
        row = dict(item)
        row.setdefault("candidate_source", "dragon_hot")
        merged[code] = row

    for item in _load_scan_mode_candidates("dragon", limit=max(limit, 8)):
        code = str(item.get("code", "") or "").strip()
        if not code:
            continue
        existing = merged.get(code, {})
        row = {**item, **existing}
        row["candidate_source"] = "dragon"
        merged[code] = row

    for item in _load_scan_mode_candidates("sub_dragon", limit=max(limit, 8)):
        code = str(item.get("code", "") or "").strip()
        if not code:
            continue
        existing = merged.get(code, {})
        row = {**item, **existing}
        row["candidate_source"] = "sub_dragon"
        merged[code] = row

    annotated = annotate_rotation_candidates(
        list(merged.values()),
        dominant_themes=dominant_themes or [],
        limit=None,
    )
    return annotated[:limit]


def _apply_candidate_timing_guards(new_candidates: list, candidate_snapshots: list) -> list:
    """Prevent stale buy ranges after intraday low has already been missed."""

    snapshot_by_code = {
        str(item.get("code", "") or "").strip(): item
        for item in candidate_snapshots or []
        if str(item.get("code", "") or "").strip()
    }
    guarded: List[dict] = []
    for candidate in new_candidates or []:
        row = dict(candidate)
        code = str(row.get("code", "") or "").strip()
        snapshot = snapshot_by_code.get(code)
        if not snapshot:
            guarded.append(row)
            continue

        timing_note = str(snapshot.get("timing_note", "") or "").strip()
        preferred_buy_range = str(snapshot.get("preferred_buy_range", "") or "").strip()
        entry_state = str(snapshot.get("entry_state", "") or "").strip()
        relay_role = str(snapshot.get("relay_role", "") or "").strip()
        row["timing_note"] = timing_note
        row["relay_role"] = relay_role
        row["current_price"] = snapshot.get("price")
        row["change_pct"] = snapshot.get("change_pct")
        row["main_net"] = snapshot.get("main_net")

        if snapshot.get("missed_entry"):
            if preferred_buy_range:
                row["buy_price_range"] = preferred_buy_range
            base_reason = str(row.get("reason", "") or "").strip()
            miss_note = "当前价格已明显脱离盘中低吸窗口，若没在分歧低点成交，现阶段不追高，等待下一次回踩确认。"
            row["reason"] = f"{base_reason} {miss_note}".strip()
        elif entry_state in {"pullback_ready", "secondary_relay"} and preferred_buy_range and not row.get("buy_price_range"):
            row["buy_price_range"] = preferred_buy_range

        guarded.append(row)
    return guarded


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

        # 硬规则1：亏损>5%强制清仓
        if pnl <= -5.0 and action.get("action") not in ("sell",):
            action["action"] = "sell"
            action["detail"] = f"硬规则风控: 亏损{pnl}%超-5%强制退出线，执行清仓"
            action["reason"] = "深度亏损超出容错区间，优先保护本金"
        # 硬规则2：亏损>3%减仓
        elif pnl <= -3.0 and action.get("action") in ("buy", "hold"):
            action["action"] = "reduce"
            action["detail"] = f"自适应风控: 亏损{pnl}%已到-3%止损线，先降到观察仓"
            action["reason"] = "龙头走弱先止损，再观察是否出现反包"
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 分析师 Prompt（市场情绪）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT_SENTIMENT_ANALYST = """你是一位专业的A股市场情绪分析师，擅长通过多维度数据判断市场情绪周期。

## 大盘研判（来自上一步分析）
{market_judge}

## 市场情绪数据
{sentiment_data}

## 用户交易偏好
用户是超短线选手，核心理念："买在分歧，卖在高潮"
- 情绪恐慌但主线板块未坏 → 低吸机会
- 情绪贪婪+人气拥挤 → 止盈信号

## 你的任务
分析当前市场情绪所处阶段，给出情绪面的操作参考。

请严格按以下 JSON 格式回复：
{{
  "emotional_cycle": "恐慌/犹豫/修复/乐观/贪婪/疯狂",
  "fear_greed_score": 0-100,
  "margin_signal": "融资情绪一句话描述",
  "breadth_signal": "涨跌广度一句话描述",
  "contrarian_opportunity": "是否存在逆向操作机会（低吸或止盈）",
  "sentiment_advice": "基于情绪面的操作建议（1-2句话）"
}}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 研究员 Prompt（基本面）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT_FUNDAMENTAL_SCAN = """你是一位A股基本面研究员。

## 重要前提
用户做低价小盘题材股，基本面是辅助参考而非主导因素。
你的重点是：财务是否有雷？增长趋势是否在好转？估值是否极端？

## {name}({code}) 基本面数据
{fundamental_data}

## 该股技术面评级摘要
{tech_rating_summary}

## 你的任务
对该股基本面给出研究评级。注意：
- 小盘股PE>100或为负是常态，不要因此直接否定
- 关注ROE趋势方向、营收/利润增速变化、是否存在ST/退市风险
- 如果基本面很差但技术面强，说明可能是炒作题材，提示风险但不否定短线交易

请严格按以下 JSON 格式回复：
{{
  "code": "{code}",
  "name": "{name}",
  "fundamental_grade": "A(优秀)/B(良好)/C(一般)/D(危险)",
  "financial_risk": "低/中/高/极高",
  "growth_trend": "加速增长/稳定增长/放缓/下滑/亏损",
  "valuation_note": "估值一句话点评（PE/PB vs 行业）",
  "risk_flag": "无/关注/ST风险/退市风险",
  "key_finding": "最重要的一个发现（1句话）"
}}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 回溯验证 Prompt（决策后标注）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT_BACKTEST_VALIDATOR = """你是一位量化回溯验证专家，负责对调仓决策进行置信度标注。

## 今日最终调仓决策
{rebalance_decision}

## 历史交易绩效（近30日）
{trade_performance}

## 历史胜率模式（近90日）
{winning_patterns}

## 扫描回测准确率
{scan_accuracy}

## 你的任务
对每条操作建议标注置信度，并检查是否匹配历史亏损模式。

注意：
- 如果某操作与历史亏损模式高度吻合（如：追高买入、持仓超3天亏损股），大幅降低置信度
- 如果某操作与历史盈利模式吻合（如：缩量回踩MA5买入、1天内止盈），提高置信度
- 如果历史数据不足（总交易<10笔），所有置信度设为50（中性）

请严格按以下 JSON 格式回复：
{{
  "overall_confidence": 0-100,
  "per_action_confidence": [
    {{
      "code": "股票代码",
      "action": "操作",
      "confidence": 0-100,
      "pattern_match": "匹配的历史模式（盈利或亏损）",
      "warning": "风险提示（如有）"
    }}
  ],
  "calibration_note": "整体校准说明（1-2句话）"
}}"""


PROMPT_HOLDING_SCAN = """你是一位专业的A股短线交易顾问，专注低价小盘题材股。

{decision_reference}

{review_lessons}

### T+1风险评估
- 如果该股今日已大涨(>3%)，不建议加仓（明天大概率回调，T+1无法当天止损）
- 评估该股是否处于"追高买入"状态：远高于MA5 = 高风险

### 评级标准（参考上面的历史胜率数据调整）
- 加仓条件：板块资金持续流入 + 个股缩量回踩MA5支撑 + 乖离率<2% + 当日涨幅<3%
- 持有条件：趋势未破 + 板块未转弱 + 盈亏在可控范围
- 减仓条件：板块资金流出 + 放量下跌破MA5 或 持仓超过历史最优天数
- 清仓条件：连续3日主力净流出 或 板块崩塌
- 地天板保护：今日急跌≥7%不建议卖出，可能是量化砸盘反转机会

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

## K线趋势分析（MA均线、MACD、RSI、量能、支撑压力位）
{trend_analysis}

## 筹码分布（获利比例、平均成本、集中度）
{chip_distribution}

## 盈亏回本预测（如果当前亏损）
{profitability_forecast}

## 你的任务
综合以上所有信息（特别关注大盘趋势、K线形态、筹码峰位置），对 {name}({code}) 给出操作评级。

### 额外要求
1. **结合大盘判断**：如果大盘处于上涨趋势且板块资金流入，即使个股短期亏损，也应评估继续持有的价值
2. **K线趋势研判**：关注均线排列（多头/空头）、MACD金叉死叉、RSI超买超卖、量价配合
3. **筹码峰分析**：如果当前价格接近筹码密集区的平均成本，抛压较大；如果在筹码密集区下方，可能有支撑
4. **持有回本预测**：如果该股处于亏损但趋势在好转（均线收敛、MACD即将金叉、板块走强），预测继续持有多少个交易日可能回本，给出 estimated_hold_days
5. **不要机械止损**：如果亏损在5%以内且趋势明确向好（K线沿MA5上攻、板块资金持续流入、筹码集中度在改善），可以建议"持有等待回本"并给出预计天数

请严格按以下 JSON 格式回复：
{{
  "code": "{code}",
  "name": "{name}",
  "rating": "加仓 / 持有 / 减仓 / 清仓",
  "score": 0-100,
  "reasons": ["原因1", "原因2", "原因3"],
  "risk_level": "低/中/高",
  "key_price_levels": {{
    "support": "支撑位（基于K线和筹码峰）",
    "resistance": "压力位（基于K线和筹码峰）",
    "stop_loss": "止损位"
  }},
  "trend_assessment": "趋势判断：上涨/震荡/下跌 + 一句话描述K线形态",
  "chip_assessment": "筹码分析：当前价vs筹码峰位置 + 抛压/支撑判断",
  "estimated_hold_days": "如果亏损但趋势向好，预计多少天可回本（无法判断则填null）",
  "hold_or_cut_reason": "详细说明为什么建议继续持有等回本 或 立即止损的逻辑"
}}"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 辩论 Prompt 模板
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT_DEBATE_CRITIQUE = """你是一位严谨保守的A股风控专家，你的工作是审查另一位交易员的调仓方案，找出其中的漏洞和风险。

## 交易员的调仓方案
{proposal}

## 当前持仓明细（注意：sellable_shares 是T+1可卖余额，今天买入的股不能卖）
{portfolio}

## 历史交易数据参考（基于真实交易，必须参考）
{decision_reference}

{review_lessons}

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
8. **低吸机会**：如果大盘下跌但主线板块资金仍在、个股只是回踩支撑，方案是否错过"买在分歧"的低吸机会？
9. **高潮兑现**：如果个股红盘冲高、题材一致加速、人气过热，方案是否应该更主动止盈，而不是继续恋战？
10. **题材切主线**：如果领涨龙头继续连板、板块内副龙/补涨股开始放量承接，方案有没有及时识别"新主线切换"和副龙低吸机会？
11. **过期买点**：如果建议买入区间已经被盘中拉升明显脱离，方案是否还在给一个已经失效的低点价位？

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
5. 每只股必须给出 target_sell_price 和 stop_loss_price，**价格必须参考持仓明细中的实时 current_price，不能用过期的价格**
6. 换股只能从热门股列表中选，不能自己编造
7. 同等条件下，优先"买在分歧、卖在高潮"：指数回落但主线未坏时允许小仓低吸；红盘冲高且人气拥挤时优先兑现
8. 如果龙头已经加速封板，优先考虑同题材里尚未完全加速、但资金开始扩散承接的副龙/补涨股
9. 如果原始低吸位已经被盘中拉升明显脱离，不要继续给过期买点；应改成等待下一次回踩确认
10. **回本持有预测**：对于亏损但趋势向好的股票（K线趋势上涨、MACD金叉、板块走强、筹码支撑有效），给出预计继续持有多少交易日可回本，作为是否止损的重要参考
11. **大盘趋势权重**：如果大盘处于上涨趋势，个股亏损但板块强势，适当放宽持有耐心；如果大盘下跌趋势，即使个股技术面好也要更严格止损

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
      "sell_timing": "建议在什么条件下卖出",
      "estimated_hold_days": "亏损股预计多少天回本（趋势好给数字，趋势差null）",
      "hold_rationale": "继续持有等回本的理由 或 立即止损的依据（K线+筹码+大盘综合判断）",
      "execution_strategy": {{
        "urgency": "high/medium/low（执行紧迫度：high=立刻成交，medium=可等几分钟，low=挂单等回调）",
        "chase_max_pct": 0.5,
        "chase_timeout": 60,
        "order_type": "aggressive/passive/limit（aggressive=对手价快速成交，passive=挂买一/卖一等待，limit=指定价不追）",
        "split_orders": false,
        "reason": "为什么选择这个执行策略"
      }}
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
      "buy_price_range": "建议买入价格区间",
      "execution_strategy": {{
        "urgency": "low",
        "chase_max_pct": 0,
        "chase_timeout": 0,
        "order_type": "limit",
        "split_orders": false,
        "reason": "新候选等回调到位再买，不追高"
      }}
    }}
  ],
  "risk_warning": "整体风险提示"
}}

execution_strategy 字段说明（每只股票必须给出，系统会据此自动执行）：
- urgency: 执行紧迫度
  - high: 止损/追跌/板块即将转弱 → 对手价立刻成交，不惜多付0.5-1%滑点
  - medium: 正常止盈/减仓 → 可等2分钟，小幅追价
  - low: 换股买入/加仓 → 挂限价单等回调，不主动追
- chase_max_pct: 最多追价百分比（买入是往上追，卖出是往下追）。
  止损紧急给1-2%，正常给0.3-0.5%，不急给0
- chase_timeout: 追单超时秒数。紧急给30-60，正常给60-120，不急给0（挂单不追）
- order_type: aggressive=按对手价（买用卖一价，卖用买一价），passive=按己方价（买用买一，卖用卖一），limit=只挂指定价不动
- split_orders: 大单是否拆分（>5000股的单子建议true，避免冲击成本）
- reason: 为什么选择这个策略（如"止损紧急需要立刻成交"或"等回踩到支撑位再买"）"""

PROMPT_REBALANCE_FINAL = """你是一位经验丰富的A股短线交易员，擅长低价小盘题材股的板块轮动策略。

## 我的交易风格（基于233笔真实交易数据优化，必须严格遵守）

### 核心策略：快进快出，小赚即走，严格止损
- 操作风格：超短线趋势交易，持股周期1-3天（历史数据：1天内胜率93%，超3天暴降至56%）
- 选股偏好：10元以下低价股、流通市值50亿以下小盘股、有热门题材概念的
- 买入条件：缩量回踩MA5支撑 + 板块资金持续流入 + 乖离率<2% + 当日涨幅<3%
- 卖出条件：盈利5-8%止盈 / 亏损5%止损 / 持仓超3天 / 板块转弱
- 绝对禁止：不买大盘蓝筹白马股，不做价值投资，不补仓亏损股

### 逆向执行偏好：买在分歧，卖在高潮
- 更高胜率的买点通常出现在大盘回落、盘面转冷、个股回踩支撑、市场"无人问津"时；前提是主线板块资金确认仍在，不能把下跌趋势误当低吸机会
- 如果个股红盘冲高、题材一致加速、讨论度和跟风情绪明显升温，要优先考虑分批止盈/减仓，做到"卖在人声鼎沸"
- 同等条件下，优先选择轻微回调或平盘承接的候选，而不是当天最热、最拥挤、最接近涨停的票
- 如果领涨龙头已经涨停或连续加速，而同板块副龙开始放量承接、但尚未脱离低吸区间，应优先考虑副龙/补涨，而不是继续追龙头
- 如果盘中最低点已经走过、当前价格明显高于低吸区间，不要继续给一个过期的静态买点，应明确写"等回踩/不追高"

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
5. **每只股票必须给出 target_sell_price（目标卖出价）和 stop_loss_price（止损价）**，价格必须基于持仓明细中的实时 current_price 来设定
6. **每只股票必须给出 sell_timing（什么条件下卖出）**，如"盈利5%或跌破MA5卖出"
7. 标注风险等级
8. 持仓天数必须严格按照 buy_date 字段准确计算到今天的交易日数，不要编造
9. 如果某只股票所处题材仍在强化、只是冲高后可能回踩，不要动不动就一刀切清仓；除非触发硬风控，否则优先分批止盈、保留底仓观察二波
10. 如果大盘极弱，可以建议空仓等待，但一旦有板块异动要给出抄底候选
11. **回本预测（重要）**：对于亏损但趋势向好的持仓（K线沿MA5上攻、MACD金叉、板块资金流入），预测继续持有大约多少个交易日可以回本，并说明判断依据（均线趋势、筹码峰支撑等）
12. **筹码峰参考**：参考各持仓股的筹码分布评级中的 chip_assessment 和 estimated_hold_days，结合大盘趋势综合判断

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
      "sell_timing": "建议在什么条件下卖出（如：盈利5%或跌破MA5时卖出）",
      "estimated_hold_days": "如果亏损，预计持有多少天可回本（趋势好则给数字，趋势差则null）",
      "hold_rationale": "继续持有的理由（K线趋势+筹码+大盘），或立即止损的理由",
      "execution_strategy": {{
        "urgency": "high/medium/low",
        "chase_max_pct": 0.5,
        "chase_timeout": 60,
        "order_type": "aggressive/passive/limit",
        "split_orders": false,
        "reason": "执行策略理由"
      }}
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
      "buy_price_range": "建议买入价格区间",
      "execution_strategy": {{
        "urgency": "low",
        "chase_max_pct": 0,
        "chase_timeout": 0,
        "order_type": "limit",
        "split_orders": false,
        "reason": "等回调到位再买"
      }}
    }}
  ],
  "risk_warning": "整体风险提示"
}}

execution_strategy 字段说明（每只股票必须给出，系统会据此自动执行）：
- urgency: high=止损紧急立刻成交 / medium=正常操作可等几分钟 / low=挂单等回调
- chase_max_pct: 最多追价百分比。止损紧急给1-2%，正常给0.3-0.5%，不急给0
- chase_timeout: 追单超时秒数。紧急30-60，正常60-120，不急0
- order_type: aggressive=对手价快速成交 / passive=挂己方价等待 / limit=只挂指定价
- split_orders: 大单(>5000股)建议拆分避免冲击
- reason: 选择该策略的理由"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_rebalance_analysis(
    config: Config = None,
    hot_concepts: list = None,
    event_signals: list = None,
) -> dict:
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

    # ── 降级日志：记录流水线中每一次降级/跳过的原因和修复建议 ──
    degradation_log: list = []
    external_hot_concepts = hot_concepts or []
    external_event_signals = event_signals or []
    event_action = "normal"
    event_summary = "无事件信号"
    event_prompt_block = ""

    try:
        if external_event_signals:
            from event_signal import get_event_action, summarize_event_signals

            event_action = get_event_action(external_event_signals)
            event_summary = summarize_event_signals(external_event_signals, max_items=5)
            event_prompt_block = (
                "\n\n## 外部事件驱动信号（news_scanner -> event_signal）\n"
                f"总体动作: {event_action}\n"
                f"信号摘要: {event_summary}\n"
                "要求: defense 时优先防守降仓；aggressive 时只允许结合技术面低吸，不追高。"
            )
            logger.info(f"[事件驱动] 动作={event_action} | {event_summary}")
        else:
            from event_signal import get_recent_event_action, load_recent_event_entries

            recent_entries = load_recent_event_entries(max_age_hours=6.0, limit=5)
            if recent_entries:
                event_action = get_recent_event_action(max_age_hours=6.0)
                event_summary = "；".join(
                    f"{entry.get('event_type', entry.get('type', 'event'))}:{entry.get('trigger', '')}"
                    f" -> {entry.get('action', 'watch')}"
                    for entry in recent_entries
                )
                event_prompt_block = (
                    "\n\n## 最近事件日志（fallback）\n"
                    f"总体动作: {event_action}\n"
                    f"信号摘要: {event_summary}\n"
                    "要求: defense 时优先防守降仓；aggressive 时只允许结合技术面低吸，不追高。"
                )
                logger.info(f"[事件驱动] 使用最近事件日志回退: {event_action} | {event_summary}")
    except Exception as e:
        logger.debug(f"[事件驱动] 外部事件信号注入跳过: {e}")

    # ── Step 0: 加载持仓（优先从 broker 真实持仓刷新，trade_log 兜底）──
    portfolio = load_portfolio()

    # 第一优先级：broker 真实持仓（THS 客户端扫描后的数据最准确）
    broker_synced = False
    try:
        from portfolio_manager import sync_portfolio_from_broker
        old_holdings = [h["code"] for h in portfolio.get("holdings", [])]
        portfolio = sync_portfolio_from_broker(portfolio)
        new_holdings = [h["code"] for h in portfolio.get("holdings", [])]
        if old_holdings != new_holdings or any(
            h.get("sellable_shares") is not None
            for h in portfolio.get("holdings", [])
        ):
            broker_synced = True
            logger.info("[Step 0] 持仓数据已从 broker（THS）刷新")
    except Exception as e:
        logger.debug(f"broker 持仓同步跳过: {e}")

    # 第二优先级：trade_log FIFO 校准（补充 buy_date 等 broker 不返回的字段）
    try:
        from portfolio_manager import sync_portfolio_from_trades
        portfolio = sync_portfolio_from_trades(portfolio)
    except Exception as e:
        if not broker_synced:
            logger.warning(f"持仓同步失败（不影响主流程）: {e}")
            degradation_log.append(_make_degradation_entry(
                "Step0_持仓同步", "portfolio_sync", error=e,
                reason="持仓同步失败，使用上次保存的持仓数据",
                fix="1) 确认 BROKER_ENABLED=true 且 THS 客户端已登录\n"
                    "2) 或检查 data/trade_log.db 是否存在且可读",
            ))
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

    # ── 实时价格覆盖：确保用盘中最新价，而非昨日收盘 ──
    try:
        tc_codes = [_stock_code_to_tencent(c) for c in holding_codes]
        realtime_quotes = _fetch_tencent_quote(tc_codes, timeout=10)
        price_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for tc_code, quote in realtime_quotes.items():
            raw_code = tc_code.replace("sh", "").replace("sz", "")
            if quote.get("price", 0) > 0:
                price_map[raw_code] = quote["price"]
                logger.info(f"实时价格覆盖: {raw_code} → {quote['price']}")
        logger.info(f"实时报价刷新完成，时间: {price_ts}")
    except Exception as e:
        price_ts = None
        logger.warning(f"实时报价获取失败（使用历史价格）: {e}")
        degradation_log.append(_make_degradation_entry(
            "Step1_实时报价", "tencent_quote", error=e,
            reason="腾讯实时行情获取失败，使用昨日收盘价（可能不准确）",
            fix="1) 检查网络连接是否通畅\n2) 确认 qt.gtimg.cn 可访问\n3) 非交易时间无实时价",
        ))

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
    if external_hot_concepts:
        prompt_market += (
            "\n\n## news_scanner 热点概念\n"
            + json.dumps(external_hot_concepts[:10], ensure_ascii=False, indent=2)
        )
    if event_prompt_block:
        prompt_market += event_prompt_block
    market_judge_raw, market_model_used = _call_scan_llm(
        prompt_market,
        "Agent1_大盘",
        return_model=True,
    )
    market_judge = _parse_llm_json(market_judge_raw)
    if external_event_signals and isinstance(market_judge, dict):
        market_judge["event_action"] = event_action
        market_judge["event_summary"] = event_summary
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
            external_hot_concepts[:10] if external_hot_concepts else sector_data.get("概念_今日_top10", []),
            ensure_ascii=False, indent=2,
        ),
        holding_sectors=json.dumps(holding_sectors, ensure_ascii=False),
    )
    if event_prompt_block:
        prompt_sector += event_prompt_block
    sector_judge_raw, sector_model_used = _call_scan_llm(
        prompt_sector,
        "Agent2_板块",
        return_model=True,
    )
    sector_judge = _parse_llm_json(sector_judge_raw)
    if external_event_signals and isinstance(sector_judge, dict):
        sector_judge["event_action"] = event_action
        sector_judge["event_summary"] = event_summary
    _save_agent_local_sample("agent2_sector", prompt_sector, sector_judge_raw, sector_judge)
    logger.info(f"  热门板块: {sector_judge.get('hot_sectors', [])}")

    # ── Step 3b: Agent 2b — 分析师：情绪面分析 ──
    logger.info("\n[Step 3b] Agent 2b: 分析师（情绪面分析）...")
    sentiment_analysis = {}
    sentiment_model_used = "skipped"
    try:
        sentiment_data = macro_data.get("sentiment_indicators", {})
        if sentiment_data:
            prompt_sentiment = PROMPT_SENTIMENT_ANALYST.format(
                market_judge=json.dumps(market_judge, ensure_ascii=False, indent=2),
                sentiment_data=json.dumps(sentiment_data, ensure_ascii=False, indent=2),
            )
            sentiment_raw, sentiment_model_used = _call_scan_llm(
                prompt_sentiment, "Agent2b_分析师", return_model=True,
            )
            sentiment_analysis = _parse_llm_json(sentiment_raw)
            _save_agent_local_sample("agent2b_sentiment", prompt_sentiment, sentiment_raw, sentiment_analysis)
            logger.info(f"  情绪周期: {sentiment_analysis.get('emotional_cycle', 'N/A')} "
                        f"(恐贪={sentiment_analysis.get('fear_greed_score', 'N/A')})")
        else:
            logger.info("  情绪数据为空，跳过分析师")
            degradation_log.append(_make_degradation_entry(
                "Step3b_情绪分析", "Agent2b_分析师", severity="warning",
                reason="情绪数据采集为空，跳过情绪面分析",
                fix="1) 检查 macro_data_collector.fetch_sentiment_indicators() 是否正常\n"
                    "2) 确认融资余额、涨跌家数等原始数据源可用",
            ))
    except Exception as e:
        logger.warning(f"  分析师(情绪)跳过: {e}")
        degradation_log.append(_make_degradation_entry(
            "Step3b_情绪分析", "Agent2b_分析师", error=e,
        ))

    # ── 即时卖出: 初始化broker（分析一只、执行一只）──
    immediate_results = []
    _immediate_broker = None
    if os.getenv("BROKER_ENABLED", "false").lower() == "true":
        try:
            from src.broker import get_broker as _get_broker_fn
            _immediate_broker = _get_broker_fn()
            if _immediate_broker and not _immediate_broker.is_connected():
                _immediate_broker = None
            if _immediate_broker:
                logger.info("[即时卖出] broker已连接，高风险持仓将立即卖出")
        except Exception:
            pass
    total_asset = portfolio.get("total_asset", 0)

    # ── Step 4: Agent 3 — 逐只持仓扫描（并行）──
    scan_cloud = os.getenv("REBALANCE_SCAN_USE_CLOUD", "true").lower() in ("true", "1", "yes")
    scan_mode = "云端并行" if scan_cloud else "本地串行"
    logger.info(f"\n[Step 4/5] Agent 3: 持仓个股扫描（{scan_mode}）...")
    holdings_ratings = []

    def _scan_one_holding(h):
        """扫描单只持仓（线程安全，可并行）"""
        code = h["code"]
        name = h.get("name", code)
        logger.info(f"  分析 {name}({code})...")

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

        tech_data = _fetch_holding_technical(code, cost_price=h.get("cost_price", 0))

        # 生成基于历史数据的决策参考模型（防AI幻觉）
        decision_ref = ""
        try:
            from trade_journal import build_decision_reference
            decision_ref = build_decision_reference(code=code, days=90)
        except Exception:
            decision_ref = "（历史数据暂不可用，请基于技术面分析）"

        # 注入Agent技能库（可成长的技能系统）
        try:
            from agent_skill_engine import build_skill_prompt
            skill_prompt = build_skill_prompt(code=code)
            decision_ref += "\n\n" + skill_prompt
        except Exception:
            pass

        # 注入复盘教训（让今天的决策吸取昨天的教训）
        _review_lessons = ""
        try:
            from src.core.trade_review import build_review_lessons
            _review_lessons = build_review_lessons(days=3, max_chars=600)
        except Exception:
            pass

        prompt_holding = PROMPT_HOLDING_SCAN.format(
            decision_reference=decision_ref,
            review_lessons=_review_lessons,
            market_judge=json.dumps(market_judge, ensure_ascii=False),
            sector_judge=json.dumps(sector_judge, ensure_ascii=False),
            stock_analysis=stock_analysis_text,
            fund_flow=json.dumps(fund_flow, ensure_ascii=False, indent=2),
            northbound_holding=json.dumps(nb_holding, ensure_ascii=False),
            comment=json.dumps(comment, ensure_ascii=False),
            stock_news=json.dumps(s_news, ensure_ascii=False, indent=2),
            trend_analysis=json.dumps(tech_data.get("trend", {}), ensure_ascii=False, indent=2),
            chip_distribution=json.dumps(tech_data.get("chip", {}), ensure_ascii=False, indent=2),
            profitability_forecast=json.dumps(tech_data.get("profitability_forecast", {}), ensure_ascii=False, indent=2),
            code=code, name=name,
        )
        rating_raw, _ = _call_scan_llm(
            prompt_holding,
            f"Agent3_{name}",
            return_model=True,
        )
        rating = _parse_llm_json(rating_raw)
        _save_agent_local_sample(f"agent3_{code}", prompt_holding, rating_raw, rating)
        return code, name, rating

    # 云端用并行（最多6线程），本地用串行（单GPU无法并发）
    max_workers = int(os.getenv("REBALANCE_SCAN_WORKERS", "6")) if scan_cloud else 1
    holdings_list = portfolio.get("holdings", [])

    # 构建 code→holding 映射（即时卖出需要持仓信息）
    _holdings_by_code = {h["code"]: h for h in holdings_list}

    if max_workers > 1 and len(holdings_list) > 1:
        t0 = time.time()
        # 并行分析，收集结果后串行检查即时卖出（broker不是线程安全的）
        _immediate_sell_candidates = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_scan_one_holding, h): h for h in holdings_list}
            for future in as_completed(futures):
                try:
                    code, name, rating = future.result()
                    if rating:
                        holdings_ratings.append(rating)
                        logger.info(f"  → {name}: {rating.get('rating', 'N/A')} (得分: {rating.get('score', 'N/A')})")
                        # 标记即时卖出候选
                        h = _holdings_by_code.get(code, {})
                        if _immediate_broker and h and _should_immediate_sell(rating, h):
                            _immediate_sell_candidates.append((h, rating))
                    else:
                        logger.warning(f"  → {name}: 分析结果解析失败")
                        degradation_log.append(_make_degradation_entry(
                            "Step4_持仓扫描", f"Agent3_{name}", severity="warning",
                            reason=f"{name}({code}) 持仓扫描结果解析失败",
                            fix="1) 检查云端API配额\n2) 查看日志中 Agent3 原始返回",
                        ))
                except Exception as e:
                    h = futures[future]
                    logger.error(f"  → {h.get('name', h['code'])}: 并行扫描异常: {e}")
                    degradation_log.append(_make_degradation_entry(
                        "Step4_持仓扫描", f"Agent3_{h.get('name', h['code'])}", error=e,
                    ))
        logger.info(f"  并行扫描完成: {len(holdings_ratings)}/{len(holdings_list)}只 耗时{time.time()-t0:.1f}s")

        # 串行执行即时卖出
        for h, rating in _immediate_sell_candidates:
            _immediate_sell(h, rating, _immediate_broker, total_asset, immediate_results)
    else:
        for h in holdings_list:
            try:
                code, name, rating = _scan_one_holding(h)
                if rating:
                    holdings_ratings.append(rating)
                    logger.info(f"  → {name}: {rating.get('rating', 'N/A')} (得分: {rating.get('score', 'N/A')})")
                    # 即时卖出：分析完一只，立即执行
                    if _immediate_broker and _should_immediate_sell(rating, h):
                        _immediate_sell(h, rating, _immediate_broker, total_asset, immediate_results)
                else:
                    logger.warning(f"  → {name}: 分析结果解析失败")
                    degradation_log.append(_make_degradation_entry(
                        "Step4_持仓扫描", f"Agent3_{name}", severity="warning",
                        reason=f"{name}({code}) 持仓扫描结果解析失败",
                        fix="1) 检查本地模型是否正常: ollama ps\n"
                            "2) 模型可能返回非JSON，查看日志中 Agent3 原始返回\n"
                            "3) 考虑简化 PROMPT_HOLDING_SCAN 模板",
                    ))
            except Exception as e:
                logger.error(f"  → {h.get('name', h['code'])}: 扫描异常: {e}")
                degradation_log.append(_make_degradation_entry(
                    "Step4_持仓扫描", f"Agent3_{h.get('name', h['code'])}", error=e,
                ))

    # ── Step 4b: Agent 3b — 研究员：基本面扫描（并行）──
    logger.info(f"\n[Step 4b] Agent 3b: 研究员（基本面扫描, {scan_mode}）...")
    fundamental_ratings = []
    fundamental_model_used = "skipped"

    # 构建 holdings_ratings 快速查找表
    _ratings_by_code = {r.get("code"): r for r in holdings_ratings if r.get("code")}

    def _scan_one_fundamental(h):
        """扫描单只持仓基本面（线程安全）"""
        code = h["code"]
        name = h.get("name", code)
        fund_data = macro_data.get("holdings_fundamental", {}).get(code, {})
        if not fund_data or not fund_data.get("source_chain"):
            return code, name, None, "no_data"

        tr = _ratings_by_code.get(code)
        tech_summary = "暂无"
        if tr:
            tech_summary = (
                f"评级:{tr.get('rating','N/A')}, "
                f"得分:{tr.get('score','N/A')}, "
                f"趋势:{tr.get('trend_assessment','N/A')}"
            )

        prompt_fund = PROMPT_FUNDAMENTAL_SCAN.format(
            code=code, name=name,
            fundamental_data=json.dumps(fund_data, ensure_ascii=False, indent=2),
            tech_rating_summary=tech_summary,
        )
        fund_raw, model_used = _call_scan_llm(
            prompt_fund, f"Agent3b_研究员_{name}", return_model=True,
        )
        fund_rating = _parse_llm_json(fund_raw)
        _save_agent_local_sample(f"agent3b_{code}", prompt_fund, fund_raw, fund_rating)
        return code, name, fund_rating, model_used

    if max_workers > 1 and len(holdings_list) > 1:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_scan_one_fundamental, h): h for h in holdings_list}
            for future in as_completed(futures):
                try:
                    code, name, fund_rating, model_used = future.result()
                    if model_used == "no_data":
                        logger.info(f"  {name}: 无基本面数据，跳过")
                        continue
                    fundamental_model_used = model_used
                    if fund_rating:
                        fundamental_ratings.append(fund_rating)
                        logger.info(f"  → {name}: {fund_rating.get('fundamental_grade', 'N/A')} "
                                    f"风险={fund_rating.get('financial_risk', 'N/A')}")
                except Exception as e:
                    h = futures[future]
                    logger.warning(f"  研究员 {h.get('name', h['code'])} 跳过: {e}")
                    degradation_log.append(_make_degradation_entry(
                        "Step4b_基本面", f"Agent3b_研究员_{h.get('name', h['code'])}", error=e,
                    ))
        logger.info(f"  并行基本面完成: {len(fundamental_ratings)}只 耗时{time.time()-t0:.1f}s")
    else:
        for h in holdings_list:
            try:
                code, name, fund_rating, model_used = _scan_one_fundamental(h)
                if model_used == "no_data":
                    logger.info(f"  {name}: 无基本面数据，跳过")
                    continue
                fundamental_model_used = model_used
                if fund_rating:
                    fundamental_ratings.append(fund_rating)
                    logger.info(f"  → {name}: {fund_rating.get('fundamental_grade', 'N/A')} "
                                f"风险={fund_rating.get('financial_risk', 'N/A')}")
            except Exception as e:
                logger.warning(f"  研究员 {h.get('name', h['code'])} 跳过: {e}")
                degradation_log.append(_make_degradation_entry(
                    "Step4b_基本面", f"Agent3b_研究员_{h.get('name', h['code'])}", error=e,
                ))

    # ── Step 5: 多模型辩论调仓（激进派 vs 保守派 → 仲裁）──
    logger.info("\n[Step 5/7] 多模型辩论调仓决策...")

    # 真实换股候选（来自全市场扫描/副龙头扫描/资金流，不让LLM编造）
    hot_picks = macro_data.get("hot_candidates", [])
    if not hot_picks:
        # 兜底：从板块数据中提取
        for sector_info in sector_data.values():
            if isinstance(sector_info, dict):
                for item in sector_info.get("top_inflow", [])[:3]:
                    hot_picks.append(item)
    hot_picks = _build_relay_candidate_pool(
        hot_picks,
        dominant_themes=sector_judge.get("hot_sectors", []),
        limit=15,
    )

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
        degradation_log.append(_make_degradation_entry(
            "Step5_风控检查", "risk_control", error=e,
            reason="风控预检跳过，止损/止盈提醒可能缺失",
            fix="1) 检查 risk_control.py 是否存在且无语法错误\n2) import risk_control 手动测试",
        ))

    # 过滤热门候选：去除当日涨幅>5%的追高股，并优先保留分歧低吸/副龙接力候选
    filtered_hot = []
    for pick in hot_picks[:15]:
        chg = pick.get("change_pct", pick.get("涨跌幅", 0))
        if isinstance(chg, str):
            try:
                chg = float(chg.replace("%", ""))
            except (ValueError, TypeError):
                chg = 0
        if chg < 5.0:  # 涨幅<5%才推荐（T+1安全）
            candidate = dict(pick)
            candidate["_rebalance_change_pct"] = float(chg or 0)
            filtered_hot.append(candidate)
    hot_picks = sorted(
        filtered_hot,
        key=lambda item: (
            {"pullback_ready": 0, "secondary_relay": 1, "watch": 2, "overextended": 3, "leader_locked": 4}.get(
                str(item.get("entry_state", "watch")),
                9,
            ),
            -float(item.get("rotation_score", 0) or 0),
            0 if -2.5 <= float(item.get("_rebalance_change_pct", 0) or 0) <= 1.5 else 1,
            abs(float(item.get("_rebalance_change_pct", 0) or 0) + 0.2),
            -float(item.get("main_net", 0) or 0),
        ),
    )[:10]
    for pick in hot_picks:
        pick.pop("_rebalance_change_pct", None)

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
            "price_updated_at": price_ts if price_ts else "未知（使用历史收盘价）",
            "price_source": "realtime_tencent" if price_ts else "daily_close",
            "holdings": portfolio_holdings,
        },
        ensure_ascii=False,
        indent=2,
    )
    hot_picks_json = json.dumps(hot_picks[:10], ensure_ascii=False, indent=2)
    last_analysis_block = _build_last_analysis_context()
    feedback_prompt_block = _build_recent_feedback_prompt_block()

    # ── 策略自改进反馈（执行质量 + 信号准确率）──
    strategy_feedback_block = ""
    try:
        from src.strategy.feedback_loop import StrategyFeedbackLoop
        fb = StrategyFeedbackLoop()
        fb_report = fb.generate_feedback_report(days=30)
        strategy_feedback_block = fb.format_for_llm_prompt(fb_report)
        if strategy_feedback_block:
            logger.info(f"[反馈环] 已注入策略反馈 ({len(strategy_feedback_block)}字)")
    except Exception as e:
        logger.debug(f"[反馈环] 策略反馈生成跳过: {e}")

    # ── 日内峰值时段分析（数据驱动优化买卖时机）──
    peak_patterns_block = ""
    try:
        from src.broker.intraday_tracker import format_peak_patterns_for_prompt
        peak_patterns_block = format_peak_patterns_for_prompt(days=30)
        if peak_patterns_block:
            logger.info(f"[峰值分析] 已注入日内峰值时段数据 ({len(peak_patterns_block)}字)")
    except Exception as e:
        logger.debug(f"[峰值分析] 跳过: {e}")

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
    # 注入上次分析结果（保持决策连贯性）
    if last_analysis_block:
        prompt_proposal = prompt_proposal + last_analysis_block
    if feedback_prompt_block:
        prompt_proposal = prompt_proposal + feedback_prompt_block
    if strategy_feedback_block:
        prompt_proposal = prompt_proposal + "\n\n" + strategy_feedback_block
    if peak_patterns_block:
        prompt_proposal = prompt_proposal + "\n\n" + peak_patterns_block
    if risk_alerts_text:
        prompt_proposal = prompt_proposal + risk_alerts_text
    # 注入情绪面 + 基本面摘要
    if sentiment_analysis:
        prompt_proposal += f"\n\n## 情绪面参考（分析师评估）\n{json.dumps(sentiment_analysis, ensure_ascii=False)}"
    if fundamental_ratings:
        prompt_proposal += f"\n\n## 基本面参考（研究员评估）\n{_summarize_fundamentals(fundamental_ratings)}"
    if external_hot_concepts:
        prompt_proposal += (
            "\n\n## news_scanner 热点概念\n"
            + json.dumps(external_hot_concepts[:10], ensure_ascii=False, indent=2)
        )
    if event_prompt_block:
        prompt_proposal += event_prompt_block

    # Step 5a prompt 最重（含所有Agent结果+候选+风控），给予更长超时
    _proposal_timeout = int(os.getenv("REBALANCE_PROPOSAL_TIMEOUT", "360"))
    proposal_raw, proposal_model_used = _call_scan_llm(
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
        degradation_log.append(_make_degradation_entry(
            "Step5a_激进派", "Agent4a_激进派", severity="error",
            reason=f"激进派(Qwen)方案解析失败，辩论流程将降级（模型: {proposal_model_used}）",
            fix="1) 检查Ollama状态: ollama ps\n"
                "2) 检查GPU显存: nvidia-smi（14B模型需要≥10GB显存）\n"
                "3) 尝试更小模型: set REBALANCE_LOCAL_MODEL=ollama/qwen2.5:7b-instruct\n"
                "4) 查看日志中Agent4a原始返回是否为空或非JSON",
        ))

    _save_agent_local_sample("agent4a_proposal", prompt_proposal, proposal_raw, proposal)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 5b: 保守派（DeepSeek-R1）— 质疑和挑刺
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("\n[Step 6/7] 保守派（DeepSeek-R1）审查质疑...")

    critique = {}
    critique_raw = "{}"
    critique_model_used = ""
    if proposal:
        # 生成决策参考数据 + 技能库
        debate_ref = ""
        try:
            from trade_journal import build_decision_reference
            debate_ref = build_decision_reference(days=90)
        except Exception:
            debate_ref = "（历史数据暂不可用）"
        try:
            from agent_skill_engine import build_skill_prompt
            debate_ref += "\n\n" + build_skill_prompt()
        except Exception:
            pass

        # 复盘教训注入辩论
        _debate_review_lessons = ""
        try:
            from src.core.trade_review import build_review_lessons
            _debate_review_lessons = build_review_lessons(days=3, max_chars=400)
        except Exception:
            pass

        prompt_critique = PROMPT_DEBATE_CRITIQUE.format(
            proposal=json.dumps(proposal, ensure_ascii=False, indent=2),
            portfolio=portfolio_json,
            decision_reference=debate_ref,
            review_lessons=_debate_review_lessons,
            market_summary=market_judge.get("summary", ""),
            sector_summary=sector_judge.get("summary", ""),
        )
        if feedback_prompt_block:
            prompt_critique = prompt_critique + feedback_prompt_block
        # 注入历史准确度
        backtest_ctx = macro_data.get("backtest_context", {})
        if backtest_ctx and backtest_ctx.get("note") != "insufficient_data":
            perf = backtest_ctx.get("trade_performance", {})
            prompt_critique += (
                f"\n\n## 历史准确度参考\n"
                f"近30日: {perf.get('total_trades',0)}笔交易, "
                f"胜率{perf.get('win_rate',0)}%, "
                f"平均盈亏{perf.get('avg_pnl_pct',0)}%"
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
            degradation_log.append(_make_degradation_entry(
                "Step5b_保守派", "Agent4b_保守派", severity="error",
                reason=f"保守派(DeepSeek-R1)审查解析失败，辩论将缺少质疑环节（模型: {critique_model_used}）",
                fix="1) 检查DeepSeek-R1模型状态: ollama ps\n"
                    "2) DeepSeek-R1推理较慢，确认超时设置≥300s\n"
                    "3) 尝试: set REBALANCE_DEBATE_MODEL=ollama/deepseek-r1:7b\n"
                    "4) 查看日志中 <think>...</think> 标签是否完整",
            ))

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
        if last_analysis_block:
            prompt_arbitrate = prompt_arbitrate + last_analysis_block
        if feedback_prompt_block:
            prompt_arbitrate = prompt_arbitrate + feedback_prompt_block
        if risk_alerts_text:
            prompt_arbitrate = prompt_arbitrate + "\n" + risk_alerts_text
        # 注入情绪面 + 基本面 + 历史准确度
        if sentiment_analysis:
            prompt_arbitrate += (
                f"\n\n## 情绪面分析（分析师）\n"
                f"情绪周期: {sentiment_analysis.get('emotional_cycle','N/A')}, "
                f"恐贪指数: {sentiment_analysis.get('fear_greed_score','N/A')}, "
                f"建议: {sentiment_analysis.get('sentiment_advice','N/A')}"
            )
        if fundamental_ratings:
            prompt_arbitrate += f"\n\n## 基本面研究摘要\n{_summarize_fundamentals(fundamental_ratings)}"
        backtest_ctx = macro_data.get("backtest_context", {})
        if backtest_ctx and backtest_ctx.get("note") != "insufficient_data":
            perf = backtest_ctx.get("trade_performance", {})
            prompt_arbitrate += (
                f"\n\n## 历史交易准确度\n"
                f"近30日: {perf.get('total_trades',0)}笔, "
                f"胜率{perf.get('win_rate',0)}%, "
                f"均盈亏{perf.get('avg_pnl_pct',0)}%\n"
                f"规则: 情绪贪婪时偏保守，情绪恐慌但基本面稳可偏激进"
            )
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
            degradation_log.append(_make_degradation_entry(
                "Step5c_仲裁", "Agent4c_仲裁", severity="error",
                reason="云端仲裁模型返回为空或解析失败，降级为本地合并（采纳激进派方案+保守派修正）",
                fix="1) 检查云端API Key: GEMINI_API_KEY 或 OPENAI_API_KEY\n"
                    "2) 检查 .env 中 REBALANCE_CLOUD_MODEL 配置\n"
                    "3) 确认网络可达: curl https://generativelanguage.googleapis.com\n"
                    "4) 查看日志中云端模型原始返回",
            ))

    elif proposal:
        # ⚠️ 只有激进派方案，保守派挂了 → 直接用方案但加风控过滤
        debate_mode = "proposal_only"
        logger.warning("  保守派审查失败，使用激进派方案 + 硬规则风控过滤...")
        rebalance = _apply_hard_rules(proposal, portfolio)
        rebalance_raw = json.dumps(rebalance, ensure_ascii=False)
        arbiter_model_used = "hard_rules"
        degradation_log.append(_make_degradation_entry(
            "Step5c_仲裁", "辩论降级", severity="error",
            reason="保守派(DeepSeek-R1)未能完成审查，跳过辩论直接使用激进派方案+硬规则风控",
            fix="1) 检查DeepSeek-R1状态: ollama ps\n"
                "2) 注意: 缺少辩论质疑，方案风险可能偏高\n"
                "3) 建议人工审核后再执行交易",
        ))

    else:
        # ❌ 全挂了 → 单模型直接决策
        debate_mode = "single_fallback"
        logger.warning("  辩论未完成，退回单模型直接决策...")
        degradation_log.append(_make_degradation_entry(
            "Step5c_仲裁", "辩论降级", severity="critical",
            reason="激进派+保守派均失败，辩论流程完全跳过，退回单模型直接决策",
            fix="1) 本地LLM可能完全不可用，检查: ollama ps && nvidia-smi\n"
                "2) 如果Ollama未运行: ollama serve\n"
                "3) 如果显存不足: 关闭其他GPU程序，或换7B模型\n"
                "4) 结果可靠性大幅降低，强烈建议人工复核",
        ))
        prompt_fallback = PROMPT_REBALANCE_FINAL.format(
            market_judge=json.dumps(market_judge, ensure_ascii=False, indent=2),
            sector_judge=json.dumps(sector_judge, ensure_ascii=False, indent=2),
            holdings_ratings=json.dumps(holdings_ratings, ensure_ascii=False, indent=2),
            portfolio=portfolio_json,
            hot_picks=hot_picks_json,
        )
        if last_analysis_block:
            prompt_fallback = prompt_fallback + last_analysis_block
        if feedback_prompt_block:
            prompt_fallback = prompt_fallback + feedback_prompt_block
        if risk_alerts_text:
            prompt_fallback = prompt_fallback + risk_alerts_text
        if external_hot_concepts:
            prompt_fallback += (
                "\n\n## news_scanner 热点概念\n"
                + json.dumps(external_hot_concepts[:10], ensure_ascii=False, indent=2)
            )
        if event_prompt_block:
            prompt_fallback += event_prompt_block
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
        degradation_log.append(_make_degradation_entry(
            "Step5c_最终兜底", "rules_only", severity="critical",
            reason="所有LLM模型（本地+云端）均失败，启用纯规则引擎生成保守建议",
            fix="1) 所有模型均不可用，这是最严重的降级\n"
                "2) 检查Ollama: ollama serve && ollama ps\n"
                "3) 检查GPU: nvidia-smi\n"
                "4) 检查云端Key: echo %GEMINI_API_KEY%\n"
                "5) 检查网络: ping google.com\n"
                "6) 纯规则建议仅基于止损止盈，不含AI分析，请勿直接执行",
        ))

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
        rebalance["new_candidates"] = _apply_candidate_timing_guards(annotated_candidates, hot_picks)
        rebalance.setdefault("_meta", {})
        rebalance["_meta"]["execution_profile_source"] = execution_profile.source
        rebalance["_meta"]["execution_profile_samples"] = execution_profile.sample_size
    except Exception as e:
        logger.warning("[执行数量规划] 生成整手数量失败，保留原始建议: %s", e)
        degradation_log.append(_make_degradation_entry(
            "Step5_执行规划", "trade_sizing", error=e,
            reason="整手数量规划失败，建议中不含具体手数，需手动计算",
            fix="1) 检查 risk_control.py 中 MAX_SINGLE_POSITION_PCT 配置\n"
                "2) 确认 trade_sizing_service 模块无错误",
        ))

    # ── Step 5d: 回溯验证 — 置信度标注 ──
    logger.info("\n[Step 5d] Agent 5: 回溯验证（置信度标注）...")
    validator_model_used = "skipped"
    try:
        backtest_ctx = macro_data.get("backtest_context", {})
        if rebalance.get("actions") and backtest_ctx:
            prompt_validator = PROMPT_BACKTEST_VALIDATOR.format(
                rebalance_decision=json.dumps(
                    {"actions": rebalance.get("actions", [])},
                    ensure_ascii=False, indent=2,
                ),
                trade_performance=json.dumps(
                    backtest_ctx.get("trade_performance", {}),
                    ensure_ascii=False, indent=2,
                ),
                winning_patterns=json.dumps(
                    backtest_ctx.get("winning_patterns", {}),
                    ensure_ascii=False, indent=2,
                ),
                scan_accuracy=json.dumps(
                    backtest_ctx.get("scan_accuracy", {}),
                    ensure_ascii=False, indent=2,
                ),
            )
            validator_raw, validator_model_used = _call_scan_llm(
                prompt_validator, "Agent5_回溯验证", return_model=True,
            )
            validation = _parse_llm_json(validator_raw)
            _save_agent_local_sample("agent5_validator", prompt_validator, validator_raw, validation)
            if validation:
                rebalance["backtest_validation"] = validation
                logger.info(f"  整体置信度: {validation.get('overall_confidence', 'N/A')}")
                for pac in validation.get("per_action_confidence", []):
                    logger.info(f"    {pac.get('code','?')}: 置信{pac.get('confidence','?')} "
                                f"| {pac.get('warning','')}")
            else:
                logger.info("  回溯验证解析失败，跳过")
                degradation_log.append(_make_degradation_entry(
                    "Step5d_回溯验证", "Agent5_回溯验证", severity="warning",
                    reason="回溯验证结果解析失败，调仓建议无置信度标注",
                    fix="1) 模型返回非JSON，查看日志中Agent5原始返回\n"
                        "2) 不影响调仓建议本身，仅缺少置信度参考",
                ))
        else:
            logger.info("  无决策或无回测数据，跳过验证")
    except Exception as e:
        logger.warning(f"  回溯验证跳过: {e}")
        degradation_log.append(_make_degradation_entry(
            "Step5d_回溯验证", "Agent5_回溯验证", error=e,
        ))

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

    # ── 降级日志汇总 ──
    if degradation_log:
        logger.warning(f"[降级汇总] 本次调仓共 {len(degradation_log)} 处降级:")
        for i, d in enumerate(degradation_log, 1):
            logger.warning(f"  [{i}] [{d['severity']}] {d['step']} | {d['agent']}: {d['reason']}")
        rebalance["degradation_log"] = degradation_log
        # 生成用户友好的降级摘要
        critical_count = sum(1 for d in degradation_log if d["severity"] == "critical")
        error_count = sum(1 for d in degradation_log if d["severity"] == "error")
        warning_count = sum(1 for d in degradation_log if d["severity"] == "warning")
        severity_parts = []
        if critical_count:
            severity_parts.append(f"{critical_count}个严重")
        if error_count:
            severity_parts.append(f"{error_count}个错误")
        if warning_count:
            severity_parts.append(f"{warning_count}个警告")
        rebalance["degradation_summary"] = (
            f"⚠️ 本次分析有{len(degradation_log)}处降级（{'、'.join(severity_parts)}），"
            f"结果可靠性{'严重降低' if critical_count else '有所降低' if error_count else '轻微影响'}。"
        )
    else:
        rebalance["degradation_log"] = []
        rebalance["degradation_summary"] = "✅ 全流程正常，无降级。"

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
        "sentiment_model_used": sentiment_model_used,
        "fundamental_model_used": fundamental_model_used,
        "validator_model_used": validator_model_used,
        "degradation_count": len(degradation_log),
        "event_action": event_action,
        "event_summary": event_summary,
        "hot_concepts_count": len(external_hot_concepts),
        **rebalance.get("_meta", {}),
    }

    # ── 保存 AI 目标价（供盘中监控自动止盈止损）──
    try:
        from src.broker.price_target_store import save_price_targets
        all_ai_actions = rebalance.get("actions", []) + rebalance.get("new_candidates", [])
        saved_count = save_price_targets(all_ai_actions, source="rebalance")
        if saved_count:
            rebalance.setdefault("_meta", {})["price_targets_saved"] = saved_count
    except Exception as e:
        logger.debug(f"[目标价] 保存跳过: {e}")

    # ── 即时卖出结果记录 ──
    if immediate_results:
        rebalance["_immediate_execution"] = [r.to_dict() for r in immediate_results]
        sold_count = sum(1 for r in immediate_results if r.is_success)
        logger.info(f"[即时卖出] 共{len(immediate_results)}笔，成功{sold_count}笔")

    # ── 券商自动执行钩子 ──
    if os.getenv("BROKER_ENABLED", "false").lower() == "true":
        try:
            from src.broker import get_broker
            from src.broker.executor import RebalanceExecutor
            broker = get_broker()
            if broker and broker.is_connected():
                confirm_mode = os.getenv("BROKER_CONFIRM_MODE", "confirm")
                executor = RebalanceExecutor(broker)
                actions = rebalance.get("actions", [])

                # 跳过已通过即时卖出执行的股票
                if immediate_results:
                    executed_codes = {r.code for r in immediate_results if r.is_success}
                    before_len = len(actions)
                    actions = [a for a in actions if a.get("code") not in executed_codes]
                    if before_len != len(actions):
                        logger.info(f"[券商] 跳过{before_len - len(actions)}只已即时卖出的股票")

                total_asset = portfolio.get("total_asset", 0)

                if confirm_mode == "auto":
                    exec_report = executor.execute(actions, mode="auto", total_asset=total_asset)
                    rebalance["_execution"] = exec_report.to_dict()
                    logger.info(f"[券商] 自动执行完成: {exec_report.format_summary()}")
                elif confirm_mode == "dry_run":
                    dry = executor._dry_run(actions, total_asset)
                    rebalance["_execution"] = {"mode": "dry_run", "actions": dry}
                    logger.info(f"[券商] dry_run: {len(dry)}笔模拟订单")
                else:
                    # confirm 模式: 只生成确认消息，不执行
                    rebalance["_execution"] = {
                        "mode": "confirm",
                        "pending": True,
                        "confirmation_message": executor.format_confirmation_message(actions),
                    }
                    logger.info("[券商] confirm模式: 等待飞书确认后执行")
            else:
                logger.info("[券商] 未连接，跳过自动执行")
        except ImportError as e:
            logger.warning(f"[券商] 模块导入失败: {e}")
        except Exception as e:
            logger.error(f"[券商] 执行钩子异常: {e}")
            rebalance["_execution"] = {"error": str(e)}

    # ── 可转债 T+0 附加模块（双模式）──
    try:
        from cb_scanner import scan_convertible_bonds
        logger.info("[CB T+0] 附加可转债扫描...")
        cb_candidates = scan_convertible_bonds(top_n=5, fetch_daily=False)
        cb_buy_list = [
            {"code": c.code, "name": c.name, "price": c.price,
             "score": c.score, "premium_rate": c.premium_rate,
             "signal": c.signal}
            for c in cb_candidates if c.signal == "buy"
        ]
        rebalance["cb_candidates"] = cb_buy_list
        logger.info(f"[CB T+0] {len(cb_buy_list)} 只可买入转债")
    except Exception as e:
        logger.debug(f"[CB T+0] 可转债扫描跳过: {e}")

    return rebalance
