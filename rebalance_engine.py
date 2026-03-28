"""
rebalance_engine.py — 多 Agent 调仓决策引擎
放在项目根目录，与 main.py / analyzer_service.py 同级

架构:
  Agent 1-3: 本地 Ollama 模型（省钱干苦力）
  Agent 4:   云端强模型（最终仲裁，一天仅一次调用）
  蒸馏采集:  每次云端调用的 prompt+response 自动存为训练样本
             积累后可用于 LoRA 微调本地模型

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
from macro_data_collector import collect_full_macro_data
from portfolio_manager import (
    load_portfolio, update_current_prices, format_rebalance_report,
)

# 蒸馏数据保存目录
DISTILL_DIR = Path("data/distillation")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM 调用封装（区分本地 / 云端）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_local_llm(prompt: str, agent_name: str = "") -> str:
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
        return result
    except Exception as e:
        logger.error(f"[{agent_name}] 本地 LLM 调用失败: {e}")
        return "{}"


def _call_cloud_llm(prompt: str, agent_name: str = "") -> str:
    """
    调用云端强模型（Agent 4 仲裁用，一天仅一次）
    优先使用 CLOUD_MODEL 环境变量，否则使用 LITELLM_FALLBACK_MODELS 的第一个
    """
    import litellm

    # 云端模型优先级：REBALANCE_CLOUD_MODEL > 主项目的 LITELLM_MODEL > 回退到本地
    cloud_model = os.getenv("REBALANCE_CLOUD_MODEL")
    if not cloud_model:
        cloud_model = os.getenv("LITELLM_MODEL")
    if not cloud_model:
        # 没有云端配置，回退到本地模型
        logger.warning(f"[{agent_name}] 未配置云端模型，回退到本地模型")
        return _call_local_llm(prompt, agent_name)

    try:
        logger.info(f"[{agent_name}] 调用云端模型: {cloud_model}")
        response = litellm.completion(
            model=cloud_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=300,
            temperature=0.3,
            num_retries=1,
        )
        result = response.choices[0].message.content
        logger.info(f"[{agent_name}] 云端模型返回 {len(result)} 字符")
        return result
    except Exception as e:
        logger.error(f"[{agent_name}] 云端 LLM 调用失败: {e}，回退到本地模型")
        return _call_local_llm(prompt, agent_name)


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
8. 持仓天数必须严格按照 buy_date 字段计算到今天的日历天数，不要编造
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
    logger.info("=" * 60)
    logger.info("开始执行多Agent调仓分析...")
    logger.info(f"本地模型: {os.getenv('REBALANCE_LOCAL_MODEL', '未配置')}")
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
    market_judge_raw = _call_local_llm(prompt_market, "Agent1_大盘")
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
    sector_judge_raw = _call_local_llm(prompt_sector, "Agent2_板块")
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
        rating_raw = _call_local_llm(prompt_holding, f"Agent3_{name}")
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

    # ── Step 5: Agent 4 — 调仓仲裁（云端强模型 + 蒸馏采集）──
    logger.info("\n[Step 5/5] Agent 4: 调仓决策仲裁（云端模型）...")

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

    prompt_final = PROMPT_REBALANCE_FINAL.format(
        market_judge=json.dumps(market_judge, ensure_ascii=False, indent=2),
        sector_judge=json.dumps(sector_judge, ensure_ascii=False, indent=2),
        holdings_ratings=json.dumps(
            holdings_ratings, ensure_ascii=False, indent=2
        ),
        portfolio=json.dumps(
            {
                "cash": portfolio.get("cash", 0),
                "total_asset": portfolio.get("total_asset", 0),
                "actual_position_ratio": portfolio.get("actual_position_ratio", 0),
                "today": datetime.now().strftime("%Y-%m-%d"),
                "holdings": [
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
                        "hold_days": (datetime.now() - datetime.strptime(hh["buy_date"], "%Y-%m-%d")).days if hh.get("buy_date") else "未知",
                    }
                    for hh in portfolio.get("holdings", [])
                ],
            },
            ensure_ascii=False, indent=2,
        ),
        hot_picks=json.dumps(hot_picks[:10], ensure_ascii=False, indent=2),
    )
    # 注入风控预警到 prompt
    if risk_alerts_text:
        prompt_final = prompt_final + risk_alerts_text

    # ★ 云端调用 + 蒸馏采集 ★
    rebalance_raw = _call_cloud_llm(prompt_final, "Agent4_仲裁")
    logger.info(f"[Agent4] 云端原始返回长度: {len(rebalance_raw)} 字符")
    if len(rebalance_raw) < 10:
        logger.error(f"[Agent4] 云端返回内容过短: {rebalance_raw!r}")
    rebalance = _parse_llm_json(rebalance_raw)
    if not rebalance:
        logger.error(f"[Agent4] JSON解析失败！原始返回前500字:\n{rebalance_raw[:500]}")

    # 保存蒸馏样本（云端模型的输出 = 本地模型未来的学习目标）
    _save_distillation_sample(
        agent_name="agent4_rebalance_final",
        prompt=prompt_final,
        response=rebalance_raw,
        parsed_json=rebalance,
        metadata={
            "holdings_count": len(holding_codes),
            "holding_codes": holding_codes,
            "market_stage": market_judge.get("market_stage", ""),
            "hot_sectors": sector_judge.get("hot_sectors", []),
        },
    )

    # 同时也让本地模型回答同一个问题，保存对比数据
    # （后续可用来评估蒸馏/微调效果）
    try:
        local_answer_raw = _call_local_llm(prompt_final, "Agent4_本地对照")
        local_answer = _parse_llm_json(local_answer_raw)
        _save_agent_local_sample(
            "agent4_local_comparison", prompt_final, local_answer_raw, local_answer
        )
        logger.info("[蒸馏] 本地对照样本已保存（用于后续对比评估）")
    except Exception:
        pass  # 对照不影响主流程

    elapsed = round(time.time() - start_time, 1)
    logger.info(f"\n调仓分析完成！耗时 {elapsed} 秒")
    logger.info(
        f"总仓位建议: {rebalance.get('overall_position_advice', 'N/A')}"
    )

    rebalance["_meta"] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "holdings_count": len(holding_codes),
        "agents_used": 4,
        "local_model": os.getenv("REBALANCE_LOCAL_MODEL", "unknown"),
        "cloud_model": cloud_model,
    }

    return rebalance