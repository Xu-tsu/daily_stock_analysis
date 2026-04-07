# Multi-Agent 辩论架构 × LangGraph 对话引擎：一个全自动 A 股交易系统的设计与实现

> 让 AI 像真实的投资团队一样辩论、决策、执行

---

## 这个系统解决了什么问题

把交易决策交给**单一 AI**，和把所有工作交给一个人一样不靠谱——"帮我看看大盘，顺便分析一下持仓，再决定买什么卖什么"，模型只会浮于表面，每个维度都不够深入。

本系统采用 **Multi-Agent 辩论 + 仲裁** 架构，让不同"角色"各司其职：

| Agent | 角色 | 模型 | 职责 |
|-------|------|------|------|
| Agent 1 | 宏观分析师 | Azure GPT-5.4-nano | 大盘研判：牛熊阶段、仓位建议 |
| Agent 2 | 板块分析师 | Azure GPT-5.4-nano | 板块轮动、主线题材判断 |
| Agent 2b | 情绪分析师 | Azure GPT-5.4-nano | 恐贪指数、情绪周期定位 |
| Agent 3 | 持仓扫描员 | 云端并行 ×6 | 逐只技术面+筹码+资金流分析 |
| Agent 3b | 基本面研究员 | 云端并行 | 财报、增长趋势、财务风险 |
| **Agent 4a** | **激进派交易员** | Qwen 14B (本地) | 提出完整调仓方案 |
| **Agent 4b** | **保守派风控** | DeepSeek-R1 14B (本地) | 对方案逐条质疑 |
| **Agent 4c** | **仲裁者** | Gemini / Azure (云端) | 综合双方意见，做出最终决策 |

每个 Agent 只关注**一个维度**，深度远超"一个 prompt 做所有事"。而"辩论 → 仲裁"机制，确保最终决策经过**多角度对抗验证**。

---

## 目录

1. [系统架构总览](#系统架构总览)
2. [Multi-Agent 辩论：核心调仓引擎](#multi-agent-辩论核心调仓引擎)
3. [LangGraph 对话引擎：飞书机器人的大脑](#langgraph-对话引擎飞书机器人的大脑)
4. [关键设计：确认工作流与做 T](#关键设计确认工作流与做-t)
5. [云端仲裁降级链：永不停机](#云端仲裁降级链永不停机)
6. [券商自动下单：从决策到执行](#券商自动下单从决策到执行)
7. [盘后智能系统：复盘 + 涨停分析](#盘后智能系统复盘--涨停分析)
8. [设计中踩过的坑](#设计中踩过的坑)

---

## 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        飞书 / 钉钉 / WebUI                       │
│                   用户指令："买入 002506 500股 5.4元"               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  LangGraph 引擎  │  意图识别 → 风控 → 确认 → 执行
                    │  (对话状态机)     │  支持多轮对话、做T、确认工作流
                    └───────┬────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
  ┌───────▼──────┐  ┌──────▼───────┐  ┌──────▼───────┐
  │  调仓分析引擎  │  │  全市场扫描    │  │  券商适配器    │
  │  (7步流水线)  │  │  (15000只筛选) │  │  (同花顺下单)  │
  └───────┬──────┘  └──────────────┘  └──────────────┘
          │
  ┌───────▼────────────────────────────────────┐
  │            Multi-Agent 辩论决策               │
  │                                              │
  │  ┌────────┐  ┌────────┐  ┌────────┐         │
  │  │ Agent1 │  │ Agent2 │  │Agent2b │         │
  │  │ 大盘   │  │ 板块   │  │ 情绪   │         │
  │  └───┬────┘  └───┬────┘  └───┬────┘         │
  │      └───────────┼───────────┘               │
  │                  │                           │
  │  ┌───────────────▼────────────────┐          │
  │  │  Agent3: 持仓扫描 (×6 并行)     │          │
  │  │  Agent3b: 基本面研究 (×6 并行)  │          │
  │  └───────────────┬────────────────┘          │
  │                  │                           │
  │  ┌───────────────▼────────────────┐          │
  │  │  🗣️ 辩论环节                    │          │
  │  │                                │          │
  │  │  Agent4a (激进派/Qwen)          │          │
  │  │    "建议清仓新里程，加仓光伏"     │          │
  │  │         │                      │          │
  │  │         ▼                      │          │
  │  │  Agent4b (保守派/DeepSeek-R1)   │          │
  │  │    "反对！新里程支撑位有效，      │          │
  │  │     光伏追高风险极大"            │          │
  │  │         │                      │          │
  │  │         ▼                      │          │
  │  │  Agent4c (仲裁者/Gemini)        │          │
  │  │    "采纳保守派意见，新里程持有；  │          │
  │  │     光伏减半建仓，设止损7.5"     │          │
  │  └────────────────────────────────┘          │
  └──────────────────────────────────────────────┘
```

> **图1：系统架构全景图** — 用户指令经 LangGraph 路由，交易决策经 Multi-Agent 辩论，最终由券商适配器自动执行

---

## Multi-Agent 辩论：核心调仓引擎

### 为什么要"辩论"而不是直接让一个模型做决策？

单一模型的问题在于**自我一致性偏差**——它提出一个方案后，很难自己推翻自己。而人类投资团队的优势恰恰在于**观点碰撞**：交易员想追涨，风控经理会泼冷水，最后由投资总监拍板。

我们的 Multi-Agent 辩论完全模拟了这个过程：

### 7 步流水线

```
Step 0: 数据准备
  ├── 从券商同步真实持仓 (broker → portfolio.json)
  ├── 覆盖实时报价
  └── 加载上次分析结果（决策连续性）

Step 1: 宏观数据采集 (纯 Python, 0 token)
  ├── 沪深300 / 创业板指 / 北向资金
  ├── 板块资金流向 / 融资融券
  └── 快讯 / 舆情

Step 2: Agent1 大盘研判 → "震荡筑底，建议仓位 45%"
Step 3: Agent2 板块轮动 → "资金从消费转向科技"
Step 3b: Agent2b 情绪分析 → "修复期，恐贪指数 50"

Step 4: Agent3 持仓扫描 (云端 ×6 并行, 12秒完成3只)
  ├── 永和智控: 66分 → 持有
  ├── 新里程: 56分 → 减仓        ← 触发即时卖出检查
  └── 千红制药: 62分 → 持有

Step 4b: Agent3b 基本面研究 (云端 ×6 并行, 2秒)

Step 5: 辩论
  5a: 激进派提出方案 (含 execution_strategy)
  5b: 保守派逐条质疑
  5c: 仲裁者综合决策 → 最终 actions[]
```

### 辩论的核心：激进派 vs 保守派

```python
# 激进派 (Agent4a) 的输出
{
    "actions": [{
        "code": "002219", "name": "新里程",
        "action": "sell",
        "reason": "技术面走弱，MA5下穿MA20，果断清仓",
        "target_sell_price": 2.50,
        "execution_strategy": {
            "urgency": "high",
            "order_type": "aggressive",
            "chase_max_pct": 0.5
        }
    }]
}

# 保守派 (Agent4b) 的质疑
{
    "critical_issues": [
        "新里程 2.42 元已接近支撑位 2.38，此时清仓等于割在地板上"
    ],
    "position_disagreements": [{
        "code": "002219",
        "original_action": "sell",
        "my_suggestion": "hold",
        "reason": "支撑位 2.38 有效，等待反弹后再决策更稳妥"
    }]
}

# 仲裁者 (Agent4c) 的最终决策
{
    "revised_actions": [{
        "code": "002219", "name": "新里程",
        "action": "hold",  # 采纳保守派意见
        "reason": "保守派指出支撑位有效，暂持有观察，跌破 2.35 再止损"
    }]
}
```

> **图2：辩论流程** — 激进派提出方案 → 保守派逐条质疑 → 仲裁者做出最终决策。每个 Agent 使用不同模型，避免思维同质化。

### 分析一只、执行一只

传统流程是"分析全部 → 生成方案 → 批量执行"，但止损信号每分钟都在变化，等 40 分钟分析完可能已错过最佳时机。

```python
# Step 4 持仓扫描循环内：
for h in holdings:
    rating = _scan_one_holding(h)  # 云端 5-15 秒

    # 高风险立即执行，不等辩论
    if rating["score"] <= 30 and rating["action"] == "sell":
        broker.sell(h["code"], h["current_price"], h["sellable_shares"])
        logger.info(f"[即时卖出] {h['name']} 评分{rating['score']}，已执行")
```

**卖出求速度（即时执行），买入求全局（走完辩论）**——这是实战中最重要的原则。

---

## LangGraph 对话引擎：飞书机器人的大脑

### 共享状态：所有节点的记忆

```python
class PortfolioGraphState(TypedDict):
    messages: Annotated[list, add_messages]  # 自动累积对话历史
    user_text: str                           # 当前用户输入
    intent: Optional[str]                    # 路由意图

    # 交易参数
    trade_action: Optional[dict]     # {"code", "shares", "price", "action"}
    risk_check: Optional[dict]       # {"allowed", "warnings", "blocked_reason"}

    # 确认工作流
    pending_confirmation: bool       # 图暂停等待用户确认
    confirmed: Optional[bool]        # 用户回复确认/取消

    # 做T工作流
    t0_phase: Optional[str]          # plan → confirm_sell → monitoring → done

    # 券商
    broker_enabled: bool
    broker_order_result: Optional[dict]

    # 输出
    response: str
```

### 意图路由：LLM 不参与的高速分类

与参考文章中用 LLM 做路由不同，我们选择了**正则 + 关键词**的硬编码路由。原因很简单：交易指令的格式是确定的（"买入 002506 500 5.4"），用 LLM 反而更慢、更不稳定。

```
entry_node (意图识别 + 持仓加载)
  │
  ├─ "持仓" → view_portfolio_node → END
  ├─ "买入/卖出/清仓" → risk_check → confirm → execute_trade → END
  ├─ "做T 002506" → validate_t0 → plan_t0 → [暂停] → execute_sell → [暂停] → buyback → END
  ├─ "调仓" → rebalance_node (触发 Multi-Agent) → END
  ├─ "涨停分析" → limit_up_analysis_node → END
  └─ 其他 → chat_node (LLM 自由对话) → END
```

> **图3：LangGraph 对话状态机** — 意图识别后条件路由，交易类指令经过风控+确认双重关卡

### 条件路由实现

```python
builder.add_conditional_edges(
    "route_intent_dummy",
    route_intent,
    {
        "view_portfolio": "view_portfolio",
        "risk_check": "risk_check",         # 买/卖/清仓
        "validate_t0": "validate_t0",       # 做T
        "rebalance": "rebalance",
        "scan": "scan",
        "chat": "chat",
        # ... 12 种意图
    },
)

# 风控后的二次路由
builder.add_conditional_edges(
    "risk_check",
    route_after_risk_check,
    {
        "confirm": "confirm",     # 正常 → 请求确认
        "end": END,               # 被风控拦截 → 直接结束
    },
)
```

---

## 关键设计：确认工作流与做 T

### 确认工作流：图的暂停与恢复

交易不能一步到位——用户说"买入"后，必须先展示订单详情，等用户确认。

```
用户: "买入 002506 500 5.4"
  │
  ▼
entry → risk_check → confirm_node
  │
  │  response: "📋 确认买入：协鑫集成 500股×5.4元=2700元"
  │  pending_confirmation = True
  │
  ▼  ← 图暂停，checkpoint 保存状态

用户: "确认"
  │
  ▼  ← invoke_graph 检测到 pending_confirmation=True
  │     进入恢复路由
  │
entry → route_confirmation → execute_trade
  │
  │  response: "✅ 已买入，剩余现金 7328 元"
  │  pending_confirmation = False
  ▼
 END
```

> **图4：确认工作流时序图** — 图在 confirm_node 暂停，用户确认后通过 checkpoint 恢复执行

### 做 T 工作流：跨消息的状态机

做 T（日内高抛低吸）需要跨越多条消息，状态机通过 `t0_phase` 字段管理：

```
用户: "做T 002506"
  → validate_t0: 检查可卖余额 200 股
  → plan_t0: "建议卖 8.68 元，目标回补 8.53 元"
  → pending_confirmation = True (暂停)

用户: "确认"
  → execute_t0_sell: 卖出 200 股 × 8.68 元
  → t0_phase = "monitoring"
  → response: "卖出完成，等价格回落后发送「回补」"

(等待价格回落...)

用户: "回补"
  → execute_t0_buyback: 买入 200 股 × 8.50 元
  → t0_phase = "done"
  → response: "做T完成！赚 0.18 元/股，共 +36 元"
```

---

## 云端仲裁降级链：永不停机

真实环境中，任何 API 都可能超时或限频。我们设计了**四级降级链**：

```
云端主力 (Azure GPT-5.4-nano)
  │ 失败
  ▼
云端备用 (Gemini 2.5 Flash)
  │ 失败
  ▼
全局默认 (LITELLM_MODEL)
  │ 失败
  ▼
本地兜底 (DeepSeek-R1 14B, Ollama)
```

```python
def _call_cloud_llm(prompt, agent_name=""):
    candidates = [CLOUD_MODEL, CLOUD_FALLBACK, LITELLM_MODEL]
    for i, model in enumerate(candidates):
        try:
            response = litellm.completion(model=model, ...)
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"[{agent_name}] {model} 失败: {e}")
            continue
    # 全部云端挂了 → 本地 DeepSeek 兜底
    return _call_debate_llm(prompt, f"{agent_name}_本地回退")
```

**降级日志**会完整记录每次降级事件，便于后续分析模型稳定性。

---

## 券商自动下单：从决策到执行

### 智能追单

A 股限价委托经常不能立即成交。系统会自动**撤单 → 重新报价 → 轮询**：

```python
def _smart_execute(self, direction, code, price, shares, limit_price, timeout):
    """提交 → 等待 → 未成交 → 撤单 → 获取实时价 → 重新报价"""
    current_price = price
    while time.time() - start < timeout:
        result = trade_fn(code, current_price, shares)
        time.sleep(poll_interval)

        if self._check_order_fill(code, shares):
            return OrderResult(status="filled", ...)

        # 未成交，获取实时价重新报价
        realtime = self._get_realtime_price(code)
        if realtime and realtime <= limit_price:
            self._cancel_latest_pending(code)
            current_price = realtime
            continue
        break
```

### AI 决定执行策略

追单参数不是写死的——每只股票的执行策略由 AI 辩论决定：

```json
{
    "execution_strategy": {
        "urgency": "high",
        "chase_max_pct": 0.5,
        "chase_timeout": 60,
        "order_type": "aggressive",
        "split_orders": false,
        "reason": "止损信号明确，尽快成交"
    }
}
```

---

## 盘后智能系统：复盘 + 涨停分析

### AI 复盘笔记

每日收盘后自动执行，分析当日盈亏原因：

```
收盘任务流程：
  15:00  分时峰值统计（执行价 vs 日内最优价偏差）
  15:01  AI 复盘笔记 → 推送飞书
  15:02  涨停深度分析 → 推送飞书
  15:03  扫描回测
  15:05  调仓分析（Multi-Agent 辩论）
```

复盘数据来源：
- `trade_log`：当日买卖记录 + 技术指标快照
- `intraday_peak_stats`：执行价与日内高低点的偏差
- `rebalance_history`：AI 决策记录 vs 实际执行
- 30 天战绩 + 90 天模式分析

### 涨停股深度分析

自动采集涨停池 → 龙虎榜 → 资金流向 → 网络搜索 → 关联股挖掘：

```
数据采集（并行）:
  ├── ak.stock_zt_pool_em → 涨停池（连板数/封板时间/行业）
  ├── ak.stock_lhb_detail_em → 龙虎榜（机构/游资买卖席位）
  └── fetch_fund_flow_rank → 主力资金流向

关联股挖掘:
  涨停股 → 所属概念 → 统计共现频次 ≥ 2 → 获取概念成分股
  → 排除已涨停 → 按涨幅排序 → 输出"明日潜在受益股"

AI 分析:
  → 情绪总览 + 核心主线 + 连板分析 + 明日机会 + 风险提示
  → 推送飞书 / 钉钉 / 邮件
```

> **图5：盘后分析流程** — 复盘笔记分析"为什么赚/亏"，涨停分析找"明天可能涨什么"

---

## 设计中踩过的坑

### ① 路由用 LLM 还是硬编码？

参考文章中 Supervisor 用 LLM 判断"下一步做什么"，这在开放性任务中是好的设计。但在交易系统中，意图是**结构化且有限的**——"买入/卖出/持仓/调仓"，用 LLM 路由反而引入不确定性（偶尔把"卖出"识别为"查询"）。

**我们的选择**：入口用正则 + 关键词硬路由（确定性 100%），仅在无法匹配时才 fallback 到 LLM 的 chat 节点。

### ② 辩论模型要用不同的吗？

必须！如果激进派和保守派都用同一个模型，它们会产出高度相似的分析——失去了辩论的意义。

我们用 Qwen 做激进派（中文理解强，敢于提出激进方案），DeepSeek-R1 做保守派（思维链 `<think>` 模式，擅长逻辑推理和找漏洞），Gemini 做仲裁（大上下文窗口，能同时消化双方观点）。

### ③ 确认工作流的图暂停

LangGraph 的 `MemorySaver` checkpointer 可以保存图执行到一半的状态。当 `pending_confirmation=True` 时，图自然结束；下一条消息到来时，通过 `thread_id` 恢复状态，检查到待确认标志，走恢复路由。

**坑**：如果用户在确认前发了一条完全不相关的消息（比如"今天天气怎么样"），必须正确处理——既不能执行交易，也不能丢失待确认状态。我们在 entry_node 中特判了这种情况，返回提示后保持 `pending_confirmation=True`。

### ④ 本地模型并发

本地 Ollama 只有一块 GPU，无法并发。如果 6 个 Agent3 同时请求，GPU 显存直接 OOM。

**解决方案**：云端扫描用 6 线程并行（5-15s），本地扫描退化为串行（100-260s）。通过 `REBALANCE_SCAN_USE_CLOUD=true` 环境变量控制。

### ⑤ 数据源信任层级

portfolio.json 是本地文件，可能和券商实际持仓不一致（手动交易、部分成交等）。

**信任链**：`券商实时持仓 > trade_log 推算 > portfolio.json`。每次调仓分析前，先调用 `sync_portfolio_from_broker()` 从同花顺客户端拉取真实仓位。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 对话引擎 | LangGraph + MemorySaver (状态机 + checkpoint) |
| 多Agent调仓 | 自研 7 步流水线 + 辩论仲裁 |
| 云端 LLM | Azure GPT-5.4-nano / Gemini 2.5 Flash (litellm 统一调用) |
| 本地 LLM | Ollama + Qwen2.5-14B / DeepSeek-R1-14B |
| 券商对接 | easytrader + 同花顺远航版 |
| 数据源 | akshare + 腾讯行情 + Tushare + 东方财富 |
| 搜索引擎 | Tavily (涨停原因搜索) |
| 消息平台 | 飞书 Stream / 钉钉 Webhook / 邮件 |
| 存储 | SQLite (trade_log / intraday_ticks) + JSON (portfolio) |
| 监控 | 盘中分时采集 + AI 目标价实时检测 + 自动止盈止损 |

---

## 扩展方向

- **微调本地模型**：系统已自动采集微调样本（JSONL 格式），积累 500+ 条后可用 LLaMA-Factory 微调 Qwen，让本地模型逐步替代云端
- **Human-in-the-loop**：LangGraph `interrupt()` 在辩论结果出来后暂停，让用户审核再执行
- **跨市场**：当前仅 A 股，架构设计支持扩展到港股/美股（已有 US market review 模块）

---

## 总结

这个项目让我深刻理解了三件事：

1. **Multi-Agent 不是噱头**——当每个 Agent 只负责一个维度时，分析深度远超单一模型。辩论机制更是让决策质量产生了质变。

2. **LangGraph 的价值在于状态管理**——交易系统的多轮对话、确认工作流、做 T 的跨消息状态机，用传统的 if-else 会写成一团乱麻，LangGraph 的条件路由 + checkpoint 让这一切变得清晰。

3. **降级链是生产系统的生命线**——云端 API 一定会挂，本地 GPU 一定会 OOM。四级降级链确保系统"永不停机"，即使所有云端都不可用，本地 DeepSeek 也能兜底出一个可用的决策。

如果你也想实践 Multi-Agent 架构，不妨从一个简单的"辩论 → 仲裁"开始——两个模型对同一个问题给出相反观点，第三个模型做仲裁。这比任何教程都能更快理解 Agent 协作的本质。

---

*作者：华尔街之狼项目组*
*技术栈：Python + LangGraph + litellm + easytrader + akshare*
