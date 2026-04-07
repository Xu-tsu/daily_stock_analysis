# 文章配图（Mermaid 格式）

> 使用 https://mermaid.live 在线渲染为 PNG/SVG，或用 VS Code Mermaid 插件预览

---

## 图1：系统架构全景图

```mermaid
graph TB
    User["👤 用户<br/>飞书 / 钉钉 / WebUI"]

    subgraph LangGraph["🧠 LangGraph 对话引擎"]
        Entry["Entry Node<br/>意图识别"]
        Risk["Risk Check<br/>风控检查"]
        Confirm["Confirm Node<br/>确认工作流"]
        Execute["Execute Trade<br/>执行交易"]
        Chat["Chat Node<br/>自由对话"]
        T0["T0 Workflow<br/>做T状态机"]
    end

    subgraph Rebalance["⚖️ Multi-Agent 调仓引擎"]
        A1["Agent1<br/>🏛️ 大盘研判"]
        A2["Agent2<br/>🔄 板块轮动"]
        A2b["Agent2b<br/>😰 情绪分析"]
        A3["Agent3 ×6<br/>📊 持仓扫描"]
        A3b["Agent3b ×6<br/>📋 基本面"]

        subgraph Debate["🗣️ 辩论环节"]
            A4a["Agent4a<br/>🔥 激进派<br/>(Qwen)"]
            A4b["Agent4b<br/>🛡️ 保守派<br/>(DeepSeek-R1)"]
            A4c["Agent4c<br/>⚖️ 仲裁者<br/>(Gemini)"]
        end
    end

    subgraph Broker["🏦 券商执行"]
        THS["同花顺<br/>easytrader"]
        Smart["智能追单<br/>撤单重报"]
    end

    subgraph PostMarket["📝 盘后系统"]
        Review["AI 复盘笔记"]
        LimitUp["涨停深度分析"]
    end

    User -->|指令| Entry
    Entry -->|买/卖| Risk
    Risk -->|通过| Confirm
    Confirm -->|确认| Execute
    Execute -->|下单| THS
    THS --> Smart
    Entry -->|调仓| A1
    Entry -->|聊天| Chat
    Entry -->|做T| T0
    A1 --> A2 --> A2b --> A3
    A3 --> A3b --> A4a
    A4a -->|方案| A4b
    A4b -->|质疑| A4c
    A4c -->|最终决策| Execute

    style Debate fill:#fff3cd,stroke:#ffc107
    style LangGraph fill:#e8f4fd,stroke:#2196f3
    style Rebalance fill:#f3e5f5,stroke:#9c27b0
    style Broker fill:#e8f5e9,stroke:#4caf50
    style PostMarket fill:#fce4ec,stroke:#e91e63
```

---

## 图2：Multi-Agent 辩论流程

```mermaid
sequenceDiagram
    participant Data as 📊 数据采集
    participant A1 as 🏛️ Agent1 大盘
    participant A2 as 🔄 Agent2 板块
    participant A3 as 📊 Agent3 持仓扫描
    participant A4a as 🔥 激进派 (Qwen)
    participant A4b as 🛡️ 保守派 (DeepSeek)
    participant A4c as ⚖️ 仲裁者 (Gemini)
    participant Exec as 🏦 券商执行

    Data->>A1: 指数+资金+快讯
    A1->>A2: 大盘判断: 震荡筑底
    A2->>A3: 板块判断: 科技→消费

    par 并行扫描 ×6
        A3->>A3: 永和智控 66分 持有
        A3->>A3: 新里程 56分 减仓
        A3->>A3: 千红制药 62分 持有
    end

    Note over A3,Exec: 评分≤30 → 即时卖出，不等辩论

    A3->>A4a: 全部分析结果
    A4a->>A4b: 方案: 清仓新里程，加仓光伏

    Note over A4b: 🤔 DeepSeek-R1 思维链<br/>《think》逐条推理《/think》

    A4b->>A4c: 质疑: 新里程支撑有效，反对清仓

    Note over A4c: ⚖️ 综合双方意见

    A4c->>Exec: 最终: 新里程持有，光伏半仓建仓
    Exec->>Exec: 智能追单 (撤单→重报→轮询)
```

---

## 图3：LangGraph 对话状态机

```mermaid
stateDiagram-v2
    [*] --> entry: 用户消息

    entry --> view_portfolio: "持仓"
    entry --> risk_check: "买入/卖出/清仓"
    entry --> validate_t0: "做T"
    entry --> rebalance: "调仓"
    entry --> scan: "扫描"
    entry --> analyze: "分析 002506"
    entry --> chat: 其他

    risk_check --> confirm: allowed=true
    risk_check --> [*]: blocked

    confirm --> [*]: pending_confirmation=true<br/>⏸️ 图暂停

    note right of confirm
        用户回复"确认"后
        图从 checkpoint 恢复
    end note

    confirm --> execute_trade: confirmed=true
    confirm --> [*]: confirmed=false

    execute_trade --> [*]: ✅ 交易完成

    validate_t0 --> plan_t0: 有可卖余额
    validate_t0 --> [*]: 无可卖余额

    plan_t0 --> [*]: pending_confirmation=true<br/>⏸️ 等待确认

    plan_t0 --> execute_t0_sell: confirmed
    execute_t0_sell --> [*]: t0_phase=monitoring<br/>等待"回补"

    execute_t0_sell --> execute_t0_buyback: "回补"
    execute_t0_buyback --> [*]: ✅ 做T完成

    view_portfolio --> [*]
    rebalance --> [*]
    scan --> [*]
    analyze --> [*]
    chat --> [*]
```

---

## 图4：确认工作流时序

```mermaid
sequenceDiagram
    actor User as 👤 用户
    participant FS as 📱 飞书
    participant Graph as 🧠 LangGraph
    participant CP as 💾 Checkpoint
    participant Broker as 🏦 券商

    User->>FS: "买入 002506 500 5.4"
    FS->>Graph: invoke_graph()
    Graph->>Graph: entry → risk_check → confirm
    Graph->>CP: 保存状态 (pending=true)
    Graph-->>FS: "📋 确认买入 500股×5.4元？"
    FS-->>User: 展示确认卡片

    Note over Graph,CP: ⏸️ 图暂停<br/>状态持久化在 checkpoint

    User->>FS: "确认"
    FS->>Graph: invoke_graph()
    Graph->>CP: 恢复状态 (pending=true)
    Graph->>Graph: 检测到确认 → execute_trade
    Graph->>Broker: sell/buy 下单
    Broker-->>Graph: 成交回报
    Graph->>CP: 清除状态 (pending=false)
    Graph-->>FS: "✅ 已买入，剩余现金 7328 元"
    FS-->>User: 交易完成通知
```

---

## 图5：盘后分析流程

```mermaid
graph LR
    Close["15:00<br/>收盘"]

    subgraph PostMarket["盘后自动任务"]
        Peak["📐 峰值统计<br/>执行价 vs 最优价"]
        Review["📝 AI 复盘<br/>盈亏原因分析"]
        ZT["🔥 涨停分析<br/>龙虎榜+关联股"]
        BT["📊 回测<br/>扫描策略验证"]
        RB["⚖️ 调仓分析<br/>Multi-Agent辩论"]
    end

    subgraph ZTDetail["涨停分析详情"]
        ZTPool["涨停池<br/>ak.stock_zt_pool_em"]
        LHB["龙虎榜<br/>ak.stock_lhb_detail_em"]
        Flow["资金流向<br/>主力净流入"]
        Search["网络搜索<br/>Tavily"]
        Concept["概念挖掘<br/>共现概念 ≥ 2"]
        AI["AI 分析<br/>主线+关联机会"]
    end

    subgraph Output["推送"]
        FS2["📱 飞书"]
        DD["💬 钉钉"]
        Email["📧 邮件"]
    end

    Close --> Peak --> Review --> ZT --> BT --> RB
    ZT --> ZTPool & LHB & Flow
    ZTPool --> Search --> Concept --> AI
    LHB --> AI
    Flow --> AI
    Review --> FS2 & DD & Email
    AI --> FS2 & DD & Email

    style PostMarket fill:#f5f5f5,stroke:#666
    style ZTDetail fill:#fff8e1,stroke:#ff9800
    style Output fill:#e8f5e9,stroke:#4caf50
```

---

## 图6：模型降级链

```mermaid
graph TD
    Request["LLM 请求"]

    M1["☁️ Azure GPT-5.4-nano<br/>(主力)"]
    M2["☁️ Gemini 2.5 Flash<br/>(备用1)"]
    M3["☁️ LITELLM_MODEL<br/>(备用2)"]
    M4["🖥️ DeepSeek-R1 14B<br/>(本地兜底)"]

    OK["✅ 返回结果"]

    Request --> M1
    M1 -->|成功| OK
    M1 -->|超时/限频| M2
    M2 -->|成功| OK
    M2 -->|失败| M3
    M3 -->|成功| OK
    M3 -->|全部云端挂了| M4
    M4 -->|成功| OK

    style M1 fill:#e3f2fd,stroke:#1976d2
    style M2 fill:#e8f5e9,stroke:#388e3c
    style M3 fill:#fff3e0,stroke:#f57c00
    style M4 fill:#fce4ec,stroke:#c62828
    style OK fill:#c8e6c9,stroke:#2e7d32
```

---

## 使用方式

1. 打开 https://mermaid.live
2. 粘贴任意图的 Mermaid 代码
3. 右上角导出为 PNG / SVG
4. 插入到文章对应位置

或者使用 VS Code 的 **Markdown Preview Mermaid Support** 插件直接预览。
