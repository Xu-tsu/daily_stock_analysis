# 記事配図（Mermaid形式）

> https://mermaid.live でPNG/SVGにレンダリング、またはVS Code Mermaid拡張でプレビュー

---

## 図1：システムアーキテクチャ全景

```mermaid
graph TB
    User["👤 ユーザー<br/>Feishu / DingTalk / WebUI"]

    subgraph LangGraph["🧠 LangGraph エンジン"]
        Entry["Entry Node<br/>意図分類"]
        Risk["Risk Check<br/>リスク管理"]
        Confirm["Confirm Node<br/>確認ワークフロー"]
        Execute["Execute Trade<br/>取引執行"]
        Chat["Chat Node<br/>自由会話"]
        T0["T0 Workflow<br/>デイトレステートマシン"]
    end

    subgraph Rebalance["⚖️ Multi-Agent リバランスエンジン"]
        A1["Agent1<br/>🏛️ マクロ判断"]
        A2["Agent2<br/>🔄 セクター分析"]
        A2b["Agent2b<br/>😰 センチメント"]
        A3["Agent3 ×6<br/>📊 銘柄スキャン"]
        A3b["Agent3b ×6<br/>📋 ファンダメンタル"]

        subgraph Debate["🗣️ ディベートフェーズ"]
            A4a["Agent4a<br/>🔥 攻撃派<br/>(Qwen)"]
            A4b["Agent4b<br/>🛡️ 守備派<br/>(DeepSeek-R1)"]
            A4c["Agent4c<br/>⚖️ 裁定者<br/>(Gemini)"]
        end
    end

    subgraph Broker["🏦 ブローカー執行"]
        THS["同花順<br/>easytrader"]
        Smart["インテリジェント<br/>追跡注文"]
    end

    subgraph PostMarket["📝 引け後システム"]
        Review["AI振り返りノート"]
        LimitUp["ストップ高<br/>深層分析"]
    end

    User -->|指示| Entry
    Entry -->|買い/売り| Risk
    Risk -->|通過| Confirm
    Confirm -->|確認| Execute
    Execute -->|発注| THS
    THS --> Smart
    Entry -->|リバランス| A1
    Entry -->|チャット| Chat
    Entry -->|デイトレ| T0
    A1 --> A2 --> A2b --> A3
    A3 --> A3b --> A4a
    A4a -->|提案| A4b
    A4b -->|反論| A4c
    A4c -->|最終判断| Execute

    style Debate fill:#fff3cd,stroke:#ffc107
    style LangGraph fill:#e8f4fd,stroke:#2196f3
    style Rebalance fill:#f3e5f5,stroke:#9c27b0
    style Broker fill:#e8f5e9,stroke:#4caf50
    style PostMarket fill:#fce4ec,stroke:#e91e63
```

---

## 図2：Multi-Agent ディベートフロー

```mermaid
sequenceDiagram
    participant Data as 📊 データ収集
    participant A1 as 🏛️ Agent1 マクロ
    participant A2 as 🔄 Agent2 セクター
    participant A3 as 📊 Agent3 銘柄スキャン
    participant A4a as 🔥 攻撃派 (Qwen)
    participant A4b as 🛡️ 守備派 (DeepSeek)
    participant A4c as ⚖️ 裁定者 (Gemini)
    participant Exec as 🏦 ブローカー

    Data->>A1: 指数+資金+ニュース
    A1->>A2: 市場判断: 底固め局面
    A2->>A3: セクター判断: テクノロジー→消費

    par 並列スキャン ×6
        A3->>A3: 永和智控 66点 保有
        A3->>A3: 新里程 56点 削減
        A3->>A3: 千紅製薬 62点 保有
    end

    Note over A3,Exec: スコア≤30 → 即時売却、ディベートを待たず

    A3->>A4a: 全分析結果
    A4a->>A4b: 提案: 新里程全売り、太陽光買い増し

    Note over A4b: 🤔 DeepSeek-R1 思考連鎖<br/>《think》逐条推論《/think》

    A4b->>A4c: 反論: 新里程サポート有効、全売り反対

    Note over A4c: ⚖️ 双方の意見を総合

    A4c->>Exec: 最終: 新里程保有、太陽光半分で新規
    Exec->>Exec: 追跡注文 (取消→再値付け→ポーリング)
```

---

## 図3：LangGraph 会話ステートマシン

```mermaid
stateDiagram-v2
    [*] --> entry: ユーザーメッセージ

    entry --> view_portfolio: "ポジション"
    entry --> risk_check: "買い/売り/全売り"
    entry --> validate_t0: "デイトレ"
    entry --> rebalance: "リバランス"
    entry --> scan: "スキャン"
    entry --> analyze: "分析 002506"
    entry --> chat: その他

    risk_check --> confirm: allowed=true
    risk_check --> [*]: blocked

    confirm --> [*]: pending_confirmation=true<br/>⏸️ グラフ一時停止

    note right of confirm
        ユーザーが「確認」と返信後
        checkpointからグラフ復元
    end note

    confirm --> execute_trade: confirmed=true
    confirm --> [*]: confirmed=false

    execute_trade --> [*]: ✅ 取引完了

    validate_t0 --> plan_t0: 売却可能残高あり
    validate_t0 --> [*]: 売却可能残高なし

    plan_t0 --> [*]: pending_confirmation=true<br/>⏸️ 確認待ち

    plan_t0 --> execute_t0_sell: 確認済
    execute_t0_sell --> [*]: t0_phase=monitoring<br/>「買い戻し」待ち

    execute_t0_sell --> execute_t0_buyback: "買い戻し"
    execute_t0_buyback --> [*]: ✅ デイトレ完了

    view_portfolio --> [*]
    rebalance --> [*]
    scan --> [*]
    analyze --> [*]
    chat --> [*]
```

---

## 図4：確認ワークフローのシーケンス

```mermaid
sequenceDiagram
    actor User as 👤 ユーザー
    participant FS as 📱 Feishu
    participant Graph as 🧠 LangGraph
    participant CP as 💾 Checkpoint
    participant Broker as 🏦 ブローカー

    User->>FS: "002506を500株5.4元で買い"
    FS->>Graph: invoke_graph()
    Graph->>Graph: entry → risk_check → confirm
    Graph->>CP: ステート保存 (pending=true)
    Graph-->>FS: "📋 注文確認 500株×5.4元？"
    FS-->>User: 確認カード表示

    Note over Graph,CP: ⏸️ グラフ一時停止<br/>ステートがcheckpointに保存

    User->>FS: "確認"
    FS->>Graph: invoke_graph()
    Graph->>CP: ステート復元 (pending=true)
    Graph->>Graph: 確認検知 → execute_trade
    Graph->>Broker: 売買発注
    Broker-->>Graph: 約定報告
    Graph->>CP: ステートクリア (pending=false)
    Graph-->>FS: "✅ 約定完了、残り現金 7328 元"
    FS-->>User: 取引完了通知
```

---

## 図5：引け後分析フロー

```mermaid
graph LR
    Close["15:00<br/>大引け"]

    subgraph PostMarket["引け後自動タスク"]
        Peak["📐 峰値統計<br/>約定価 vs 最適価"]
        Review["📝 AI振り返り<br/>損益原因分析"]
        ZT["🔥 ストップ高分析<br/>龍虎リスト+関連銘柄"]
        BT["📊 バックテスト<br/>スキャン戦略検証"]
        RB["⚖️ リバランス分析<br/>Multi-Agentディベート"]
    end

    subgraph ZTDetail["ストップ高分析詳細"]
        ZTPool["ストップ高プール<br/>ak.stock_zt_pool_em"]
        LHB["龍虎リスト<br/>ak.stock_lhb_detail_em"]
        Flow["資金フロー<br/>主力純流入"]
        Search["Web検索<br/>Tavily"]
        Concept["テーマ発掘<br/>共起テーマ ≥ 2"]
        AI["AI分析<br/>主線+関連チャンス"]
    end

    subgraph Output["プッシュ通知"]
        FS2["📱 Feishu"]
        DD["💬 DingTalk"]
        Email["📧 メール"]
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

## 図6：モデルデグレードチェーン

```mermaid
graph TD
    Request["LLM リクエスト"]

    M1["☁️ Azure GPT-5.4-nano<br/>(主力)"]
    M2["☁️ Gemini 2.5 Flash<br/>(予備1)"]
    M3["☁️ LITELLM_MODEL<br/>(予備2)"]
    M4["🖥️ DeepSeek-R1 14B<br/>(ローカルフォールバック)"]

    OK["✅ 結果を返却"]

    Request --> M1
    M1 -->|成功| OK
    M1 -->|タイムアウト/レート制限| M2
    M2 -->|成功| OK
    M2 -->|失敗| M3
    M3 -->|成功| OK
    M3 -->|全クラウドダウン| M4
    M4 -->|成功| OK

    style M1 fill:#e3f2fd,stroke:#1976d2
    style M2 fill:#e8f5e9,stroke:#388e3c
    style M3 fill:#fff3e0,stroke:#f57c00
    style M4 fill:#fce4ec,stroke:#c62828
    style OK fill:#c8e6c9,stroke:#2e7d32
```

---

## 使い方

1. https://mermaid.live を開く
2. 任意の図のMermaidコードを貼り付け
3. 右上からPNG / SVGでエクスポート
4. 記事の該当箇所に挿入

または VS Code の **Markdown Preview Mermaid Support** 拡張で直接プレビュー可能です。
