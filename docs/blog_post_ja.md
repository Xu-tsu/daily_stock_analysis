# Multi-Agent ディベート × LangGraph 会話エンジンで全自動A株トレーディングシステムを作ってみた

> AIに投資チームのように議論・意思決定・執行させる

---

## このシステムが解決する問題

トレーディング判断を**単一のAI**に任せると、すべてを一人に丸投げするのと同じで精度が落ちます。「マクロ環境を見て、保有銘柄も分析して、売買も決めて」と一気に頼むと、LLMはすべての観点を浅くしか処理できません。

本システムでは **Multi-Agent ディベート + 裁定** アーキテクチャを採用し、異なる「役割」に分業させることでこの問題を解決しています。

| Agent | 役割 | モデル | 担当 |
|-------|------|--------|------|
| Agent 1 | マクロアナリスト | Azure GPT-5.4-nano | 市場環境判断・ポジション比率提案 |
| Agent 2 | セクターアナリスト | Azure GPT-5.4-nano | セクターローテーション・主要テーマ判断 |
| Agent 2b | センチメントアナリスト | Azure GPT-5.4-nano | Fear & Greed指数・感情サイクル分析 |
| Agent 3 | ポートフォリオスキャナー | クラウド並列 ×6 | 銘柄ごとのテクニカル+需給+資金フロー分析 |
| Agent 3b | ファンダメンタルリサーチャー | クラウド並列 | 決算・成長トレンド・財務リスク |
| **Agent 4a** | **アグレッシブトレーダー** | Qwen 14B (ローカル) | リバランス提案の作成 |
| **Agent 4b** | **コンサバリスクマネージャー** | DeepSeek-R1 14B (ローカル) | 提案への逐条反論 |
| **Agent 4c** | **アービター（裁定者）** | Gemini / Azure (クラウド) | 双方の意見を総合し最終判断 |

各Agentが**一つの観点だけ**に集中するため、「1つのプロンプトで全部やる」よりはるかに深い分析が可能です。さらに「ディベート → 裁定」メカニズムにより、最終判断が**多角的な対立検証**を経ることが保証されます。

---

## 目次

1. [アーキテクチャ全体像](#アーキテクチャ全体像)
2. [Multi-Agent ディベート：コアリバランスエンジン](#multi-agent-ディベートコアリバランスエンジン)
3. [LangGraph 会話エンジン：Feishu Botの頭脳](#langgraph-会話エンジンfeishu-botの頭脳)
4. [設計のポイント：確認ワークフローとデイトレード](#設計のポイント確認ワークフローとデイトレード)
5. [クラウド裁定デグレードチェーン：ダウンタイムゼロ](#クラウド裁定デグレードチェーンダウンタイムゼロ)
6. [ブローカー自動発注：判断から約定まで](#ブローカー自動発注判断から約定まで)
7. [引け後分析システム：振り返り + ストップ高分析](#引け後分析システム振り返り--ストップ高分析)
8. [設計で踏んだ落とし穴](#設計で踏んだ落とし穴)

---

## アーキテクチャ全体像

```
┌──────────────────────────────────────────────────────────────────┐
│                     Feishu / DingTalk / WebUI                    │
│              ユーザー指示：「002506を500株5.4元で買い」              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                     ┌───────▼────────┐
                     │  LangGraph エンジン │  意図分類 → リスク管理 → 確認 → 執行
                     │  (会話ステートマシン) │  マルチターン対話・デイトレ・確認WF対応
                     └───────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
   ┌───────▼───────┐ ┌──────▼───────┐ ┌───────▼──────┐
   │ リバランスエンジン │ │ 全市場スキャン  │ │ ブローカーAdapter│
   │ (7ステップ)    │ │ (15000銘柄)   │ │ (同花順発注)    │
   └───────┬───────┘ └──────────────┘ └──────────────┘
           │
   ┌───────▼─────────────────────────────────────┐
   │          Multi-Agent ディベート意思決定          │
   │                                               │
   │  ┌─────────┐ ┌─────────┐ ┌──────────┐       │
   │  │ Agent1  │ │ Agent2  │ │ Agent2b  │       │
   │  │ マクロ   │ │ セクター │ │ センチメント│       │
   │  └────┬────┘ └────┬────┘ └────┬─────┘       │
   │       └───────────┼───────────┘              │
   │                   │                          │
   │  ┌────────────────▼──────────────────┐       │
   │  │ Agent3: 保有銘柄スキャン (×6 並列)   │       │
   │  │ Agent3b: ファンダメンタル分析 (×6 並列)│       │
   │  └────────────────┬──────────────────┘       │
   │                   │                          │
   │  ┌────────────────▼──────────────────┐       │
   │  │  ディベートフェーズ                  │       │
   │  │                                   │       │
   │  │  Agent4a (攻撃派/Qwen)             │       │
   │  │    「新里程を全売り、太陽光を買い増し」 │       │
   │  │          │                        │       │
   │  │          ▼                        │       │
   │  │  Agent4b (守備派/DeepSeek-R1)      │       │
   │  │    「反対！新里程はサポート有効、      │       │
   │  │     太陽光は高値掴みリスク大」        │       │
   │  │          │                        │       │
   │  │          ▼                        │       │
   │  │  Agent4c (裁定者/Gemini)           │       │
   │  │    「守備派を採用、新里程は保有継続；  │       │
   │  │     太陽光は半分だけ新規、損切7.5設定」│       │
   │  └───────────────────────────────────┘       │
   └──────────────────────────────────────────────┘
```

> **図1：システムアーキテクチャ全景** — ユーザー指示はLangGraphでルーティングされ、売買判断はMulti-Agentディベートを経て、ブローカーAdapterが自動執行する

---

## Multi-Agent ディベート：コアリバランスエンジン

### なぜ「ディベート」なのか？1つのモデルで直接判断すればいいのでは？

単一モデルの問題は**自己一貫性バイアス**にあります。自分で出した提案を自分で覆すのは困難です。一方、人間の投資チームの強みは**意見の衝突**にあります。トレーダーが「買い増したい」と言えば、リスク管理者が「待て」と止め、最終的に投資責任者が判断を下す。

本システムのMulti-Agentディベートは、このプロセスを完全に再現しています。

### 7ステップパイプライン

```
Step 0: データ準備
  ├── ブローカーから実際のポジションを同期 (broker → portfolio.json)
  ├── リアルタイム価格で上書き
  └── 前回の分析結果をロード（判断の連続性確保）

Step 1: マクロデータ収集 (純Python, 0トークン)
  ├── CSI300 / ChiNext / 北向資金
  ├── セクター資金フロー / 信用取引
  └── ニュース速報 / 市場センチメント

Step 2: Agent1 市場環境判断 → 「底固め局面、推奨ポジション比率45%」
Step 3: Agent2 セクターローテーション → 「資金が消費から→テクノロジーへ」
Step 3b: Agent2b センチメント → 「回復期、Fear&Greed指数50」

Step 4: Agent3 保有銘柄スキャン (クラウド×6並列, 3銘柄を12秒で完了)
  ├── 永和智控: 66点 → 保有継続
  ├── 新里程: 56点 → 削減          ← 即時売却チェック発動
  └── 千紅製薬: 62点 → 保有継続

Step 4b: Agent3b ファンダメンタル (クラウド×6並列, 2秒)

Step 5: ディベート
  5a: 攻撃派が提案 (execution_strategy付き)
  5b: 守備派が逐条反論
  5c: 裁定者が最終決定 → 確定actions[]
```

### ディベートの核心：攻撃派 vs 守備派

```python
# 攻撃派 (Agent4a) の出力
{
    "actions": [{
        "code": "002219", "name": "新里程",
        "action": "sell",
        "reason": "テクニカル弱含み、MA5がMA20を下抜け、即座に全売り",
        "target_sell_price": 2.50,
        "execution_strategy": {
            "urgency": "high",
            "order_type": "aggressive",
            "chase_max_pct": 0.5
        }
    }]
}

# 守備派 (Agent4b) の反論
{
    "critical_issues": [
        "新里程2.42元はサポートライン2.38に近接、ここで売りは底値で投げ売り同然"
    ],
    "position_disagreements": [{
        "code": "002219",
        "original_action": "sell",
        "my_suggestion": "hold",
        "reason": "サポート2.38が有効、反発を待ってから判断がより安全"
    }]
}

# 裁定者 (Agent4c) の最終判断
{
    "revised_actions": [{
        "code": "002219", "name": "新里程",
        "action": "hold",  # 守備派の意見を採用
        "reason": "守備派がサポートの有効性を指摘、一旦保有で様子見、2.35割れで損切り"
    }]
}
```

> **図2：ディベートフロー** — 攻撃派が提案 → 守備派が逐条反論 → 裁定者が最終判断。各Agentは異なるモデルを使用し、思考の同質化を防ぐ。

### 分析1銘柄 → 即時執行

従来のフローは「全銘柄分析 → 一括提案 → 一括執行」ですが、損切りシグナルは毎分変化しており、40分かけて全分析が終わる頃には最適なタイミングを逃している可能性があります。

```python
# Step 4 保有銘柄スキャンのループ内：
for h in holdings:
    rating = _scan_one_holding(h)  # クラウド 5-15秒

    # ハイリスクは即時執行、ディベートを待たない
    if rating["score"] <= 30 and rating["action"] == "sell":
        broker.sell(h["code"], h["current_price"], h["sellable_shares"])
        logger.info(f"[即時売却] {h['name']} スコア{rating['score']}、執行済")
```

**売却はスピード重視（即時執行）、買いは全体像重視（ディベート完了まで待つ）**——これが実戦で最も重要な原則です。

---

## LangGraph 会話エンジン：Feishu Botの頭脳

### 共有ステート：全ノードの記憶

```python
class PortfolioGraphState(TypedDict):
    messages: Annotated[list, add_messages]  # 会話履歴を自動蓄積
    user_text: str                           # 現在のユーザー入力
    intent: Optional[str]                    # ルーティング先の意図

    # 取引パラメータ
    trade_action: Optional[dict]     # {"code", "shares", "price", "action"}
    risk_check: Optional[dict]       # {"allowed", "warnings", "blocked_reason"}

    # 確認ワークフロー
    pending_confirmation: bool       # グラフ一時停止中（ユーザー確認待ち）
    confirmed: Optional[bool]        # ユーザーが確認/キャンセル

    # デイトレードワークフロー
    t0_phase: Optional[str]          # plan → confirm_sell → monitoring → done

    # ブローカー
    broker_enabled: bool
    broker_order_result: Optional[dict]

    # 出力
    response: str
```

### 意図ルーティング：LLMを使わない高速分類

参考記事ではSupervisorがLLMでルーティングを判断していますが、本システムでは**正規表現 + キーワード**によるハードコードルーティングを選択しました。理由はシンプルです：取引指示のフォーマットは確定的（「買い 002506 500 5.4」）であり、LLMを使うとむしろ遅くなり不安定になります。

```
entry_node (意図分類 + ポートフォリオロード)
  │
  ├─ "ポジション" → view_portfolio_node → END
  ├─ "買い/売り/全売り" → risk_check → confirm → execute_trade → END
  ├─ "デイトレ 002506" → validate_t0 → plan_t0 → [停止] → execute_sell → [停止] → buyback → END
  ├─ "リバランス" → rebalance_node (Multi-Agentトリガー) → END
  ├─ "ストップ高分析" → limit_up_analysis_node → END
  └─ その他 → chat_node (LLM自由会話) → END
```

> **図3：LangGraph 会話ステートマシン** — 意図分類後に条件ルーティング、取引系指示はリスク管理+確認の二重ゲートを通過

### 条件ルーティングの実装

```python
builder.add_conditional_edges(
    "route_intent_dummy",
    route_intent,
    {
        "view_portfolio": "view_portfolio",
        "risk_check": "risk_check",         # 買い/売り/全売り
        "validate_t0": "validate_t0",       # デイトレ
        "rebalance": "rebalance",
        "scan": "scan",
        "chat": "chat",
        # ... 12種類の意図
    },
)

# リスクチェック後の二次ルーティング
builder.add_conditional_edges(
    "risk_check",
    route_after_risk_check,
    {
        "confirm": "confirm",     # 通過 → 確認を要求
        "end": END,               # ブロック → 即終了
    },
)
```

---

## 設計のポイント：確認ワークフローとデイトレード

### 確認ワークフロー：グラフの一時停止と再開

取引は一発で実行してはいけません。ユーザーが「買い」と言った後、まず注文詳細を表示し、確認を待つ必要があります。

```
ユーザー: 「002506を500株5.4元で買い」
  │
  ▼
entry → risk_check → confirm_node
  │
  │  response: 「注文確認：協鑫集成 500株×5.4元=2700元」
  │  pending_confirmation = True
  │
  ▼  ← グラフ一時停止、checkpoint にステート保存

ユーザー: 「確認」
  │
  ▼  ← invoke_graph が pending_confirmation=True を検知
  │     リカバリールートに入る
  │
entry → route_confirmation → execute_trade
  │
  │  response: 「約定完了、残り現金 7328 元」
  │  pending_confirmation = False
  ▼
 END
```

> **図4：確認ワークフローのシーケンス** — confirm_nodeでグラフが一時停止し、ユーザー確認後にcheckpointから復元して取引を実行

### デイトレ（T+0）ワークフロー：メッセージをまたぐステートマシン

デイトレ（日中の高値売り→安値買い戻し）は複数のメッセージにまたがるため、`t0_phase`フィールドでステートマシンを管理します：

```
ユーザー: 「002506をデイトレ」
  → validate_t0: 売却可能株数200株を確認
  → plan_t0: 「8.68元で売り、8.53元で買い戻し目標」
  → pending_confirmation = True (一時停止)

ユーザー: 「確認」
  → execute_t0_sell: 200株 × 8.68元で売却
  → t0_phase = "monitoring"
  → response: 「売却完了。価格が下落したら「買い戻し」と送信してください」

(価格下落を待機...)

ユーザー: 「買い戻し」
  → execute_t0_buyback: 200株 × 8.50元で買い戻し
  → t0_phase = "done"
  → response: 「デイトレ完了！1株あたり+0.18元、合計+36元」
```

---

## クラウド裁定デグレードチェーン：ダウンタイムゼロ

本番環境では、どのAPIもタイムアウトやレート制限が発生し得ます。そこで**4段階のデグレードチェーン**を設計しました：

```
クラウド主力 (Azure GPT-5.4-nano)
  │ 失敗
  ▼
クラウド予備 (Gemini 2.5 Flash)
  │ 失敗
  ▼
グローバルデフォルト (LITELLM_MODEL)
  │ 失敗
  ▼
ローカルフォールバック (DeepSeek-R1 14B, Ollama)
```

```python
def _call_cloud_llm(prompt, agent_name=""):
    candidates = [CLOUD_MODEL, CLOUD_FALLBACK, LITELLM_MODEL]
    for i, model in enumerate(candidates):
        try:
            response = litellm.completion(model=model, ...)
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"[{agent_name}] {model} 失敗: {e}")
            continue
    # 全クラウドがダウン → ローカル DeepSeek でフォールバック
    return _call_debate_llm(prompt, f"{agent_name}_ローカル回帰")
```

**デグレードログ**により、各デグレードイベントが完全に記録され、モデルの安定性を事後分析できます。

---

## ブローカー自動発注：判断から約定まで

### インテリジェント追跡注文

A株の指値注文は即座に約定しないことがよくあります。システムは自動的に**取消 → 再値付け → ポーリング**を行います：

```python
def _smart_execute(self, direction, code, price, shares, limit_price, timeout):
    """発注 → 待機 → 未約定 → 取消 → リアルタイム価格取得 → 再発注"""
    current_price = price
    while time.time() - start < timeout:
        result = trade_fn(code, current_price, shares)
        time.sleep(poll_interval)

        if self._check_order_fill(code, shares):
            return OrderResult(status="filled", ...)

        # 未約定、リアルタイム価格で再発注
        realtime = self._get_realtime_price(code)
        if realtime and realtime <= limit_price:
            self._cancel_latest_pending(code)
            current_price = realtime
            continue
        break
```

### AIが執行戦略を決定

追跡注文のパラメータはハードコードではなく、各銘柄の執行戦略をAIディベートが決定します：

```json
{
    "execution_strategy": {
        "urgency": "high",
        "chase_max_pct": 0.5,
        "chase_timeout": 60,
        "order_type": "aggressive",
        "split_orders": false,
        "reason": "損切りシグナルが明確、早期約定を優先"
    }
}
```

---

## 引け後分析システム：振り返り + ストップ高分析

### AI振り返りノート

毎日大引け後に自動実行し、当日の損益原因を分析します：

```
大引け後タスクフロー：
  15:00  ティック峰値統計（約定価 vs 日中最適価の乖離）
  15:01  AI振り返りノート → Feishu通知
  15:02  ストップ高深層分析 → Feishu通知
  15:03  スキャンバックテスト
  15:05  リバランス分析（Multi-Agentディベート）
```

振り返りデータソース：
- `trade_log`：当日の売買記録 + テクニカル指標スナップショット
- `intraday_peak_stats`：約定価と日中高安値の乖離
- `rebalance_history`：AI判断記録 vs 実際の執行結果
- 30日間の成績 + 90日間のパターン分析

### ストップ高銘柄の深層分析

ストップ高プール → 龍虎リスト → 資金フロー → Web検索 → 関連銘柄発掘を自動化：

```
データ収集（並列）:
  ├── ak.stock_zt_pool_em → ストップ高プール（連続日数/板張り時間/業種）
  ├── ak.stock_lhb_detail_em → 龍虎リスト（機関投資家/投機筋の売買席位）
  └── fetch_fund_flow_rank → 主力資金フロー

関連銘柄発掘:
  ストップ高銘柄 → 所属テーマ → 共起頻度 ≥ 2 → テーマ構成銘柄取得
  → ストップ高済を除外 → 騰落率順 → 「翌日のポテンシャル銘柄」を出力

AI分析:
  → センチメント概要 + コア主線テーマ + 連続ストップ高分析 + 翌日チャンス + リスク警告
  → Feishu / DingTalk / メールにプッシュ通知
```

> **図5：引け後分析フロー** — 振り返りノートが「なぜ勝った/負けた」を分析し、ストップ高分析が「明日何が上がるか」を探す

---

## 設計で踏んだ落とし穴

### ① ルーティングはLLMかハードコードか？

参考記事ではSupervisorがLLMで「次に何をすべきか」を判断しています。オープンなタスクではこれは良い設計です。しかしトレーディングシステムでは、意図は**構造化されており有限**です——「買い/売り/ポジション確認/リバランス」。LLMルーティングはむしろ不確定性を持ち込みます（稀に「売り」を「照会」と誤認識することがある）。

**私たちの選択**：エントリーは正規表現 + キーワードによるハードルーティング（確実性100%）、マッチしない場合のみLLMのchatノードにフォールバック。

### ② ディベートモデルは異なるものを使うべきか？

必須です！攻撃派と守備派に同じモデルを使うと、高度に類似した分析結果が出力され、ディベートの意味がなくなります。

本システムではQwenを攻撃派（中国語理解力が高く、攻めの提案を出しやすい）、DeepSeek-R1を守備派（思考連鎖 `<think>` モード搭載、論理的推論と欠陥発見に優れる）、Geminiを裁定者（大コンテキストウィンドウ、双方の意見を同時に消化可能）に配置しています。

### ③ 確認ワークフローのグラフ一時停止

LangGraphの `MemorySaver` checkpointerにより、グラフ実行途中のステートを保存できます。`pending_confirmation=True` でグラフが自然終了し、次のメッセージ到着時に `thread_id` でステートを復元、確認待ちフラグを検知してリカバリールートに入ります。

**落とし穴**：確認待ち中にユーザーが全く関係ないメッセージを送った場合（例：「今日の天気は？」）、正しくハンドリングする必要があります——取引を実行してはならないし、確認待ちステートも失ってはなりません。entry_nodeでこのケースを特別処理し、プロンプトを返しつつ `pending_confirmation=True` を維持しています。

### ④ ローカルモデルの並行処理

ローカルのOllamaはGPU1枚のみで並行処理ができません。6つのAgent3が同時にリクエストすると、GPUメモリがOOMになります。

**解決策**：クラウドスキャンは6スレッド並列（5-15秒）、ローカルスキャンは逐次処理にデグレード（100-260秒）。`REBALANCE_SCAN_USE_CLOUD=true` 環境変数で制御。

### ⑤ データソースの信頼階層

portfolio.jsonはローカルファイルであり、ブローカーの実際のポジションと不整合が起きうる場合があります（手動取引、部分約定など）。

**信頼チェーン**：`ブローカーリアルタイムポジション > trade_logからの推算 > portfolio.json`。リバランス分析のたびに、まず `sync_portfolio_from_broker()` で同花順クライアントから実際のポジションを取得します。

---

## 技術スタック

| コンポーネント | 技術 |
|--------------|------|
| 会話エンジン | LangGraph + MemorySaver (ステートマシン + checkpoint) |
| Multi-Agentリバランス | 自作7ステップパイプライン + ディベート裁定 |
| クラウドLLM | Azure GPT-5.4-nano / Gemini 2.5 Flash (litellm統一呼出) |
| ローカルLLM | Ollama + Qwen2.5-14B / DeepSeek-R1-14B |
| ブローカー連携 | easytrader + 同花順遠航版 |
| データソース | akshare + テンセント相場 + Tushare + 東方財富 |
| 検索エンジン | Tavily (ストップ高理由検索) |
| メッセージプラットフォーム | Feishu Stream / DingTalk Webhook / メール |
| ストレージ | SQLite (trade_log / intraday_ticks) + JSON (portfolio) |
| モニタリング | 日中ティック収集 + AI目標価リアルタイム検知 + 自動利確/損切 |

---

## 拡張の方向性

- **ローカルモデルのファインチューニング**：システムがJSONL形式でファインチューニングサンプルを自動収集しており、500件以上蓄積後にLLaMA-FactoryでQwenをファインチューニングし、ローカルモデルでクラウドを段階的に置換可能
- **Human-in-the-loop**：LangGraph `interrupt()` でディベート結果をユーザーに確認させてから執行
- **クロスマーケット**：現在はA株のみだが、アーキテクチャ設計は香港株/米国株への拡張をサポート（US market reviewモジュール搭載済み）

---

## まとめ

このプロジェクトを通じて、3つのことを深く理解しました：

1. **Multi-Agentは単なるバズワードではない** — 各Agentが1つの観点だけを担当することで、単一モデルをはるかに超える分析深度が実現できます。ディベートメカニズムは意思決定の品質に質的な変化をもたらしました。

2. **LangGraphの価値はステート管理にある** — トレーディングシステムのマルチターン会話、確認ワークフロー、デイトレードのメッセージ横断ステートマシンは、従来のif-elseでは混沌としたコードになりますが、LangGraphの条件ルーティング + checkpointですべてが明快になります。

3. **デグレードチェーンは本番システムの生命線** — クラウドAPIは必ずダウンし、ローカルGPUは必ずOOMします。4段階のデグレードチェーンにより「ダウンタイムゼロ」を確保し、全クラウドが使えなくてもローカルDeepSeekが実用的な判断を出せます。

Multi-Agentアーキテクチャを実践したい方は、まずシンプルな「ディベート → 裁定」から始めてみてください。2つのモデルが同じ問題に対して相反する意見を出し、3つ目のモデルが裁定する。どんなチュートリアルよりも早く、Agentの協調の本質を理解できるはずです。

---

*著者：華爾街之狼プロジェクトチーム*
*技術スタック：Python + LangGraph + litellm + easytrader + akshare*
