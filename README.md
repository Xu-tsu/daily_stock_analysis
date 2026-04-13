<div align="center">

# 📈 AI 株式インテリジェント分析システム

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Ready-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/)

**AI 大規模言語モデルによる A株/香港株/米国株のインテリジェント分析システム**

ウォッチリストを毎日自動分析 → 「意思決定ダッシュボード」を生成 → 複数チャネルにプッシュ通知

[**クイックスタート**](#-クイックスタート) · [**主要機能**](#-主要機能) · [**マルチモデル討論**](#-マルチモデル討論アーキテクチャ) · [**対話型ボット**](#-飛書feishu対話型ボット)

日本語 | [简体中文](README.md) | [English](README_EN.md) | [繁體中文](README_CHT.md)

</div>

---

## ✨ 主要機能

| モジュール | 機能 | 説明 |
|-----------|------|------|
| AI | 意思決定ダッシュボード | 一文結論 + 精密な売買ポイント + 操作チェックリスト |
| 分析 | 多次元分析 | テクニカル（MA/MACD/RSI/強気配列）+ 出来高分布 + センチメント + リアルタイム相場 |
| 市場 | グローバル市場 | A株、香港株、米国株および米株指数（SPX、DJI、IXIC等） |
| ファンダメンタルズ | 構造化集約 | バリュエーション / 成長性 / 業績 / 機関投資家動向 / 資金フロー、fail-open 降格 |
| 戦略 | 市場戦略 | A株「三段式復盤」と米株「Regime Strategy」内蔵 |
| 復盤 | 大盤復盤 | 日次市場概況・セクター騰落・主線テーマ追跡 |
| 監視 | 寄付オークション監視 | 09:15-09:25 の寄付段階の出来高と売買強弱を追跡 |
| バックテスト | AI 検証 | 過去の分析精度を自動評価、方向的中率・利確/損切り命中率 |
| Agent Q&A | 戦略対話 | マルチターン戦略問答、11種類の内蔵戦略（Web/Bot/API） |
| 通知 | マルチチャネル | WeChat Work、飛書、Telegram、Discord、DingTalk、メール |
| 自動化 | 定時実行 | GitHub Actions / cron / Docker、サーバー不要 |

### 技術スタック

| 種類 | サポート |
|------|---------|
| AI モデル | Gemini、OpenAI互換、DeepSeek、Qwen、Claude、Ollama等（LiteLLM統一呼出、マルチKey負荷分散） |
| 相場データ | AkShare、Tushare、Pytdx、Baostock、YFinance（5ソース自動フォールバック） |
| ニュース検索 | Tavily、SerpAPI、Bocha、Brave、MiniMax |
| 対話エンジン | LangGraph（18ノード状態グラフ、確認ワークフロー付き） |
| ローカル推論 | Ollama（Qwen 14B + DeepSeek-R1 14B、RTX 5070 12GBで同時稼働） |

### 内蔵トレーディングルール

| ルール | 説明 |
|-------|------|
| 高値追い禁止 | 乖離率 > 5% で自動警告、強トレンド銘柄は自動緩和 |
| トレンドトレード | MA5 > MA10 > MA20 強気配列確認 |
| 精密ポイント | エントリー価格・損切り価格・目標価格を提示 |
| チェックリスト | 各条件を「合格 / 注意 / 不合格」でマーキング |

### 实盘事件链路

- `python main.py` 的 A 股实盘链路现在会把 `news_scanner -> event_signal -> run_stock_scan / run_rebalance_analysis` 串起来，新闻热点不再只用于一次性选股打分。
- 事件信号会生成板块级关注池，并把防御/进攻动作直接传给建仓和调仓；遇到 `defense` 级事件时，自动建仓仓位会在自适应仓位基础上再减半。
- 自适应仓位会根据真实卖出后的盈亏持续更新，后续建仓不再依赖写死的牛/熊固定仓位。

---

## 🤖 マルチモデル討論アーキテクチャ

単一LLMのバイアスを排除するため、**複数モデルが「討論」して投資判断を導出**する仕組みを実装。

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Qwen 14B   │    │ DeepSeek-R1  │    │  Azure GPT /    │
│ （攻撃的提案）│ →  │ （保守的批評）  │ →  │  Gemini（裁定）  │
│  ローカル GPU │    │  ローカル GPU  │    │   クラウド API    │
└─────────────┘    └──────────────┘    └─────────────────┘
```

| ステージ | モデル | 役割 |
|---------|--------|------|
| Agent 1-3 | Qwen 2.5 14B（ローカル） | データ収集・テクニカル分析・スクリーニング |
| Agent 4a | Qwen 2.5 14B | 攻撃的なリバランス提案を生成 |
| Agent 4b | DeepSeek-R1 14B | 保守的な視点で批評（Chain-of-Thought推論） |
| Agent 4c | Azure GPT / Gemini（クラウド） | 双方の議論を踏まえた最終裁定 |

**5段階フォールバック**: クラウドAPI制限時に自動降格

```
完全討論 → ローカル統合 → 提案のみ → ルールのみ → ハードルール
```

**蒸留パイプライン**: クラウドLLMの応答をJSONLで自動保存し、将来のファインチューニングに備える（目標500+サンプル）

---

## 💬 飛書（Feishu）対話型ボット

LangGraph 状態グラフ（18ノード）による**対話型持株管理エンジン**。自然言語で売買・分析・ポジション管理が可能。

| 機能 | コマンド例 | 説明 |
|------|-----------|------|
| 持株確認 | `持仓` | 現在の全ポジションを表示 |
| 買い注文 | `买入 002506 500 5.4` | 銘柄コード + 数量 + 価格 |
| 銘柄名で操作 | `买入 协鑫集成 500 5.2` | 名前→コード自動解決（ピンイン/あいまい検索対応） |
| 名前で分析 | `赤天化呢？` | 銘柄名だけで即時テクニカル分析 |
| 持株修正 | `协鑫集成是5.2 有13手` | 口語でポジション数量・原価を更新 |
| T+0デイトレ | `做T 002506` | 高値売り→安値買い戻しワークフロー |
| リバランス | `调仓` | マルチモデル討論による調整提案（非同期実行） |
| 市場スキャン | `扫描` | 全市場テクニカルスクリーニング |
| 戦績確認 | `战绩` | 過去の取引記録・勝率 |
| 戦略確認 | `策略` | 現在のリスク管理パラメータ表示 |

**設計上の特徴**:
- **確認ワークフロー**: 約定前に必ずユーザーの承認を要求（Human-in-the-loop）
- **T+1管理**: A株ルール通り、当日購入分は売却不可として自動分離
- **FIFO原価追跡**: 取引ログから先入先出法で正確な取得原価・保有日数を自動計算
- **非同期処理**: 重い処理は即座に「処理中...」を返し、完了後にチャットへ自動プッシュ

---

## 🛡️ リスク管理

| 項目 | 内容 |
|------|------|
| ハード損切り | -8% で強制決済 |
| 段階警告 | -5% で見直し（上昇トレンド中は保有継続可） |
| トレーリングストップ | 含み益が一定以上でストップラインを自動引上げ |
| ポジション上限 | 単一銘柄15%まで、最大5銘柄 |
| T+1管理 | 当日購入分は自動的に売却不可として管理 |
| FIFO追跡 | 取引ログから先入先出法で原価・保有日数を自動同期 |
| 保有日数ルール | 3日超＋上昇トレンド→トレーリングストップで保有、7日超＋利益5%未満→売却検討 |

---

## 🚀 クイックスタート

### 方法1: GitHub Actions（ゼロコスト）

```
1. Fork → 2. Settings > Secrets 設定 → 3. Actions 有効化 → 4. Run workflow
```

**必須Secrets:**

| Secret名 | 説明 |
|----------|------|
| `GEMINI_API_KEY` or `OPENAI_API_KEY` | AI モデルKey（最低1つ） |
| `STOCK_LIST` | ウォッチリスト（例: `600519,AAPL,hk00700`） |
| 通知チャネル | `TELEGRAM_BOT_TOKEN` / `DISCORD_WEBHOOK_URL` / `FEISHU_WEBHOOK_URL` 等から1つ以上 |

デフォルトで毎営業日18:00（北京時間）に自動実行。

### 方法2: ローカル実行

```bash
git clone https://github.com/ZhuLinsen/daily_stock_analysis.git
cd daily_stock_analysis
pip install -r requirements.txt
cp .env.example .env && vim .env
```

```bash
python main.py                          # 一回実行
python main.py --schedule               # 定時モード（毎日自動）
python main.py --stocks AAPL,TSLA       # 特定銘柄のみ
python main.py --monitor --interval 5   # 寄付オークション監視
python main.py --rebalance              # リバランス分析
python train_blind_agent.py             # 用2025训练 blind agent 并回测 2026YTD
python backtest_blind_agent_distilled.py # blind agent + 在线高手skill偏置回测
python backtest_blind_agent_v4b.py      # blind agent 结构升级版 V4b 回测
python backtest_blind_agent_v4c.py      # 环境路由 + V4b 结构层联合回测（用于和 V4b 做同口径对比）
python backtest_blind_agent_v4d.py      # MA5 + MA30 + 月线 的多周期个股分析回测（用于和 V4b 做同口径对比）
python backtest_blind_agent_v4e.py      # 四种原生策略按环境和最近胜率自适应切换的回测
python backtest_blind_agent_v4f.py      # 在 V4b 上叠加轻量自适应策略选择的回测
python backtest_blind_agent_agentic.py  # 真正的 agentic_backtest：AI teams 逐日 proposal/critique/arbitrate，并输出每日思考记录
python distill_expert_skills.py         # 蒸馏高手成交记录为 YAML strategy skills
python main.py --serve                  # Web UI + API起動
```

蒸馏生成的 strategy 会写到 `strategies/distilled/`，如果你想让 agent 优先加载这批高手经验，可以把 `AGENT_STRATEGY_DIR=./strategies/distilled` 写进 `.env`。

`AGENT_STRATEGY_ROUTING=auto` 现在不只影响 multi-agent 的 strategy stage；单 Agent 分析路径也会结合当前 `trend_result + market_context` 自动切换更合适的策略技能，只有显式指定策略时才会覆盖这层自动路由。
`python main.py` 的自动建仓默认回到训练后的 `blind_base` 链路：保留 blind agent 的自适应仓位和事件防御，不再默认叠加 `V4b` 的结构过滤。当前推荐这么做，是因为 `2024 + 2025 + 2026YTD` 的三窗口严格回测里，`blind_base` 的复合收益高于 `V4b`。
`backtest_blind_agent.py` 的冷静期状态机现在只会对同一波 `连亏>=4` 触发一次冷静日；冷静期结束后会重新评估新候选，不会再因为旧亏损序列被永久锁死在空仓。
`backtest_blind_agent_v4d.py` 会把个股分析从纯日线扩展到 `MA5 + MA30 + 月线趋势`，但它目前是实验性同口径回测版本，是否替代 `blind_base` / `V4b` 应以回测结果为准。
`backtest_blind_agent_v4e.py` / `backtest_blind_agent_v4f.py` 会让 agent 在 `低吸 / 回踩 / 突破 / 接力` 之间按环境和最近模式表现做自适应选择；其中 `V4f` 是更保守的叠加层版本，避免错误切换把主策略完全带偏。
`backtest_blind_agent_agentic.py` 会把回测改成真正的 “AI 内核”：每天只给 AI teams 当天及以前的数据，由进攻派 proposal、风控 critique、仲裁派 final decision 共同决定下一交易日 `buy / sell / hold / empty`，并把三段原始响应、结构化决策和候选池写入 `thought_logs.jsonl`、`thought_summary.csv` 供复盘检查。

**銘柄コード形式:**
A株 = 6桁（`600519`）、香港株 = `hk` + 5桁（`hk00700`）、米国株 = ティッカー（`AAPL`）

---

## 📱 出力サンプル

```
🎯 2026-03-31 意思決定ダッシュボード

共3銘柄分析完了 | 🟢買い:1 🟡様子見:1 🔴売り:1

🟢 AAPL (Apple Inc.)
📌 コア結論: 強気 | テクニカル良好+ポジティブ材料

🎯 スナイパーポイント
  理想エントリー: $183-184
  損切り:         $177
  利確目標:       $195

✅ チェックリスト
  ✅ 強気トレンド確認
  ✅ MA5サポート付近
  ✅ 出来高がトレンドを確認
  ⚠️ 市場ボラティリティに注意
```

---

## 🧩 Web サービス（オプション）

```bash
python main.py --serve       # API + 分析実行
python main.py --serve-only  # APIのみ
```

`http://127.0.0.1:8000` / APIドキュメント: `/docs`

設定管理、分析トリガー、リアルタイム進捗、Agent戦略Q&A、バックテスト検証に対応。

---

## 📄 ライセンス

MIT License

## ⚠️ 免責事項

本ツールは**情報提供および教育目的**のみです。分析結果はAIによって生成されたものであり、投資助言ではありません。投資判断はご自身の責任で行ってください。本ソフトウェアの使用によるいかなる金銭的損失についても、開発者は責任を負いません。
