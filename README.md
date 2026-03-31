<div align="center">

# 📈 股票智能分析系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Ready-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/)

> 基于 AI 大模型的 A股/港股/美股自选股智能分析系统，每日自动分析并推送「决策仪表盘」到企业微信/飞书/Telegram/Discord/邮箱

[**功能特性**](#-功能特性) · [**快速开始**](#-快速开始) · [**多模型辩论**](#-多模型辩论架构) · [**对话式交易**](#-飞书对话式交易bot) · [**推送效果**](#-推送效果)

简体中文 | [English](README_EN.md) | [繁體中文](README_CHT.md) | [日本語](README_JA.md)

</div>

---

## ✨ 功能特性

| 模块 | 功能 | 说明 |
|------|------|------|
| AI | 决策仪表盘 | 一句话核心结论 + 精确买卖点位 + 操作检查清单 |
| 分析 | 多维度分析 | 技术面（MA/MACD/RSI/多头排列）+ 筹码分布 + 舆情情报 + 实时行情 |
| 市场 | 全球市场 | A股、港股、美股及指数（SPX、DJI、IXIC 等） |
| 基本面 | 结构化聚合 | 估值/成长/业绩/机构动向/资金流/板块涨跌，fail-open 降级 |
| 策略 | 市场策略系统 | 内置 A股「三段式复盘」与美股「Regime Strategy」 |
| 复盘 | 大盘复盘 | 每日市场概览、板块涨跌、主线题材追踪 |
| 监控 | 集合竞价监控 | 09:15-09:25 跟踪候选股/持仓的竞价成交额与买卖盘强弱 |
| 回测 | AI 回测验证 | 自动评估历史分析准确率，方向胜率、止盈止损命中率 |
| Agent | 策略问股 | 多轮策略问答，11 种内置策略（Web/Bot/API 全链路） |
| 通知 | 多渠道推送 | 企业微信、飞书、Telegram、Discord、钉钉、邮件 |
| 自动化 | 定时运行 | GitHub Actions / cron / Docker，无需服务器 |

### 技术栈

| 类型 | 支持 |
|------|------|
| AI 模型 | Gemini、OpenAI 兼容、DeepSeek、通义千问、Claude、Ollama 等（LiteLLM 统一调用，多 Key 负载均衡）|
| 行情数据 | AkShare、Tushare、Pytdx、Baostock、YFinance（5 源自动 fallback） |
| 新闻搜索 | Tavily、SerpAPI、Bocha、Brave、MiniMax |
| 对话引擎 | LangGraph（18 节点状态图，确认工作流） |
| 本地推理 | Ollama（Qwen 14B + DeepSeek-R1 14B，RTX 5070 12GB 同时运行） |

### 内置交易纪律

| 规则 | 说明 |
|------|------|
| 严禁追高 | 乖离率 > 5% 自动警告，强势趋势股自动放宽 |
| 趋势交易 | MA5 > MA10 > MA20 多头排列 |
| 精确点位 | 买入价、止损价、目标价 |
| 检查清单 | 每项条件标记「满足 / 注意 / 不满足」 |

---

## 🤖 多模型辩论架构

单一 LLM 易产生偏见（如对某只股票始终建议卖出）。本系统让**多个模型「辩论」后达成共识**：

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Qwen 14B   │    │ DeepSeek-R1  │    │  Azure GPT /    │
│ （激进派提案）│ →  │ （保守派批评）  │ →  │  Gemini（裁定）  │
│  本地 GPU    │    │  本地 GPU     │    │   云端 API       │
└─────────────┘    └──────────────┘    └─────────────────┘
```

| 阶段 | 模型 | 职责 |
|------|------|------|
| Agent 1-3 | Qwen 2.5 14B（本地） | 大盘研判、板块轮动、持仓扫描 |
| Agent 4a | Qwen 2.5 14B | 激进派调仓提案 |
| Agent 4b | DeepSeek-R1 14B | 保守派质疑（Chain-of-Thought 推理） |
| Agent 4c | Azure GPT / Gemini（云端） | 综合双方观点做最终裁定 |

**5 级降级链**: 云端限流时自动回退

```
完全辩论 → 本地合并 → 仅提案 → 规则判断 → 硬规则
```

**蒸馏管线**: 云端 LLM 的响应自动保存为 JSONL，用于未来微调本地模型（目标 500+ 样本）。

---

## 💬 飞书对话式交易 Bot

基于 LangGraph 状态图（18 节点）的**对话式持仓管理引擎**，支持自然语言交易：

| 功能 | 命令示例 | 说明 |
|------|---------|------|
| 查看持仓 | `持仓` | 显示全部持仓、盈亏、持仓天数 |
| 买入 | `买入 002506 500 5.4` | 代码 + 数量 + 价格 |
| 用名称买入 | `买入 协鑫集成 500 5.2` | 名称→代码自动解析（拼音/模糊匹配） |
| 名称查询 | `赤天化呢？` | 用股票名即时触发技术分析 |
| 修正持仓 | `协鑫集成是5.2 有13手` | 口语修正持仓数量和成本 |
| 做T | `做T 002506` | T+0 高卖低买工作流 |
| 调仓 | `调仓` | 触发多模型辩论调仓分析（异步） |
| 市场扫描 | `扫描` | 全市场技术面筛选 |
| 战绩 | `战绩` | 历史交易记录和胜率 |

**核心特性**:
- **确认工作流**: 交易执行前必须用户确认（Human-in-the-loop）
- **T+1 管理**: A 股当日买入不可卖出，自动分离可卖余额
- **FIFO 追踪**: 从交易日志自动先进先出计算真实成本和持仓天数
- **异步执行**: 耗时操作（调仓/扫描）立即返回「处理中」，完成后异步推送结果（防 WebSocket 超时）
- **批量口令**: `协鑫集成5.24元13手，赤天化4.3加仓3手` 一句话批量更新

---

## 🛡️ 风控体系

| 项目 | 内容 |
|------|------|
| 硬止损 | -8% 强制清仓 |
| 风险复核 | -5% 触发复核（非机械清仓，结合市场/板块/趋势综合判断） |
| 移动止盈 | 盈利达标后自动抬高止损线 |
| 仓位上限 | 单只 ≤15%，最多 5 只 |
| T+1 管理 | 当日买入自动标记为不可卖 |
| FIFO 追踪 | 从交易日志先进先出匹配真实成本/持仓天数 |
| 保有日数 | 3天+上升趋势→移动止盈持有；7天+利润<5%→考虑卖出 |
| 自适应 | 市场偏强+主线确认→保留底仓做T；市场偏弱→优先清仓 |

---

## 🚀 快速开始

### 方式一：GitHub Actions（零成本）

```
1. Fork 本仓库 → 2. Settings > Secrets 配置 → 3. Actions 启用 → 4. Run workflow
```

**必要 Secrets:**

| Secret | 说明 |
|--------|------|
| `GEMINI_API_KEY` 或 `OPENAI_API_KEY` | AI 模型 Key（至少一个） |
| `STOCK_LIST` | 自选股代码（如 `600519,AAPL,hk00700`） |
| 通知渠道 | `TELEGRAM_BOT_TOKEN` / `DISCORD_WEBHOOK_URL` / `FEISHU_WEBHOOK_URL` 等至少一个 |

> 详细配置见 [LLM 配置指南](docs/LLM_CONFIG_GUIDE.md)

默认每个工作日 18:00（北京时间）自动执行，非交易日自动跳过。

### 方式二：本地运行

```bash
git clone https://github.com/ZhuLinsen/daily_stock_analysis.git && cd daily_stock_analysis
pip install -r requirements.txt
cp .env.example .env && vim .env
```

```bash
python main.py                          # 一次性分析
python main.py --schedule               # 定时模式（每日自动）
python main.py --stocks AAPL,TSLA       # 指定股票
python main.py --monitor --interval 5   # 集合竞价监控
python main.py --rebalance              # 调仓分析（多模型辩论）
python main.py --serve                  # Web UI + API
python main.py --webui                  # Web 界面 + 定时分析
```

**股票代码格式**: A股 = 6位数（`600519`）、港股 = `hk` + 5位（`hk00700`）、美股 = 字母（`AAPL`）

> Docker 部署见 [完整指南](docs/full-guide.md)

---

## 📱 推送效果

### 决策仪表盘
```
🎯 2026-03-31 决策仪表盘
共分析3只股票 | 🟢买入:1 🟡观望:1 🔴卖出:1

🟢 协鑫集成 (002506)
📌 核心结论: 看多 | 光伏板块轮动+量价配合

🎯 狙击点位
  理想买入: 5.10-5.20
  止损:     4.85
  目标价:   5.80

✅ 检查清单
  ✅ 多头排列确认
  ✅ 量能配合
  ⚠️ 注意板块持续性
```

### 大盘复盘
```
🎯 大盘复盘
📊 上证 3250.12 (+0.85%) | 深证 10521.36 (+1.02%) | 创业板 2156.78 (+1.35%)
上涨 3920 | 下跌 1349 | 涨停 155
🔥 领涨: 互联网服务、文化传媒、小金属
```

### 多模型辩论报告
```
🤖 调仓辩论
Agent4a 激进派: 建议加仓协鑫集成至 1500 股（板块轮动+量价配合）
Agent4b 保守派: 持仓天数仅 1 天，建议观望确认趋势（RSI 偏高风险）
Agent4c 裁定:   维持当前仓位，设移动止盈 5.60，跌破 4.85 止损
```

---

## 🖥️ Web 界面

```bash
python main.py --webui       # Web 界面 + 定时分析
python main.py --webui-only  # 仅启动 Web 界面
```

访问 `http://127.0.0.1:8000`，包含：配置管理、分析触发、历史报告、Agent 策略问股、回测验证。

---

## 📖 文档

- [完整配置指南](docs/full-guide.md) · [FAQ](docs/FAQ.md) · [LLM 配置](docs/LLM_CONFIG_GUIDE.md) · [部署指南](docs/DEPLOY.md)
- [飞书 Bot 配置](docs/bot/feishu-bot-config.md) · [钉钉 Bot 配置](docs/bot/dingding-bot-config.md) · [Bot 命令一览](docs/bot-command.md)
- [更新日志](docs/CHANGELOG.md) · [贡献指南](docs/CONTRIBUTING.md)

---

## 📄 License

[MIT License](LICENSE)

## ⚠️ 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。作者不对使用本项目产生的任何损失负责。
