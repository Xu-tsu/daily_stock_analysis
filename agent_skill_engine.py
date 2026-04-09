# -*- coding: utf-8 -*-
"""
agent_skill_engine.py — Agent 技能成长引擎
=============================================

核心理念：让Agent基于真实交易数据持续学习、自我进化

架构：
  ┌─────────────────────────────────────────┐
  │           Agent Skill Engine             │
  │                                          │
  │  1. SkillBook    — 技能库（可成长）       │
  │  2. PatternMiner — 模式挖掘（自动发现）   │
  │  3. SkillGrader  — 技能评分（优胜劣汰）   │
  │  4. PromptBuilder— 提示构建（注入Agent）   │
  └─────────────────────────────────────────┘

技能类型：
  - signal_skill    : 信号类（地天板/连板/金叉/超跌反弹...）
  - timing_skill    : 时机类（什么环境下买入胜率最高）
  - risk_skill      : 风控类（什么条件下该跑/该扛）
  - sector_skill    : 板块类（哪个板块最近赚钱多）

成长机制：
  每日收盘后自动运行 evolve()：
    1. 从 trade_journal 获取最新交易结果
    2. 更新每个技能的胜率/盈亏比/置信度
    3. 挖掘新模式（如果发现新的高胜率组合 → 自动生成新技能）
    4. 淘汰失效技能（连续30天胜率<35% → 降级）
    5. 保存到本地JSON（可被Agent读取）

使用:
  from agent_skill_engine import SkillEngine
  engine = SkillEngine()
  engine.evolve()                          # 每日进化
  prompt = engine.build_prompt(code="000790")  # 注入Agent
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SKILL_DB_PATH = Path("data/agent_skills")
SKILL_FILE = SKILL_DB_PATH / "skill_book.json"
EVOLUTION_LOG = SKILL_DB_PATH / "evolution_log.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 技能数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _default_skill(name: str, display: str, category: str, conditions: dict, description: str = "") -> dict:
    """创建一个技能条目"""
    return {
        "name": name,
        "display_name": display,
        "category": category,           # signal / timing / risk / sector
        "conditions": conditions,        # 触发条件（可量化的）
        "description": description,
        "stats": {
            "total_trades": 0,
            "win_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "avg_hold_days": 0.0,
            "profit_factor": 0.0,        # 盈亏比
            "max_win": 0.0,
            "max_loss": 0.0,
            "recent_5_results": [],       # 最近5笔结果 [True/False...]
        },
        "confidence": 0.0,               # 0~1, 基于样本量和胜率
        "grade": "C",                     # S/A/B/C/D/F
        "enabled": True,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "updated_at": datetime.now().strftime("%Y-%m-%d"),
        "version": 1,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 内置种子技能（基于数据挖掘结论）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEED_SKILLS = [
    # 信号类
    _default_skill(
        "floor_to_sky", "地天板", "signal",
        {"prev_chg": "<=-8", "today_chg": ">=9.5"},
        "昨日跌停+今日涨停，量化砸盘后反转。神剑股份2025.12验证。"
    ),
    _default_skill(
        "sky_floor_sky", "天地天板", "signal",
        {"prev2_chg": ">=9.5", "prev_chg": "<=-8", "today_chg": ">=9.5"},
        "前天涨停→昨天跌停→今天涨停，30%振幅，极端反转。"
    ),
    _default_skill(
        "board_relay", "涨停接力", "signal",
        {"prev_chg": ">=9.5", "today_chg": ">=3", "consec_limit": ">=1"},
        "昨日涨停+今日继续强势。数据挖掘：次日均+2.11%, 42%概率≥5%。"
    ),
    _default_skill(
        "double_board", "二连板", "signal",
        {"consec_limit": ">=2", "today_chg": ">=9.5"},
        "连续两天涨停，龙头确认信号。"
    ),
    _default_skill(
        "pre_dragon", "龙头预埋", "signal",
        {"has_recent_limit": True, "today_chg": "<3", "price_pos_20d": "<0.7"},
        "近7日有过涨停但当前回调到位。提前埋伏等下一波拉升。"
    ),
    _default_skill(
        "oversold_bounce", "超跌反弹", "signal",
        {"rsi": "<30", "bias5": "<-5", "ma_trend": "空头排列"},
        "RSI超卖+偏离MA5超5%，均值回归机会。"
    ),
    _default_skill(
        "volume_breakout", "爆量突破", "signal",
        {"today_chg": ">=5", "vol_ratio": ">2", "is_20d_high": True},
        "放量突破20日新高，趋势启动。"
    ),

    # 时机类
    _default_skill(
        "bull_chase", "牛市追涨", "timing",
        {"market_regime": "bull", "strategy": "dragon"},
        "牛市环境下追涨停/连板，胜率显著高于震荡市。"
    ),
    _default_skill(
        "sideways_reversion", "震荡均值回归", "timing",
        {"market_regime": "sideways", "strategy": "mean_reversion"},
        "震荡市买超跌反弹，不追涨。"
    ),
    _default_skill(
        "bear_defense", "熊市防守", "timing",
        {"market_regime": "bear", "strategy": "minimal"},
        "熊市极少出手，只做超跌反弹，轻仓。"
    ),

    # 风控类
    _default_skill(
        "quant_trap_protect", "量化砸盘保护", "risk",
        {"today_chg": "<=-7", "has_recent_limit": True},
        "盘中急跌≥7%且近期有过涨停，疑似量化砸盘，不卖出等反转。"
    ),
    _default_skill(
        "hold_limit_up", "涨停不卖", "risk",
        {"today_chg": ">=9.5", "holding": True},
        "持仓股涨停不卖，等连板。"
    ),
    _default_skill(
        "fund_outflow_exit", "资金出逃退出", "risk",
        {"main_net_3d": "<0", "consecutive_outflow_days": ">=3"},
        "连续3日主力净流出，趋势恶化信号。"
    ),

    # 板块类
    _default_skill(
        "sector_rotation", "板块轮动", "sector",
        {"sector_inflow_days": ">=3"},
        "连续3天资金净流入的板块，可能是新主线。"
    ),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 技能引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SkillEngine:
    """Agent技能成长引擎"""

    def __init__(self):
        self.skills: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """加载技能库"""
        SKILL_DB_PATH.mkdir(parents=True, exist_ok=True)

        if SKILL_FILE.exists():
            try:
                with open(SKILL_FILE, "r", encoding="utf-8") as f:
                    self.skills = json.load(f)
                logger.info(f"[SkillEngine] 加载 {len(self.skills)} 个技能")
            except Exception as e:
                logger.warning(f"[SkillEngine] 加载失败: {e}")
                self.skills = {}

        # 合并种子技能（不覆盖已有的）
        for seed in SEED_SKILLS:
            if seed["name"] not in self.skills:
                self.skills[seed["name"]] = seed
                logger.info(f"[SkillEngine] 新增种子技能: {seed['display_name']}")

    def _save(self):
        """保存技能库"""
        SKILL_DB_PATH.mkdir(parents=True, exist_ok=True)
        with open(SKILL_FILE, "w", encoding="utf-8") as f:
            json.dump(self.skills, f, ensure_ascii=False, indent=2)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 每日进化
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def evolve(self) -> dict:
        """
        每日收盘后运行：更新技能统计 + 挖掘新模式 + 评级

        Returns:
            进化报告
        """
        logger.info("[SkillEngine] 开始每日进化...")
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "skills_updated": 0,
            "new_skills_discovered": 0,
            "skills_promoted": [],
            "skills_demoted": [],
        }

        # 1. 从trade_journal获取历史数据更新技能统计
        report["skills_updated"] = self._update_skill_stats()

        # 2. 挖掘新模式
        new_count = self._mine_new_patterns()
        report["new_skills_discovered"] = new_count

        # 3. 重新评级
        promoted, demoted = self._regrade_all()
        report["skills_promoted"] = promoted
        report["skills_demoted"] = demoted

        # 4. 保存
        self._save()
        self._save_evolution_log(report)

        logger.info(
            f"[SkillEngine] 进化完成: "
            f"更新{report['skills_updated']}个技能, "
            f"发现{report['new_skills_discovered']}个新模式, "
            f"升级{len(promoted)}, 降级{len(demoted)}"
        )
        return report

    def _update_skill_stats(self) -> int:
        """从交易记录更新技能统计"""
        updated = 0
        try:
            from trade_journal import analyze_winning_patterns, get_performance_summary

            patterns = analyze_winning_patterns(days=90)
            if patterns.get("sample_size", 0) < 3:
                return 0

            perf = get_performance_summary(days=90)

            # 更新MA趋势相关技能
            ma_data = patterns.get("ma_analysis", {})
            for trend, stats in ma_data.items():
                if not trend or trend == "未知":
                    continue
                skill_name = f"ma_{trend}".replace(" ", "_").lower()
                if skill_name not in self.skills:
                    # 自动创建MA趋势技能
                    self.skills[skill_name] = _default_skill(
                        skill_name, f"MA{trend}", "timing",
                        {"ma_trend": trend},
                        f"MA趋势={trend}时的买入表现"
                    )
                self.skills[skill_name]["stats"].update({
                    "total_trades": stats["count"],
                    "win_trades": int(stats["count"] * stats["win_rate"] / 100),
                    "win_rate": stats["win_rate"],
                    "avg_return": stats["avg_pnl"],
                })
                updated += 1

            # 更新MACD信号相关技能
            macd_data = patterns.get("macd_analysis", {})
            for sig, stats in macd_data.items():
                if not sig or sig == "未知":
                    continue
                skill_name = f"macd_{sig}".replace(" ", "_").lower()
                if skill_name not in self.skills:
                    self.skills[skill_name] = _default_skill(
                        skill_name, f"MACD{sig}", "timing",
                        {"macd_signal": sig},
                        f"MACD信号={sig}时的买入表现"
                    )
                self.skills[skill_name]["stats"].update({
                    "total_trades": stats["count"],
                    "win_trades": int(stats["count"] * stats["win_rate"] / 100),
                    "win_rate": stats["win_rate"],
                    "avg_return": stats["avg_pnl"],
                })
                updated += 1

            # 更新持仓天数技能
            hold_data = patterns.get("hold_period_analysis", {})
            for period, stats in hold_data.items():
                if stats.get("count", 0) == 0:
                    continue
                skill_name = f"hold_{period}".replace(" ", "_").replace("-", "_").lower()
                if skill_name not in self.skills:
                    self.skills[skill_name] = _default_skill(
                        skill_name, f"持仓{period}", "timing",
                        {"hold_period": period},
                        f"持仓{period}的表现"
                    )
                self.skills[skill_name]["stats"].update({
                    "total_trades": stats["count"],
                    "win_trades": int(stats["count"] * stats["win_rate"] / 100),
                    "win_rate": stats["win_rate"],
                    "avg_return": stats["avg_pnl"],
                })
                updated += 1

            # 更新资金流向技能
            flow_data = patterns.get("fund_flow_analysis", {})
            for flow, stats in flow_data.items():
                if stats.get("count", 0) == 0:
                    continue
                skill_name = f"flow_{flow[:4]}".replace(" ", "_").lower()
                if skill_name not in self.skills:
                    self.skills[skill_name] = _default_skill(
                        skill_name, flow, "timing",
                        {"fund_flow": flow},
                        f"{flow}的表现"
                    )
                self.skills[skill_name]["stats"].update({
                    "total_trades": stats["count"],
                    "win_trades": int(stats["count"] * stats["win_rate"] / 100),
                    "win_rate": stats["win_rate"],
                    "avg_return": stats["avg_pnl"],
                })
                updated += 1

            # 更新板块技能
            sector_data = patterns.get("sector_analysis", {})
            for sector, stats in sector_data.items():
                if not sector or sector == "未知" or stats.get("count", 0) < 2:
                    continue
                skill_name = f"sector_{sector}".replace(" ", "_").lower()
                if skill_name not in self.skills:
                    self.skills[skill_name] = _default_skill(
                        skill_name, f"板块:{sector}", "sector",
                        {"sector": sector},
                        f"板块={sector}的交易表现"
                    )
                self.skills[skill_name]["stats"].update({
                    "total_trades": stats["count"],
                    "win_trades": int(stats["count"] * stats["win_rate"] / 100),
                    "win_rate": stats["win_rate"],
                    "avg_return": stats["avg_pnl"],
                    "total_pnl": stats.get("total_pnl", 0),
                })
                updated += 1

        except Exception as e:
            logger.warning(f"[SkillEngine] 更新技能统计失败: {e}")

        return updated

    def _mine_new_patterns(self) -> int:
        """挖掘新的高胜率模式组合"""
        new_count = 0
        try:
            from trade_journal import _conn

            conn = _conn()
            # 找MA+MACD组合的胜率
            combos = conn.execute("""
                SELECT ma_trend, macd_signal,
                       COUNT(*) as cnt,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       AVG(pnl_pct) as avg_pnl
                FROM trade_log
                WHERE trade_type = 'sell' AND pnl IS NOT NULL
                  AND trade_date >= date('now', '-90 days')
                  AND ma_trend IS NOT NULL AND macd_signal IS NOT NULL
                GROUP BY ma_trend, macd_signal
                HAVING COUNT(*) >= 3
                ORDER BY AVG(pnl_pct) DESC
            """).fetchall()
            conn.close()

            for combo in combos:
                combo = dict(combo)
                ma = combo["ma_trend"]
                macd = combo["macd_signal"]
                cnt = combo["cnt"]
                wins = combo["wins"]
                avg_pnl = combo["avg_pnl"] or 0
                win_rate = (wins / cnt * 100) if cnt > 0 else 0

                skill_name = f"combo_{ma}_{macd}".replace(" ", "_").lower()

                if skill_name not in self.skills and win_rate >= 55 and cnt >= 3:
                    # 发现新的高胜率组合！
                    self.skills[skill_name] = _default_skill(
                        skill_name,
                        f"组合:{ma}+{macd}",
                        "signal",
                        {"ma_trend": ma, "macd_signal": macd},
                        f"MA{ma}+MACD{macd}组合，历史胜率{win_rate:.0f}%"
                    )
                    self.skills[skill_name]["stats"] = {
                        "total_trades": cnt,
                        "win_trades": wins,
                        "win_rate": round(win_rate, 1),
                        "avg_return": round(avg_pnl, 2),
                        "avg_hold_days": 0,
                        "profit_factor": 0,
                        "max_win": 0,
                        "max_loss": 0,
                        "recent_5_results": [],
                    }
                    new_count += 1
                    logger.info(
                        f"[SkillEngine] 发现新模式: {ma}+{macd} "
                        f"胜率{win_rate:.0f}% 均收益{avg_pnl:.2f}% (样本{cnt})"
                    )

        except Exception as e:
            logger.debug(f"[SkillEngine] 模式挖掘失败: {e}")

        return new_count

    def _regrade_all(self) -> tuple:
        """重新评级所有技能"""
        promoted = []
        demoted = []

        for name, skill in self.skills.items():
            stats = skill.get("stats", {})
            total = stats.get("total_trades", 0)
            win_rate = stats.get("win_rate", 0)
            avg_return = stats.get("avg_return", 0)

            old_grade = skill.get("grade", "C")

            # 计算置信度（样本量加权）
            if total >= 30:
                confidence = min(1.0, total / 50) * (win_rate / 100)
            elif total >= 10:
                confidence = (total / 30) * (win_rate / 100) * 0.8
            elif total >= 3:
                confidence = (total / 10) * (win_rate / 100) * 0.5
            else:
                confidence = 0.1  # 新技能，低置信度

            skill["confidence"] = round(confidence, 3)

            # 评级
            if total >= 10 and win_rate >= 65 and avg_return >= 3:
                new_grade = "S"
            elif total >= 8 and win_rate >= 60 and avg_return >= 2:
                new_grade = "A"
            elif total >= 5 and win_rate >= 50 and avg_return >= 0:
                new_grade = "B"
            elif total >= 3 and win_rate >= 40:
                new_grade = "C"
            elif total >= 5 and win_rate < 35:
                new_grade = "D"
                skill["enabled"] = False  # 自动禁用低胜率技能
            elif total >= 10 and win_rate < 30:
                new_grade = "F"
                skill["enabled"] = False
            else:
                new_grade = "C"  # 样本不足，保持中性

            skill["grade"] = new_grade
            skill["updated_at"] = datetime.now().strftime("%Y-%m-%d")

            if new_grade < old_grade:  # 字母序反转
                promoted.append(f"{skill['display_name']}: {old_grade}→{new_grade}")
            elif new_grade > old_grade:
                demoted.append(f"{skill['display_name']}: {old_grade}→{new_grade}")

        return promoted, demoted

    def _save_evolution_log(self, report: dict):
        """保存进化日志"""
        logs = []
        if EVOLUTION_LOG.exists():
            try:
                with open(EVOLUTION_LOG, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []

        logs.append(report)
        # 只保留最近90天
        logs = logs[-90:]

        with open(EVOLUTION_LOG, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Agent Prompt 构建
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_prompt(self, code: str = None, include_all: bool = False) -> str:
        """
        构建注入Agent的技能参考文本

        只注入 enabled=True 且 grade>=C 的技能
        按 confidence 降序排列，优先展示高置信度技能
        """
        lines = []
        lines.append("## Agent技能库（基于真实交易数据，持续进化中）")
        lines.append("")

        # 按类别组织
        categories = {
            "signal": ("[Signal] 信号技能", []),
            "timing": ("[Timing] 时机技能", []),
            "risk": ("[Risk] 风控技能", []),
            "sector": ("[Sector] 板块技能", []),
        }

        for name, skill in sorted(
            self.skills.items(),
            key=lambda x: x[1].get("confidence", 0),
            reverse=True
        ):
            if not skill.get("enabled", True) and not include_all:
                continue
            if skill.get("grade", "C") in ("D", "F") and not include_all:
                continue

            cat = skill.get("category", "signal")
            if cat in categories:
                categories[cat][1].append(skill)

        for cat_key, (cat_title, skills) in categories.items():
            if not skills:
                continue
            lines.append(f"### {cat_title}")
            for s in skills[:10]:  # 每类最多10个
                stats = s.get("stats", {})
                total = stats.get("total_trades", 0)
                wr = stats.get("win_rate", 0)
                avg_r = stats.get("avg_return", 0)
                grade = s.get("grade", "C")
                conf = s.get("confidence", 0)

                # 格式：[等级] 技能名 — 条件 | 胜率 | 均收益 | 置信度
                if total >= 3:
                    lines.append(
                        f"  [{grade}] **{s['display_name']}** — "
                        f"胜率{wr:.0f}% 均收益{avg_r:+.2f}% "
                        f"(样本{total}, 置信{conf:.0%})"
                    )
                    if s.get("description"):
                        lines.append(f"      → {s['description'][:60]}")
                else:
                    lines.append(
                        f"  [{grade}] **{s['display_name']}** — "
                        f"待验证 ({s.get('description', '')[:40]})"
                    )
            lines.append("")

        # 该股相关技能
        if code:
            stock_skills = self._get_stock_relevant_skills(code)
            if stock_skills:
                lines.append(f"### [Match] 该股({code})相关技能")
                for s in stock_skills:
                    stats = s.get("stats", {})
                    lines.append(
                        f"  [{s['grade']}] {s['display_name']} — "
                        f"胜率{stats.get('win_rate', 0):.0f}% "
                        f"均收益{stats.get('avg_return', 0):+.2f}%"
                    )
                lines.append("")

        # 决策规则
        lines.append("### [Rules] 技能驱动决策规则")
        lines.append("- **S/A级技能触发** → 高权重参考，可作为核心买入/持有理由")
        lines.append("- **B级技能触发** → 辅助参考，需结合其他条件")
        lines.append("- **C级技能触发** → 低权重参考，样本不足需谨慎")
        lines.append("- **D/F级已禁用** → 历史表现差，不应参考")
        lines.append("- 多个技能同时触发时，优先听从置信度最高的技能")
        lines.append("- 如果S级风控技能（如量化砸盘保护）触发，必须执行，不得违反")

        return "\n".join(lines)

    def _get_stock_relevant_skills(self, code: str) -> list:
        """获取与该股相关的技能（基于该股历史交易记录匹配技能条件）"""
        relevant = []
        try:
            from trade_journal import _conn
            conn = _conn()
            # 获取该股最近一次交易的技术状态
            last = conn.execute("""
                SELECT ma_trend, macd_signal, rsi, vol_pattern, sector
                FROM trade_log
                WHERE code = ?
                ORDER BY trade_date DESC LIMIT 1
            """, (code,)).fetchone()
            conn.close()

            if last:
                last = dict(last)
                for name, skill in self.skills.items():
                    conds = skill.get("conditions", {})
                    # 匹配MA趋势
                    if conds.get("ma_trend") and conds["ma_trend"] == last.get("ma_trend"):
                        relevant.append(skill)
                    # 匹配MACD
                    elif conds.get("macd_signal") and conds["macd_signal"] == last.get("macd_signal"):
                        relevant.append(skill)
                    # 匹配板块
                    elif conds.get("sector") and conds["sector"] == last.get("sector"):
                        relevant.append(skill)

        except Exception:
            pass

        # 按confidence排序
        relevant.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return relevant[:5]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 查询接口
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_top_skills(self, n: int = 5, category: str = None) -> list:
        """获取top N技能（按confidence排序）"""
        skills = list(self.skills.values())
        if category:
            skills = [s for s in skills if s.get("category") == category]
        skills = [s for s in skills if s.get("enabled", True)]
        skills.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return skills[:n]

    def get_skill_summary(self) -> str:
        """获取技能库概要"""
        total = len(self.skills)
        enabled = sum(1 for s in self.skills.values() if s.get("enabled", True))
        by_grade = {}
        for s in self.skills.values():
            g = s.get("grade", "C")
            by_grade[g] = by_grade.get(g, 0) + 1

        grade_str = " ".join(f"{g}:{c}" for g, c in sorted(by_grade.items()))
        return f"技能库: {total}个(启用{enabled}) | 评级分布: {grade_str}"

    def match_skills(self, **conditions) -> list:
        """匹配当前条件触发的技能"""
        matched = []
        for name, skill in self.skills.items():
            if not skill.get("enabled", True):
                continue
            conds = skill.get("conditions", {})
            if not conds:
                continue

            # 简单条件匹配
            match = True
            for key, expected in conds.items():
                actual = conditions.get(key)
                if actual is None:
                    match = False
                    break
                # 支持比较运算符
                if isinstance(expected, str):
                    if expected.startswith(">="):
                        if float(actual) < float(expected[2:]):
                            match = False
                    elif expected.startswith("<="):
                        if float(actual) > float(expected[2:]):
                            match = False
                    elif expected.startswith(">"):
                        if float(actual) <= float(expected[1:]):
                            match = False
                    elif expected.startswith("<"):
                        if float(actual) >= float(expected[1:]):
                            match = False
                    elif str(actual) != expected:
                        match = False
                elif actual != expected:
                    match = False

            if match:
                matched.append(skill)

        matched.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return matched


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 便捷接口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_engine: Optional[SkillEngine] = None

def get_skill_engine() -> SkillEngine:
    """获取全局技能引擎实例"""
    global _engine
    if _engine is None:
        _engine = SkillEngine()
    return _engine


def evolve_skills() -> dict:
    """每日进化（在收盘任务中调用）"""
    return get_skill_engine().evolve()


def build_skill_prompt(code: str = None) -> str:
    """构建Agent技能prompt"""
    return get_skill_engine().build_prompt(code=code)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    engine = SkillEngine()

    print("\n" + "=" * 60)
    print(engine.get_skill_summary())
    print("=" * 60)

    # 进化
    report = engine.evolve()
    print(f"\n进化报告: {json.dumps(report, ensure_ascii=False, indent=2)}")

    # 展示prompt
    print("\n" + "=" * 60)
    print("Agent Prompt Preview:")
    print("=" * 60)
    print(engine.build_prompt())
