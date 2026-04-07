"""
策略自改进反馈环 — 对比历史建议 vs 实际盈亏，生成改进建议注入到LLM prompt
"""
import logging
import sqlite3
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = "data/scanner_history.db"


class StrategyFeedbackLoop:
    """策略反馈循环

    读取 execution_log + trade_log，对比每笔:
    - 引擎建议的动作 vs 实际执行结果
    - 目标价 vs 实际卖出价
    - 哪些信号/条件赚钱，哪些亏钱
    """

    def generate_feedback_report(self, days: int = 30) -> dict:
        """生成策略反馈报告"""
        try:
            from src.broker.execution_tracker import _get_conn
            conn = _get_conn()  # 自动建表
            conn.row_factory = sqlite3.Row

            # 执行记录
            exec_rows = conn.execute("""
                SELECT * FROM execution_log
                WHERE trade_date >= date('now', '-' || ? || ' days', 'localtime')
                ORDER BY created_at
            """, (days,)).fetchall()

            # 交易记录（含盈亏）
            trade_rows = conn.execute("""
                SELECT * FROM trade_log
                WHERE trade_date >= date('now', '-' || ? || ' days', 'localtime')
                  AND trade_type = 'sell' AND pnl IS NOT NULL
                ORDER BY trade_date
            """, (days,)).fetchall()

            conn.close()

            report = {
                "days": days,
                "execution_stats": self._analyze_executions(exec_rows),
                "trade_outcomes": self._analyze_trade_outcomes(trade_rows),
                "signal_accuracy": self._analyze_signal_accuracy(trade_rows),
                "confidence_calibration": self._analyze_confidence(exec_rows, trade_rows),
            }
            return report

        except Exception as e:
            logger.error(f"[反馈] 生成报告失败: {e}")
            return {"error": str(e)}

    def format_for_llm_prompt(self, report: dict) -> str:
        """将反馈报告格式化为LLM prompt注入文本"""
        if report.get("error") or not report.get("trade_outcomes"):
            return ""

        lines = [f"## 你的历史战绩 (近{report.get('days', 30)}天)"]

        outcomes = report.get("trade_outcomes", {})
        if outcomes.get("total_trades", 0) > 0:
            lines.append(
                f"- 总交易: {outcomes['total_trades']}笔, "
                f"胜率: {outcomes.get('win_rate', 0):.1f}%"
            )
            lines.append(
                f"- 平均盈利: {outcomes.get('avg_win_pct', 0):+.2f}%, "
                f"平均亏损: {outcomes.get('avg_loss_pct', 0):.2f}%"
            )
            lines.append(f"- 盈亏比: {outcomes.get('profit_factor', 0):.2f}")

        signals = report.get("signal_accuracy", {})
        best = signals.get("best_signal")
        worst = signals.get("worst_signal")
        if best:
            lines.append(f"- 最佳信号: {best['name']} (胜率{best['win_rate']:.0f}%)")
        if worst:
            lines.append(f"- 最差信号: {worst['name']} (胜率{worst['win_rate']:.0f}%)")

        exec_stats = report.get("execution_stats", {})
        if exec_stats.get("avg_slippage_pct"):
            lines.append(f"- 平均滑点: {exec_stats['avg_slippage_pct']:+.3f}%")

        if exec_stats.get("target_hit_rate") is not None:
            lines.append(
                f"- 止损触发率: {exec_stats.get('stop_loss_hit_rate', 0):.0f}% "
                f"(在到达目标价之前先被止损)"
            )

        calibration = report.get("confidence_calibration", {})
        if calibration.get("note"):
            lines.append(f"- 校准提示: {calibration['note']}")

        lines.append("- 规则: 对近期亏钱多的信号降低权重，对赢钱多的信号可适当加大仓位")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _analyze_executions(self, rows) -> dict:
        """分析执行质量"""
        if not rows:
            return {}

        total = len(rows)
        filled = [r for r in rows if r["order_status"] in ("filled", "submitted")]
        slippages = [r["slippage_pct"] for r in filled if r["slippage_pct"] != 0]

        # 目标价 vs 止损 命中率
        with_target = [r for r in rows if r["target_sell_price"] > 0]
        with_stop = [r for r in rows if r["stop_loss_price"] > 0]

        return {
            "total_executions": total,
            "fill_rate": round(len(filled) / total * 100, 1) if total else 0,
            "avg_slippage_pct": round(sum(slippages) / len(slippages), 3) if slippages else 0,
            "orders_with_target": len(with_target),
            "orders_with_stop": len(with_stop),
        }

    def _analyze_trade_outcomes(self, rows) -> dict:
        """分析交易盈亏结果"""
        if not rows:
            return {}

        total = len(rows)
        wins = [r for r in rows if (r["pnl"] or 0) > 0]
        losses = [r for r in rows if (r["pnl"] or 0) < 0]

        avg_win = 0
        avg_loss = 0
        if wins:
            avg_win = sum(r["pnl_pct"] or 0 for r in wins) / len(wins)
        if losses:
            avg_loss = sum(r["pnl_pct"] or 0 for r in losses) / len(losses)

        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        return {
            "total_trades": total,
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": round(len(wins) / total * 100, 1) if total else 0,
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
        }

    def _analyze_signal_accuracy(self, rows) -> dict:
        """按技术信号分组统计胜率"""
        if not rows:
            return {}

        signal_stats = defaultdict(lambda: {"wins": 0, "total": 0})

        for r in rows:
            # 使用 trade_log 中的 ma_trend, macd_signal 字段
            ma = r["ma_trend"] if "ma_trend" in r.keys() else None
            macd = r["macd_signal"] if "macd_signal" in r.keys() else None
            is_win = (r["pnl"] or 0) > 0

            if ma:
                signal_stats[f"MA_{ma}"]["total"] += 1
                if is_win:
                    signal_stats[f"MA_{ma}"]["wins"] += 1
            if macd:
                signal_stats[f"MACD_{macd}"]["total"] += 1
                if is_win:
                    signal_stats[f"MACD_{macd}"]["wins"] += 1

        # 找最佳/最差信号（至少3次）
        results = {}
        qualified = {
            k: {"name": k, "win_rate": v["wins"] / v["total"] * 100, "count": v["total"]}
            for k, v in signal_stats.items() if v["total"] >= 3
        }

        if qualified:
            best = max(qualified.values(), key=lambda x: x["win_rate"])
            worst = min(qualified.values(), key=lambda x: x["win_rate"])
            results["best_signal"] = best
            results["worst_signal"] = worst
            results["all_signals"] = list(qualified.values())

        return results

    def _analyze_confidence(self, exec_rows, trade_rows) -> dict:
        """分析置信度校准（置信度高的交易是否真的赢得多）"""
        # 这需要 backtest_validation 数据，暂时返回基础提示
        if not trade_rows:
            return {}

        total = len(trade_rows)
        wins = sum(1 for r in trade_rows if (r["pnl"] or 0) > 0)
        win_rate = wins / total * 100 if total else 0

        note = ""
        if win_rate < 40:
            note = "近期胜率偏低(<40%)，建议减仓并等待更确定的机会"
        elif win_rate > 70:
            note = "近期胜率较高(>70%)，注意不要过度自信增大风险敞口"
        else:
            note = f"近期胜率{win_rate:.0f}%，表现正常"

        return {"win_rate": round(win_rate, 1), "note": note}
