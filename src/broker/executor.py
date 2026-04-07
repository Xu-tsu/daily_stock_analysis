"""
批量执行器 — 将调仓引擎的 actions[] 批量下单到券商
"""
import logging
import os
import time
import uuid
from typing import List, Optional

from src.broker.base import BrokerAdapter
from src.broker.models import OrderResult, ExecutionReport
from src.broker.safety import check_order_allowed, increment_order_count, is_trading_halted

logger = logging.getLogger(__name__)


class RebalanceExecutor:
    """调仓批量执行器

    接收 rebalance_engine 输出的 actions[]，
    按「先卖后买」顺序在券商下单。
    """

    def __init__(self, broker: BrokerAdapter):
        self._broker = broker

    def execute(
        self,
        actions: List[dict],
        mode: str = "confirm",
        total_asset: float = 0,
    ) -> ExecutionReport:
        """执行调仓动作列表

        Args:
            actions: rebalance_engine 输出的 actions 列表
            mode: "confirm"(需确认) / "auto"(直接执行) / "dry_run"(只记录不下单)
            total_asset: 总资产（用于安全检查）

        Returns:
            ExecutionReport
        """
        session_id = f"rebalance_{uuid.uuid4().hex[:8]}"
        report = ExecutionReport(session_id=session_id, mode=mode)

        if not actions:
            logger.info("[执行器] 无操作指令")
            return report

        # 过滤掉 hold 动作
        actionable = [a for a in actions if a.get("action") in ("buy", "sell", "reduce")]
        if not actionable:
            logger.info("[执行器] 所有持仓均为持有，无需操作")
            return report

        # 分成卖出组和买入组（先卖后买）
        sells = [a for a in actionable if a.get("action") in ("sell", "reduce")]
        buys = [a for a in actionable if a.get("action") == "buy"]

        logger.info(
            f"[执行器] 模式={mode} | 卖出{len(sells)}笔 买入{len(buys)}笔 "
            f"(跳过{len(actions) - len(actionable)}笔hold)"
        )

        if mode == "dry_run":
            report.orders = self._dry_run(sells + buys, total_asset)
            return report

        # 先卖出
        for action in sells:
            result = self._execute_single(action, total_asset)
            report.orders.append(result)
            if result.is_success:
                increment_order_count()
            time.sleep(1)  # 委托间隔，避免过快

        # 再买入
        for action in buys:
            if is_trading_halted():
                report.orders.append(OrderResult(
                    code=action.get("code", ""),
                    direction="buy", status="rejected",
                    message="急停开关已触发，跳过买入",
                ))
                continue
            result = self._execute_single(action, total_asset)
            report.orders.append(result)
            if result.is_success:
                increment_order_count()
            time.sleep(1)

        logger.info(
            f"[执行器] 完成: {report.filled_count}笔成交 "
            f"{report.rejected_count}笔拒绝 滑点={report.total_slippage_pct:+.3f}%"
        )
        return report

    def _execute_single(self, action: dict, total_asset: float) -> OrderResult:
        """执行单笔动作"""
        code = action.get("code", "")
        name = action.get("name", code)
        act = action.get("action", "")
        price = float(action.get("suggested_price", action.get("current_price", 0)))
        shares = int(action.get("suggested_shares", 0))

        # 如果没有 suggested_shares，尝试从 detail 提取或使用全部持仓
        if shares <= 0:
            shares = self._infer_shares(action)

        if shares <= 0 or price <= 0:
            return OrderResult(
                code=code, name=name, direction="sell" if act in ("sell", "reduce") else "buy",
                status="rejected", requested_price=price, requested_shares=shares,
                message=f"无法确定数量或价格 (shares={shares}, price={price})",
            )

        direction = "sell" if act in ("sell", "reduce") else "buy"

        # 安全检查
        allowed, reason = check_order_allowed(code, price, shares, total_asset)
        if not allowed:
            return OrderResult(
                code=code, name=name, direction=direction,
                status="rejected", requested_price=price, requested_shares=shares,
                message=f"安全限制: {reason}",
            )

        # 下单
        if direction == "sell":
            result = self._broker.sell(code, price, shares)
        else:
            result = self._broker.buy(code, price, shares)

        result.name = name
        return result

    def _dry_run(self, actions: List[dict], total_asset: float) -> List[OrderResult]:
        """Dry run — 模拟执行，不实际下单"""
        results = []
        for action in actions:
            code = action.get("code", "")
            name = action.get("name", code)
            act = action.get("action", "")
            price = float(action.get("suggested_price", action.get("current_price", 0)))
            shares = int(action.get("suggested_shares", 0))
            if shares <= 0:
                shares = self._infer_shares(action)

            direction = "sell" if act in ("sell", "reduce") else "buy"

            allowed, reason = check_order_allowed(code, price, shares, total_asset)
            if not allowed:
                results.append(OrderResult(
                    code=code, name=name, direction=direction,
                    status="rejected", requested_price=price, requested_shares=shares,
                    message=f"[dry_run] 安全限制: {reason}",
                ))
            else:
                results.append(OrderResult(
                    code=code, name=name, direction=direction,
                    status="filled", requested_price=price, requested_shares=shares,
                    actual_price=price, actual_shares=shares,
                    message="[dry_run] 模拟成交",
                ))
            logger.info(
                f"[dry_run] {direction} {name}({code}) {shares}股 @ {price:.2f} "
                f"→ {results[-1].status}"
            )
        return results

    @staticmethod
    def _infer_shares(action: dict) -> int:
        """从 action 中推断交易股数"""
        # 优先用 suggested_shares
        s = action.get("suggested_shares", 0)
        if s and int(s) > 0:
            return int(s)
        # 尝试从 detail 里提取数字（如 "建议数量: 300股"）
        detail = action.get("detail", "")
        import re
        m = re.search(r"(\d+)\s*股", detail)
        if m:
            return int(m.group(1))
        return 0

    def format_confirmation_message(self, actions: List[dict]) -> str:
        """生成确认消息（Feishu 用）"""
        actionable = [a for a in actions if a.get("action") in ("buy", "sell", "reduce")]
        if not actionable:
            return "无需操作，所有持仓均为持有。"

        sells = [a for a in actionable if a.get("action") in ("sell", "reduce")]
        buys = [a for a in actionable if a.get("action") == "buy"]

        lines = ["📦 **调仓执行确认**", ""]

        if sells:
            lines.append("🔴 **卖出/减仓:**")
            for a in sells:
                price = a.get("suggested_price", a.get("current_price", "?"))
                shares = a.get("suggested_shares", "?")
                act_cn = "清仓" if a.get("action") == "sell" else "减仓"
                lines.append(f"  • {a.get('name','')}({a.get('code','')}) {act_cn} {shares}股 @ {price}")
            lines.append("")

        if buys:
            lines.append("🟢 **买入:**")
            for a in buys:
                price = a.get("suggested_price", a.get("current_price", "?"))
                shares = a.get("suggested_shares", "?")
                lines.append(f"  • {a.get('name','')}({a.get('code','')}) {shares}股 @ {price}")
            lines.append("")

        lines.append("回复「确认执行」下单，「取消」放弃")
        return "\n".join(lines)
