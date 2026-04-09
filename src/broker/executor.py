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

        # 记录到交易日志（复盘用）
        self._record_to_journal(report, actions)

        # 持仓核对：用委托明细和真实持仓验证交易是否生效
        if report.filled_count > 0:
            self._verify_execution(report, actions)

        return report

    def _record_to_journal(self, report: ExecutionReport, actions: List[dict]):
        """将成交记录写入trade_journal（供复盘分析用）"""
        try:
            from trade_journal import record_buy, record_sell
        except ImportError:
            return

        # 建立 code→action 映射获取AI分析数据
        action_map = {a.get("code", ""): a for a in actions}

        for order in report.orders:
            if not order.is_success:
                continue

            code = order.code
            name = getattr(order, "name", code)
            action_info = action_map.get(code, {})
            price = order.actual_price or action_info.get("price", 0)
            shares = order.actual_shares or action_info.get("shares", 0)

            try:
                if order.direction == "buy":
                    record_buy(
                        code=code, name=name, shares=shares, price=price,
                        ma_trend=str(action_info.get("trend_assessment", ""))[:20],
                        macd_signal="",
                        rsi=0,
                        vol_pattern="",
                        tech_score=action_info.get("score", 0),
                        sector=action_info.get("sector", ""),
                        source="rebalance",
                        note=f"调仓买入 {str(action_info.get('reason', ''))[:50]}",
                    )
                elif order.direction == "sell":
                    buy_price = float(action_info.get("cost_price", 0) or 0)
                    record_sell(
                        code=code, name=name, shares=shares, price=price,
                        buy_price=buy_price,
                        ma_trend=str(action_info.get("trend_assessment", ""))[:20],
                        macd_signal="",
                        rsi=0,
                        vol_pattern="",
                        tech_score=action_info.get("score", 0),
                        sector=action_info.get("sector", ""),
                        source="rebalance",
                        note=f"调仓卖出 {str(action_info.get('reason', ''))[:50]}",
                    )
            except Exception as e:
                logger.warning(f"[执行器] 交易日志记录失败 {code}: {e}")

    def _verify_execution(self, report: ExecutionReport, actions: List[dict]):
        """核对委托明细和真实持仓，确认交易结果。

        数据来源:
        1. broker.get_today_orders() — 委托列表（含成交状态/成交数量）
        2. broker.get_positions() — 当前真实持仓
        3. report.orders — 本次下单结果
        """
        try:
            # 等待 THS 刷新
            time.sleep(2)

            # ── 1. 读取委托明细 ──
            orders = self._broker.get_today_orders()
            if not orders:
                logger.warning("[核对] 无法获取今日委托")
                return

            # 按股票代码汇总已成交数量
            filled_by_code = {}  # {code: {"buy": shares, "sell": shares}}
            for o in orders:
                if o.status in ("filled", "partial"):
                    code = o.code
                    if code not in filled_by_code:
                        filled_by_code[code] = {"buy": 0, "sell": 0}
                    filled_by_code[code][o.direction] += o.filled_shares

            # ── 2. 读取真实持仓 ──
            positions = self._broker.get_positions()
            pos_by_code = {p.code: p for p in positions}

            # ── 3. 核对每笔操作 ──
            mismatches = []
            for order_result in report.orders:
                if not order_result.is_success:
                    continue

                code = order_result.code
                name = order_result.name or code
                direction = order_result.direction
                expected_shares = order_result.actual_shares or order_result.requested_shares

                # 从委托明细确认
                entrust_filled = filled_by_code.get(code, {}).get(direction, 0)
                pos = pos_by_code.get(code)

                if entrust_filled >= expected_shares:
                    logger.info(
                        f"[核对] ✅ {name}({code}) {direction} "
                        f"{expected_shares}股 → 委托已成交 {entrust_filled}股"
                    )
                elif entrust_filled > 0:
                    logger.warning(
                        f"[核对] ⚠️ {name}({code}) {direction} "
                        f"期望{expected_shares}股 → 实际成交 {entrust_filled}股 (部分成交)"
                    )
                    mismatches.append({
                        "code": code, "name": name, "direction": direction,
                        "expected": expected_shares, "actual": entrust_filled,
                        "status": "partial",
                    })
                else:
                    logger.error(
                        f"[核对] ❌ {name}({code}) {direction} "
                        f"{expected_shares}股 → 委托未成交！"
                    )
                    mismatches.append({
                        "code": code, "name": name, "direction": direction,
                        "expected": expected_shares, "actual": 0,
                        "status": "not_filled",
                    })

                # 持仓验证
                if pos:
                    logger.info(
                        f"[核对]    持仓: {pos.shares}股 可卖:{pos.sellable_shares}股 "
                        f"成本:{pos.cost_price:.2f} 现价:{pos.current_price:.2f}"
                    )
                elif direction == "sell":
                    logger.info(f"[核对]    {name} 已不在持仓中（清仓确认）")

            if mismatches:
                report.verification_mismatches = mismatches
                logger.warning(
                    f"[核对] 共 {len(mismatches)} 笔不一致，需关注"
                )
            else:
                logger.info("[核对] 全部核对通过 ✅")

        except Exception as e:
            logger.warning(f"[核对] 持仓核对异常（不影响交易）: {e}")

    def _execute_single(self, action: dict, total_asset: float) -> OrderResult:
        """执行单笔动作"""
        code = action.get("code", "")
        name = action.get("name", code)
        act = action.get("action", "")

        # 价格获取优先级: suggested_price > reference_price > current_price > target_sell/buy_price > broker实时
        price = float(
            action.get("suggested_price", 0)
            or action.get("reference_price", 0)
            or action.get("current_price", 0)
            or 0
        )
        # 如果 AI 给了 target_sell_price / stop_loss_price / target_buy_price，也可用
        if price <= 0:
            if act in ("sell", "reduce"):
                price = float(action.get("target_sell_price", 0) or action.get("stop_loss_price", 0) or 0)
            else:
                price = float(action.get("target_buy_price", 0) or 0)

        # 最终兜底：从 broker 实时持仓获取价格
        if price <= 0 and code:
            try:
                positions = self._broker.get_positions()
                for p in positions:
                    if p.code == code and p.current_price > 0:
                        price = p.current_price
                        logger.info(f"[执行器] {name}({code}) 价格从broker实时获取: {price}")
                        break
            except Exception:
                pass

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

        # 下单：根据 AI execution_strategy 决定执行方式
        strategy = action.get("execution_strategy", {})
        order_type = str(strategy.get("order_type", "")).lower()
        chase_pct = float(strategy.get("chase_max_pct", 0) or 0)
        chase_timeout = int(strategy.get("chase_timeout", 0) or 0)
        split = strategy.get("split_orders", False)

        # 大单拆分（AI 建议 split_orders=true 时，分批下单）
        if split and shares > 5000 and hasattr(self._broker, "smart_buy"):
            result = self._execute_split(
                direction, code, price, shares,
                chase_pct, chase_timeout, order_type,
            )
        # aggressive/passive → 智能追单，AI 给的参数
        elif chase_pct > 0 and chase_timeout > 0 and hasattr(self._broker, "smart_buy"):
            # AI 给了追单参数
            if order_type == "aggressive":
                # 对手价：买用卖一(+0.01)，卖用买一(-0.01)
                price = round(price + (0.01 if direction == "buy" else -0.01), 2)
            if direction == "sell":
                result = self._broker.smart_sell(
                    code, price, shares,
                    max_chase_pct=chase_pct, timeout=chase_timeout,
                )
            else:
                result = self._broker.smart_buy(
                    code, price, shares,
                    max_chase_pct=chase_pct, timeout=chase_timeout,
                )
        # limit 或无策略 → 普通限价委托
        else:
            if direction == "sell":
                result = self._broker.sell(code, price, shares)
            else:
                result = self._broker.buy(code, price, shares)

        if strategy.get("reason"):
            result.message = f"{result.message} | AI策略: {strategy['reason'][:50]}"

        result.name = name
        return result

    def _execute_split(self, direction, code, price, total_shares,
                        chase_pct, chase_timeout, order_type) -> OrderResult:
        """拆单执行：将大单拆成多笔小单，减少市场冲击。"""
        batch_size = 2000  # 每批最多 2000 股
        remaining = total_shares
        filled_total = 0

        while remaining > 0:
            batch = min(remaining, batch_size)
            if chase_pct > 0 and chase_timeout > 0:
                if direction == "sell":
                    r = self._broker.smart_sell(code, price, batch,
                                                max_chase_pct=chase_pct,
                                                timeout=min(chase_timeout, 60))
                else:
                    r = self._broker.smart_buy(code, price, batch,
                                               max_chase_pct=chase_pct,
                                               timeout=min(chase_timeout, 60))
            else:
                if direction == "sell":
                    r = self._broker.sell(code, price, batch)
                else:
                    r = self._broker.buy(code, price, batch)

            if r.is_success:
                filled_total += r.actual_shares or batch
            remaining -= batch
            if remaining > 0:
                time.sleep(2)  # 批间间隔
            # 更新实时价
            if hasattr(self._broker, "_get_realtime_price"):
                new_p = self._broker._get_realtime_price(code)
                if new_p > 0:
                    price = round(new_p, 2)

        return OrderResult(
            code=code, direction=direction,
            status="filled" if filled_total >= total_shares else "partial",
            requested_price=price, requested_shares=total_shares,
            actual_price=price, actual_shares=filled_total,
            message=f"拆单执行: {filled_total}/{total_shares}股",
        )

    def _dry_run(self, actions: List[dict], total_asset: float) -> List[OrderResult]:
        """Dry run — 模拟执行，不实际下单"""
        results = []
        for action in actions:
            code = action.get("code", "")
            name = action.get("name", code)
            act = action.get("action", "")
            price = float(action.get("suggested_price", 0) or action.get("reference_price", 0) or action.get("current_price", 0))
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
                price = a.get("suggested_price") or a.get("reference_price") or a.get("current_price", "?")
                shares = a.get("suggested_shares", "?")
                act_cn = "清仓" if a.get("action") == "sell" else "减仓"
                lines.append(f"  • {a.get('name','')}({a.get('code','')}) {act_cn} {shares}股 @ {price}")
            lines.append("")

        if buys:
            lines.append("🟢 **买入:**")
            for a in buys:
                price = a.get("suggested_price") or a.get("reference_price") or a.get("current_price", "?")
                shares = a.get("suggested_shares", "?")
                lines.append(f"  • {a.get('name','')}({a.get('code','')}) {shares}股 @ {price}")
            lines.append("")

        lines.append("回复「确认执行」下单，「取消」放弃")
        return "\n".join(lines)
