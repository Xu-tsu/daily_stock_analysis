"""
持仓同步 — THS账户 ↔ 本地 portfolio.json
"""
import logging
from dataclasses import dataclass, field
from typing import List

from src.broker.base import BrokerAdapter
from src.broker.models import Position, AccountBalance

logger = logging.getLogger(__name__)


@dataclass
class SyncDiff:
    """同步差异"""
    code: str
    name: str
    field: str          # "shares" / "new" / "removed" / "cost_price"
    local_val: str
    broker_val: str


@dataclass
class SyncReport:
    """同步报告"""
    synced: bool = False
    diffs: List[SyncDiff] = field(default_factory=list)
    positions_added: int = 0
    positions_removed: int = 0
    positions_updated: int = 0
    cash_diff: float = 0.0
    message: str = ""

    def format_summary(self) -> str:
        if not self.synced:
            return f"❌ 同步失败: {self.message}"
        if not self.diffs and self.cash_diff == 0:
            return "✅ 持仓同步: 数据一致，无需修正"
        lines = ["🔄 **持仓同步完成**"]
        if self.positions_added:
            lines.append(f"  新增: {self.positions_added}只")
        if self.positions_removed:
            lines.append(f"  移除: {self.positions_removed}只")
        if self.positions_updated:
            lines.append(f"  更新: {self.positions_updated}只")
        if abs(self.cash_diff) > 1:
            lines.append(f"  现金差异: {self.cash_diff:+,.0f}元")
        for d in self.diffs[:10]:
            lines.append(f"  • {d.name}({d.code}): {d.field} {d.local_val}→{d.broker_val}")
        return "\n".join(lines)


class PositionSyncer:
    """持仓同步器"""

    def __init__(self, broker: BrokerAdapter):
        self._broker = broker

    def sync_to_local(self, portfolio: dict) -> SyncReport:
        """从券商同步持仓到本地 portfolio dict

        以券商为准，更新本地持仓。返回 SyncReport + 修改后的 portfolio。

        Args:
            portfolio: 当前本地 portfolio dict（会被原地修改）

        Returns:
            SyncReport
        """
        report = SyncReport()

        try:
            broker_positions = self._broker.get_positions()
            broker_balance = self._broker.get_balance()
        except Exception as e:
            report.message = f"获取券商数据失败: {e}"
            return report

        if not broker_positions and not broker_balance.total_asset:
            report.message = "券商返回空数据（可能未登录或非交易时间）"
            return report

        # 构建券商持仓 map
        broker_map = {p.code: p for p in broker_positions}
        local_holdings = portfolio.get("holdings", [])
        local_map = {h["code"]: h for h in local_holdings}

        new_holdings = []

        # 1. 更新/保留在券商中存在的持仓
        for code, bp in broker_map.items():
            if code in local_map:
                lh = local_map[code]
                # 检查差异
                if lh.get("shares", 0) != bp.shares:
                    report.diffs.append(SyncDiff(
                        code=code, name=bp.name, field="shares",
                        local_val=str(lh.get("shares", 0)),
                        broker_val=str(bp.shares),
                    ))
                    report.positions_updated += 1
                # 用券商数据覆盖
                lh["shares"] = bp.shares
                lh["sellable_shares"] = bp.sellable_shares
                lh["current_price"] = bp.current_price
                lh["market_value"] = bp.market_value
                lh["cost_price"] = bp.cost_price if bp.cost_price > 0 else lh.get("cost_price", 0)
                lh["pnl_pct"] = bp.pnl_pct
                lh["name"] = bp.name or lh.get("name", code)
                new_holdings.append(lh)
            else:
                # 券商有但本地没有 → 新增
                report.diffs.append(SyncDiff(
                    code=code, name=bp.name, field="new",
                    local_val="无", broker_val=f"{bp.shares}股",
                ))
                report.positions_added += 1
                new_holdings.append({
                    "code": code,
                    "name": bp.name,
                    "shares": bp.shares,
                    "sellable_shares": bp.sellable_shares,
                    "cost_price": bp.cost_price,
                    "current_price": bp.current_price,
                    "market_value": bp.market_value,
                    "pnl_pct": bp.pnl_pct,
                    "sector": "",
                    "buy_date": "",
                    "strategy_tag": "broker_sync",
                })

        # 2. 标记本地有但券商没有的（已卖完）
        for code, lh in local_map.items():
            if code not in broker_map:
                report.diffs.append(SyncDiff(
                    code=code, name=lh.get("name", code), field="removed",
                    local_val=f"{lh.get('shares', 0)}股", broker_val="0",
                ))
                report.positions_removed += 1
                # 不加入 new_holdings（等于删除）

        # 3. 同步现金
        if broker_balance.cash > 0:
            old_cash = portfolio.get("cash", 0)
            report.cash_diff = broker_balance.cash - old_cash
            portfolio["cash"] = broker_balance.cash
            portfolio["total_asset"] = broker_balance.total_asset

        portfolio["holdings"] = new_holdings
        report.synced = True
        logger.info(
            f"[同步] 完成: +{report.positions_added} -{report.positions_removed} "
            f"~{report.positions_updated} 现金差={report.cash_diff:+,.0f}"
        )
        return report
