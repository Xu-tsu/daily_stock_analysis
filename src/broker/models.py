"""
券商数据模型 — 与具体券商解耦的标准数据结构
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional


@dataclass
class OrderResult:
    """单笔委托执行结果"""
    order_id: str = ""
    code: str = ""
    name: str = ""
    direction: str = ""             # "buy" / "sell"
    status: str = "error"           # "submitted" / "filled" / "partial" / "rejected" / "error"
    requested_price: float = 0.0
    requested_shares: int = 0
    actual_price: float = 0.0
    actual_shares: int = 0
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def slippage_pct(self) -> float:
        """滑点百分比 (实际-请求)/请求"""
        if self.requested_price > 0 and self.actual_price > 0:
            return round((self.actual_price - self.requested_price) / self.requested_price * 100, 3)
        return 0.0

    @property
    def is_success(self) -> bool:
        return self.status in ("filled", "partial", "submitted")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["slippage_pct"] = self.slippage_pct
        d["is_success"] = self.is_success
        return d


@dataclass
class Position:
    """持仓信息"""
    code: str = ""
    name: str = ""
    shares: int = 0
    sellable_shares: int = 0        # T+1可卖数量
    cost_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AccountBalance:
    """账户资金信息"""
    total_asset: float = 0.0        # 总资产
    cash: float = 0.0               # 可用现金
    market_value: float = 0.0       # 持仓市值
    frozen: float = 0.0             # 冻结资金

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Order:
    """委托单"""
    order_id: str = ""
    code: str = ""
    name: str = ""
    direction: str = ""             # "buy" / "sell"
    price: float = 0.0
    shares: int = 0
    status: str = ""                # "pending" / "filled" / "partial" / "cancelled"
    filled_shares: int = 0
    filled_price: float = 0.0
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionReport:
    """批量执行报告"""
    session_id: str = ""
    mode: str = ""                  # "confirm" / "auto" / "dry_run"
    orders: List[OrderResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def filled_count(self) -> int:
        return sum(1 for o in self.orders if o.status == "filled")

    @property
    def rejected_count(self) -> int:
        return sum(1 for o in self.orders if o.status == "rejected")

    @property
    def total_slippage_pct(self) -> float:
        filled = [o for o in self.orders if o.is_success and o.slippage_pct != 0]
        if not filled:
            return 0.0
        return round(sum(o.slippage_pct for o in filled) / len(filled), 3)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "total_orders": len(self.orders),
            "filled_count": self.filled_count,
            "rejected_count": self.rejected_count,
            "avg_slippage_pct": self.total_slippage_pct,
            "orders": [o.to_dict() for o in self.orders],
        }

    def format_summary(self) -> str:
        """格式化为中文摘要"""
        lines = [f"📦 执行报告 ({self.mode})"]
        lines.append(f"  总委托: {len(self.orders)}笔 | 成交: {self.filled_count} | 拒绝: {self.rejected_count}")
        if self.total_slippage_pct:
            lines.append(f"  平均滑点: {self.total_slippage_pct:+.3f}%")
        for o in self.orders:
            emoji = "✅" if o.is_success else "❌"
            dir_cn = "买" if o.direction == "buy" else "卖"
            lines.append(
                f"  {emoji} {dir_cn} {o.name}({o.code}) "
                f"{o.actual_shares or o.requested_shares}股 "
                f"@ {o.actual_price or o.requested_price:.2f}"
            )
            if o.message:
                lines.append(f"     {o.message}")
        return "\n".join(lines)
