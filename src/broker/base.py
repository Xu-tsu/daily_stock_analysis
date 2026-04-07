"""
券商适配器抽象基类 — 定义统一接口，支持多券商实现
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from src.broker.models import OrderResult, Position, AccountBalance, Order


class BrokerAdapter(ABC):
    """券商适配器接口

    所有券商（同花顺、东方财富、中信等）都实现此接口。
    调用方只依赖此抽象类，不依赖具体实现。
    """

    @abstractmethod
    def connect(self) -> bool:
        """连接券商客户端，返回是否成功"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """检查是否已连接"""
        ...

    @abstractmethod
    def buy(self, code: str, price: float, shares: int) -> OrderResult:
        """买入委托"""
        ...

    @abstractmethod
    def sell(self, code: str, price: float, shares: int) -> OrderResult:
        """卖出委托"""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤销委托"""
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """获取当前持仓"""
        ...

    @abstractmethod
    def get_balance(self) -> AccountBalance:
        """获取账户资金"""
        ...

    @abstractmethod
    def get_today_orders(self) -> List[Order]:
        """获取当日委托"""
        ...

    def get_position_by_code(self, code: str) -> Optional[Position]:
        """根据代码获取单只持仓"""
        for pos in self.get_positions():
            if pos.code == code:
                return pos
        return None
