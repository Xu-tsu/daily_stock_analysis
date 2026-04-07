"""
同花顺适配器 — 通过 easytrader 操作同花顺客户端（模拟盘/实盘）

前置条件:
  1. 安装 easytrader: pip install easytrader
  2. 同花顺客户端已打开并登录（模拟盘或实盘）
  3. 下单程序 xiadan.exe 路径正确
  4. 设置环境变量: THS_EXE_PATH=G:\\同花顺远航版\\transaction\\xiadan.exe
"""
import logging
import os
import time
from typing import List, Optional

from src.broker.base import BrokerAdapter
from src.broker.models import OrderResult, Position, AccountBalance, Order

logger = logging.getLogger(__name__)


def _retry(max_attempts: int = 3, delay: float = 1.0):
    """简单重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"[THS] {func.__name__} 第{attempt+1}次失败: {e}，"
                            f"{delay*(attempt+1):.0f}秒后重试..."
                        )
                        time.sleep(delay * (attempt + 1))
            raise last_err
        return wrapper
    return decorator


class THSBrokerAdapter(BrokerAdapter):
    """同花顺 easytrader 适配器"""

    def __init__(self):
        self._user = None
        self._connected = False
        self._exe_path = os.getenv("THS_EXE_PATH", r"C:\同花顺\xiadan.exe")

    def connect(self) -> bool:
        try:
            import easytrader
            self._user = easytrader.use("ths")
            self._user.connect(self._exe_path)
            # 启用键盘输入模式（解决64位Python控制32位THS的剪贴板兼容问题）
            self._user.enable_type_keys_for_editor()
            self._connected = True
            logger.info(f"[THS] 连接成功（键盘输入模式）: {self._exe_path}")
            return True
        except Exception as e:
            self._connected = False
            logger.error(f"[THS] 连接失败: {e}")
            return False

    def disconnect(self) -> None:
        self._user = None
        self._connected = False
        logger.info("[THS] 已断开")

    def is_connected(self) -> bool:
        if not self._connected or not self._user:
            return False
        try:
            self._user.balance
            return True
        except Exception:
            self._connected = False
            return False

    def _ensure_connected(self) -> bool:
        """确保已连接，未连接则尝试重连"""
        if self.is_connected():
            return True
        logger.warning("[THS] 连接丢失，尝试重连...")
        return self.connect()

    @_retry(max_attempts=3, delay=1.0)
    def buy(self, code: str, price: float, shares: int) -> OrderResult:
        price = round(price, 2)  # A股价格精度2位小数
        if not self._ensure_connected():
            return OrderResult(
                code=code, direction="buy", status="error",
                requested_price=price, requested_shares=shares,
                message="同花顺客户端未连接",
            )
        try:
            result = self._execute_with_captcha_check(
                self._user.buy, code, price, shares, "buy"
            )
            logger.info(f"[THS] 买入委托: {code} {shares}股 @ {price} → {result}")
            order = self._parse_order_result(result, code, "buy", price, shares)
            if order.is_success:
                self._refresh_after_trade("买入", code)
            return order
        except Exception as e:
            logger.error(f"[THS] 买入失败: {code} {shares}股 @ {price}: {e}")
            return OrderResult(
                code=code, direction="buy", status="error",
                requested_price=price, requested_shares=shares,
                message=str(e)[:200],
            )

    @_retry(max_attempts=3, delay=1.0)
    def sell(self, code: str, price: float, shares: int) -> OrderResult:
        price = round(price, 2)  # A股价格精度2位小数
        if not self._ensure_connected():
            return OrderResult(
                code=code, direction="sell", status="error",
                requested_price=price, requested_shares=shares,
                message="同花顺客户端未连接",
            )
        try:
            result = self._execute_with_captcha_check(
                self._user.sell, code, price, shares, "sell"
            )
            logger.info(f"[THS] 卖出委托: {code} {shares}股 @ {price} → {result}")
            order = self._parse_order_result(result, code, "sell", price, shares)
            if order.is_success:
                self._refresh_after_trade("卖出", code)
            return order
        except Exception as e:
            logger.error(f"[THS] 卖出失败: {code} {shares}股 @ {price}: {e}")
            return OrderResult(
                code=code, direction="sell", status="error",
                requested_price=price, requested_shares=shares,
                message=str(e)[:200],
            )

    def cancel_order(self, order_id: str) -> bool:
        if not self._ensure_connected():
            return False
        try:
            self._user.cancel_entrust(order_id)
            logger.info(f"[THS] 撤单成功: {order_id}")
            return True
        except Exception as e:
            logger.error(f"[THS] 撤单失败: {order_id}: {e}")
            return False

    @_retry(max_attempts=2, delay=0.5)
    def get_positions(self) -> List[Position]:
        if not self._ensure_connected():
            return []
        try:
            raw_positions = self._user.position
            positions = []
            for p in raw_positions:
                pos = Position(
                    code=str(p.get("证券代码", p.get("stock_code", ""))).zfill(6),
                    name=p.get("证券名称", p.get("stock_name", "")),
                    shares=int(p.get("股票余额", p.get("current_amount", 0))),
                    sellable_shares=int(p.get("可用余额", p.get("enable_amount", 0))),
                    cost_price=float(p.get("成本价", p.get("cost_price", 0))),
                    current_price=float(p.get("市价", p.get("current_price", p.get("last_price", 0)))),
                    market_value=float(p.get("市值", p.get("market_value", 0))),
                    pnl=float(p.get("盈亏", p.get("income_balance", 0))),
                    pnl_pct=float(p.get("盈亏比例(%)", p.get("income_balance_ratio", 0))),
                )
                if pos.code and pos.shares > 0:
                    positions.append(pos)
            logger.info(f"[THS] 获取持仓: {len(positions)}只")
            return positions
        except Exception as e:
            logger.error(f"[THS] 获取持仓失败: {e}")
            return []

    @_retry(max_attempts=2, delay=0.5)
    def get_balance(self) -> AccountBalance:
        if not self._ensure_connected():
            return AccountBalance()
        try:
            raw = self._user.balance
            # easytrader 可能返回 dict 或 list[dict]
            b = raw[0] if isinstance(raw, list) and raw else raw
            balance = AccountBalance(
                total_asset=float(b.get("总资产", b.get("asset_balance", 0))),
                cash=float(b.get("可用金额", b.get("可用余额", b.get("enable_balance", 0)))),
                market_value=float(b.get("股票市值", b.get("market_value", 0))),
                frozen=float(b.get("冻结金额", b.get("冻结资金", b.get("frozen_balance", 0)))),
            )
            logger.info(
                f"[THS] 账户: 总资产={balance.total_asset:,.0f} "
                f"现金={balance.cash:,.0f} 市值={balance.market_value:,.0f}"
            )
            return balance
        except Exception as e:
            logger.error(f"[THS] 获取账户失败: {e}")
            return AccountBalance()

    @_retry(max_attempts=2, delay=0.5)
    def get_today_orders(self) -> List[Order]:
        if not self._ensure_connected():
            return []
        try:
            raw_orders = self._user.today_entrusts
            orders = []
            for o in raw_orders:
                orders.append(Order(
                    order_id=str(o.get("合同编号", o.get("entrust_no", ""))),
                    code=str(o.get("证券代码", o.get("stock_code", ""))).zfill(6),
                    name=o.get("证券名称", o.get("stock_name", "")),
                    direction="buy" if "买" in str(o.get("操作", o.get("entrust_bs", ""))) else "sell",
                    price=float(o.get("委托价格", o.get("entrust_price", 0))),
                    shares=int(o.get("委托数量", o.get("entrust_amount", 0))),
                    status=self._map_order_status(o.get("备注", o.get("entrust_status", ""))),
                    filled_shares=int(o.get("成交数量", o.get("business_amount", 0))),
                    filled_price=float(o.get("成交价格", o.get("business_price", 0))),
                    created_at=str(o.get("委托时间", o.get("entrust_time", ""))),
                ))
            return orders
        except Exception as e:
            logger.error(f"[THS] 获取今日委托失败: {e}")
            return []

    # ── 验证码检测 ──

    def _execute_with_captcha_check(self, trade_fn, code, price, shares, direction):
        """包装buy/sell，检测验证码拦截并轮询重试。

        同花顺验证码弹窗会被 easytrader 的 TradePopDialogHandler 自动关闭
        （未知弹窗 → _close()），导致订单未提交但返回 {"message":"success"}。
        检测策略：对比下单前后 today_entrusts 数量。
        恢复策略：发飞书告警 → 轮询重试（用户在THS手动完成验证后，重试即成功）。
        """
        # 1. 快照当前委托数
        try:
            before_count = len(self._user.today_entrusts)
        except Exception:
            before_count = -1  # 无法验证，跳过检测

        # 2. 执行下单
        result = trade_fn(code, price=price, amount=shares)

        # 3. 有委托号 → 确认成功
        if isinstance(result, dict) and result.get("entrust_no"):
            return result

        # 4. 无法验证 → 直接返回
        if before_count < 0:
            return result

        # 5. 对比委托数：增加了说明订单提交成功
        time.sleep(0.5)
        try:
            after_count = len(self._user.today_entrusts)
        except Exception:
            return result

        if after_count > before_count:
            return result  # 委托已提交

        # 6. 委托未提交 → 疑似验证码拦截
        dir_cn = "买入" if direction == "buy" else "卖出"
        logger.warning(
            f"[THS] 疑似验证码拦截: {dir_cn} {code} {shares}股 @ {price} "
            f"(委托数: {before_count} → {after_count})"
        )

        # 7. 截图+飞书告警
        screenshot = self._capture_screen()
        self._send_captcha_alert(screenshot, code, price, shares, direction)

        # 8. 轮询重试：等待用户在THS客户端手动完成验证码，然后重试下单
        #    easytrader会关闭验证码弹窗，所以不能等弹窗消失，
        #    而是周期性重试下单并检查委托数是否增加
        timeout = int(os.getenv("CAPTCHA_WAIT_TIMEOUT", "180"))
        poll_interval = 15  # 每15秒重试一次
        start = time.time()

        while time.time() - start < timeout:
            logger.info(
                f"[THS] 等待验证码处理... "
                f"({int(time.time() - start)}/{timeout}s)"
            )
            time.sleep(poll_interval)

            try:
                # 快照委托数
                retry_before = len(self._user.today_entrusts)
                # 重试下单
                retry_result = trade_fn(code, price=price, amount=shares)
                time.sleep(0.5)
                retry_after = len(self._user.today_entrusts)

                # 有委托号 → 成功
                if isinstance(retry_result, dict) and retry_result.get("entrust_no"):
                    logger.info(f"[THS] 验证码已通过，{dir_cn}成功 (entrust_no)")
                    return retry_result

                # 委托数增加 → 成功
                if retry_after > retry_before:
                    logger.info(f"[THS] 验证码已通过，{dir_cn}成功 (委托数+1)")
                    return retry_result

                logger.debug(f"[THS] 重试未成功，继续等待... (委托数: {retry_before}→{retry_after})")

            except Exception as e:
                logger.debug(f"[THS] 重试异常: {e}，继续等待...")

        # 超时
        raise Exception(
            f"验证码超时未处理({timeout}秒)，{dir_cn} {code} {shares}股 @ {price} 交易取消"
        )

    def _capture_screen(self) -> Optional[bytes]:
        """截取当前屏幕（包含验证码弹窗）"""
        import io
        try:
            img = self._user.app.top_window().capture_as_image()
        except Exception:
            try:
                from PIL import ImageGrab
                img = ImageGrab.grab()
            except Exception as e:
                logger.error(f"[THS] 截图失败: {e}")
                return None
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _send_captcha_alert(self, image_bytes, code, price, shares, direction):
        """通过回调发送验证码警报到飞书"""
        try:
            from src.broker.captcha_alert import send_captcha_alert
            dir_cn = "买入" if direction == "buy" else "卖出"
            msg = (
                f"⚠️ 验证码拦截\n"
                f"{dir_cn} {code} {shares}股 @ {price}\n"
                f"请在同花顺客户端完成验证码，系统将自动重试（3分钟超时）"
            )
            send_captcha_alert(image_bytes or b"", msg)
        except Exception as e:
            logger.error(f"[THS] 发送验证码警报失败: {e}")

    def _refresh_after_trade(self, direction_cn: str, code: str):
        """交易成功后自动刷新同花顺持仓显示"""
        try:
            time.sleep(0.3)
            self._user.refresh()
            logger.info(f"[THS] {direction_cn} {code} 后已自动刷新持仓")
        except Exception as e:
            logger.debug(f"[THS] 自动刷新失败（不影响交易）: {e}")

    # ── 内部辅助方法 ──

    def _parse_order_result(
        self, raw, code: str, direction: str,
        req_price: float, req_shares: int,
    ) -> OrderResult:
        """解析 easytrader 返回的委托结果

        easytrader 返回格式不统一：
          - {"entrust_no": "xxx"}          — 有委托号
          - {"message": "success"}         — 成功但无委托号（模拟盘常见）
          - {"error": "xxx"}               — 失败
          - 空 dict / None                 — 异常
        """
        if not raw:
            return OrderResult(
                code=code, direction=direction, status="error",
                requested_price=req_price, requested_shares=req_shares,
                message="空返回",
            )
        if isinstance(raw, dict):
            # 有委托号 → 提交成功
            if raw.get("entrust_no"):
                return OrderResult(
                    order_id=str(raw["entrust_no"]),
                    code=code, direction=direction, status="submitted",
                    requested_price=req_price, requested_shares=req_shares,
                    actual_price=req_price, actual_shares=req_shares,
                    message="委托已提交",
                )
            # message=success → 成功（模拟盘不返回委托号）
            msg = str(raw.get("message", "")).lower()
            if "success" in msg or "成功" in msg:
                return OrderResult(
                    order_id=str(raw.get("entrust_no", "")),
                    code=code, direction=direction, status="submitted",
                    requested_price=req_price, requested_shares=req_shares,
                    actual_price=req_price, actual_shares=req_shares,
                    message="委托已提交（模拟盘）",
                )
            # 明确的错误
            if raw.get("error"):
                return OrderResult(
                    code=code, direction=direction, status="rejected",
                    requested_price=req_price, requested_shares=req_shares,
                    message=str(raw["error"])[:200],
                )
            # 其他 dict → 尝试当成功处理（easytrader 未报错即成功）
            return OrderResult(
                code=code, direction=direction, status="submitted",
                requested_price=req_price, requested_shares=req_shares,
                actual_price=req_price, actual_shares=req_shares,
                message=str(raw)[:200],
            )
        # 非 dict（字符串等）→ 兜底
        return OrderResult(
            code=code, direction=direction, status="submitted",
            requested_price=req_price, requested_shares=req_shares,
            actual_price=req_price, actual_shares=req_shares,
            message=str(raw)[:200],
        )

    @staticmethod
    def _map_order_status(raw_status: str) -> str:
        """映射同花顺委托状态到标准状态"""
        s = str(raw_status)
        if "已成" in s:
            return "filled"
        if "部分" in s or "部成" in s:
            return "partial"
        if "已撤" in s or "撤单" in s:
            return "cancelled"
        if "已报" in s or "待报" in s:
            return "pending"
        return "pending"
