"""
同花顺适配器 — 通过 easytrader 操作同花顺客户端（模拟盘/实盘）

前置条件:
  1. 安装 easytrader: pip install easytrader
  2. 同花顺客户端已打开并登录（模拟盘或实盘）
  3. 下单程序 xiadan.exe 路径正确
  4. 设置环境变量: THS_EXE_PATH=G:\\同花顺远航版\\transaction\\xiadan.exe
"""
import ctypes
import ctypes.wintypes
import logging
import os
import re
import time
from typing import List, Optional

import win32con
import win32gui

from src.broker.base import BrokerAdapter
from src.broker.models import OrderResult, Position, AccountBalance, Order

logger = logging.getLogger(__name__)

# ── 价格小数位工具 ──
_CB_PREFIXES = ("110", "113", "123", "127", "128", "118")
_THS_SECURITY_CONTROL_ID = 1032
_THS_PRICE_CONTROL_ID = 1033
_THS_AMOUNT_CONTROL_ID = 1034


def _price_decimals(code: str) -> int:
    """转债3位小数，股票2位小数。"""
    return 3 if str(code).strip().startswith(_CB_PREFIXES) else 2


def _normalize_trade_input_text(control_id: int, text, code: str = "") -> str:
    """规范 THS 交易输入框文本，确保最终写入控件的是合法字符串。"""
    raw = "" if text is None else str(text).strip()

    if control_id == _THS_SECURITY_CONTROL_ID:
        digits = re.sub(r"\D", "", raw)
        return (digits or raw)[-6:]

    if control_id == _THS_AMOUNT_CONTROL_ID:
        try:
            return str(int(float(raw)))
        except (TypeError, ValueError):
            return raw

    if control_id == _THS_PRICE_CONTROL_ID and raw:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return raw
        return f"{value:.{_price_decimals(code or '')}f}"

    return raw


def _read_edit_text(hwnd: int, buf_size: int = 256) -> str:
    """读取 Edit 控件文本，用于写入后的回读校验。"""
    buf = ctypes.create_unicode_buffer(buf_size)
    ctypes.windll.user32.SendMessageW(hwnd, win32con.WM_GETTEXT, buf_size, buf)
    return buf.value.strip()


def _write_edit_text(hwnd: int, text: str) -> None:
    """使用 WM_SETTEXT 精确写入 Edit 控件。"""
    ctypes.windll.user32.SendMessageW(hwnd, win32con.WM_SETTEXT, 0, text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全局验证码处理（纯 win32 API + OCR）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 懒加载 OCR 实例（ddddocr 初始化较慢，只做一次）
_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        try:
            import ddddocr
            _ocr_instance = ddddocr.DdddOcr(show_ad=False)
            logger.info("[THS-CAPTCHA] ddddocr OCR 初始化成功")
        except Exception as e:
            logger.error(f"[THS-CAPTCHA] ddddocr 加载失败: {e}")
    return _ocr_instance


def dismiss_ths_captcha() -> bool:
    """查找并自动处理同花顺"拷贝数据验证码"弹窗。

    纯 win32gui + OCR 实现。

    弹窗真实结构（通过诊断工具确认）：
      父窗口: class='#32770', title=''（标题为空！）
      子控件:
        - Button  "确定"
        - Button  "取消"
        - Static  "检测到您正在拷贝数据..."
        - Static  ""         (72x32, 验证码图片，GetWindowText 返回空)
        - Static  "验证码错误!!"
        - Static  "提示"
        - Edit    ""         (输入框)
        - Static  "先输入验证码："
    """
    # 1. 枚举所有可见的 #32770 对话框（标题为空）
    candidate_hwnds = []

    def _enum_windows(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            try:
                cls = win32gui.GetClassName(hwnd)
                if cls == "#32770":
                    candidate_hwnds.append(hwnd)
            except Exception:
                pass
        return True

    try:
        win32gui.EnumWindows(_enum_windows, None)
    except Exception:
        return False

    for hwnd in candidate_hwnds:
        if _try_handle_captcha_dialog(hwnd):
            return True

    return False


def _try_handle_captcha_dialog(hwnd: int) -> bool:
    """尝试处理单个 #32770 对话框。"""
    children = []

    def _enum_children(child_hwnd, _):
        children.append(child_hwnd)
        return True

    try:
        win32gui.EnumChildWindows(hwnd, _enum_children, None)
    except Exception:
        return False

    edit_hwnd = None
    confirm_hwnd = None
    captcha_img_hwnd = None
    is_captcha_dialog = False

    for child in children:
        try:
            cls = win32gui.GetClassName(child)
            text = win32gui.GetWindowText(child)
        except Exception:
            continue

        # 识别是否为验证码弹窗
        if cls == "Static" and ("拷贝数据" in text or "验证码" in text):
            is_captcha_dialog = True

        # 找 Edit 输入框
        if cls == "Edit":
            edit_hwnd = child

        # 找确定按钮
        if cls == "Button" and text in ("确定", "确认"):
            confirm_hwnd = child

        # 找验证码图片控件：Static + 文本为空 + 尺寸合适(~72x32)
        if cls == "Static" and not text:
            try:
                rect = win32gui.GetWindowRect(child)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if 50 <= w <= 200 and 20 <= h <= 60:
                    captcha_img_hwnd = child
            except Exception:
                pass

    if not is_captcha_dialog:
        return False

    if not edit_hwnd:
        logger.warning("[THS-CAPTCHA] 弹窗已识别但找不到输入框")
        return False

    if not captcha_img_hwnd:
        logger.warning("[THS-CAPTCHA] 弹窗已识别但找不到验证码图片控件")
        return False

    # ── OCR 识别验证码 ──
    captcha_code = _ocr_captcha_control(captcha_img_hwnd)
    if not captcha_code:
        logger.warning("[THS-CAPTCHA] OCR 识别失败")
        return False

    logger.info(f"[THS-CAPTCHA] OCR 识别验证码: {captcha_code}")

    # ── 输入验证码 ──
    try:
        win32gui.SendMessage(edit_hwnd, win32con.WM_SETTEXT, 0, captcha_code)
        time.sleep(0.3)

        # 验证是否写入成功
        buf = ctypes.create_unicode_buffer(32)
        ctypes.windll.user32.SendMessageW(
            edit_hwnd, win32con.WM_GETTEXT, 32, buf
        )
        actual = buf.value
        if actual != captcha_code:
            logger.warning(f"[THS-CAPTCHA] WM_SETTEXT 写入不匹配: "
                           f"期望={captcha_code} 实际={actual}，用 WM_CHAR 重试")
            win32gui.SendMessage(edit_hwnd, win32con.WM_SETTEXT, 0, "")
            for ch in captcha_code:
                win32gui.PostMessage(edit_hwnd, win32con.WM_CHAR, ord(ch), 0)
            time.sleep(0.3)
    except Exception as e:
        logger.error(f"[THS-CAPTCHA] 输入验证码失败: {e}")
        return False

    # ── 点击确定 ──
    try:
        if confirm_hwnd:
            win32gui.SendMessage(confirm_hwnd, win32con.BM_CLICK, 0, 0)
        else:
            win32gui.PostMessage(edit_hwnd, win32con.WM_KEYDOWN,
                                win32con.VK_RETURN, 0)
            win32gui.PostMessage(edit_hwnd, win32con.WM_KEYUP,
                                win32con.VK_RETURN, 0)
    except Exception as e:
        logger.error(f"[THS-CAPTCHA] 点击确定失败: {e}")
        return False

    time.sleep(0.5)
    logger.info(f"[THS-CAPTCHA] 验证码 {captcha_code} 已自动输入并确定 ✓")
    return True


def _ocr_captcha_control(hwnd: int) -> str:
    """截取指定控件区域并 OCR 识别验证码数字。"""
    import io
    from PIL import ImageGrab

    try:
        rect = win32gui.GetWindowRect(hwnd)
        # 截取控件区域
        img = ImageGrab.grab(bbox=(rect[0], rect[1], rect[2], rect[3]))

        # 转 bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # OCR 识别
        ocr = _get_ocr()
        if ocr is None:
            return ""
        result = ocr.classification(img_bytes)
        # 只保留数字
        digits = re.sub(r"[^\d]", "", result)
        if 4 <= len(digits) <= 6:
            return digits

        logger.debug(f"[THS-CAPTCHA] OCR 原始结果: {result!r} → 数字: {digits!r}")
        return ""
    except Exception as e:
        logger.error(f"[THS-CAPTCHA] OCR 异常: {e}")
        return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 验证码守护线程（最可靠方案）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_captcha_watcher_running = False


def start_captcha_watcher(interval: float = 1.0):
    """启动后台守护线程，每隔 interval 秒检测并自动处理验证码弹窗。

    不依赖 easytrader 的任何回调/patch，独立运行。
    弹窗一出现就立即处理，不管是读数据还是下单触发的。
    """
    global _captcha_watcher_running
    if _captcha_watcher_running:
        return

    import threading

    def _watcher_loop():
        global _captcha_watcher_running
        logger.info(f"[THS-CAPTCHA] 守护线程已启动 (间隔 {interval}s)")
        while _captcha_watcher_running:
            try:
                if dismiss_ths_captcha():
                    logger.info("[THS-CAPTCHA] 守护线程自动处理了一个验证码")
            except Exception as e:
                logger.debug(f"[THS-CAPTCHA] 守护线程异常: {e}")
            time.sleep(interval)
        logger.info("[THS-CAPTCHA] 守护线程已停止")

    _captcha_watcher_running = True
    t = threading.Thread(target=_watcher_loop, daemon=True, name="CaptchaWatcher")
    t.start()


def stop_captcha_watcher():
    """停止守护线程"""
    global _captcha_watcher_running
    _captcha_watcher_running = False


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
        self._active_trade_code = ""
        self._last_trade_code = ""
        # 持仓缓存：避免每分钟多次调用THS触发验证码
        self._positions_cache = None       # List[Position]
        self._positions_cache_ts = 0.0     # 上次刷新时间戳
        self._positions_cache_ttl = 30.0   # 缓存有效期（秒）

    def connect(self) -> bool:
        try:
            import easytrader
            self._user = easytrader.use("ths")
            self._user.connect(self._exe_path)
            # 启用键盘输入模式（解决64位Python控制32位THS的剪贴板兼容问题）
            self._user.enable_type_keys_for_editor()
            # ── 修复：monkey-patch easytrader 的输入函数 ──
            # 原版 type_keys 模式下 editor.select() 只聚焦不选全，
            # THS 自动填入的旧价格没被清掉，新价格叠加上去导致超精度。
            # 修复：打字前先 Ctrl+A 全选 → Delete 清空 → 再打新内容。
            _original_type_edit = self._user._type_edit_control_keys.__func__

            def _fixed_type_edit_control_keys(trader_self, control_id, text):
                code_hint = self._active_trade_code or self._last_trade_code
                normalized_text = _normalize_trade_input_text(control_id, text, code_hint)
                if control_id == _THS_SECURITY_CONTROL_ID:
                    self._last_trade_code = normalized_text

                editor = trader_self._main.child_window(
                    control_id=control_id, class_name="Edit"
                )
                editor.set_focus()
                time.sleep(0.05)

                # 证券代码输入更依赖键盘事件触发行情联想，因此保留 type_keys。
                if control_id == _THS_SECURITY_CONTROL_ID:
                    editor.type_keys("^a{DELETE}", set_foreground=False)
                    time.sleep(0.05)
                    editor.type_keys(normalized_text, set_foreground=False)
                    return

                # ── 价格控件特殊处理：THS 输入证券代码后会异步回填"当前价"
                # （常带 3~5 位小数），如果我们在回填完成前写入，值会被它覆盖。
                # 策略：先等 THS 回填完毕，再写；写完后多次回读校验，被覆盖就重写。
                if control_id == _THS_PRICE_CONTROL_ID:
                    # 先等 THS 自己把行情价回填进去（最多 400ms）
                    time.sleep(0.35)

                    final_text = ""
                    for attempt in range(4):
                        try:
                            _write_edit_text(editor.handle, normalized_text)
                        except Exception as e:
                            logger.debug(f"[THS] 价格 WM_SETTEXT 失败 attempt={attempt}: {e}")

                        # 等 150ms，给 THS 一次"又回填"的机会
                        time.sleep(0.15)
                        try:
                            final_text = _read_edit_text(editor.handle)
                        except Exception:
                            final_text = ""

                        if final_text == normalized_text:
                            return  # 成功写入且未被覆盖

                        # 被覆盖或写入失败，稍等再试（递增退避）
                        time.sleep(0.1 * (attempt + 1))

                    # WM_SETTEXT 反复被覆盖 → 用 type_keys 兜底
                    try:
                        editor.type_keys("^a{DELETE}", set_foreground=False)
                        time.sleep(0.1)
                        editor.type_keys(normalized_text, set_foreground=False)
                        time.sleep(0.2)
                        final_text = _read_edit_text(editor.handle)
                    except Exception as e:
                        logger.debug(f"[THS] 价格 type_keys 兜底失败: {e}")
                        final_text = ""

                    if final_text != normalized_text:
                        logger.error(
                            f"[THS] 价格输入最终不一致 control_id={control_id}: "
                            f"expect={normalized_text!r} actual={final_text!r} "
                            f"(可能是 THS 自动回填 tick 精度价格覆盖了我们的写入)"
                        )
                    return

                # 数量 / 其他控件：短路径写入
                actual_text = ""
                try:
                    _write_edit_text(editor.handle, normalized_text)
                    time.sleep(0.05)
                    actual_text = _read_edit_text(editor.handle)
                except Exception as e:
                    logger.debug(f"[THS] WM_SETTEXT 写入失败 control_id={control_id}: {e}")

                if actual_text != normalized_text:
                    editor.type_keys("^a{DELETE}", set_foreground=False)
                    time.sleep(0.05)
                    editor.type_keys(normalized_text, set_foreground=False)
                    time.sleep(0.05)
                    try:
                        actual_text = _read_edit_text(editor.handle)
                    except Exception:
                        actual_text = ""

                if actual_text and actual_text != normalized_text:
                    logger.warning(
                        f"[THS] 输入框回读不一致 control_id={control_id}: "
                        f"expect={normalized_text!r} actual={actual_text!r}"
                    )

            import types
            self._user._type_edit_control_keys = types.MethodType(
                _fixed_type_edit_control_keys, self._user
            )
            self._connected = True
            logger.info(f"[THS] 连接成功（键盘输入模式+输入补丁）: {self._exe_path}")
            # 注入验证码拦截补丁
            self._patch_pop_dialog_handler()
            # 启动验证码守护线程：每秒检测弹窗，弹窗一出现立刻 OCR + 输入 + 确定
            start_captcha_watcher(interval=1.0)
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
        # ── 价格精度：本地保留 float（OrderResult 字段要 float），
        # 但传给 easytrader 的是字符串 —— easyutils.round_price_by_code 对 str
        # 直接原样返回，这样下游拿到的永远是干净的 "9.54"/"123.456"，
        # 不会出现浮点 repr 残余（如 8.070000001）。
        d = _price_decimals(code)
        price = float(f"{float(price):.{d}f}")  # 截断到 d 位小数的 float
        self._active_trade_code = str(code).strip()
        if not self._ensure_connected():
            self._active_trade_code = ""
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
        finally:
            self._active_trade_code = ""

    @_retry(max_attempts=3, delay=1.0)
    def sell(self, code: str, price: float, shares: int) -> OrderResult:
        # ── 价格精度：本地保留 float（OrderResult 字段要 float），
        # 字符串化延后到调用 easytrader 那一刻进行。
        d = _price_decimals(code)
        price = float(f"{float(price):.{d}f}")
        self._active_trade_code = str(code).strip()

        if not self._ensure_connected():
            self._active_trade_code = ""
            return OrderResult(
                code=code, direction="sell", status="error",
                requested_price=price, requested_shares=shares,
                message="同花顺客户端未连接",
            )

        # ── 可卖余额检查：已清仓则直接拒绝，防止重复卖出 ──
        try:
            positions = self.get_positions()
            pos = next((p for p in positions if p.code == code), None)
            if pos is None or pos.sellable_shares <= 0:
                msg = f"{code} 可卖余额为0，跳过卖出"
                logger.warning(f"[THS] {msg}")
                return OrderResult(
                    code=code, direction="sell", status="rejected",
                    requested_price=price, requested_shares=shares,
                    message=msg,
                )
            # 卖出数量不能超过可卖余额
            if shares > pos.sellable_shares:
                logger.warning(f"[THS] {code} 请求卖{shares}股 > 可卖{pos.sellable_shares}股，自动修正")
                shares = pos.sellable_shares
        except Exception as e:
            logger.debug(f"[THS] 可卖余额检查跳过: {e}")

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
        finally:
            self._active_trade_code = ""

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
    def get_positions(self, force_refresh: bool = False) -> List[Position]:
        # ── 缓存：30秒内不重复调THS，避免频繁触发验证码 ──
        now = time.time()
        if (
            not force_refresh
            and self._positions_cache is not None
            and (now - self._positions_cache_ts) < self._positions_cache_ttl
        ):
            logger.debug(f"[THS] 使用持仓缓存 ({now - self._positions_cache_ts:.0f}s前刷新)")
            return self._positions_cache

        if not self._ensure_connected():
            return self._positions_cache or []
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
            # 更新缓存
            self._positions_cache = positions
            self._positions_cache_ts = now
            logger.info(f"[THS] 获取持仓: {len(positions)}只")
            return positions
        except Exception as e:
            logger.error(f"[THS] 获取持仓失败: {e}")
            return self._positions_cache or []

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

    # ── 验证码检测 + 自动处理 ──

    def _patch_pop_dialog_handler(self):
        """Monkeypatch easytrader 的三个验证码触发点。

        触发点1: Copy._get_clipboard_data() — 读持仓/余额/委托时
          easytrader 用 Ctrl+A Ctrl+C 复制表格 → THS 弹验证码
          easytrader 自带的 captcha_recognize() 函数未定义，必然失败
          → 替换为我们的 dismiss_ths_captcha() + ddddocr

        触发点2: TradePopDialogHandler.handle("提示") — 下单时
          easytrader 先点确定（没输入验证码）再抛 TradeError
          → 拦截，先调用 dismiss_ths_captcha()

        触发点3: PopDialogHandler.handle("提示") — 其他操作时
          同上
        """
        # ── Patch 1: Copy._get_clipboard_data（最关键） ──
        try:
            from easytrader.grid_strategies import Copy
            import pywinauto.clipboard

            _original_get_clipboard = Copy._get_clipboard_data

            def _patched_get_clipboard_data(self_copy):
                """替换 easytrader 坏掉的验证码流程，用 dismiss_ths_captcha()"""
                # 检查是否有验证码弹窗
                try:
                    top = self_copy._trader.app.top_window()
                    if top.window(class_name="Static", title_re="验证码").exists(timeout=1):
                        logger.info("[THS-Patch] 剪贴板读取触发验证码，调用自动处理...")
                        for attempt in range(3):
                            if dismiss_ths_captcha():
                                logger.info(f"[THS-Patch] 验证码自动处理成功 (第{attempt+1}次)")
                                time.sleep(0.5)
                                break
                            time.sleep(0.5)
                        else:
                            logger.warning("[THS-Patch] 3次自动处理均失败")
                    else:
                        Copy._need_captcha_reg = False
                except Exception as e:
                    logger.debug(f"[THS-Patch] 验证码检测异常: {e}")

                # 读剪贴板
                count = 5
                while count > 0:
                    try:
                        return pywinauto.clipboard.GetData()
                    except Exception as e:
                        count -= 1
                        logger.debug(f"[THS-Patch] 剪贴板读取重试: {e}")
                return ""

            Copy._get_clipboard_data = _patched_get_clipboard_data
            logger.info("[THS] 验证码补丁已注入 Copy._get_clipboard_data")

        except Exception as e:
            logger.warning(f"[THS] Copy 补丁注入失败: {e}")

        # ── Patch 2+3: PopDialogHandler + TradePopDialogHandler ──
        try:
            from easytrader.pop_dialog_handler import (
                TradePopDialogHandler, PopDialogHandler,
            )

            _original_trade_handle = TradePopDialogHandler.handle
            _original_pop_handle = PopDialogHandler.handle

            def _make_patched(original_fn):
                def _patched_handle(self_handler, title):
                    if title == "提示":
                        try:
                            content = self_handler._extract_content()
                        except Exception:
                            content = ""

                        if "拷贝数据" in content or "验证码" in content:
                            logger.info(f"[THS-Patch] 拦截验证码弹窗: {content[:60]}")
                            if dismiss_ths_captcha():
                                return None  # 已处理，让 easytrader 继续
                            logger.warning("[THS-Patch] 自动处理失败，走原逻辑")

                    return original_fn(self_handler, title)
                return _patched_handle

            TradePopDialogHandler.handle = _make_patched(_original_trade_handle)
            PopDialogHandler.handle = _make_patched(_original_pop_handle)
            logger.info("[THS] 验证码补丁已注入 (Trade + Pop)")

        except Exception as e:
            logger.warning(f"[THS] 弹窗补丁注入失败: {e}")

    def _dismiss_copy_captcha(self) -> bool:
        """自动处理验证码弹窗（委托给全局 win32 API 实现）"""
        return dismiss_ths_captcha()

    def _execute_with_captcha_check(self, trade_fn, code, price, shares, direction):
        """包装buy/sell，检测验证码拦截并自动处理+轮询重试。

        同花顺"拷贝数据验证码"会在剪贴板操作时弹出。
        处理策略：
        1. 先尝试自动识别并填写验证码
        2. 自动失败则截图告警+轮询等待用户手动处理
        """
        # 0. 下单前先检查是否有残留验证码弹窗
        self._dismiss_copy_captcha()

        # ── 关键修复：防止 float 精度泄漏到 THS 输入框 ──
        # easytrader type_keys 会把 float 转 str 打字，round(8.06808,2) 在
        # 二进制浮点下可能 repr 出多余小数位（如 8.07000000000001）。
        # 统一格式化为字符串：easyutils.round_price_by_code 对 str 入参直接返回，
        # 再配合 monkey-patch 的 _normalize_trade_input_text，最终写入的永远是精确值。
        d = _price_decimals(code)
        try:
            price_val = float(price) if not isinstance(price, (int, float)) else price
        except (TypeError, ValueError):
            price_val = price
        if isinstance(price_val, (int, float)):
            price = f"{float(price_val):.{d}f}"

        # 1. 快照当前委托数
        try:
            before_count = len(self._user.today_entrusts)
        except Exception as e:
            err_msg = str(e)
            if "拷贝数据" in err_msg or "验证码" in err_msg:
                # 读取委托列表时就触发了验证码
                if self._dismiss_copy_captcha():
                    try:
                        before_count = len(self._user.today_entrusts)
                    except Exception:
                        before_count = -1
                else:
                    before_count = -1
            else:
                before_count = -1

        # 2. 执行下单
        try:
            result = trade_fn(code, price=price, amount=shares)
        except Exception as e:
            err_msg = str(e)
            if "拷贝数据" in err_msg or "验证码" in err_msg:
                # 下单过程中触发了验证码，自动处理后重试
                logger.warning(f"[THS] 下单触发验证码拦截，尝试自动处理...")
                if self._dismiss_copy_captcha():
                    time.sleep(0.5)
                    result = trade_fn(code, price=price, amount=shares)
                else:
                    raise  # 自动处理失败，抛出原异常
            else:
                raise

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

        # 7. 先尝试自动处理验证码
        if self._dismiss_copy_captcha():
            logger.info("[THS] 验证码已自动处理，重试下单...")
            time.sleep(0.5)
            try:
                retry_result = trade_fn(code, price=price, amount=shares)
                if isinstance(retry_result, dict) and retry_result.get("entrust_no"):
                    logger.info(f"[THS] 自动验证码后{dir_cn}成功")
                    return retry_result
                time.sleep(0.5)
                retry_after = len(self._user.today_entrusts)
                if retry_after > before_count:
                    logger.info(f"[THS] 自动验证码后{dir_cn}成功 (委托数+1)")
                    return retry_result
            except Exception as e:
                logger.warning(f"[THS] 自动验证码后重试失败: {e}")

        # 8. 自动处理失败，截图+飞书告警+轮询等待人工
        screenshot = self._capture_screen()
        self._send_captcha_alert(screenshot, code, price, shares, direction)

        timeout = int(os.getenv("CAPTCHA_WAIT_TIMEOUT", "180"))
        poll_interval = 15
        start = time.time()

        while time.time() - start < timeout:
            logger.info(
                f"[THS] 等待验证码处理... "
                f"({int(time.time() - start)}/{timeout}s)"
            )
            time.sleep(poll_interval)

            # 每轮先尝试自动处理
            self._dismiss_copy_captcha()
            time.sleep(0.5)

            try:
                retry_before = len(self._user.today_entrusts)
                retry_result = trade_fn(code, price=price, amount=shares)
                time.sleep(0.5)
                retry_after = len(self._user.today_entrusts)

                if isinstance(retry_result, dict) and retry_result.get("entrust_no"):
                    logger.info(f"[THS] 验证码已通过，{dir_cn}成功 (entrust_no)")
                    return retry_result

                if retry_after > retry_before:
                    logger.info(f"[THS] 验证码已通过，{dir_cn}成功 (委托数+1)")
                    return retry_result

                logger.debug(f"[THS] 重试未成功，继续等待... (委托数: {retry_before}→{retry_after})")

            except Exception as e:
                err_msg = str(e)
                if "拷贝数据" in err_msg or "验证码" in err_msg:
                    self._dismiss_copy_captcha()
                logger.debug(f"[THS] 重试异常: {e}，继续等待...")

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

    # ── 智能追单：未成交自动撤单改价 ──

    def smart_buy(self, code: str, price: float, shares: int,
                  max_chase_pct: float = 0.5, timeout: int = 60) -> OrderResult:
        """智能买入：委托后监控成交，未成交则撤单按最新价重挂。

        参数由 AI execution_strategy 决定：
            max_chase_pct: AI 根据紧迫度决定最多追高多少（止损紧急=1-2%，正常=0.3-0.5%）
            timeout: AI 根据紧迫度决定追单超时（紧急=30-60s，正常=60-120s）
        """
        d = _price_decimals(code)
        ceiling = round(price * (1 + max_chase_pct / 100), d)
        return self._smart_execute("buy", code, price, shares, ceiling, timeout)

    def smart_sell(self, code: str, price: float, shares: int,
                   max_chase_pct: float = 0.5, timeout: int = 60) -> OrderResult:
        """智能卖出：委托后监控成交，未成交则撤单按最新价重挂。

        参数由 AI execution_strategy 决定：
            max_chase_pct: AI 根据紧迫度决定最多降价多少
            timeout: AI 根据紧迫度决定追单超时
        """
        d = _price_decimals(code)
        floor = round(price * (1 - max_chase_pct / 100), d)
        return self._smart_execute("sell", code, price, shares, floor, timeout)

    def _smart_execute(self, direction: str, code: str, price: float,
                       shares: int, limit_price: float, timeout: int) -> OrderResult:
        """核心追单循环：挂单 → 等待 → 未成交则撤单改价 → 重挂。"""
        poll_interval = int(os.getenv("CHASE_POLL_INTERVAL", "5"))
        dir_cn = "买入" if direction == "buy" else "卖出"
        trade_fn = self.buy if direction == "buy" else self.sell

        # 第一次下单
        d = _price_decimals(code)
        current_price = round(price, d)
        result = trade_fn(code, current_price, shares)
        if not result.is_success:
            return result

        order_id = result.order_id
        logger.info(
            f"[追单] {dir_cn} {code} {shares}股 @ {current_price} "
            f"(限价={limit_price}, 超时={timeout}s)"
        )

        start = time.time()
        while time.time() - start < timeout:
            time.sleep(poll_interval)

            # 刷新持仓/委托
            self._user.refresh()

            # 查找该委托状态
            filled, remaining = self._check_order_fill(code, direction, shares)

            if remaining <= 0:
                logger.info(f"[追单] {dir_cn} {code} 全部成交")
                result.status = "filled"
                result.actual_shares = shares
                result.actual_price = current_price
                result.message = "智能追单: 全部成交"
                return result

            if filled > 0:
                logger.info(f"[追单] {dir_cn} {code} 部分成交 {filled}/{shares}")

            # 获取实时价格
            new_price = self._get_realtime_price(code)
            if new_price <= 0:
                continue

            # 检查是否超出追价范围
            if direction == "buy" and new_price > limit_price:
                logger.warning(
                    f"[追单] {dir_cn} {code} 现价{new_price} > 追价上限{limit_price}，"
                    f"放弃追单"
                )
                break
            if direction == "sell" and new_price < limit_price:
                logger.warning(
                    f"[追单] {dir_cn} {code} 现价{new_price} < 追价下限{limit_price}，"
                    f"放弃追单"
                )
                break

            # 价格没变，继续等
            new_price = round(new_price, d)
            if new_price == current_price:
                continue

            # 撤单 → 改价重挂
            logger.info(
                f"[追单] {dir_cn} {code} 改价: {current_price} → {new_price} "
                f"(剩余{remaining}股)"
            )

            # 撤原单
            if order_id:
                self.cancel_order(order_id)
            else:
                # 没有 order_id（模拟盘），尝试撤该股票最新挂单
                self._cancel_latest_pending(code, direction)
            time.sleep(0.5)

            # 重新下单（剩余未成交部分）
            current_price = new_price
            retry_result = trade_fn(code, current_price, remaining)
            if retry_result.is_success:
                order_id = retry_result.order_id
                shares = remaining  # 后续只追踪剩余部分
            else:
                logger.warning(f"[追单] 改价重挂失败: {retry_result.message}")
                result.message = f"智能追单: 部分成交{filled}股，改价失败"
                result.actual_shares = filled
                return result

        # 超时或放弃：统计最终成交
        final_filled, final_remaining = self._check_order_fill(code, direction, shares)
        total_filled = shares - final_remaining if final_remaining >= 0 else 0

        if total_filled > 0:
            result.status = "partial" if final_remaining > 0 else "filled"
            result.actual_shares = total_filled
            result.message = f"智能追单: 成交{total_filled}股，未成交{final_remaining}股"
        else:
            result.message = f"智能追单: 超时{timeout}s未成交，保留原挂单"

        logger.info(f"[追单] {dir_cn} {code} 结束: {result.message}")
        return result

    def _check_order_fill(self, code: str, direction: str, total_shares: int):
        """检查某只股票某方向的成交情况。返回 (已成交, 未成交)。"""
        try:
            orders = self.get_today_orders()
            filled = 0
            pending = 0
            for o in orders:
                if o.code == code and o.direction == direction:
                    if o.status == "filled":
                        filled += o.filled_shares or o.shares
                    elif o.status == "partial":
                        filled += o.filled_shares
                        pending += o.shares - o.filled_shares
                    elif o.status == "pending":
                        pending += o.shares
            remaining = max(0, total_shares - filled)
            return filled, remaining
        except Exception as e:
            logger.debug(f"[追单] 查询成交失败: {e}")
            return 0, total_shares

    def _cancel_latest_pending(self, code: str, direction: str):
        """撤销该股票最新的挂单（模拟盘没有 order_id 时使用）。"""
        try:
            orders = self.get_today_orders()
            for o in reversed(orders):
                if o.code == code and o.direction == direction and o.status == "pending":
                    if o.order_id:
                        self.cancel_order(o.order_id)
                        logger.info(f"[追单] 撤销挂单: {o.order_id}")
                    return
        except Exception as e:
            logger.debug(f"[追单] 撤单失败: {e}")

    def _get_realtime_price(self, code: str) -> float:
        """获取股票实时价格（用于追单改价）。"""
        try:
            from macro_data_collector import _stock_code_to_tencent, _fetch_tencent_quote
            tc = _stock_code_to_tencent(code)
            quotes = _fetch_tencent_quote([tc], timeout=5)
            q = quotes.get(tc, {})
            return float(q.get("price", 0) or 0)
        except Exception as e:
            logger.debug(f"[追单] 获取实时价失败 {code}: {e}")
            return 0

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
