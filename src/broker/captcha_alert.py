"""
验证码警报桥接 — broker模块通过回调发送通知，不直接依赖飞书
"""
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# callback(image_bytes: bytes, message: str) -> bool
CaptchaAlertCallback = Callable[[bytes, str], bool]
_alert_callback: Optional[CaptchaAlertCallback] = None


def set_captcha_alert_callback(callback: CaptchaAlertCallback) -> None:
    """注册验证码警报回调（由飞书启动时调用）"""
    global _alert_callback
    _alert_callback = callback
    logger.info("[CaptchaAlert] 回调已注册")


def send_captcha_alert(image_bytes: bytes, message: str) -> bool:
    """发送验证码警报（截图+文字）"""
    if _alert_callback is None:
        logger.warning("[CaptchaAlert] 未注册回调，无法发送验证码警报")
        return False
    try:
        return _alert_callback(image_bytes, message)
    except Exception as e:
        logger.error(f"[CaptchaAlert] 发送失败: {e}")
        return False
