# -*- coding: utf-8 -*-
"""
Daily scheduler helpers.

This project's fixed daily checkpoints are defined in A-share market time
(Asia/Shanghai), while the `schedule` library runs against the host's local
wall-clock time. This module converts market times to local times at
registration so overseas hosts still trigger `10:15` / `12:30` checkpoints at
the intended A-share moments.
"""

import logging
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)
CN_MARKET_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")


def _get_local_timezone():
    """Return the host timezone used by the local schedule loop."""
    return datetime.now().astimezone().tzinfo or CN_MARKET_TIMEZONE


def _normalize_hhmm(time_str: str) -> str:
    """Validate and normalize a HH:MM string."""
    parsed = datetime.strptime(time_str, "%H:%M")
    return parsed.strftime("%H:%M")


def _format_timezone_name(tz) -> str:
    if tz is None:
        return "local"
    return tz.tzname(None) or str(tz)


def convert_market_time_to_local_time(
    time_str: str,
    market_timezone=CN_MARKET_TIMEZONE,
    local_timezone=None,
    reference: Optional[datetime] = None,
) -> str:
    """
    Convert an A-share market clock time to the host local clock time.

    `schedule` only understands local wall-clock time, so `10:15` and `12:30`
    need to be interpreted as A-share time first and then converted.
    """
    normalized = _normalize_hhmm(time_str)
    local_tz = local_timezone or _get_local_timezone()
    current_local = reference.astimezone(local_tz) if reference else datetime.now(local_tz)
    current_market = current_local.astimezone(market_timezone)
    hour, minute = [int(part) for part in normalized.split(":")]
    target_market = current_market.replace(hour=hour, minute=minute, second=0, microsecond=0)
    target_local = target_market.astimezone(local_tz)
    return target_local.strftime("%H:%M")


class GracefulShutdown:
    """Handle SIGTERM / SIGINT and stop after the current job finishes."""

    def __init__(self):
        self.shutdown_requested = False
        self._lock = threading.Lock()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        with self._lock:
            if not self.shutdown_requested:
                logger.info("收到退出信号 (%s)，等待当前任务完成后再退出...", signum)
                self.shutdown_requested = True

    @property
    def should_shutdown(self) -> bool:
        with self._lock:
            return self.shutdown_requested


class Scheduler:
    """
    Lightweight daily task scheduler.

    By default, scheduled times are interpreted as A-share market time.
    """

    def __init__(
        self,
        schedule_time: str = "18:00",
        market_timezone=CN_MARKET_TIMEZONE,
        local_timezone=None,
    ):
        try:
            import schedule

            self.schedule = schedule
        except ImportError:
            logger.error("schedule 库未安装，请执行: pip install schedule")
            raise ImportError("请安装 schedule 库: pip install schedule")

        self.schedule_time = _normalize_hhmm(schedule_time)
        self.market_timezone = market_timezone
        self.local_timezone = local_timezone or _get_local_timezone()
        self.shutdown_handler = GracefulShutdown()
        self._task_callback: Optional[Callable] = None
        self._running = False

    def set_daily_tasks(
        self,
        tasks: Sequence[Tuple[str, Callable, str]],
        run_immediately: bool = False,
        immediate_task_names: Optional[Sequence[str]] = None,
    ):
        """Register multiple daily jobs."""
        if not tasks:
            return

        self._task_callback = tasks[0][1]
        immediate_names = list(immediate_task_names or [])
        if run_immediately and not immediate_names:
            immediate_names = [tasks[0][2]]

        for time_str, task, task_name in tasks:
            market_time = _normalize_hhmm(time_str)
            local_time = convert_market_time_to_local_time(
                market_time,
                market_timezone=self.market_timezone,
                local_timezone=self.local_timezone,
            )
            self.schedule.every().day.at(local_time).do(
                self._safe_run_task,
                task_name=task_name,
                task_callback=task,
            )
            logger.info(
                "已设置每日定时任务 [%s]，A股时间: %s (%s) -> 本机时间: %s (%s)",
                task_name,
                market_time,
                _format_timezone_name(self.market_timezone),
                local_time,
                _format_timezone_name(self.local_timezone),
            )

        if run_immediately:
            for _, task, task_name in tasks:
                if task_name in immediate_names:
                    logger.info("立即执行任务 [%s]...", task_name)
                    self._safe_run_task(task_name=task_name, task_callback=task)

    def set_daily_task(self, task: Callable, run_immediately: bool = True):
        """Backward-compatible single daily task helper."""
        self.set_daily_tasks(
            tasks=[(self.schedule_time, task, "daily_task")],
            run_immediately=run_immediately,
            immediate_task_names=["daily_task"],
        )

    def _safe_run_task(
        self,
        task_name: str = "daily_task",
        task_callback: Optional[Callable] = None,
    ):
        """Run a scheduled task with error handling."""
        callback = task_callback or self._task_callback
        if callback is None:
            return

        try:
            logger.info("=" * 50)
            logger.info(
                "定时任务开始执行 [%s] - %s",
                task_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            logger.info("=" * 50)

            callback()

            logger.info(
                "定时任务执行完成 [%s] - %s",
                task_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

        except Exception as exc:
            logger.exception("定时任务执行失败 [%s]: %s", task_name, exc)

    def run(self):
        """Run the scheduler loop until a shutdown signal is received."""
        self._running = True
        logger.info("调度器开始运行...")
        logger.info("下次执行时间: %s", self._get_next_run_time())

        while self._running and not self.shutdown_handler.should_shutdown:
            self.schedule.run_pending()
            time.sleep(30)

            now = datetime.now()
            if now.minute == 0 and now.second < 30:
                logger.info("调度器运行中... 下次执行: %s", self._get_next_run_time())

        logger.info("调度器已停止")

    def _get_next_run_time(self) -> str:
        """Return the next scheduled run in both local and A-share time."""
        jobs = self.schedule.get_jobs()
        if not jobs:
            return "未设置"

        next_run = min(job.next_run for job in jobs)
        local_dt = next_run.replace(tzinfo=self.local_timezone)
        market_dt = local_dt.astimezone(self.market_timezone)
        return (
            f"{local_dt.strftime('%Y-%m-%d %H:%M:%S')} ({_format_timezone_name(self.local_timezone)}) / "
            f"{market_dt.strftime('%Y-%m-%d %H:%M:%S')} ({_format_timezone_name(self.market_timezone)})"
        )

    def stop(self):
        self._running = False


def run_with_schedule(
    task: Callable,
    schedule_time: str = "18:00",
    run_immediately: bool = True,
    extra_daily_tasks: Optional[Sequence[Tuple[str, Callable, str]]] = None,
):
    """Convenience helper for daily schedule mode."""
    scheduler = Scheduler(schedule_time=schedule_time)
    if extra_daily_tasks:
        scheduler.set_daily_tasks(
            tasks=[(schedule_time, task, "daily_analysis"), *list(extra_daily_tasks)],
            run_immediately=run_immediately,
            immediate_task_names=["daily_analysis"],
        )
    else:
        scheduler.set_daily_task(task, run_immediately=run_immediately)
    scheduler.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    )

    def test_task():
        print(f"任务执行中... {datetime.now()}")
        time.sleep(2)
        print("任务完成!")

    print("启动测试调度器（按 Ctrl+C 退出）")
    run_with_schedule(test_task, schedule_time="23:59", run_immediately=True)
