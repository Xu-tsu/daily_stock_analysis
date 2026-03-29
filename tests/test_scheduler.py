# -*- coding: utf-8 -*-
"""Unit tests for multi-checkpoint scheduler wiring."""

import sys
import unittest
from datetime import datetime, timedelta


class _FakeJobBuilder:
    def __init__(self, module):
        self._module = module
        self._time_str = None

    @property
    def day(self):
        return self

    def at(self, time_str):
        self._time_str = time_str
        return self

    def do(self, callback, *args, **kwargs):
        self._module.jobs.append(
            {
                "time": self._time_str,
                "callback": callback,
                "args": args,
                "kwargs": kwargs,
                "next_run": datetime.now() + timedelta(minutes=len(self._module.jobs) + 1),
            }
        )
        return self


class _FakeScheduleModule:
    def __init__(self):
        self.jobs = []

    def every(self):
        return _FakeJobBuilder(self)

    def get_jobs(self):
        return [type("FakeJob", (), {"next_run": item["next_run"]})() for item in self.jobs]

    def run_pending(self):
        return None


class SchedulerMultiTaskTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._original_schedule = sys.modules.get("schedule")
        sys.modules["schedule"] = _FakeScheduleModule()

    def tearDown(self) -> None:
        if self._original_schedule is None:
            sys.modules.pop("schedule", None)
        else:
            sys.modules["schedule"] = self._original_schedule

    def test_set_daily_tasks_registers_multiple_jobs_and_runs_only_named_immediate_task(self) -> None:
        from src.scheduler import Scheduler

        executed = []
        scheduler = Scheduler(schedule_time="18:00")
        scheduler.set_daily_tasks(
            tasks=[
                ("18:00", lambda: executed.append("daily"), "daily_analysis"),
                ("10:15", lambda: executed.append("morning"), "morning_review"),
                ("12:30", lambda: executed.append("afternoon"), "afternoon_review"),
            ],
            run_immediately=True,
            immediate_task_names=["daily_analysis"],
        )

        self.assertEqual(executed, ["daily"])
        self.assertEqual([job["time"] for job in scheduler.schedule.jobs], ["18:00", "10:15", "12:30"])

    def test_set_daily_task_keeps_backward_compatible_single_job_behavior(self) -> None:
        from src.scheduler import Scheduler

        executed = []
        scheduler = Scheduler(schedule_time="18:00")
        scheduler.set_daily_task(lambda: executed.append("single"), run_immediately=True)

        self.assertEqual(executed, ["single"])
        self.assertEqual(len(scheduler.schedule.jobs), 1)
        self.assertEqual(scheduler.schedule.jobs[0]["time"], "18:00")


if __name__ == "__main__":
    unittest.main()
