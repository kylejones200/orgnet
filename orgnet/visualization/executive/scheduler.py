"""Scheduled report generation for executive dashboards.

Supports cron-like scheduling of automated reports (daily, weekly, etc.)
using APScheduler when the optional dependency is installed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from orgnet.utils.logging import get_logger

if TYPE_CHECKING:
    from orgnet.visualization.executive.dashboard import ExecutiveDashboard
    from orgnet.visualization.executive.reporting import AutomatedReporter

logger = get_logger(__name__)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    BackgroundScheduler = None  # type: ignore
    CronTrigger = None  # type: ignore


class ScheduledReportRunner:
    """Run automated reports on a schedule.

    Requires: pip install agentv[scheduler]  (adds apscheduler)
    """

    def __init__(
        self,
        reporter: AutomatedReporter,
        output_dir: str = "reports",
        *,
        json_schedule: str | None = "0 9 * * 1-5",  # 9am weekdays
        summary_schedule: str | None = None,
    ):
        """Initialize scheduled report runner.

        Args:
            reporter: AutomatedReporter instance
            output_dir: Directory to write scheduled reports
            json_schedule: Cron expression for JSON reports (default: 9am weekdays).
                None to disable.
            summary_schedule: Cron expression for summary reports. None to disable.
        """
        if not HAS_APSCHEDULER:
            raise ImportError(
                "Scheduled reports require apscheduler. Install with: pip install agentv[scheduler]"
            )
        self.reporter = reporter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_schedule = json_schedule
        self.summary_schedule = summary_schedule
        self._scheduler = BackgroundScheduler()
        self._started = False

    def _run_json_report(self) -> None:
        """Generate and save JSON report."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            path = self.output_dir / f"executive_dashboard_{ts}.json"
            self.reporter.generate_json_report(str(path))
            logger.info(f"Scheduled JSON report saved to {path}")
        except Exception as e:
            logger.exception(f"Scheduled JSON report failed: {e}")

    def _run_summary_report(self) -> None:
        """Generate and optionally save summary report."""
        try:
            summary = self.reporter.generate_summary_report()
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            path = self.output_dir / f"executive_summary_{ts}.json"
            import json

            with open(path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Scheduled summary report saved to {path}")
        except Exception as e:
            logger.exception(f"Scheduled summary report failed: {e}")

    def start(self) -> None:
        """Start the scheduler."""
        if self._started:
            logger.warning("Scheduler already started")
            return
        if self.json_schedule:
            self._scheduler.add_job(
                self._run_json_report,
                CronTrigger.from_crontab(self.json_schedule),
                id="json_report",
            )
            logger.info(f"Scheduled JSON reports: {self.json_schedule}")
        if self.summary_schedule:
            self._scheduler.add_job(
                self._run_summary_report,
                CronTrigger.from_crontab(self.summary_schedule),
                id="summary_report",
            )
            logger.info(f"Scheduled summary reports: {self.summary_schedule}")
        self._scheduler.start()
        self._started = True
        logger.info("Scheduled report runner started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._started:
            return
        self._scheduler.shutdown(wait=False)
        self._started = False
        logger.info("Scheduled report runner stopped")

    def run_now(self, report_type: str = "json") -> str | dict:
        """Run a report immediately (synchronous).

        Args:
            report_type: "json" or "summary"

        Returns:
            Path to saved file (json) or summary dict (summary)
        """
        if report_type == "json":
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            path = self.output_dir / f"executive_dashboard_{ts}.json"
            self.reporter.generate_json_report(str(path))
            return str(path)
        elif report_type == "summary":
            return self.reporter.generate_summary_report()
        else:
            raise ValueError(f"Unknown report_type: {report_type}")


def create_scheduled_runner(
    dashboard: ExecutiveDashboard,
    output_dir: str = "reports",
    **kwargs,
) -> ScheduledReportRunner | None:
    """Create a ScheduledReportRunner if apscheduler is available.

    Args:
        dashboard: ExecutiveDashboard instance
        output_dir: Output directory for reports
        **kwargs: Passed to ScheduledReportRunner

    Returns:
        ScheduledReportRunner or None if apscheduler not installed
    """
    if not HAS_APSCHEDULER:
        logger.warning("apscheduler not installed; scheduled reports unavailable")
        return None
    from orgnet.visualization.executive.reporting import AutomatedReporter

    reporter = AutomatedReporter(dashboard)
    return ScheduledReportRunner(reporter, output_dir, **kwargs)
