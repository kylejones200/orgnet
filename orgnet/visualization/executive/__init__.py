"""Executive dashboard and automated reporting."""

from orgnet.visualization.executive.dashboard import ExecutiveDashboard
from orgnet.visualization.executive.reporting import AutomatedReporter

__all__ = ["ExecutiveDashboard", "AutomatedReporter"]

try:
    from orgnet.visualization.executive.scheduler import (
        ScheduledReportRunner,
        create_scheduled_runner,
    )

    __all__ += ["ScheduledReportRunner", "create_scheduled_runner"]
except ImportError:
    pass
