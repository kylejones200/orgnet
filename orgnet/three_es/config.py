"""
Configuration for the orgnet analytics library.
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class Config:
    """Shared settings for library and tooling (e.g. Alembic, local scripts)."""

    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///orgnet.db")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"


class AnomalyThresholds:
    """Thresholds for anomaly detection."""

    COMMUNICATION_DROP_THRESHOLD = 0.5  # 50% drop
    ISOLATION_THRESHOLD = 2  # connections
    METRIC_DECLINE_THRESHOLD = 10  # points
    CENTRALIZATION_THRESHOLD = 0.5
    AFTER_HOURS_RATIO_THRESHOLD = 0.3

    # anomsmith (time series on metric history)
    ANOMSMITH_SCORE_QUANTILE = 0.95
    MIN_METRIC_HISTORY_POINTS = 5
