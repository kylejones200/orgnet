"""Audit logging for organizational network analysis.

This module provides audit logging functionality to track analysis runs,
data access, and configuration changes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class AuditLogger:
    """Logs audit events for orgnet analysis."""

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize audit logger.

        Args:
            log_file: Path to audit log file. If None, uses default location.
        """
        if log_file is None:
            log_file = ".audit/orgnet_audit.log"

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _compute_config_hash(self, config: Dict) -> str:
        """
        Compute hash of configuration for tracking.

        Args:
            config: Configuration dictionary

        Returns:
            Hash string
        """
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def log_analysis_run(
        self,
        user: Optional[str] = None,
        dataset: Optional[str] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        num_people: Optional[int] = None,
        num_interactions: Optional[int] = None,
        output_files: Optional[List[str]] = None,
        status: str = "completed",
        error: Optional[str] = None,
    ):
        """
        Log an analysis run.

        Args:
            user: User identifier (optional)
            dataset: Dataset identifier or path
            config: Configuration dictionary
            config_path: Path to configuration file
            num_people: Number of people analyzed
            num_interactions: Number of interactions analyzed
            output_files: List of output files generated
            status: Status of the run ('completed', 'failed', 'partial')
            error: Error message if status is 'failed'
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "analysis_run",
            "user": user or "unknown",
            "dataset": dataset,
            "config_path": config_path,
            "config_hash": self._compute_config_hash(config) if config else None,
            "num_people": num_people,
            "num_interactions": num_interactions,
            "output_files": output_files or [],
            "status": status,
            "error": error,
        }

        self._write_log(event)

    def log_data_access(
        self,
        user: Optional[str] = None,
        data_source: Optional[str] = None,
        data_path: Optional[str] = None,
        access_type: str = "read",
    ):
        """
        Log data access event.

        Args:
            user: User identifier
            data_source: Type of data source (email, slack, etc.)
            data_path: Path to data file
            access_type: Type of access ('read', 'write', 'delete')
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "data_access",
            "user": user or "unknown",
            "data_source": data_source,
            "data_path": data_path,
            "access_type": access_type,
        }

        self._write_log(event)

    def log_config_change(
        self,
        user: Optional[str] = None,
        config_path: Optional[str] = None,
        old_config_hash: Optional[str] = None,
        new_config_hash: Optional[str] = None,
    ):
        """
        Log configuration change.

        Args:
            user: User identifier
            config_path: Path to configuration file
            old_config_hash: Hash of old configuration
            new_config_hash: Hash of new configuration
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "config_change",
            "user": user or "unknown",
            "config_path": config_path,
            "old_config_hash": old_config_hash,
            "new_config_hash": new_config_hash,
        }

        self._write_log(event)

    def log_privacy_action(
        self,
        user: Optional[str] = None,
        action: str = "anonymize",
        num_records: Optional[int] = None,
        details: Optional[Dict] = None,
    ):
        """
        Log privacy-related action.

        Args:
            user: User identifier
            action: Type of privacy action ('anonymize', 'aggregate', 'retention_filter', etc.)
            num_records: Number of records affected
            details: Additional details about the action
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "privacy_action",
            "user": user or "unknown",
            "action": action,
            "num_records": num_records,
            "details": details or {},
        }

        self._write_log(event)

    def _write_log(self, event: Dict):
        """
        Write event to audit log file.

        Args:
            event: Event dictionary to log
        """
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def read_audit_log(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Read audit log entries.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            DataFrame with audit log entries
        """
        if not self.log_file.exists():
            return pd.DataFrame()

        events = []
        with open(self.log_file, "r") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    event_time = datetime.fromisoformat(event["timestamp"])

                    if start_date and event_time < start_date:
                        continue
                    if end_date and event_time > end_date:
                        continue

                    events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse audit log line: {e}")

        if not events:
            return pd.DataFrame()

        df = pd.DataFrame(events)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_analysis_summary(self, days: int = 30) -> Dict:
        """
        Get summary of analysis runs.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with summary statistics
        """
        cutoff = datetime.now() - pd.Timedelta(days=days)
        df = self.read_audit_log(start_date=cutoff)

        if df.empty:
            return {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "unique_users": 0,
                "unique_datasets": 0,
            }

        analysis_runs = df[df["event_type"] == "analysis_run"]

        return {
            "total_runs": len(analysis_runs),
            "successful_runs": len(analysis_runs[analysis_runs["status"] == "completed"]),
            "failed_runs": len(analysis_runs[analysis_runs["status"] == "failed"]),
            "unique_users": (
                analysis_runs["user"].nunique() if "user" in analysis_runs.columns else 0
            ),
            "unique_datasets": (
                analysis_runs["dataset"].nunique() if "dataset" in analysis_runs.columns else 0
            ),
        }
