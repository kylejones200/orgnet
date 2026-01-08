"""Privacy and ethics utilities for organizational network analysis.

This module provides functions for anonymization, aggregation, and privacy
compliance in orgnet analysis.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class PrivacyManager:
    """Manages privacy and anonymization for orgnet data."""

    def __init__(
        self,
        anonymize_individuals: bool = False,
        aggregate_before_storage: bool = True,
        retention_days: int = 90,
        min_group_size: int = 5,
        hash_salt: Optional[str] = None,
    ):
        """
        Initialize privacy manager.

        Args:
            anonymize_individuals: If True, hash personal identifiers
            aggregate_before_storage: If True, aggregate data before storing
            retention_days: Number of days to retain data
            min_group_size: Minimum group size to display (smaller groups are dropped)
            hash_salt: Optional salt for hashing (for consistent hashing across runs)
        """
        self.anonymize_individuals = anonymize_individuals
        self.aggregate_before_storage = aggregate_before_storage
        self.retention_days = retention_days
        self.min_group_size = min_group_size
        self.hash_salt = hash_salt or "orgnet_default_salt"
        self.hash_mapping: Dict[str, str] = {}

    def hash_identifier(self, identifier: str) -> str:
        """
        Hash a personal identifier.

        Args:
            identifier: Original identifier (email, name, etc.)

        Returns:
            Hashed identifier
        """
        if identifier in self.hash_mapping:
            return self.hash_mapping[identifier]

        # Create hash with salt
        hash_input = f"{self.hash_salt}:{identifier}".encode("utf-8")
        hash_value = hashlib.sha256(hash_input).hexdigest()[:16]  # Use first 16 chars
        self.hash_mapping[identifier] = hash_value
        return hash_value

    def anonymize_dataframe(self, df: pd.DataFrame, id_columns: List[str]) -> pd.DataFrame:
        """
        Anonymize identifiers in a DataFrame.

        Args:
            df: DataFrame to anonymize
            id_columns: List of column names containing identifiers to hash

        Returns:
            DataFrame with hashed identifiers
        """
        if not self.anonymize_individuals:
            return df

        df = df.copy()
        for col in id_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self.hash_identifier)

        return df

    def filter_by_retention(
        self, df: pd.DataFrame, timestamp_column: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Filter DataFrame to only include data within retention period.

        Args:
            df: DataFrame with timestamp data
            timestamp_column: Name of timestamp column

        Returns:
            Filtered DataFrame
        """
        if timestamp_column not in df.columns:
            logger.warning(
                f"Timestamp column {timestamp_column} not found, skipping retention filter"
            )
            return df

        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        df = df.copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        filtered = df[df[timestamp_column] >= cutoff_date]

        logger.info(
            f"Retention filter: {len(df)} rows -> {len(filtered)} rows "
            f"(cutoff: {cutoff_date.date()})"
        )
        return filtered

    def drop_small_groups(
        self, df: pd.DataFrame, group_column: str, min_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Drop groups smaller than minimum size.

        Args:
            df: DataFrame to filter
            group_column: Column name to group by
            min_size: Minimum group size (uses self.min_group_size if None)

        Returns:
            Filtered DataFrame
        """
        if min_size is None:
            min_size = self.min_group_size

        if group_column not in df.columns:
            return df

        group_counts = df[group_column].value_counts()
        valid_groups = group_counts[group_counts >= min_size].index
        filtered = df[df[group_column].isin(valid_groups)]

        dropped = len(df) - len(filtered)
        if dropped > 0:
            logger.info(
                f"Dropped {dropped} rows from groups smaller than {min_size} "
                f"in column {group_column}"
            )

        return filtered

    def remove_sensitive_fields(
        self, df: pd.DataFrame, sensitive_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Remove sensitive fields from DataFrame.

        Args:
            df: DataFrame to clean
            sensitive_columns: List of column names to remove.
                If None, uses default sensitive fields.

        Returns:
            DataFrame with sensitive columns removed
        """
        if sensitive_columns is None:
            # Default sensitive fields
            sensitive_columns = [
                "email",
                "phone",
                "address",
                "ssn",
                "salary",
                "personal_email",
                "home_address",
            ]

        df = df.copy()
        columns_to_drop = [col for col in sensitive_columns if col in df.columns]

        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logger.info(f"Removed sensitive columns: {columns_to_drop}")

        return df

    def aggregate_interactions(
        self, interactions: List[Dict], group_by: List[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate interactions to reduce individual-level detail.

        Args:
            interactions: List of interaction dictionaries
            group_by: Columns to group by. If None, aggregates by sender/receiver pair and period.

        Returns:
            Aggregated DataFrame
        """
        if not self.aggregate_before_storage:
            return pd.DataFrame(interactions)

        df = pd.DataFrame(interactions)

        if group_by is None:
            # Default aggregation: by sender, receiver, and time period
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["period"] = df["timestamp"].dt.to_period("W")  # Weekly aggregation
                group_by = ["sender_id", "receiver_id", "period"]
            else:
                group_by = ["sender_id", "receiver_id"]

        # Aggregate
        agg_dict = {}
        if "count" in df.columns:
            agg_dict["count"] = "sum"
        else:
            agg_dict = {col: "count" for col in df.columns if col not in group_by}

        aggregated = df.groupby(group_by, as_index=False).agg(agg_dict)

        logger.info(f"Aggregated {len(df)} interactions -> {len(aggregated)} aggregated records")

        return aggregated

    def apply_privacy_policy(
        self,
        df: pd.DataFrame,
        id_columns: Optional[List[str]] = None,
        timestamp_column: Optional[str] = None,
        group_column: Optional[str] = None,
        sensitive_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply all privacy policies to a DataFrame.

        Args:
            df: DataFrame to process
            id_columns: Columns containing identifiers to hash
            timestamp_column: Column with timestamps for retention filtering
            group_column: Column for small group filtering
            sensitive_columns: Columns to remove

        Returns:
            Processed DataFrame
        """
        result = df.copy()

        # Remove sensitive fields
        if sensitive_columns:
            result = self.remove_sensitive_fields(result, sensitive_columns)

        # Filter by retention
        if timestamp_column:
            result = self.filter_by_retention(result, timestamp_column)

        # Anonymize identifiers
        if id_columns and self.anonymize_individuals:
            result = self.anonymize_dataframe(result, id_columns)

        # Drop small groups
        if group_column:
            result = self.drop_small_groups(result, group_column)

        return result

    def get_hash_mapping(self) -> Dict[str, str]:
        """
        Get the mapping of original identifiers to hashed values.

        Returns:
            Dictionary mapping original -> hashed identifiers
        """
        return self.hash_mapping.copy()


def create_privacy_manager_from_config(config: Dict) -> PrivacyManager:
    """
    Create PrivacyManager from configuration dictionary.

    Args:
        config: Configuration dictionary with privacy settings

    Returns:
        PrivacyManager instance
    """
    privacy_config = config.get("privacy", {})
    return PrivacyManager(
        anonymize_individuals=privacy_config.get("anonymize_individuals", False),
        aggregate_before_storage=privacy_config.get("aggregate_before_storage", True),
        retention_days=privacy_config.get("retention_days", 90),
        min_group_size=privacy_config.get("min_group_size", 5),
        hash_salt=privacy_config.get("hash_salt"),
    )
