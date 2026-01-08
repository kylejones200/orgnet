"""Utility functions for standardizing metric outputs."""

from __future__ import annotations

import pandas as pd
from typing import Optional


def standardize_metric_output(
    df: pd.DataFrame,
    value_column: str,
    id_column: str = "node_id",
    top_n: Optional[int] = None,
    top_percentile: float = 0.05,
) -> pd.DataFrame:
    """
    Standardize metric output with ranks and flags.

    Args:
        df: DataFrame with metric values
        value_column: Name of column with metric values
        id_column: Name of column with identifiers
        top_n: Number of top nodes to flag (if None, uses percentile)
        top_percentile: Percentile threshold for top flag (default 0.05 = top 5%)

    Returns:
        DataFrame with standardized columns:
        - id_column: Person/node identifier
        - value: Metric value
        - rank: Rank (1 = highest)
        - top_n_flag: Boolean flag for top N
        - top_percentile_flag: Boolean flag for top percentile
    """
    df = df.copy()

    # Ensure value column exists
    if value_column not in df.columns:
        raise ValueError(f"Value column '{value_column}' not found in DataFrame")

    # Sort by value (descending)
    df = df.sort_values(value_column, ascending=False).reset_index(drop=True)

    # Add rank (1 = highest value)
    df["rank"] = range(1, len(df) + 1)

    # Rename value column to standard name
    if value_column != "value":
        df["value"] = df[value_column]

    # Add top N flag
    if top_n is not None:
        df["top_n_flag"] = df["rank"] <= top_n
    else:
        df["top_n_flag"] = False

    # Add top percentile flag
    threshold_rank = max(1, int(len(df) * top_percentile))
    df["top_percentile_flag"] = df["rank"] <= threshold_rank

    # Select and reorder columns
    output_columns = [id_column, "value", "rank", "top_n_flag", "top_percentile_flag"]
    # Keep any additional columns from original
    additional_cols = [col for col in df.columns if col not in output_columns]
    output_columns.extend(additional_cols)

    return df[output_columns]
