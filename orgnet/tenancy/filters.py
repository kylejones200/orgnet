"""DataFrame and record scoping by tenant."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from orgnet.tenancy.context import current_tenant_id


def filter_dataframe_for_tenant(
    df: pd.DataFrame,
    tenant_column: str,
    *,
    tenant_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return rows for a single tenant.

    If ``tenant_id`` is omitted, uses :func:`orgnet.tenancy.context.current_tenant_id`.
    """
    tid = tenant_id if tenant_id is not None else current_tenant_id()
    if tid is None:
        raise ValueError("tenant_id is required when no tenant_scope is active")
    if tenant_column not in df.columns:
        raise KeyError(f"Column {tenant_column!r} not in DataFrame")
    return df.loc[df[tenant_column] == tid].copy()


def assert_tenant_column(df: pd.DataFrame, tenant_column: str) -> None:
    """Raise if the tenant column is missing (defensive guard before persistence)."""
    if tenant_column not in df.columns:
        raise KeyError(f"Expected tenant column {tenant_column!r} in DataFrame")
