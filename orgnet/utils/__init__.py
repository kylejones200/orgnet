"""Utility modules."""

from orgnet.utils.logging import get_logger
from orgnet.utils.performance import (
    NUMBA_AVAILABLE,
    POLARS_AVAILABLE,
    PARQUET_AVAILABLE,
    JOBLIB_AVAILABLE,
    save_parquet,
    load_parquet,
    optimize_dataframe_ops,
    polars_groupby,
    polars_join,
    ParquetCache,
    cached_computation,
    parallel_map,
    parallel_groupby_apply,
    z_score_numba,
    rolling_mean_numba,
    compute_edge_weights_numba,
)

__all__ = [
    "get_logger",
    "NUMBA_AVAILABLE",
    "POLARS_AVAILABLE",
    "PARQUET_AVAILABLE",
    "JOBLIB_AVAILABLE",
    "save_parquet",
    "load_parquet",
    "optimize_dataframe_ops",
    "polars_groupby",
    "polars_join",
    "ParquetCache",
    "cached_computation",
    "parallel_map",
    "parallel_groupby_apply",
    "z_score_numba",
    "rolling_mean_numba",
    "compute_edge_weights_numba",
]
