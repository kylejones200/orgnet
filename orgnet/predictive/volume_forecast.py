"""Email volume forecasting (from Enron project)."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from orgnet.utils.logging import get_logger

# Try to import optional dependencies
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

logger = get_logger(__name__)


@dataclass
class VolumeForecastModel:
    """Container for volume forecasting model."""

    method: str = "moving_average"  # 'moving_average', 'exponential_smoothing', 'ml'
    model: Optional[any] = None
    avg_volume: float = 0.0
    trend: float = 0.0  # Volume trend (positive = increasing)


def prepare_volume_data(
    df: pd.DataFrame, date_col: str = "timestamp", time_period: str = "day"
) -> pd.Series:
    """
    Prepare time series data for volume forecasting.

    Args:
        df: DataFrame with email data
        date_col: Column name for dates
        time_period: Time period ('hour', 'day', 'week', 'month')

    Returns:
        Series with time-indexed email counts
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if len(df) == 0:
        return pd.Series(dtype=float)

    # Group by time period
    if time_period == "hour":
        df["period"] = df[date_col].dt.floor("H")
    elif time_period == "day":
        df["period"] = df[date_col].dt.date
    elif time_period == "week":
        df["period"] = df[date_col].dt.to_period("W")
    elif time_period == "month":
        df["period"] = df[date_col].dt.to_period("M")
    else:
        raise ValueError(f"Unknown time period: {time_period}")

    # Count emails per period
    if len(df) > 10000:
        df_pl = pl.from_pandas(df) if POLARS_AVAILABLE else None
        if df_pl is not None:
            volume_series = df_pl.group_by("period").agg(pl.count().alias("count"))
            volume_series = volume_series.to_pandas().set_index("period")["count"]
        else:
            volume_series = df.groupby("period").size()
    else:
        volume_series = df.groupby("period").size()
    volume_series.index = pd.to_datetime(volume_series.index)
    volume_series = volume_series.sort_index()

    return volume_series


def forecast_moving_average(
    volume_series: pd.Series, periods: int = 7, forecast_horizon: int = 7
) -> Tuple[np.ndarray, Dict]:
    """
    Forecast using moving average.

    Args:
        volume_series: Time series of email volumes
        periods: Number of periods for moving average
        forecast_horizon: Number of periods to forecast

    Returns:
        Tuple of (forecast_values, metrics_dict)
    """
    if len(volume_series) < periods:
        # Not enough data, use simple average
        avg = volume_series.mean() if len(volume_series) > 0 else 0.0
        forecast = np.array([avg] * forecast_horizon)
        return forecast, {"method": "simple_average", "avg_volume": avg}

    # Calculate moving average
    ma = volume_series.rolling(window=periods).mean()
    last_ma = ma.iloc[-1] if not ma.empty else volume_series.mean()

    # Simple forecast: use last moving average
    forecast = np.array([last_ma] * forecast_horizon)

    # Calculate trend
    if len(volume_series) >= 2:
        recent = volume_series.tail(periods)
        trend = (recent.iloc[-1] - recent.iloc[0]) / len(recent) if len(recent) > 1 else 0.0
    else:
        trend = 0.0

    metrics = {"method": "moving_average", "avg_volume": last_ma, "trend": trend, "window": periods}

    return forecast, metrics


def forecast_exponential_smoothing(
    volume_series: pd.Series, alpha: float = 0.3, forecast_horizon: int = 7
) -> Tuple[np.ndarray, Dict]:
    """
    Forecast using exponential smoothing.

    Args:
        volume_series: Time series of email volumes
        alpha: Smoothing parameter (0-1)
        forecast_horizon: Number of periods to forecast

    Returns:
        Tuple of (forecast_values, metrics_dict)
    """
    if len(volume_series) == 0:
        return np.array([0.0] * forecast_horizon), {
            "method": "exponential_smoothing",
            "avg_volume": 0.0,
        }

    # Simple exponential smoothing
    smoothed = [volume_series.iloc[0]]
    for value in volume_series.iloc[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])

    last_smoothed = smoothed[-1]
    forecast = np.array([last_smoothed] * forecast_horizon)

    metrics = {"method": "exponential_smoothing", "avg_volume": last_smoothed, "alpha": alpha}

    return forecast, metrics


def forecast_volume(
    df: pd.DataFrame,
    model: Optional[VolumeForecastModel] = None,
    date_col: str = "timestamp",
    time_period: str = "day",
    forecast_horizon: int = 7,
) -> Tuple[np.ndarray, Dict]:
    """
    Forecast email volumes.

    Args:
        df: DataFrame with email data
        model: Optional VolumeForecastModel
        date_col: Column name for dates
        time_period: Time period for aggregation
        forecast_horizon: Number of periods to forecast

    Returns:
        Tuple of (forecast_values, metrics_dict)
    """
    if model is None:
        model = VolumeForecastModel()

    # Prepare volume data
    volume_series = prepare_volume_data(df, date_col=date_col, time_period=time_period)

    if len(volume_series) == 0:
        return np.array([0.0] * forecast_horizon), {"method": model.method, "avg_volume": 0.0}

    if model.method == "moving_average":
        forecast, metrics = forecast_moving_average(
            volume_series, forecast_horizon=forecast_horizon
        )
    elif model.method == "exponential_smoothing":
        forecast, metrics = forecast_exponential_smoothing(
            volume_series, forecast_horizon=forecast_horizon
        )
    else:
        # Default to moving average
        forecast, metrics = forecast_moving_average(
            volume_series, forecast_horizon=forecast_horizon
        )

    return forecast, metrics
