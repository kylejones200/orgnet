"""Simple metric figures via plotsmith (optional ``viz`` extra)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_team_scores_timeseries(
    frame: pd.DataFrame,
    *,
    date_col: str = "calculation_period_end",
    value_cols: tuple[str, ...] = (
        "energy_score",
        "engagement_score",
        "exploration_score",
    ),
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """
    Line chart of Three E's scores over time using PlotSmith styling.

    ``frame`` must include ``date_col`` and the requested ``value_cols`` (wide format).
    """
    import pandas as pd

    try:
        import plotsmith
    except ImportError as e:  # pragma: no cover - exercised when viz extra missing
        raise ImportError(
            "plotsmith is required for plotting. Install with: "
            "pip install 'orgnet[three-es-viz]'"
        ) from e

    missing = [c for c in (date_col, *value_cols) if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    plot_df = frame[[date_col, *value_cols]].copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col], utc=True, errors="coerce")
    plot_df = plot_df.dropna(subset=[date_col]).sort_values(date_col)
    plot_df = plot_df.set_index(date_col)[list(value_cols)]

    return plotsmith.plot_timeseries(
        plot_df,
        title=title or "Team scores over time",
        xlabel="Date",
        ylabel="Score",
        save_path=save_path,
        figsize=figsize,
    )
