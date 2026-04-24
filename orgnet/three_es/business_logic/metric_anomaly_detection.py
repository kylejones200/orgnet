"""
Time-series anomaly detection on team metrics history using anomsmith.

Builds a univariate series from ``TeamMetrics`` snapshots (e.g. ``overall_score`` ordered
by ``calculated_at``) and runs anomsmith scoring + thresholding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from sqlalchemy.orm import Session

from anomsmith import IQRScorer, ThresholdRule, ZScoreScorer, detect_anomalies

from ..config import AnomalyThresholds
from ..data_access.repositories import TeamMetricsRepository
from ..database.models import TeamMetrics

MetricName = Literal["overall_score", "energy_score", "engagement_score", "exploration_score"]
ScorerKind = Literal["zscore", "iqr"]


def _metric_series(rows: List[TeamMetrics], metric: MetricName) -> pd.Series:
    index = pd.DatetimeIndex([r.calculated_at for r in rows])
    values = [float(getattr(r, metric) or 0.0) for r in rows]
    return pd.Series(values, index=index, name=metric)


def _make_scorer(kind: ScorerKind) -> Union[ZScoreScorer, IQRScorer]:
    if kind == "iqr":
        return IQRScorer()
    return ZScoreScorer()


def detect_team_metric_anomalies(
    session: Session,
    team_id: int,
    metric: MetricName = "overall_score",
    *,
    history_limit: int = 100,
    quantile: Optional[float] = None,
    min_points: Optional[int] = None,
    scorer_kind: ScorerKind = "zscore",
) -> Dict[str, Any]:
    """
    Flag unusual points in a team's stored metric history.

    Args:
        session: SQLAlchemy session.
        team_id: Team primary key.
        metric: Which ``TeamMetrics`` column to treat as the series.
        history_limit: Max number of snapshots (oldest-first window tail).
        quantile: Threshold quantile for ``ThresholdRule`` (default from config).
        min_points: Minimum rows required to run detection (default from config).
        scorer_kind: ``zscore`` (default) or ``iqr`` for a more robust spread measure.

    Returns:
        Dict with ``status`` ``ok`` or ``skipped``, per-timestep ``details`` when ok,
        and counts ``n_points`` / ``n_flagged``.
    """
    q = float(quantile if quantile is not None else AnomalyThresholds.ANOMSMITH_SCORE_QUANTILE)
    need = int(
        min_points if min_points is not None else AnomalyThresholds.MIN_METRIC_HISTORY_POINTS
    )

    repo = TeamMetricsRepository(session)
    rows = repo.get_history_chronological(team_id, history_limit)
    if len(rows) < need:
        return {
            "team_id": team_id,
            "metric": metric,
            "status": "skipped",
            "reason": f"need at least {need} metric snapshots, got {len(rows)}",
            "n_points": len(rows),
            "n_flagged": 0,
            "details": [],
        }

    y = _metric_series(rows, metric)
    scorer = _make_scorer(scorer_kind)
    scorer.fit(y.values.astype(float))
    rule = ThresholdRule(method="quantile", value=q, quantile=q)
    df = detect_anomalies(y, scorer, rule)

    details: List[Dict[str, Any]] = []
    for ts, row in df.iterrows():
        details.append(
            {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "value": float(y.loc[ts]),
                "score": float(row["score"]),
                "anomaly": bool(row["flag"]),
            }
        )

    return {
        "team_id": team_id,
        "metric": metric,
        "status": "ok",
        "scorer": scorer_kind,
        "quantile": q,
        "n_points": len(y),
        "n_flagged": int(df["flag"].sum()),
        "details": details,
    }
