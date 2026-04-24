"""Tests for anomsmith-backed metric history anomaly detection."""

from datetime import datetime, timedelta, timezone

import pytest

from orgnet.three_es.business_logic.metric_anomaly_detection import detect_team_metric_anomalies
from orgnet.three_es.data_access.repositories import TeamMetricsRepository


@pytest.fixture
def metrics_repo(session):
    return TeamMetricsRepository(session)


def test_skipped_when_insufficient_history(session, sample_team, metrics_repo):
    team, _members = sample_team
    out = detect_team_metric_anomalies(session, team.id, "overall_score")
    assert out["status"] == "skipped"
    assert out["n_flagged"] == 0


def test_detects_spike_in_overall_score(session, sample_team, metrics_repo):
    team, _members = sample_team
    base = datetime.now(timezone.utc) - timedelta(days=30)
    for i in range(8):
        overall = 60.0 if i < 7 else 99.0
        metrics_repo.create(
            team_id=team.id,
            energy_score=overall,
            engagement_score=overall,
            exploration_score=overall,
            overall_score=overall,
            calculation_period_start=base + timedelta(days=i),
            calculation_period_end=base + timedelta(days=i + 1),
            calculated_at=base + timedelta(days=i),
            total_communications=10,
            participation_rate=0.5,
            gini_coefficient=0.4,
        )

    out = detect_team_metric_anomalies(
        session,
        team.id,
        "overall_score",
        quantile=0.9,
        min_points=5,
    )
    assert out["status"] == "ok"
    assert out["n_points"] == 8
    assert out["n_flagged"] >= 1
    flagged = [d for d in out["details"] if d["anomaly"]]
    assert any(d["value"] == 99.0 for d in flagged)
