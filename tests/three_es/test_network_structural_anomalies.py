"""Tests for anomsmith-backed structural network anomaly detection."""

from __future__ import annotations

from orgnet.three_es.business_logic.network_structural_anomalies import (
    detect_team_network_structural_anomalies,
)


class TestDetectTeamNetworkStructuralAnomalies:
    def test_skipped_with_single_member(self, session, team_repo, member_repo, now):
        team = team_repo.create("Solo", "solo")
        member_repo.create("Only", "only@test.com", team.id, "Eng")
        out = detect_team_network_structural_anomalies(session, team.id, now, now)
        assert out["status"] == "skipped"
        assert out["n_members"] == 1

    def test_flags_hub_like_pattern(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ) -> None:
        team, members = sample_team
        hub = members[2]
        for i in [0, 1, 3, 4]:
            comm_repo.create(
                sender_id=members[i].id,
                receiver_id=hub.id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )
            comm_repo.create(
                sender_id=hub.id,
                receiver_id=members[i].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )
        out = detect_team_network_structural_anomalies(
            session,
            team.id,
            thirty_days_ago,
            now,
            quantile=0.85,
            random_state=0,
        )
        assert out["status"] == "ok"
        assert out["n_members"] == 5
        assert out["n_edges"] >= 1
        assert out["networkx_metrics"] in ("included", "skipped", "disabled")
        hub_row = next(m for m in out["members"] if m["member_id"] == hub.id)
        wds = [float(m["weighted_degree"]) for m in out["members"]]
        assert float(hub_row["weighted_degree"]) == max(wds)
        assert out["n_flagged"] >= 1 or hub_row["score"] == max(m["score"] for m in out["members"])
