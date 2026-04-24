"""
Tests for NetworkAnalyzer.
"""

import pytest
from orgnet.three_es.business_logic import NetworkAnalyzer


class TestNetworkConstruction:
    """Tests for building communication networks"""

    def test_empty_network(self, session, sample_team, thirty_days_ago, now):
        """Test building network with no communications"""
        team, members = sample_team
        analyzer = NetworkAnalyzer(session)

        G = analyzer.build_communication_network(team.id, thirty_days_ago, now)

        assert G.number_of_nodes() == 5  # All members added as nodes
        assert G.number_of_edges() == 0  # No communications = no edges

    def test_network_with_communications(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test network includes communications as edges"""
        team, members = sample_team
        analyzer = NetworkAnalyzer(session)

        # Add communications
        comm_repo.create(
            sender_id=members[0].id,
            receiver_id=members[1].id,
            team_id=team.id,
            communication_type="face-to-face",
            timestamp=mid_date,
        )

        G = analyzer.build_communication_network(team.id, thirty_days_ago, now)

        assert G.number_of_nodes() == 5
        assert G.number_of_edges() >= 1
        assert G.has_edge(members[0].id, members[1].id)

    def test_network_weights_multiple_communications(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test that multiple communications increase edge weight"""
        team, members = sample_team
        analyzer = NetworkAnalyzer(session)

        # Add 5 communications between same pair
        for _ in range(5):
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[1].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )

        G = analyzer.build_communication_network(team.id, thirty_days_ago, now)

        assert G[members[0].id][members[1].id]["weight"] == 5


class TestNetworkMetrics:
    """Tests for network metrics calculation"""

    def test_density_fully_connected(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test density of fully connected network"""
        team, members = sample_team
        analyzer = NetworkAnalyzer(session)

        # Create fully connected network
        for i, sender in enumerate(members):
            for j, receiver in enumerate(members):
                if i != j:
                    comm_repo.create(
                        sender_id=sender.id,
                        receiver_id=receiver.id,
                        team_id=team.id,
                        communication_type="face-to-face",
                        timestamp=mid_date,
                    )

        result = analyzer.analyze_network_metrics(team.id, thirty_days_ago, now)

        assert result["density"] == pytest.approx(1.0, rel=0.01)
        assert result["is_connected"] == True

    def test_disconnected_network(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test detection of disconnected network"""
        team, members = sample_team
        analyzer = NetworkAnalyzer(session)

        # Only connect first two members, leave others isolated
        comm_repo.create(
            sender_id=members[0].id,
            receiver_id=members[1].id,
            team_id=team.id,
            communication_type="face-to-face",
            timestamp=mid_date,
        )

        result = analyzer.analyze_network_metrics(team.id, thirty_days_ago, now)

        assert result["is_connected"] == False

    def test_bottleneck_detection(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test detection of communication bottlenecks"""
        team, members = sample_team
        analyzer = NetworkAnalyzer(session)

        # Create star topology with member[2] as central hub
        for i in [0, 1, 3, 4]:
            comm_repo.create(
                sender_id=members[i].id,
                receiver_id=members[2].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )
            comm_repo.create(
                sender_id=members[2].id,
                receiver_id=members[i].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )

        result = analyzer.analyze_network_metrics(team.id, thirty_days_ago, now)

        # Central hub should be identified as bottleneck
        assert members[2].id in result.get("potential_bottlenecks", [])


class TestCommunityDetection:
    """Tests for community detection"""

    def test_single_community(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test well-connected team has single community"""
        team, members = sample_team
        analyzer = NetworkAnalyzer(session)

        # Create well-connected network
        for sender in members:
            for receiver in members:
                if sender.id != receiver.id:
                    comm_repo.create(
                        sender_id=sender.id,
                        receiver_id=receiver.id,
                        team_id=team.id,
                        communication_type="face-to-face",
                        timestamp=mid_date,
                    )

        result = analyzer.detect_communities(team.id, thirty_days_ago, now)

        assert result["num_communities"] == 1
        assert result["is_siloed"] == False

    def test_insufficient_members_for_communities(
        self, session, team_repo, member_repo, thirty_days_ago, now
    ):
        """Test handling of too-small teams"""
        team = team_repo.create("Small Team", "Only 2 members")
        member1 = member_repo.create("M1", "m1@test.com", team.id, "Eng")
        member2 = member_repo.create("M2", "m2@test.com", team.id, "Eng")

        analyzer = NetworkAnalyzer(session)
        result = analyzer.detect_communities(team.id, thirty_days_ago, now)

        assert "error" in result


class TestCentralityAnalysis:
    """Tests for advanced centrality metrics"""

    def test_hub_identification(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test identification of highly connected members"""
        team, members = sample_team
        analyzer = NetworkAnalyzer(session)

        # Member 0 connects to everyone
        for i in range(1, 5):
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[i].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )
            comm_repo.create(
                sender_id=members[i].id,
                receiver_id=members[0].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )

        result = analyzer.calculate_advanced_centrality(team.id, thirty_days_ago, now)

        # Member 0 should be top hub
        assert result["key_roles"]["hubs"][0]["member_id"] == members[0].id

    def test_centrality_with_empty_network(self, session, sample_team, thirty_days_ago, now):
        """Test centrality calculation with no communications"""
        team, members = sample_team
        analyzer = NetworkAnalyzer(session)

        result = analyzer.calculate_advanced_centrality(team.id, thirty_days_ago, now)

        # Should handle gracefully
        assert "centrality_metrics" in result or "error" in result
