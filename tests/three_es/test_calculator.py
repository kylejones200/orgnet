"""
Comprehensive tests for ThreeEsCalculator.
"""

import pytest

from orgnet.three_es.business_logic import ThreeEsCalculator


class TestEnergyCalculation:
    """Tests for energy score calculation"""

    def test_energy_zero_with_no_communications(self, session, sample_team, thirty_days_ago, now):
        """Energy should be 0 when there are no communications"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        result = calculator.calculate_energy(team.id, thirty_days_ago, now)

        assert result["energy_score"] == 0.0
        assert result["total_communications"] == 0
        assert result["face_to_face_ratio"] == 0.0

    def test_energy_increases_with_face_to_face(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Energy should be higher with face-to-face communications"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Add face-to-face communications with timestamp in range
        for i in range(20):
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[1].id,
                team_id=team.id,
                communication_type="face-to-face",
                duration_minutes=30,
                timestamp=mid_date,
            )

        result = calculator.calculate_energy(team.id, thirty_days_ago, now)

        assert result["energy_score"] > 0
        assert result["total_communications"] == 20
        assert result["face_to_face_ratio"] == 1.0

    def test_energy_lower_with_email(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Energy with email only should be lower than face-to-face"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Add 20 face-to-face for team 1
        for i in range(20):
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[1].id,
                team_id=team.id,
                communication_type="face-to-face",
                duration_minutes=30,
                timestamp=mid_date,
            )

        f2f_result = calculator.calculate_energy(team.id, thirty_days_ago, now)

        # Create new team for email test
        from orgnet.three_es.data_access.repositories import TeamRepository

        team_repo = TeamRepository(session)
        email_team = team_repo.create("Email Team", "Email only team")

        # Add same number of email communications
        for i in range(20):
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[1].id,
                team_id=email_team.id,
                communication_type="email",
                duration_minutes=30,
                timestamp=mid_date,
            )

        email_result = calculator.calculate_energy(email_team.id, thirty_days_ago, now)

        # Face-to-face should have higher energy
        assert f2f_result["energy_score"] > email_result["energy_score"]

    def test_energy_score_capped_at_100(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Energy score should never exceed 100"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Add excessive communications
        for i in range(1000):
            comm_repo.create(
                sender_id=members[i % len(members)].id,
                receiver_id=members[(i + 1) % len(members)].id,
                team_id=team.id,
                communication_type="face-to-face",
                duration_minutes=60,
                timestamp=mid_date,
            )

        result = calculator.calculate_energy(team.id, thirty_days_ago, now)

        assert result["energy_score"] <= 100.0


class TestEngagementCalculation:
    """Tests for engagement score calculation"""

    def test_engagement_zero_with_no_communications(
        self, session, sample_team, thirty_days_ago, now
    ):
        """Engagement should be 0 when there are no communications"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        result = calculator.calculate_engagement(team.id, thirty_days_ago, now)

        assert result["engagement_score"] == 0.0
        assert result["participation_rate"] == 0.0

    def test_engagement_perfect_with_balanced_communication(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Engagement should be high when all members communicate equally"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Create fully connected, balanced communications
        for sender in members:
            for receiver in members:
                if sender.id != receiver.id:
                    # Each pair communicates exactly twice
                    for _ in range(2):
                        comm_repo.create(
                            sender_id=sender.id,
                            receiver_id=receiver.id,
                            team_id=team.id,
                            communication_type="face-to-face",
                            duration_minutes=15,
                            timestamp=mid_date,
                        )

        result = calculator.calculate_engagement(team.id, thirty_days_ago, now)

        assert result["engagement_score"] > 70  # Should be high
        assert result["participation_rate"] == 1.0  # All members participated
        assert result["gini_coefficient"] < 0.2  # Very balanced

    def test_engagement_low_with_dominated_communication(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Engagement should be lower when one person dominates"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Member 0 sends 95% of communications
        for i in range(95):
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[i % 4 + 1].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )

        # Others send just a few
        for i in range(1, 5):
            comm_repo.create(
                sender_id=members[i].id,
                receiver_id=members[0].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )

        result = calculator.calculate_engagement(team.id, thirty_days_ago, now)

        assert result["gini_coefficient"] > 0.5  # High inequality
        assert result["balance_score"] < 0.5  # Low balance

    def test_engagement_participation_rate(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Participation rate should reflect percentage of active members"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Only 3 out of 5 members communicate
        for i in range(10):
            comm_repo.create(
                sender_id=members[i % 3].id,  # Only first 3 members
                receiver_id=members[(i + 1) % 3].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )

        result = calculator.calculate_engagement(team.id, thirty_days_ago, now)

        assert result["participation_rate"] == pytest.approx(0.6, rel=0.01)  # 3/5 = 0.6

    def test_gini_coefficient_calculation(self, session):
        """Test Gini coefficient calculation directly"""
        calculator = ThreeEsCalculator(session)

        # Perfect equality
        assert calculator._calculate_gini_coefficient([1, 1, 1, 1]) == pytest.approx(0, abs=0.01)

        # Perfect inequality (one has everything)
        assert calculator._calculate_gini_coefficient([0, 0, 0, 10]) == pytest.approx(0.75, abs=0.1)

        # Moderate inequality
        gini = calculator._calculate_gini_coefficient([1, 2, 3, 4, 5])
        assert 0.2 < gini < 0.4

    def test_two_way_communication_detection(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test detection of two-way communication patterns"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Create two-way communications between pairs
        for i in range(10):
            # A -> B
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[1].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )
            # B -> A
            comm_repo.create(
                sender_id=members[1].id,
                receiver_id=members[0].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )

        result = calculator.calculate_engagement(team.id, thirty_days_ago, now)

        assert result["two_way_communication_score"] > 0


class TestExplorationCalculation:
    """Tests for exploration score calculation"""

    def test_exploration_zero_with_no_cross_team(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Exploration should be 0 when all communications are internal"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Add only internal communications
        for i in range(20):
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[1].id,
                team_id=team.id,
                communication_type="face-to-face",
                is_cross_team=0,
                timestamp=mid_date,
            )

        result = calculator.calculate_exploration(team.id, thirty_days_ago, now)

        assert result["exploration_score"] == 0.0
        assert result["cross_team_communications"] == 0

    def test_exploration_increases_with_cross_team(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Exploration should increase with cross-team communications"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Add mix of internal and cross-team
        for i in range(10):
            # Internal
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[1].id,
                team_id=team.id,
                communication_type="face-to-face",
                is_cross_team=0,
                timestamp=mid_date,
            )
            # Cross-team
            comm_repo.create(
                sender_id=members[0].id,
                team_id=team.id,
                communication_type="face-to-face",
                is_cross_team=1,
                timestamp=mid_date,
            )

        result = calculator.calculate_exploration(team.id, thirty_days_ago, now)

        assert result["exploration_score"] > 0
        assert result["cross_team_communications"] == 10
        assert result["exploration_ratio"] == pytest.approx(0.5, rel=0.01)  # 10/20

    def test_exploration_member_participation(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test tracking which members are exploring"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Only 2 members do cross-team communications
        for i in range(5):
            comm_repo.create(
                sender_id=members[0].id,
                team_id=team.id,
                communication_type="face-to-face",
                is_cross_team=1,
                timestamp=mid_date,
            )
            comm_repo.create(
                sender_id=members[1].id,
                team_id=team.id,
                communication_type="face-to-face",
                is_cross_team=1,
                timestamp=mid_date,
            )

        result = calculator.calculate_exploration(team.id, thirty_days_ago, now)

        assert result["members_exploring"] == 2
        assert result["member_exploration_rate"] == pytest.approx(0.4, rel=0.01)  # 2/5


class TestOverallPerformance:
    """Tests for overall performance calculation"""

    def test_overall_weighted_average(self, session):
        """Test overall score uses correct weighted average"""
        calculator = ThreeEsCalculator(session)

        overall = calculator.calculate_overall_performance(
            energy=80.0, engagement=90.0, exploration=70.0
        )

        # Weights: energy=0.35, engagement=0.40, exploration=0.25
        expected = (80 * 0.35) + (90 * 0.40) + (70 * 0.25)
        assert overall == pytest.approx(expected, rel=0.01)

    def test_overall_handles_zeros(self, session):
        """Test overall score with zero values"""
        calculator = ThreeEsCalculator(session)

        overall = calculator.calculate_overall_performance(0, 0, 0)
        assert overall == 0.0

    def test_overall_handles_perfect_scores(self, session):
        """Test overall score with perfect scores"""
        calculator = ThreeEsCalculator(session)

        overall = calculator.calculate_overall_performance(100, 100, 100)
        assert overall == 100.0


class TestCalculateAllMetrics:
    """Tests for the main calculate_all_metrics method"""

    def test_calculates_all_three_es(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test that all three E's are calculated"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Add some communications
        for i in range(10):
            comm_repo.create(
                sender_id=members[i % len(members)].id,
                receiver_id=members[(i + 1) % len(members)].id,
                team_id=team.id,
                communication_type="face-to-face",
                duration_minutes=15,
                timestamp=mid_date,
            )

        result = calculator.calculate_all_metrics(
            team_id=team.id, start_date=thirty_days_ago, end_date=now, save_to_db=False
        )

        assert "energy" in result
        assert "engagement" in result
        assert "exploration" in result
        assert "overall_score" in result
        assert result["team_id"] == team.id

    def test_saves_to_database_when_requested(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test that metrics are saved to database when save_to_db=True"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        # Add communications
        for i in range(5):
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[1].id,
                team_id=team.id,
                communication_type="face-to-face",
                timestamp=mid_date,
            )

        result = calculator.calculate_all_metrics(
            team_id=team.id, start_date=thirty_days_ago, end_date=now, save_to_db=True
        )

        # Verify it was saved
        from orgnet.three_es.data_access.repositories import TeamMetricsRepository

        metrics_repo = TeamMetricsRepository(session)
        saved_metrics = metrics_repo.get_latest_by_team(team.id)

        assert saved_metrics is not None
        assert saved_metrics.energy_score == result["energy"]["energy_score"]

    def test_default_date_range(self, session, sample_team, comm_repo):
        """Test that default date range is 30 days"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        result = calculator.calculate_all_metrics(team_id=team.id, save_to_db=False)

        # Verify period is set
        assert "calculation_period" in result
        assert "start" in result["calculation_period"]
        assert "end" in result["calculation_period"]


class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_single_member_team(
        self, session, team_repo, member_repo, comm_repo, thirty_days_ago, now
    ):
        """Test calculation with only one team member"""
        team = team_repo.create("Solo Team", "One member team")
        member_repo.create(
            name="Solo Member",
            email="solo@test.com",
            team_id=team.id,
            role="Engineer",
        )

        calculator = ThreeEsCalculator(session)
        result = calculator.calculate_all_metrics(team.id, thirty_days_ago, now, save_to_db=False)

        # Should handle gracefully
        assert result["energy"]["energy_score"] == 0.0
        assert result["engagement"]["engagement_score"] == 0.0

    def test_empty_team(self, session, team_repo, thirty_days_ago, now):
        """Test calculation with team that has no members"""
        team = team_repo.create("Empty Team", "No members")

        calculator = ThreeEsCalculator(session)
        result = calculator.calculate_all_metrics(team.id, thirty_days_ago, now, save_to_db=False)

        assert result["energy"]["energy_score"] == 0.0
        assert result["engagement"]["engagement_score"] == 0.0
        assert result["exploration"]["exploration_score"] == 0.0

    def test_zero_duration_communications(
        self, session, sample_team, comm_repo, thirty_days_ago, now, mid_date
    ):
        """Test handling of communications with no duration"""
        team, members = sample_team
        calculator = ThreeEsCalculator(session)

        for i in range(10):
            comm_repo.create(
                sender_id=members[0].id,
                receiver_id=members[1].id,
                team_id=team.id,
                communication_type="chat",
                duration_minutes=0,  # Zero duration
                timestamp=mid_date,
            )

        result = calculator.calculate_energy(team.id, thirty_days_ago, now)

        # Should handle gracefully
        assert result["energy_score"] >= 0
        assert result["total_duration_minutes"] == 0.0
