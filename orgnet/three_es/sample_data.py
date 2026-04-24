"""
Sample Data Generator
====================
Generates sample data for testing and demonstration purposes
"""

from datetime import datetime, timedelta, timezone
import random
from typing import Dict
from sqlalchemy.orm import Session

from .database.models import Team, TeamMember, Communication
from .data_access.repositories import TeamRepository, TeamMemberRepository, CommunicationRepository


class SampleDataGenerator:
    """Generate sample organizational data for testing"""

    def __init__(self, session: Session):
        self.session = session
        self.team_repo = TeamRepository(session)
        self.member_repo = TeamMemberRepository(session)
        self.comm_repo = CommunicationRepository(session)

    def generate_sample_team(
        self, name: str = "Engineering Team Alpha", num_members: int = 8
    ) -> Team:
        """
        Generate a sample team with members

        Args:
            name: Team name
            num_members: Number of team members to create

        Returns:
            Created Team object
        """
        # Create team
        team = self.team_repo.create(
            name=name,
            description="A high-performance engineering team focused on product development",
        )

        # Sample member names and roles
        sample_members = [
            ("Alice Johnson", "Team Lead"),
            ("Bob Smith", "Senior Engineer"),
            ("Carol Williams", "Engineer"),
            ("David Brown", "Engineer"),
            ("Eve Davis", "Designer"),
            ("Frank Miller", "Product Manager"),
            ("Grace Wilson", "QA Engineer"),
            ("Henry Moore", "Engineer"),
            ("Ivy Taylor", "Data Analyst"),
            ("Jack Anderson", "DevOps Engineer"),
        ]

        # Create members
        created_members = []
        for i in range(min(num_members, len(sample_members))):
            name, role = sample_members[i]
            email = f"{name.lower().replace(' ', '.')}@company.com"
            member = self.member_repo.create(name=name, email=email, team_id=team.id, role=role)
            created_members.append(member)

        return team

    def generate_sample_communications(
        self, team_id: int, days: int = 30, intensity: str = "high"
    ) -> int:
        """
        Generate sample communications for a team

        Args:
            team_id: ID of the team
            days: Number of days to generate data for
            intensity: 'low', 'medium', or 'high' communication intensity

        Returns:
            Number of communications created
        """
        members = self.member_repo.get_by_team(team_id)
        if not members:
            return 0

        member_ids = [m.id for m in members]

        # Set communication frequency based on intensity
        intensity_map = {"low": 2, "medium": 5, "high": 10}
        comms_per_day = intensity_map.get(intensity, 5)

        # Communication types and their weights (face-to-face preferred per research)
        comm_types = [
            ("face-to-face", 0.4),
            ("meeting", 0.25),
            ("chat", 0.15),
            ("email", 0.10),
            ("video-call", 0.10),
        ]

        count = 0
        end_date = datetime.now(timezone.utc)

        for day in range(days):
            current_date = end_date - timedelta(days=day)

            # Skip weekends
            if current_date.weekday() >= 5:
                continue

            # Generate communications for this day
            num_comms = random.randint(comms_per_day - 2, comms_per_day + 3)

            for _ in range(num_comms):
                # Select sender
                sender_id = random.choice(member_ids)

                # Decide if group communication or one-on-one
                is_group = random.random() < 0.3  # 30% chance of group comm

                if is_group:
                    receiver_id = None
                else:
                    # Select receiver (different from sender)
                    possible_receivers = [m for m in member_ids if m != sender_id]
                    receiver_id = random.choice(possible_receivers) if possible_receivers else None

                # Select communication type (weighted)
                comm_type = random.choices(
                    [ct[0] for ct in comm_types], weights=[ct[1] for ct in comm_types]
                )[0]

                # Duration (in minutes)
                if comm_type in ["face-to-face", "chat"]:
                    duration = random.randint(5, 30)
                elif comm_type in ["meeting", "video-call"]:
                    duration = random.randint(15, 60)
                else:
                    duration = random.randint(2, 10)

                # Cross-team communication (10% chance)
                is_cross_team = random.random() < 0.1

                # Random time during work hours
                hour = random.randint(9, 17)
                minute = random.randint(0, 59)
                timestamp = current_date.replace(hour=hour, minute=minute, second=0)

                # Create communication
                comm = Communication(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    team_id=team_id,
                    communication_type=comm_type,
                    duration_minutes=duration,
                    is_group_communication=1 if is_group else 0,
                    is_cross_team=1 if is_cross_team else 0,
                    timestamp=timestamp,
                )
                self.session.add(comm)
                count += 1

        self.session.commit()
        return count

    def generate_complete_sample(
        self,
        team_name: str = "Engineering Team Alpha",
        num_members: int = 8,
        days: int = 30,
        intensity: str = "high",
    ) -> Dict:
        """
        Generate a complete sample dataset with team, members, and communications

        Returns:
            Dictionary with created objects info
        """
        team = self.generate_sample_team(team_name, num_members)
        num_comms = self.generate_sample_communications(team.id, days, intensity)

        return {
            "team_id": team.id,
            "team_name": team.name,
            "num_members": len(team.members),
            "num_communications": num_comms,
            "days": days,
        }


def generate_sample_data(session: Session, num_teams: int = 3):
    """
    Generate multiple sample teams with varying performance characteristics

    Args:
        session: Database session
        num_teams: Number of teams to create
    """
    generator = SampleDataGenerator(session)

    # Create teams with different performance levels
    teams_config = [
        ("High Performance Team", 8, 30, "high"),
        ("Average Team", 6, 30, "medium"),
        ("Struggling Team", 5, 30, "low"),
    ]

    results = []
    for i in range(min(num_teams, len(teams_config))):
        config = teams_config[i]
        result = generator.generate_complete_sample(*config)
        results.append(result)
        print(
            f"Created: {result['team_name']} with {result['num_members']} members and {result['num_communications']} communications"
        )

    return results
