"""
Data Access Layer (Layer 2)
===========================
This layer provides repository pattern for database operations.
Abstracts database queries and provides a clean interface for the business logic layer.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import datetime
from typing import List, Optional, Dict
from ..database.models import Team, TeamMember, Communication, TeamMetrics, ExternalTeam


class TeamRepository:
    """Repository for Team operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str, description: str = None) -> Team:
        """Create a new team"""
        team = Team(name=name, description=description)
        self.session.add(team)
        self.session.commit()
        self.session.refresh(team)
        return team

    def get_by_id(self, team_id: int) -> Optional[Team]:
        """Get team by ID"""
        return self.session.query(Team).filter(Team.id == team_id).first()

    def get_by_name(self, name: str) -> Optional[Team]:
        """Get team by name"""
        return self.session.query(Team).filter(Team.name == name).first()

    def get_all(self) -> List[Team]:
        """Get all teams"""
        return self.session.query(Team).all()

    _UPDATE_ALLOWED = {"name", "description"}

    def update(self, team_id: int, **kwargs) -> Optional[Team]:
        """Update team attributes. Only name and description are updatable."""
        team = self.get_by_id(team_id)
        if team:
            for key, value in kwargs.items():
                if key in self._UPDATE_ALLOWED:
                    setattr(team, key, value)
            self.session.commit()
            self.session.refresh(team)
        return team

    def delete(self, team_id: int) -> bool:
        """Delete a team"""
        team = self.get_by_id(team_id)
        if team:
            self.session.delete(team)
            self.session.commit()
            return True
        return False


class TeamMemberRepository:
    """Repository for TeamMember operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str, email: str, team_id: int, role: str = None) -> TeamMember:
        """Create a new team member"""
        member = TeamMember(name=name, email=email, team_id=team_id, role=role)
        self.session.add(member)
        self.session.commit()
        self.session.refresh(member)
        return member

    def get_by_id(self, member_id: int) -> Optional[TeamMember]:
        """Get team member by ID"""
        return self.session.query(TeamMember).filter(TeamMember.id == member_id).first()

    def get_by_email(self, email: str) -> Optional[TeamMember]:
        """Get team member by email"""
        return self.session.query(TeamMember).filter(TeamMember.email == email).first()

    def get_by_team(self, team_id: int) -> List[TeamMember]:
        """Get all members of a team"""
        return self.session.query(TeamMember).filter(TeamMember.team_id == team_id).all()

    def get_all(self) -> List[TeamMember]:
        """Get all team members"""
        return self.session.query(TeamMember).all()

    def update(self, member_id: int, **kwargs) -> Optional[TeamMember]:
        """Update team member attributes"""
        member = self.get_by_id(member_id)
        if member:
            for key, value in kwargs.items():
                if hasattr(member, key):
                    setattr(member, key, value)
            self.session.commit()
            self.session.refresh(member)
        return member

    def delete(self, member_id: int) -> bool:
        """Delete a team member"""
        member = self.get_by_id(member_id)
        if member:
            self.session.delete(member)
            self.session.commit()
            return True
        return False


class CommunicationRepository:
    """Repository for Communication operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        sender_id: int,
        team_id: int,
        communication_type: str,
        receiver_id: int = None,
        duration_minutes: float = None,
        is_group_communication: int = 0,
        is_cross_team: int = 0,
        message_content: str = None,
        timestamp: datetime = None,
    ) -> Communication:
        """Create a new communication record"""
        attrs = dict(
            sender_id=sender_id,
            receiver_id=receiver_id,
            team_id=team_id,
            communication_type=communication_type,
            duration_minutes=duration_minutes,
            is_group_communication=is_group_communication,
            is_cross_team=is_cross_team,
            message_content=message_content,
        )
        if timestamp is not None:
            attrs["timestamp"] = timestamp
        comm = Communication(**attrs)
        self.session.add(comm)
        self.session.commit()
        self.session.refresh(comm)
        return comm

    def get_by_id(self, comm_id: int) -> Optional[Communication]:
        """Get communication by ID"""
        return self.session.query(Communication).filter(Communication.id == comm_id).first()

    def get_by_team(
        self, team_id: int, start_date: datetime = None, end_date: datetime = None
    ) -> List[Communication]:
        """Get all communications for a team within a date range"""
        query = self.session.query(Communication).filter(Communication.team_id == team_id)

        if start_date:
            query = query.filter(Communication.timestamp >= start_date)
        if end_date:
            query = query.filter(Communication.timestamp <= end_date)

        return query.order_by(Communication.timestamp.desc()).all()

    def get_by_member(
        self, member_id: int, start_date: datetime = None, end_date: datetime = None
    ) -> List[Communication]:
        """Get all communications involving a specific member"""
        query = self.session.query(Communication).filter(
            or_(Communication.sender_id == member_id, Communication.receiver_id == member_id)
        )

        if start_date:
            query = query.filter(Communication.timestamp >= start_date)
        if end_date:
            query = query.filter(Communication.timestamp <= end_date)

        return query.order_by(Communication.timestamp.desc()).all()

    def get_cross_team_communications(
        self, team_id: int, start_date: datetime = None, end_date: datetime = None
    ) -> List[Communication]:
        """Get cross-team (exploration) communications"""
        query = self.session.query(Communication).filter(
            and_(Communication.team_id == team_id, Communication.is_cross_team == 1)
        )

        if start_date:
            query = query.filter(Communication.timestamp >= start_date)
        if end_date:
            query = query.filter(Communication.timestamp <= end_date)

        return query.all()

    def get_communication_stats(
        self, team_id: int, start_date: datetime = None, end_date: datetime = None
    ) -> Dict:
        """Get communication statistics for a team"""
        query = self.session.query(Communication).filter(Communication.team_id == team_id)

        if start_date:
            query = query.filter(Communication.timestamp >= start_date)
        if end_date:
            query = query.filter(Communication.timestamp <= end_date)

        total = query.count()
        face_to_face = query.filter(Communication.communication_type == "face-to-face").count()
        group_comms = query.filter(Communication.is_group_communication == 1).count()
        cross_team = query.filter(Communication.is_cross_team == 1).count()

        avg_duration = (
            self.session.query(func.avg(Communication.duration_minutes))
            .filter(Communication.team_id == team_id)
            .scalar()
            or 0
        )

        return {
            "total_communications": total,
            "face_to_face": face_to_face,
            "group_communications": group_comms,
            "cross_team_communications": cross_team,
            "avg_duration_minutes": float(avg_duration),
        }

    def delete(self, comm_id: int) -> bool:
        """Delete a communication record"""
        comm = self.get_by_id(comm_id)
        if comm:
            self.session.delete(comm)
            self.session.commit()
            return True
        return False


class TeamMetricsRepository:
    """Repository for TeamMetrics operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        team_id: int,
        energy_score: float,
        engagement_score: float,
        exploration_score: float,
        overall_score: float,
        calculation_period_start: datetime,
        calculation_period_end: datetime,
        **kwargs,
    ) -> TeamMetrics:
        """Create a new metrics record"""
        metrics = TeamMetrics(
            team_id=team_id,
            energy_score=energy_score,
            engagement_score=engagement_score,
            exploration_score=exploration_score,
            overall_score=overall_score,
            calculation_period_start=calculation_period_start,
            calculation_period_end=calculation_period_end,
            **kwargs,
        )
        self.session.add(metrics)
        self.session.commit()
        self.session.refresh(metrics)
        return metrics

    def get_by_id(self, metrics_id: int) -> Optional[TeamMetrics]:
        """Get metrics by ID"""
        return self.session.query(TeamMetrics).filter(TeamMetrics.id == metrics_id).first()

    def get_latest_by_team(self, team_id: int) -> Optional[TeamMetrics]:
        """Get the most recent metrics for a team"""
        return (
            self.session.query(TeamMetrics)
            .filter(TeamMetrics.team_id == team_id)
            .order_by(TeamMetrics.calculated_at.desc())
            .first()
        )

    def get_by_team(self, team_id: int, limit: int = 10) -> List[TeamMetrics]:
        """Get metrics history for a team"""
        return (
            self.session.query(TeamMetrics)
            .filter(TeamMetrics.team_id == team_id)
            .order_by(TeamMetrics.calculated_at.desc())
            .limit(limit)
            .all()
        )

    def get_history_chronological(self, team_id: int, limit: int = 100) -> List[TeamMetrics]:
        """Oldest-first metrics rows for time-series analysis (e.g. anomsmith)."""
        return (
            self.session.query(TeamMetrics)
            .filter(TeamMetrics.team_id == team_id)
            .order_by(TeamMetrics.calculated_at.asc())
            .limit(limit)
            .all()
        )

    def get_all_latest(self) -> List[TeamMetrics]:
        """Get latest metrics for all teams"""
        subquery = (
            self.session.query(
                TeamMetrics.team_id, func.max(TeamMetrics.calculated_at).label("max_date")
            )
            .group_by(TeamMetrics.team_id)
            .subquery()
        )

        return (
            self.session.query(TeamMetrics)
            .join(
                subquery,
                and_(
                    TeamMetrics.team_id == subquery.c.team_id,
                    TeamMetrics.calculated_at == subquery.c.max_date,
                ),
            )
            .all()
        )


class ExternalTeamRepository:
    """Repository for ExternalTeam operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str, department: str = None, description: str = None) -> ExternalTeam:
        """Create a new external team"""
        ext_team = ExternalTeam(name=name, department=department, description=description)
        self.session.add(ext_team)
        self.session.commit()
        self.session.refresh(ext_team)
        return ext_team

    def get_by_id(self, team_id: int) -> Optional[ExternalTeam]:
        """Get external team by ID"""
        return self.session.query(ExternalTeam).filter(ExternalTeam.id == team_id).first()

    def get_all(self) -> List[ExternalTeam]:
        """Get all external teams"""
        return self.session.query(ExternalTeam).all()
