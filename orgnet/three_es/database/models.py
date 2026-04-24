"""
Database Layer (Layer 1)
========================
This layer contains all database models and schema definitions.
Uses SQLAlchemy ORM for database abstraction.
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship, validates
from datetime import datetime, timezone

Base = declarative_base()


class Team(Base):
    """Represents a team in the organization"""

    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    members = relationship("TeamMember", back_populates="team", cascade="all, delete-orphan")
    communications = relationship(
        "Communication", back_populates="team", cascade="all, delete-orphan"
    )
    metrics = relationship("TeamMetrics", back_populates="team", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Team(id={self.id}, name='{self.name}')>"


class TeamMember(Base):
    """Represents a member of a team"""

    __tablename__ = "team_members"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    role = Column(String(50))
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    team = relationship("Team", back_populates="members")
    sent_communications = relationship(
        "Communication", foreign_keys="Communication.sender_id", back_populates="sender"
    )
    received_communications = relationship(
        "Communication", foreign_keys="Communication.receiver_id", back_populates="receiver"
    )

    def __repr__(self):
        return f"<TeamMember(id={self.id}, name='{self.name}', team_id={self.team_id})>"


class Communication(Base):
    """Represents a communication event between team members"""

    __tablename__ = "communications"

    # Add indexes for performance
    __table_args__ = (
        Index("idx_team_timestamp", "team_id", "timestamp"),
        Index("idx_sender_timestamp", "sender_id", "timestamp"),
        Index("idx_cross_team", "team_id", "is_cross_team", "timestamp"),
        Index("idx_communication_type", "communication_type"),
    )

    id = Column(Integer, primary_key=True)
    sender_id = Column(Integer, ForeignKey("team_members.id"), nullable=False)
    receiver_id = Column(Integer, ForeignKey("team_members.id"))
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    # Communication details
    communication_type = Column(String(50))  # face-to-face, email, chat, meeting, etc.
    duration_minutes = Column(Float)  # Duration of interaction
    is_group_communication = Column(Integer, default=0)  # 0=one-on-one, 1=group
    is_cross_team = Column(Integer, default=0)  # 0=internal, 1=external/exploration

    message_content = Column(Text)  # Optional: for analyzing content
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # NEW: Track external team if cross-team communication
    external_team_id = Column(Integer, ForeignKey("external_teams.id"))

    # NEW: Communication context/tags for better analysis
    context = Column(String(50))  # "standup", "1:1", "planning", "review", etc.

    # Relationships
    sender = relationship(
        "TeamMember", foreign_keys=[sender_id], back_populates="sent_communications"
    )
    receiver = relationship(
        "TeamMember", foreign_keys=[receiver_id], back_populates="received_communications"
    )
    team = relationship("Team", back_populates="communications")
    external_team = relationship("ExternalTeam")

    @validates("timestamp")
    def validate_timestamp(self, key, timestamp):
        """Ensure timestamp is not in the future"""
        if timestamp and timestamp > datetime.now(timezone.utc):
            raise ValueError("Cannot create communications with future timestamps")
        return timestamp

    @validates("duration_minutes")
    def validate_duration(self, key, duration):
        """Ensure duration is reasonable"""
        if duration is not None:
            if duration < 0:
                raise ValueError("Duration cannot be negative")
            if duration > 480:  # 8 hours
                raise ValueError("Duration cannot exceed 8 hours (480 minutes)")
        return duration

    def __repr__(self):
        return f"<Communication(id={self.id}, type='{self.communication_type}', timestamp={self.timestamp})>"


class TeamMetrics(Base):
    """Stores calculated Three E's metrics for teams"""

    __tablename__ = "team_metrics"

    # Add index for performance
    __table_args__ = (Index("idx_team_calculated", "team_id", "calculated_at"),)

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    # The Three E's
    energy_score = Column(Float)  # Magnitude of communication
    engagement_score = Column(Float)  # Degree of interaction and contribution
    exploration_score = Column(Float)  # Cross-team engagement

    # Overall performance indicator
    overall_score = Column(Float)

    # Metadata
    calculation_period_start = Column(DateTime)
    calculation_period_end = Column(DateTime)
    calculated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Additional metrics
    total_communications = Column(Integer)
    avg_response_time = Column(Float)
    participation_rate = Column(Float)

    # NEW: Store Gini coefficient for balance tracking
    gini_coefficient = Column(Float)

    # Relationships
    team = relationship("Team", back_populates="metrics")

    def __repr__(self):
        return f"<TeamMetrics(team_id={self.team_id}, energy={self.energy_score}, engagement={self.engagement_score}, exploration={self.exploration_score})>"


class ExternalTeam(Base):
    """Represents external teams for tracking exploration"""

    __tablename__ = "external_teams"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    department = Column(String(100))
    description = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<ExternalTeam(id={self.id}, name='{self.name}')>"


def init_db(database_url):
    """Initialize the database with the given URL"""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine
