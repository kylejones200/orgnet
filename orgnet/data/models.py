"""Data models for organizational network analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class InteractionType(Enum):
    """Types of interactions between people."""

    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    MEETING = "meeting"
    DOCUMENT = "document"
    CODE = "code"
    CALENDAR = "calendar"


@dataclass
class Person:
    """Represents a person in the organization."""

    id: str
    name: str
    email: str
    department: Optional[str] = None
    role: Optional[str] = None
    manager_id: Optional[str] = None
    team: Optional[str] = None
    location: Optional[str] = None
    timezone: Optional[str] = None
    tenure_days: Optional[int] = None
    job_level: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Interaction:
    """Represents an interaction between people."""

    id: str
    source_id: str  # Person ID
    target_id: str  # Person ID
    interaction_type: InteractionType
    timestamp: datetime
    channel: Optional[str] = None
    thread_id: Optional[str] = None
    response_time_seconds: Optional[float] = None
    is_reciprocal: bool = False
    content: Optional[str] = None  # May be None for privacy
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Represents a document and its collaboration."""

    id: str
    title: str
    author_ids: List[str]
    editor_ids: List[str]
    created_at: datetime
    last_modified: datetime
    document_type: Optional[str] = None
    platform: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Meeting:
    """Represents a meeting."""

    id: str
    organizer_id: str
    attendee_ids: List[str]
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    is_recurring: bool = False
    meeting_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeCommit:
    """Represents code collaboration."""

    id: str
    author_id: str
    repository: str
    file_paths: List[str]
    timestamp: datetime
    reviewer_ids: List[str] = field(default_factory=list)
    is_merge: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HRISRecord:
    """Represents HRIS data for a person."""

    person_id: str
    department: str
    role: str
    manager_id: Optional[str]
    team: str
    start_date: datetime
    location: Optional[str] = None
    job_level: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
