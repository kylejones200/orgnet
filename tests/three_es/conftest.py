"""
Pytest fixtures and configuration for tests.
"""

import os
import sys
from pathlib import Path

# NetSmith is a separate repo; PyPI wheels omit the Python ONA tree. Prepend ``src/`` when:
# - NETSMITH_ROOT points at a local checkout, or
# - a sibling ``../netsmith`` exists next to this repo (e.g. ~/orgnet and ~/netsmith).
_orgnet_root = Path(__file__).resolve().parents[2]
_netsmith_candidates = []
_env = os.environ.get("NETSMITH_ROOT")
if _env:
    _netsmith_candidates.append(Path(_env) / "src")
_netsmith_candidates.append(_orgnet_root.parent / "netsmith" / "src")
for _p in _netsmith_candidates:
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from orgnet.three_es.database.models import Base
from orgnet.three_es.data_access.repositories import (
    TeamRepository,
    TeamMemberRepository,
    CommunicationRepository,
)


@pytest.fixture
def engine():
    """Create a test database engine"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create a test database session"""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def team_repo(session):
    """Team repository fixture"""
    return TeamRepository(session)


@pytest.fixture
def member_repo(session):
    """Member repository fixture"""
    return TeamMemberRepository(session)


@pytest.fixture
def comm_repo(session):
    """Communication repository fixture"""
    return CommunicationRepository(session)


@pytest.fixture
def sample_team(team_repo, member_repo):
    """Create a sample team with 5 members"""
    team = team_repo.create("Test Team", "A test team")

    members = []
    for i in range(5):
        member = member_repo.create(
            name=f"Member {i}", email=f"member{i}@test.com", team_id=team.id, role="Engineer"
        )
        members.append(member)

    return team, members


@pytest.fixture
def now():
    """Current datetime fixture"""
    return datetime.now(timezone.utc)


@pytest.fixture
def thirty_days_ago(now):
    """30 days ago datetime fixture"""
    return now - timedelta(days=30)


@pytest.fixture
def mid_date(thirty_days_ago):
    """Timestamp in the middle of the 30-day range (for comm creation in range)"""
    return thirty_days_ago + timedelta(days=15)
