"""
Example usage: in-process library calls (no HTTP server).

Run: pip install -e ".[three-es]" && python scripts/three_es_examples.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_orgnet_root = Path(__file__).resolve().parents[1]
_netsmith_candidates: list[Path] = []
if os.environ.get("NETSMITH_ROOT"):
    _netsmith_candidates.append(Path(os.environ["NETSMITH_ROOT"]) / "src")
_netsmith_candidates.append(_orgnet_root.parent / "netsmith" / "src")
for _p in _netsmith_candidates:
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orgnet.three_es.business_logic import NetworkAnalyzer, ThreeEsCalculator
from orgnet.three_es.data_access.repositories import (
    CommunicationRepository,
    TeamMemberRepository,
    TeamRepository,
)
from orgnet.three_es.database.models import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_examples() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        team_repo = TeamRepository(session)
        member_repo = TeamMemberRepository(session)
        comm_repo = CommunicationRepository(session)

        team = team_repo.create("Innovation Lab", "Cross-functional innovation team")
        members_data = [
            ("Sarah Chen", "sarah.chen@company.com", "Team Lead"),
            ("Mike Johnson", "mike.johnson@company.com", "Engineer"),
            ("Lisa Park", "lisa.park@company.com", "Designer"),
            ("David Kim", "david.kim@company.com", "Product Manager"),
        ]
        members = []
        for name, email, role in members_data:
            members.append(member_repo.create(name=name, email=email, team_id=team.id, role=role))

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        mid = start + timedelta(days=15)

        comm_repo.create(
            sender_id=members[0].id,
            receiver_id=members[1].id,
            team_id=team.id,
            communication_type="face-to-face",
            duration_minutes=20.0,
            timestamp=mid,
        )
        comm_repo.create(
            sender_id=members[0].id,
            receiver_id=None,
            team_id=team.id,
            communication_type="meeting",
            duration_minutes=60.0,
            is_group_communication=1,
            timestamp=mid,
        )
        comm_repo.create(
            sender_id=members[2].id,
            receiver_id=None,
            team_id=team.id,
            communication_type="face-to-face",
            duration_minutes=30.0,
            is_cross_team=1,
            timestamp=mid,
        )
        session.commit()

        calc = ThreeEsCalculator(session)
        metrics = calc.calculate_all_metrics(team.id, start, end, save_to_db=True)
        logger.info("Metrics for team %s:", team.id)
        logger.info("  Energy: %.2f", metrics["energy"]["energy_score"])
        logger.info("  Engagement: %.2f", metrics["engagement"]["engagement_score"])
        logger.info("  Exploration: %.2f", metrics["exploration"]["exploration_score"])
        logger.info("  Overall: %.2f", metrics["overall_score"])

        analyzer = NetworkAnalyzer(session)
        network = analyzer.analyze_network_metrics(team.id, start, end)
        logger.info(
            "Network: density=%.3f nodes=%s edges=%s",
            network.get("density", 0),
            network.get("num_nodes"),
            network.get("num_edges"),
        )
        logger.info("Examples completed successfully.")
    finally:
        session.close()


if __name__ == "__main__":
    run_examples()
