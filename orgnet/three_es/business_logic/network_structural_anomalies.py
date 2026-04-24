"""
Structural (graph) anomaly detection on team communications using anomsmith.

Call :func:`detect_team_network_structural_anomalies` from your own HTTP layer,
batch job, or notebook. It mirrors ``NetworkAnalyzer`` aggregation (dyadic rows
with ``receiver_id``) and enriches optional NetworkX centralities when installed.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from anomsmith.primitives.thresholding import ThresholdRule
from anomsmith.workflows.network import (
    aggregate_undirected_edges,
    detect_network_node_anomalies,
    node_features_from_edges,
)

from ..data_access.repositories import CommunicationRepository, TeamMemberRepository


def _finite_float(val: Any) -> Any:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return None
    if hasattr(val, "item"):
        try:
            x = float(val.item())
        except (TypeError, ValueError):
            return None
        return x if math.isfinite(x) else None
    if isinstance(val, (float, int)):
        x = float(val)
        return x if math.isfinite(x) else None
    return val


def detect_team_network_structural_anomalies(
    session: Session,
    team_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    *,
    quantile: float = 0.9,
    contamination: float = 0.1,
    include_networkx_metrics: bool = True,
    random_state: Optional[int] = 0,
) -> Dict[str, Any]:
    """
    Flag members whose graph-derived features are unusual vs the team.

    Loads communications in ``[start_date, end_date]``, aggregates undirected
    edges, builds structural + optional NetworkX features, and runs anomsmith's
    isolation-forest node detector.

    Args:
        session: SQLAlchemy session.
        team_id: Team primary key.
        start_date: Inclusive window start (default: 30 days before ``end_date``).
        end_date: Inclusive window end (default: UTC now).
        quantile: Score quantile for ``ThresholdRule`` (higher = fewer flags).
        contamination: Isolation forest contamination prior.
        include_networkx_metrics: If True, merge betweenness / closeness /
            eigenvector columns when ``networkx`` is installed (via anomsmith's
            ``[network]`` extra or this package's dependency).
        random_state: RNG seed for the forest.

    Returns:
        JSON-serializable dict with ``status`` ``ok`` or ``skipped``, member-level
        ``members`` rows (scores, flags, feature columns), and window metadata.
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(days=30)

    member_repo = TeamMemberRepository(session)
    comm_repo = CommunicationRepository(session)
    members = member_repo.get_by_team(team_id)
    comms = comm_repo.get_by_team(team_id, start_date, end_date)

    member_ids = [m.id for m in members]
    names = {m.id: m.name for m in members}

    if len(member_ids) < 2:
        return {
            "team_id": team_id,
            "status": "skipped",
            "reason": "need at least 2 team members",
            "n_members": len(member_ids),
            "n_communication_rows": 0,
            "n_edges": 0,
            "n_flagged": 0,
            "members": [],
        }

    rows: List[Dict[str, Any]] = []
    for c in comms:
        rid = getattr(c, "receiver_id", None)
        if rid is None:
            continue
        rows.append({"sender_id": c.sender_id, "receiver_id": rid})

    df = pd.DataFrame(rows)
    if df.empty:
        edges = pd.DataFrame(columns=["u", "v", "weight"])
    else:
        edges = aggregate_undirected_edges(df)

    features = node_features_from_edges(edges, member_ids)
    nx_status = "disabled"
    nx_note: Optional[str] = None
    if include_networkx_metrics:
        try:
            from anomsmith.workflows.network import (  # noqa: E402
                node_graph_metrics_networkx,
            )

            nx_df = node_graph_metrics_networkx(edges, member_ids)
            features = pd.concat([features, nx_df], axis=1)
            nx_status = "included"
        except ImportError as exc:
            nx_status = "skipped"
            nx_note = str(exc)

    rule = ThresholdRule(method="quantile", value=0.0, quantile=float(quantile))
    scored = detect_network_node_anomalies(
        features,
        rule,
        contamination=float(contamination),
        random_state=random_state,
    )

    details: List[Dict[str, Any]] = []
    for mid in scored.index:
        row = scored.loc[mid]
        entry: Dict[str, Any] = {
            "member_id": int(mid),
            "member_name": names.get(int(mid)),
            "flagged": bool(int(row["flag"]) == 1),
            "score": float(row["score"]),
        }
        for col in scored.columns:
            if col in ("score", "flag"):
                continue
            entry[col] = _finite_float(row[col])
        details.append(entry)

    return {
        "team_id": team_id,
        "status": "ok",
        "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "n_members": len(member_ids),
        "n_communication_rows": len(rows),
        "n_edges": int(len(edges)),
        "quantile": float(quantile),
        "n_flagged": int(scored["flag"].sum()),
        "networkx_metrics": nx_status,
        "networkx_note": nx_note,
        "members": details,
    }
