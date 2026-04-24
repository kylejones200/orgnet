"""
Business Logic Layer (Layer 3)
==============================
Orchestrates Three E's scoring and network analysis.

Pure math lives in netsmith.ona — this layer owns ORM access and persistence.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from sqlalchemy.orm import Session

from netsmith.api.compute import communities as ns_communities
from netsmith.api.compute import degree as ns_degree
from netsmith.api.compute import pagerank as ns_pagerank
from netsmith.engine.contracts import EdgeList
from netsmith.ona import Communication as NsComm
from netsmith.ona import detect_silos, score_team
from netsmith.ona.three_es import gini_coefficient

from ..data_access.repositories import (
    CommunicationRepository,
    TeamMemberRepository,
    TeamMetricsRepository,
    TeamRepository,
)
from ..database.models import Communication, Team, TeamMember
from .netsmith_graph_helpers import (
    closeness_centrality_wf,
    edge_pairs_uint64_from_el,
    graph_is_connected,
    weighted_modularity,
)

logger = logging.getLogger(__name__)


# ── ORM → netsmith.ona adapter ────────────────────────────────────────────────


def _to_ns_comms(orm_comms: List[Communication]) -> List[NsComm]:
    """Convert SQLAlchemy Communication rows to netsmith.ona.Communication dataclasses."""
    return [
        NsComm(
            sender_id=c.sender_id,
            receiver_id=c.receiver_id,
            duration_minutes=c.duration_minutes or 0.0,
            comm_type=c.communication_type or "email",
            is_cross_team=bool(c.is_cross_team),
        )
        for c in orm_comms
    ]


# ── ThreeEsCalculator ─────────────────────────────────────────────────────────


class ThreeEsCalculator:
    """
    Coordinates Three E's metric calculation for a team.

    Fetches data via repositories, delegates scoring to netsmith.ona,
    and persists results back to the database.
    """

    def __init__(self, session: Session):
        self.session = session
        self.team_repo = TeamRepository(session)
        self.member_repo = TeamMemberRepository(session)
        self.comm_repo = CommunicationRepository(session)
        self.metrics_repo = TeamMetricsRepository(session)

    # ── Individual E calculators (kept for API compatibility) ─────────────────
    # Key names are mapped back to the original API contract so callers don't break.

    def calculate_energy(self, team_id: int, start_date: datetime, end_date: datetime) -> Dict:
        from netsmith.ona.three_es import energy_score as _energy

        members = self.member_repo.get_by_team(team_id)
        comms = self.comm_repo.get_by_team(team_id, start_date, end_date)
        days = max((end_date - start_date).days, 1)
        score, d = _energy(_to_ns_comms(comms), [m.id for m in members], days)
        return {
            "energy_score": score,
            "total_communications": d["total_comms"],
            "avg_communications_per_member": round(d["total_comms"] / max(len(members), 1), 2),
            "total_duration_minutes": d["total_duration_min"],
            "face_to_face_ratio": d["face_to_face_ratio"],
            "frequency_normalized": d.get("freq_normalized", 0.0),
            "duration_normalized": d.get("duration_normalized", 0.0),
        }

    def calculate_engagement(self, team_id: int, start_date: datetime, end_date: datetime) -> Dict:
        from netsmith.ona.three_es import engagement_score as _engagement

        members = self.member_repo.get_by_team(team_id)
        comms = self.comm_repo.get_by_team(team_id, start_date, end_date)
        score, d = _engagement(_to_ns_comms(comms), [m.id for m in members])
        return {
            "engagement_score": score,
            "participation_rate": d["participation_rate"],
            "balance_score": d["balance_score"],
            "two_way_communication_score": d["two_way_rate"],
            "gini_coefficient": d["gini"],
        }

    def calculate_exploration(self, team_id: int, start_date: datetime, end_date: datetime) -> Dict:
        from netsmith.ona.three_es import exploration_score as _exploration

        members = self.member_repo.get_by_team(team_id)
        comms = self.comm_repo.get_by_team(team_id, start_date, end_date)
        score, d = _exploration(_to_ns_comms(comms), [m.id for m in members])
        return {
            "exploration_score": score,
            "cross_team_communications": d["cross_team_count"],
            "exploration_ratio": d["exploration_ratio"],
            "members_exploring": d["explorers"],
            "member_exploration_rate": d["explorer_rate"],
        }

    def calculate_overall_performance(
        self, energy: float, engagement: float, exploration: float
    ) -> float:
        from netsmith.ona.three_es import overall_score

        return overall_score(energy, engagement, exploration)

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        return gini_coefficient(values)

    # ── Main entry point ──────────────────────────────────────────────────────

    def calculate_all_metrics(
        self,
        team_id: int,
        start_date: datetime = None,
        end_date: datetime = None,
        save_to_db: bool = True,
    ) -> Dict:
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        members = self.member_repo.get_by_team(team_id)
        comms = self.comm_repo.get_by_team(team_id, start_date, end_date)
        days = max((end_date - start_date).days, 1)

        result = score_team(_to_ns_comms(comms), [m.id for m in members], days)

        if save_to_db:
            stats = self.comm_repo.get_communication_stats(team_id, start_date, end_date)
            self.metrics_repo.create(
                team_id=team_id,
                energy_score=result.energy,
                engagement_score=result.engagement,
                exploration_score=result.exploration,
                overall_score=result.overall,
                calculation_period_start=start_date,
                calculation_period_end=end_date,
                total_communications=stats["total_communications"],
                participation_rate=result.detail["engagement"].get("participation_rate", 0.0),
                gini_coefficient=result.detail["engagement"].get("gini", 0.0),
            )
            logger.info(f"Saved metrics for team {team_id}: overall={result.overall:.2f}")

        return {
            "team_id": team_id,
            "calculation_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "energy": {"energy_score": result.energy, **result.detail["energy"]},
            "engagement": {"engagement_score": result.engagement, **result.detail["engagement"]},
            "exploration": {
                "exploration_score": result.exploration,
                **result.detail["exploration"],
            },
            "overall_score": result.overall,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }


# ── NetworkAnalyzer ───────────────────────────────────────────────────────────
# Uses netsmith for degree, PageRank, Louvain, connectivity, WF closeness, and modularity.
# NetworkX retained only for betweenness/eigenvector (not yet in netsmith_rs / ona stack).


class NetworkAnalyzer:
    """Graph topology analysis powered by netsmith (and netsmith_rs) plus NetworkX."""

    def __init__(self, session: Session):
        self.session = session
        self.comm_repo = CommunicationRepository(session)
        self.member_repo = TeamMemberRepository(session)

    # ── Internal builders ─────────────────────────────────────────────────────

    def _load_data(
        self, team_id: int, start_date: datetime, end_date: datetime
    ) -> tuple[list, list, dict[int, dict]]:
        """Return (members, communications, node_meta) for a team + window."""
        members = self.member_repo.get_by_team(team_id)
        comms = self.comm_repo.get_by_team(team_id, start_date, end_date)
        node_meta = {m.id: {"name": m.name, "role": m.role} for m in members}
        return members, comms, node_meta

    def _to_edge_list(
        self, members: list, comms: list
    ) -> tuple[EdgeList | None, dict[int, int], dict[int, int]]:
        """
        Build a netsmith EdgeList from ORM objects.

        Returns (EdgeList | None, id_to_idx, idx_to_id).
        None when there are no edges.
        """
        member_set = {m.id for m in members}
        id_to_idx = {mid: i for i, mid in enumerate(sorted(member_set))}
        idx_to_id = {i: mid for mid, i in id_to_idx.items()}

        edge_weights: dict[tuple[int, int], int] = {}
        for c in comms:
            if c.receiver_id and c.receiver_id in member_set:
                a, b = id_to_idx[c.sender_id], id_to_idx[c.receiver_id]
                key = (min(a, b), max(a, b))  # canonical undirected key
                edge_weights[key] = edge_weights.get(key, 0) + 1

        if not edge_weights:
            return None, id_to_idx, idx_to_id

        u = np.array([k[0] for k in edge_weights], dtype=np.int64)
        v = np.array([k[1] for k in edge_weights], dtype=np.int64)
        w = np.array(list(edge_weights.values()), dtype=np.float64)
        return EdgeList(u=u, v=v, w=w, directed=False, n_nodes=len(members)), id_to_idx, idx_to_id

    def _to_nx(self, members: list, comms: list) -> nx.Graph:
        """Build a NetworkX graph (used only for betweenness/eigenvector)."""
        member_set = {m.id for m in members}
        G = nx.Graph()
        for m in members:
            G.add_node(m.id, name=m.name, role=m.role)
        for c in comms:
            if c.receiver_id and c.receiver_id in member_set:
                if G.has_edge(c.sender_id, c.receiver_id):
                    G[c.sender_id][c.receiver_id]["weight"] += 1
                else:
                    G.add_edge(c.sender_id, c.receiver_id, weight=1)
        return G

    def _louvain_labels_nx_fallback(
        self, members: list, comms: list, id_to_idx: dict[int, int]
    ) -> np.ndarray:
        """Louvain partition as int labels per graph index (0 .. n-1), via NetworkX."""
        from networkx.algorithms.community import louvain_communities

        G = self._to_nx(members, comms)
        n = len(id_to_idx)
        if G.number_of_edges() == 0:
            return np.zeros(n, dtype=np.int64)
        partition = louvain_communities(G, weight="weight", seed=0)
        mid_to_c: dict[int, int] = {}
        for ci, block in enumerate(partition):
            for node in block:
                mid_to_c[int(node)] = ci
        out = np.zeros(n, dtype=np.int64)
        for mid, idx in id_to_idx.items():
            out[idx] = mid_to_c.get(mid, 0)
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    def build_communication_network(
        self, team_id: int, start_date: datetime = None, end_date: datetime = None
    ) -> nx.Graph:
        """Return a NetworkX graph of member communications (legacy public API)."""
        members, comms, _ = self._load_data(team_id, start_date, end_date)
        return self._to_nx(members, comms)

    def analyze_network_metrics(
        self, team_id: int, start_date: datetime = None, end_date: datetime = None
    ) -> Dict:
        members, comms, node_meta = self._load_data(team_id, start_date, end_date)
        if not members:
            return {"error": "No data available for analysis"}

        el, id_to_idx, idx_to_id = self._to_edge_list(members, comms)
        n_nodes = len(members)
        n_edges = len(el.u) if el is not None else 0
        density = round((2 * n_edges) / max(n_nodes * (n_nodes - 1), 1), 3)

        if el is None:
            return {
                "density": density,
                "num_nodes": n_nodes,
                "num_edges": 0,
                "note": "No edges — members have not communicated",
            }

        # netsmith: degree sequence
        degrees = ns_degree(el, backend="python")
        edges_u64 = edge_pairs_uint64_from_el(el)
        is_connected = graph_is_connected(n_nodes, edges_u64)

        # NetworkX: betweenness (not yet in netsmith public API)
        G = self._to_nx(members, comms)
        try:
            betweenness = nx.betweenness_centrality(G)
            bv = np.array(list(betweenness.values()))
            most_central = max(betweenness.items(), key=lambda x: x[1])
            bottlenecks = [n for n, c in betweenness.items() if c > bv.mean() + bv.std()]
            return {
                "density": density,
                "is_connected": is_connected,
                "num_nodes": n_nodes,
                "num_edges": n_edges,
                "avg_degree": round(float(degrees.mean()), 2),
                "most_central_member_id": most_central[0],
                "centrality_score": round(most_central[1], 3),
                "potential_bottlenecks": bottlenecks,
                "avg_betweenness": round(float(bv.mean()), 3),
            }
        except (nx.NetworkXError, ValueError) as e:
            return {
                "density": density,
                "is_connected": is_connected,
                "num_nodes": n_nodes,
                "num_edges": n_edges,
                "avg_degree": round(float(degrees.mean()), 2),
                "note": f"Network too sparse for full analysis: {e}",
            }

    def detect_communities(
        self, team_id: int, start_date: datetime = None, end_date: datetime = None
    ) -> Dict:
        members, comms, node_meta = self._load_data(team_id, start_date, end_date)
        if len(members) < 3:
            return {"error": "Need at least 3 members for community detection"}

        el, id_to_idx, idx_to_id = self._to_edge_list(members, comms)
        if el is None:
            return {"error": "No edges — members have not communicated"}

        # netsmith: Louvain (Rust when available). PyPI netsmith_rs wheels may omit
        # louvain_partition_rust while src netsmith still calls it — fall back to NetworkX.
        try:
            community_ids = np.asarray(
                ns_communities(el, method="louvain", backend="python"), dtype=np.int64
            )
        except (AttributeError, RuntimeError):
            community_ids = self._louvain_labels_nx_fallback(members, comms, id_to_idx)

        # Group idx → community
        groups: dict[int, list[int]] = {}
        for idx, cid in enumerate(community_ids):
            groups.setdefault(int(cid), []).append(idx_to_id[idx])

        community_list = [
            {
                "community_id": i + 1,
                "member_ids": member_ids,
                "member_names": [node_meta[mid]["name"] for mid in member_ids],
                "size": len(member_ids),
            }
            for i, member_ids in enumerate(groups.values())
        ]

        comm_arr = np.asarray(community_ids, dtype=np.int64)
        mod = round(
            weighted_modularity(
                int(el.n_nodes),
                np.asarray(el.u, dtype=np.int64),
                np.asarray(el.v, dtype=np.int64),
                np.asarray(el.w, dtype=np.float64),
                comm_arr,
            ),
            3,
        )

        return {
            "num_communities": len(community_list),
            "communities": community_list,
            "modularity": mod,
            "is_siloed": mod > 0.4,
            "interpretation": self._interpret_communities(len(community_list), mod),
        }

    def calculate_advanced_centrality(
        self, team_id: int, start_date: datetime = None, end_date: datetime = None
    ) -> Dict:
        members, comms, node_meta = self._load_data(team_id, start_date, end_date)
        if not members:
            return {"error": "No data available"}

        el, id_to_idx, idx_to_id = self._to_edge_list(members, comms)
        if el is None:
            return {"error": "No edges — members have not communicated"}

        # netsmith: degree + PageRank (vectorised, Rust-accelerated when available)
        degrees = ns_degree(el, backend="python")
        pr_scores = ns_pagerank(el, backend="python")

        n = len(members)
        max_degree = max(n - 1, 1)
        degree_cent = {idx_to_id[i]: round(float(degrees[i]) / max_degree, 3) for i in range(n)}
        pagerank_cent = {idx_to_id[i]: round(float(pr_scores[i]), 4) for i in range(n)}

        edges_u64 = edge_pairs_uint64_from_el(el)
        closeness_arr = closeness_centrality_wf(n, edges_u64)
        closeness_cent = {idx_to_id[i]: round(float(closeness_arr[i]), 3) for i in range(n)}

        # NetworkX: betweenness + eigenvector (not yet in netsmith_rs)
        G = self._to_nx(members, comms)
        try:
            betweenness_cent = nx.betweenness_centrality(G)
            try:
                eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                eigenvector_cent = {nid: 0.0 for nid in G.nodes()}
        except Exception:
            betweenness_cent = eigenvector_cent = {m.id: 0.0 for m in members}

        top_n = min(3, n)

        def member_info(mid: int) -> dict:
            return {
                "member_id": mid,
                **node_meta.get(mid, {"name": f"Member {mid}", "role": "Unknown"}),
            }

        return {
            "centrality_metrics": {
                "degree": degree_cent,
                "pagerank": pagerank_cent,
                "betweenness": {k: round(v, 3) for k, v in betweenness_cent.items()},
                "closeness": {k: round(v, 3) for k, v in closeness_cent.items()},
                "eigenvector": {k: round(v, 3) for k, v in eigenvector_cent.items()},
            },
            "key_roles": {
                "connectors": [
                    {**member_info(k), "score": round(v, 3)}
                    for k, v in sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[
                        :top_n
                    ]
                ],
                "influencers": [
                    {**member_info(k), "score": round(v, 4)}
                    for k, v in sorted(pagerank_cent.items(), key=lambda x: x[1], reverse=True)[
                        :top_n
                    ]
                ],
                "hubs": [
                    {**member_info(k), "score": round(v, 3)}
                    for k, v in sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[
                        :top_n
                    ]
                ],
            },
            "insights": self._centrality_insights(degree_cent, betweenness_cent, pagerank_cent),
        }

    # ── Interpretation helpers ────────────────────────────────────────────────

    def _interpret_communities(self, num_communities: int, modularity: float) -> str:
        if num_communities == 1:
            return "Team is well-integrated with no distinct sub-groups."
        interpretations = [
            (
                modularity > 0.5,
                f"Warning: {num_communities} distinct silos (modularity {modularity:.2f}). Consider cross-group activities.",
            ),
            (
                modularity > 0.3,
                f"Team has {num_communities} sub-groups with good cross-communication (modularity {modularity:.2f}).",
            ),
            (
                True,
                f"Team has {num_communities} informal groupings with excellent cross-communication (modularity {modularity:.2f}).",
            ),
        ]
        return next(msg for cond, msg in interpretations if cond)

    def _centrality_insights(
        self, degree_cent: Dict, betweenness_cent: Dict, pagerank_cent: Dict
    ) -> List[str]:
        insights = []
        if betweenness_cent and max(betweenness_cent.values(), default=0) > 0.5:
            insights.append(
                "Warning: Network is highly centralized — consider distributing communication responsibilities"
            )
        isolated = sum(1 for v in degree_cent.values() if v == 0)
        if isolated:
            insights.append(f"Warning: {isolated} member(s) are isolated with no connections")
        bv = np.array(list(betweenness_cent.values())) if betweenness_cent else np.array([])
        if bv.size > 3 and bv.mean() > 0 and bv.std() / bv.mean() < 0.5:
            insights.append(
                "Positive: Communication responsibility is well-distributed across team"
            )
        # PageRank concentration check
        pr = np.array(list(pagerank_cent.values())) if pagerank_cent else np.array([])
        if pr.size > 1 and float(pr.max()) > 3.0 / pr.size:
            insights.append(
                "Warning: Influence is concentrated — one or few members dominate information flow"
            )
        return insights or ["Network structure appears healthy"]
