"""Cross-modal validation of insights."""

import pandas as pd
from enum import Enum
from typing import Dict, List, Optional

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for validated findings."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CrossModalValidator:
    """Validates insights across multiple data sources (modalities)."""

    def __init__(self, graph_layers: Optional[Dict[str, any]] = None):
        """
        Initialize cross-modal validator.

        Args:
            graph_layers: Dictionary mapping layer name to graph
        """
        self.graph_layers = graph_layers or {}

    def validate_finding(self, finding: str, evidence_by_modality: Dict[str, bool]) -> Dict:
        """
        Validate a finding across multiple modalities.

        Args:
            finding: Description of the finding
            evidence_by_modality: Dictionary mapping modality name to evidence (True/False)

        Returns:
            Dictionary with validation results
        """
        modalities = list(evidence_by_modality.keys())
        evidence_count = sum(1 for v in evidence_by_modality.values() if v)
        total_modalities = len(modalities)

        agreement_ratio = evidence_count / total_modalities if total_modalities > 0 else 0

        # Determine confidence using threshold-based mapping
        confidence_thresholds = [(0.75, ConfidenceLevel.HIGH), (0.5, ConfidenceLevel.MEDIUM)]

        confidence = ConfidenceLevel.LOW  # Default
        for threshold, level in confidence_thresholds:
            if agreement_ratio >= threshold:
                confidence = level
                break

        return {
            "finding": finding,
            "confidence": confidence.value,
            "agreement_ratio": agreement_ratio,
            "evidence_count": evidence_count,
            "total_modalities": total_modalities,
            "evidence_by_modality": evidence_by_modality,
        }

    def validate_silo_detection(
        self, department1: str, department2: str, layer_graphs: Dict[str, any]
    ) -> Dict:
        """
        Validate silo detection across multiple layers.

        Args:
            department1: First department
            department2: Second department
            layer_graphs: Dictionary mapping layer name to graph

        Returns:
            Validation result
        """
        evidence = {}

        for layer_name, graph in layer_graphs.items():
            # Count cross-boundary edges using vectorized approach
            edges_list = list(graph.edges())
            total_edges = len(edges_list)

            if total_edges == 0:
                evidence[layer_name] = True  # No edges = silo
                continue

            cross_edges = sum(
                1
                for u, v in edges_list
                if u in graph.nodes()
                and v in graph.nodes()
                and (
                    (
                        graph.nodes[u].get("department", "") == department1
                        and graph.nodes[v].get("department", "") == department2
                    )
                    or (
                        graph.nodes[u].get("department", "") == department2
                        and graph.nodes[v].get("department", "") == department1
                    )
                )
            )

            # Low cross-boundary volume indicates silo
            cross_boundary_ratio = cross_edges / total_edges
            evidence[layer_name] = cross_boundary_ratio < 0.05  # Less than 5% cross-boundary

        finding = f"Silo detected between {department1} and {department2}"
        return self.validate_finding(finding, evidence)

    def validate_collaboration_strength(
        self, node1: str, node2: str, layer_graphs: Dict[str, any]
    ) -> Dict:
        """
        Validate collaboration strength across layers.

        Args:
            node1: First node
            node2: Second node
            layer_graphs: Dictionary mapping layer name to graph

        Returns:
            Validation result
        """
        evidence = {}

        for layer_name, graph in layer_graphs.items():
            if graph.has_edge(node1, node2):
                weight = graph[node1][node2].get("weight", 1.0)
                # Normalize by comparing to average edge weight (vectorized)
                edges_list = list(graph.edges())
                num_edges = len(edges_list)
                avg_weight = (
                    sum(graph[u][v].get("weight", 1.0) for u, v in edges_list) / num_edges
                    if num_edges > 0
                    else 1.0
                )

                evidence[layer_name] = weight > avg_weight * 1.5  # 50% above average
            else:
                evidence[layer_name] = False

        finding = f"Strong collaboration between {node1} and {node2}"
        return self.validate_finding(finding, evidence)

    def generate_validation_report(self, findings: List[Dict]) -> pd.DataFrame:
        """
        Generate validation report for multiple findings.

        Args:
            findings: List of validation result dictionaries

        Returns:
            DataFrame with validation report
        """
        return pd.DataFrame(findings)
