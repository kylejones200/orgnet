"""Efficiency and deduplication modules for COO-focused analytics.

This package provides tools for identifying redundant work and facilitating
cross-pollination, helping COOs improve organizational efficiency and innovation.
"""

from orgnet.efficiency.cross_pollination import CrossPollinationEngine
from orgnet.efficiency.deduplication_roi import DeduplicationROICalculator
from orgnet.efficiency.knowledge_graph import KnowledgeGraph
from orgnet.efficiency.redundancy_detection import RedundancyDetector

__all__ = [
    "RedundancyDetector",
    "CrossPollinationEngine",
    "DeduplicationROICalculator",
    "KnowledgeGraph",
]
