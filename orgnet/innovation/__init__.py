"""Innovation and knowledge flow analytics.

This package provides innovation analytics and internal talent marketplace:
- InnovationIndex: Productizes bridging/bonding analysis as innovation score
- ExpertiseInferenceEngine: Internal Talent Marketplace for finding hidden experts
- StructuralHoleMonetizer: ROI calculator for structural holes
- InnovationBehaviorAnalyzer: Behavioral indicators of innovation
- CollaborationDepthAnalyzer: Depth and velocity of cross-functional collaboration
- LearningVelocityAnalyzer: Learning behavior and knowledge growth velocity
- PerspectiveDiversityAnalyzer: Diversity of perspectives and inclusion
- InnovationPerformanceScorer: Performance-review-ready innovation scores
"""

try:
    from orgnet.innovation.behavior import InnovationBehaviorAnalyzer
    from orgnet.innovation.collaboration_depth import CollaborationDepthAnalyzer
    from orgnet.innovation.diversity import PerspectiveDiversityAnalyzer
    from orgnet.innovation.expertise import ExpertiseInferenceEngine
    from orgnet.innovation.index import InnovationIndex
    from orgnet.innovation.learning import LearningVelocityAnalyzer
    from orgnet.innovation.performance_scorer import InnovationPerformanceScorer
    from orgnet.innovation.structural_holes import StructuralHoleMonetizer

    __all__ = [
        "InnovationIndex",
        "ExpertiseInferenceEngine",
        "StructuralHoleMonetizer",
        "InnovationBehaviorAnalyzer",
        "CollaborationDepthAnalyzer",
        "LearningVelocityAnalyzer",
        "PerspectiveDiversityAnalyzer",
        "InnovationPerformanceScorer",
    ]
except ImportError:
    __all__ = []
