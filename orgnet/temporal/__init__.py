"""Temporal analysis modules."""

from orgnet.temporal.change_detection import ChangeDetector
from orgnet.temporal.onboarding import OnboardingAnalyzer
from orgnet.temporal.new_hire_integration import NewHireIntegrationTracker
from orgnet.temporal.cross_team_density import CrossTeamDensityTracker

try:
    from orgnet.temporal.ts2net import (
        Ts2NetConfig,
        features_for_series,
        edge_temporal_features,
        person_temporal_features,
        merge_edge_temporal_features,
    )

    __all__ = [
        "ChangeDetector",
        "OnboardingAnalyzer",
        "NewHireIntegrationTracker",
        "CrossTeamDensityTracker",
        "Ts2NetConfig",
        "features_for_series",
        "edge_temporal_features",
        "person_temporal_features",
        "merge_edge_temporal_features",
    ]
except ImportError:
    __all__ = [
        "ChangeDetector",
        "OnboardingAnalyzer",
        "NewHireIntegrationTracker",
        "CrossTeamDensityTracker",
    ]
