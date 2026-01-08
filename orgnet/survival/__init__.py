"""Survival analysis for organizational networks.

This module provides survival analysis capabilities, integrating orgnet metrics
and ts2net features to predict events like employee exit, team breakup, etc.
"""

from orgnet.survival.discrete import (
    DiscreteTimeSurvivalModel,
    build_person_period_table,
    fit_discrete_time_model,
    attach_survival_outputs,
)
from orgnet.survival.models import (
    build_person_time_table,
    fit_survival_model,
    score_hazard,
    compute_hazard_flags,
)

__all__ = [
    # Discrete time models
    "DiscreteTimeSurvivalModel",
    "build_person_period_table",
    "fit_discrete_time_model",
    "attach_survival_outputs",
    # Main interface
    "build_person_time_table",
    "fit_survival_model",
    "score_hazard",
    "compute_hazard_flags",
]
