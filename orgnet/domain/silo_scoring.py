"""Silo risk classification and ramification text.

Single source of truth for the thresholds and descriptions used when
scoring communication silos.  Application layer computes raw metrics;
this module translates them into risk labels and human-readable copy.
"""

from __future__ import annotations

# ── Thresholds ────────────────────────────────────────────────────────────────

RISK_HIGH_INTERNAL_RATIO = 0.90
RISK_MEDIUM_INTERNAL_RATIO = 0.78
RISK_LOW_EXTERNAL_PER_NODE = 0.35
RISK_MIN_MEMBERS_FOR_STRUCTURAL = 4

SOCIAL_ISOLATION_THRESHOLD = 0.75
OPERATIONAL_LOW_EXTERNAL_THRESHOLD = 0.50
TECHNICAL_HIGH_DENSITY_THRESHOLD = 0.25


# ── Classification ─────────────────────────────────────────────────────────────


def classify_silo_risk(
    pct_internal_w: float,
    ext_per_node: float,
    member_count: int,
) -> str:
    """Return 'high', 'medium', or 'low' silo risk label."""
    large_enough = member_count >= RISK_MIN_MEMBERS_FOR_STRUCTURAL
    if pct_internal_w >= RISK_HIGH_INTERNAL_RATIO and large_enough:
        return "high"
    if pct_internal_w >= RISK_MEDIUM_INTERNAL_RATIO or (
        ext_per_node < RISK_LOW_EXTERNAL_PER_NODE and large_enough
    ):
        return "medium"
    return "low"


# ── Ramification text ─────────────────────────────────────────────────────────


def describe_social_risk(pct_internal_w: float) -> str:
    """One-sentence social-risk framing for the silo card."""
    tail = (
        "That pattern elevates isolation risk."
        if pct_internal_w >= SOCIAL_ISOLATION_THRESHOLD
        else "Some outward reach remains."
    )
    return f"{pct_internal_w:.0%} of this cluster's weighted ties stay inside the group. {tail}"


def describe_operational_risk(ext_per_node: float, member_count: int) -> str:
    """One-sentence operational-risk framing for the silo card."""
    if ext_per_node < OPERATIONAL_LOW_EXTERNAL_THRESHOLD and member_count > 2:
        return (
            "Low cross-group surface area — duplicated initiatives are harder to spot "
            "without bridges."
        )
    return "Moderate outward collaboration; watch for theme overlap with distant clusters."


def describe_technical_risk(internal_density: float) -> str:
    """One-sentence technical-risk framing for the silo card."""
    if internal_density > TECHNICAL_HIGH_DENSITY_THRESHOLD:
        return (
            f"Subgraph density {internal_density:.2f} — parallel workstreams are plausible "
            "if bridges stay thin."
        )
    return "Looser in-cluster mesh; cohesion is moderate rather than clique-like."
