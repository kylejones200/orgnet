"""Multi-Modal Organizational Health Score.

This module creates a unified "Organizational Health Score" by fusing signals
from multiple data sources (Slack, email, GitHub, calendar, code commits).
This is a major differentiator: high Slack activity + low code commits =
high meeting/coordination overhead.
"""

from datetime import datetime, timedelta

import networkx as nx
import pandas as pd

from orgnet.utils.logging import get_logger
from orgnet.visualization.dashboards import DashboardGenerator

logger = get_logger(__name__)


class MultiModalHealthScore:
    """Multi-Modal Organizational Health Score.

    Fuses "digital exhaust" from Slack, Teams, email, calendar, and code commits
    into a unified "Organizational Health Score" that weighs signals from different
    sources to provide comprehensive organizational insights.
    """

    def __init__(
        self,
        graph: nx.Graph,
        email_data: pd.DataFrame | None = None,
        slack_data: pd.DataFrame | None = None,
        calendar_data: pd.DataFrame | None = None,
        code_data: pd.DataFrame | None = None,
        github_data: pd.DataFrame | None = None,
    ):
        """Initialize multi-modal health score calculator.

        Args:
            graph: Organizational network graph
            email_data: Optional email event data
            slack_data: Optional Slack event data
            calendar_data: Optional calendar/meeting data
            code_data: Optional code commit data
            github_data: Optional GitHub activity data
        """
        self.graph = graph
        self.email_data = email_data
        self.slack_data = slack_data
        self.calendar_data = calendar_data
        self.code_data = code_data
        self.github_data = github_data

        # Core dashboard generator
        self.dashboard_generator = DashboardGenerator(graph)

    def compute_health_score(
        self,
        lookback_days: int = 30,
        slack_weight: float = 0.25,
        email_weight: float = 0.25,
        code_weight: float = 0.25,
        calendar_weight: float = 0.25,
    ) -> dict:
        """Compute unified organizational health score from multiple modalities.

        This creates a unified "Organizational Health Score" by weighing signals
        from different sources (e.g., high Slack activity + low code commits =
        high meeting/coordination overhead).

        Args:
            lookback_days: Number of days to analyze (default: 30)
            slack_weight: Weight for Slack activity (default: 0.25)
            email_weight: Weight for email activity (default: 0.25)
            code_weight: Weight for code activity (default: 0.25)
            calendar_weight: Weight for calendar activity (default: 0.25)

        Returns:
            Dictionary with unified health score and component scores
        """
        logger.info("Computing Multi-Modal Organizational Health Score...")

        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        # Compute health scores for each modality
        modality_scores = {}

        # 1. Slack Health Score
        slack_score = self._compute_slack_health_score(cutoff_date)
        modality_scores["slack"] = slack_score

        # 2. Email Health Score
        email_score = self._compute_email_health_score(cutoff_date)
        modality_scores["email"] = email_score

        # 3. Code/Development Health Score
        code_score = self._compute_code_health_score(cutoff_date)
        modality_scores["code"] = code_score

        # 4. Calendar/Meeting Health Score
        calendar_score = self._compute_calendar_health_score(cutoff_date)
        modality_scores["calendar"] = calendar_score

        # 5. Network Health Score (from graph)
        network_score = self._compute_network_health_score()
        modality_scores["network"] = network_score

        # Detect patterns across modalities
        patterns = self._detect_cross_modal_patterns(
            slack_score, email_score, code_score, calendar_score
        )

        # Compute unified health score
        unified_score = self._compute_unified_score(
            modality_scores,
            slack_weight,
            email_weight,
            code_weight,
            calendar_weight,
        )

        # Health classification
        health_level = self._classify_health_level(unified_score)

        # Key insights
        insights = self._generate_health_insights(unified_score, modality_scores, patterns)

        return {
            "timestamp": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "unified_health_score": unified_score,
            "health_level": health_level,
            "modality_scores": modality_scores,
            "patterns": patterns,
            "insights": insights,
            "recommendations": self._generate_health_recommendations(
                unified_score, modality_scores, patterns
            ),
        }

    def _compute_slack_health_score(self, cutoff_date: datetime) -> dict:
        """Compute Slack health score.

        Args:
            cutoff_date: Cutoff date for analysis

        Returns:
            Dictionary with Slack health score and metrics
        """
        if self.slack_data is None or self.slack_data.empty:
            return {
                "score": 0.5,  # Neutral if no data
                "activity_level": 0.0,
                "engagement": 0.0,
                "data_available": False,
            }

        slack_recent = self.slack_data[
            pd.to_datetime(self.slack_data.get("timestamp", pd.Series())) >= cutoff_date
        ]

        if slack_recent.empty:
            return {
                "score": 0.5,
                "activity_level": 0.0,
                "engagement": 0.0,
                "data_available": True,
            }

        # Activity metrics
        total_messages = len(slack_recent)
        unique_users = (
            slack_recent.get("user_id", pd.Series()).nunique()
            if "user_id" in slack_recent.columns
            else 0
        )
        channels = (
            slack_recent.get("channel", pd.Series()).nunique()
            if "channel" in slack_recent.columns
            else 0
        )

        # Engagement score (higher activity = higher engagement, but too high = overload)
        activity_level = min(total_messages / 10000.0, 1.0)  # Normalize to 0-1
        engagement = min((unique_users * channels) / 1000.0, 1.0)  # Normalize to 0-1

        # Health score: balance between activity and engagement
        # Too high activity without engagement = coordination overhead
        slack_score = (activity_level * 0.5) + (engagement * 0.5)

        return {
            "score": slack_score,
            "activity_level": activity_level,
            "engagement": engagement,
            "total_messages": total_messages,
            "unique_users": unique_users,
            "channels": channels,
            "data_available": True,
        }

    def _compute_email_health_score(self, cutoff_date: datetime) -> dict:
        """Compute email health score.

        Args:
            cutoff_date: Cutoff date for analysis

        Returns:
            Dictionary with email health score and metrics
        """
        if self.email_data is None or self.email_data.empty:
            return {
                "score": 0.5,
                "volume": 0.0,
                "response_time": 0.0,
                "data_available": False,
            }

        email_recent = self.email_data[
            pd.to_datetime(self.email_data.get("timestamp", pd.Series())) >= cutoff_date
        ]

        if email_recent.empty:
            return {
                "score": 0.5,
                "volume": 0.0,
                "response_time": 0.0,
                "data_available": True,
            }

        # Activity metrics
        total_emails = len(email_recent)
        unique_senders = (
            email_recent.get("sender", pd.Series()).nunique()
            if "sender" in email_recent.columns
            else 0
        )

        # Email volume (normalize)
        volume = min(total_emails / 5000.0, 1.0)

        # Health score: moderate email volume is healthy
        # Too high = overload, too low = disconnection
        # Optimal range: 0.3-0.7
        if 0.3 <= volume <= 0.7:
            email_score = 0.8  # Optimal range
        elif volume > 0.7:
            email_score = 1.0 - (volume - 0.7) * 2  # Penalize overload
        else:
            email_score = volume / 0.3  # Penalize low activity

        email_score = max(0.0, min(1.0, email_score))  # Clip to [0, 1]

        return {
            "score": email_score,
            "volume": volume,
            "total_emails": total_emails,
            "unique_senders": unique_senders,
            "data_available": True,
        }

    def _compute_code_health_score(self, cutoff_date: datetime) -> dict:
        """Compute code/development health score.

        Args:
            cutoff_date: Cutoff date for analysis

        Returns:
            Dictionary with code health score and metrics
        """
        # Combine code_data and github_data
        code_recent = pd.DataFrame()
        if self.code_data is not None and not self.code_data.empty:
            code_recent = self.code_data[
                pd.to_datetime(self.code_data.get("timestamp", pd.Series())) >= cutoff_date
            ]
        if self.github_data is not None and not self.github_data.empty:
            github_recent = self.github_data[
                pd.to_datetime(self.github_data.get("timestamp", pd.Series())) >= cutoff_date
            ]
            if not github_recent.empty:
                if code_recent.empty:
                    code_recent = github_recent
                else:
                    code_recent = pd.concat([code_recent, github_recent], ignore_index=True)

        if code_recent.empty:
            return {
                "score": 0.5,
                "commit_activity": 0.0,
                "developer_count": 0.0,
                "data_available": False,
            }

        # Activity metrics
        total_commits = len(code_recent)
        unique_authors = (
            code_recent.get("author_id", pd.Series()).nunique()
            if "author_id" in code_recent.columns
            else (
                code_recent.get("author", pd.Series()).nunique()
                if "author" in code_recent.columns
                else 0
            )
        )

        # Commit activity (normalize)
        commit_activity = min(total_commits / 1000.0, 1.0)

        # Developer activity (normalize)
        developer_count = min(unique_authors / 50.0, 1.0)

        # Code health score: balance between commits and developers
        code_score = (commit_activity * 0.6) + (developer_count * 0.4)

        return {
            "score": code_score,
            "commit_activity": commit_activity,
            "developer_count": developer_count,
            "total_commits": total_commits,
            "unique_authors": unique_authors,
            "data_available": True,
        }

    def _compute_calendar_health_score(self, cutoff_date: datetime) -> dict:
        """Compute calendar/meeting health score.

        Args:
            cutoff_date: Cutoff date for analysis

        Returns:
            Dictionary with calendar health score and metrics
        """
        if self.calendar_data is None or self.calendar_data.empty:
            return {
                "score": 0.5,
                "meeting_intensity": 0.0,
                "coordination_overhead": 0.0,
                "data_available": False,
            }

        calendar_recent = self.calendar_data[
            pd.to_datetime(self.calendar_data.get("timestamp", pd.Series())) >= cutoff_date
        ]

        if calendar_recent.empty:
            return {
                "score": 0.5,
                "meeting_intensity": 0.0,
                "coordination_overhead": 0.0,
                "data_available": True,
            }

        # Meeting metrics
        total_meetings = len(calendar_recent)
        total_meeting_hours = (
            calendar_recent.get("duration_hours", pd.Series()).sum()
            if "duration_hours" in calendar_recent.columns
            else total_meetings * 1.0  # Assume 1 hour per meeting
        )

        # Meeting intensity (meetings per person per week)
        unique_attendees = (
            calendar_recent.get("attendees", pd.Series()).str.split(",").explode().nunique()
            if "attendees" in calendar_recent.columns
            else total_meetings * 2  # Assume 2 attendees per meeting
        )

        # Normalize meeting intensity
        meeting_intensity = min(total_meetings / 1000.0, 1.0)

        # Coordination overhead: high meeting hours = high overhead
        # Assume 40 hours/week work time, normalize
        weeks_covered = 4.0  # Approximate for 30-day lookback
        hours_per_person_per_week = (
            (total_meeting_hours / unique_attendees) / weeks_covered
            if unique_attendees > 0
            else 0.0
        )
        coordination_overhead = min(hours_per_person_per_week / 20.0, 1.0)  # 20 hours/week = 100%

        # Calendar health score: lower is better (fewer meetings = more productive)
        # But too low = poor coordination
        if coordination_overhead < 0.2:
            calendar_score = 0.6  # Too few meetings
        elif coordination_overhead > 0.6:
            calendar_score = 1.0 - (coordination_overhead - 0.6) * 2.5  # Too many meetings
        else:
            calendar_score = 0.8  # Optimal range

        calendar_score = max(0.0, min(1.0, calendar_score))

        return {
            "score": calendar_score,
            "meeting_intensity": meeting_intensity,
            "coordination_overhead": coordination_overhead,
            "total_meetings": total_meetings,
            "total_meeting_hours": total_meeting_hours,
            "hours_per_person_per_week": hours_per_person_per_week,
            "data_available": True,
        }

    def _compute_network_health_score(self) -> dict:
        """Compute network health score from graph.

        Args:
            Returns:
            Dictionary with network health score
        """
        # Use existing dashboard generator
        health_dashboard = self.dashboard_generator.generate_health_dashboard()

        # Extract key network metrics
        density = health_dashboard.get("network_density", 0.0)
        modularity = health_dashboard.get("modularity", 0.0)
        largest_component_pct = health_dashboard.get("largest_component_pct", 0.0) / 100.0

        # Network health: balance density, connectivity, and modularity
        # Higher density = better connectivity (but too high = overconnected)
        # Lower modularity = less siloing (but some modularity is healthy)
        # Higher largest component = better connectivity

        density_score = min(density * 20.0, 1.0)  # Normalize density
        modularity_score = 1.0 - min(modularity * 1.5, 1.0)  # Lower modularity = better
        connectivity_score = largest_component_pct

        # Network health score
        network_score = (
            (density_score * 0.3) + (modularity_score * 0.3) + (connectivity_score * 0.4)
        )

        return {
            "score": network_score,
            "density": density,
            "modularity": modularity,
            "largest_component_pct": largest_component_pct * 100.0,
            "density_score": density_score,
            "modularity_score": modularity_score,
            "connectivity_score": connectivity_score,
        }

    def _detect_cross_modal_patterns(
        self, slack_score: dict, email_score: dict, code_score: dict, calendar_score: dict
    ) -> list[dict]:
        """Detect patterns across modalities.

        This is the key differentiator: detecting patterns like
        "high Slack activity + low code commits = high meeting/coordination overhead"

        Args:
            slack_score: Slack health score
            email_score: Email health score
            code_score: Code health score
            calendar_score: Calendar health score

        Returns:
            List of detected patterns
        """
        patterns = []

        # Pattern 1: High Slack + Low Code = Coordination Overhead
        slack_activity = slack_score.get("activity_level", 0.0)
        code_activity = code_score.get("commit_activity", 0.0)

        if slack_activity > 0.7 and code_activity < 0.3:
            patterns.append(
                {
                    "pattern_type": "coordination_overhead",
                    "severity": "high",
                    "description": (
                        "High Slack activity combined with low code commits indicates "
                        "high meeting/coordination overhead. Teams may be spending too "
                        "much time in meetings rather than productive work."
                    ),
                    "slack_activity": slack_activity,
                    "code_activity": code_activity,
                    "coordination_overhead": slack_activity - code_activity,
                    "recommendation": (
                        "Consider implementing 'No-Meeting Wednesdays' or reducing "
                        "Slack dependency. Focus on asynchronous communication and code productivity."
                    ),
                }
            )

        # Pattern 2: High Email + High Calendar = Meeting Overload
        email_volume = email_score.get("volume", 0.0)
        calendar_overhead = calendar_score.get("coordination_overhead", 0.0)

        if email_volume > 0.7 and calendar_overhead > 0.6:
            patterns.append(
                {
                    "pattern_type": "meeting_overload",
                    "severity": "high",
                    "description": (
                        "High email volume combined with high meeting overhead indicates "
                        "excessive communication overhead. Teams may be over-communicating."
                    ),
                    "email_volume": email_volume,
                    "calendar_overhead": calendar_overhead,
                    "recommendation": (
                        "Review meeting cadence and email culture. Consider consolidating "
                        "meetings and reducing email volume."
                    ),
                }
            )

        # Pattern 3: Low Code + High Calendar = Low Productivity
        if code_activity < 0.3 and calendar_overhead > 0.6:
            patterns.append(
                {
                    "pattern_type": "low_productivity",
                    "severity": "medium",
                    "description": (
                        "Low code activity combined with high meeting overhead suggests "
                        "teams may be spending too much time in meetings rather than "
                        "productive development work."
                    ),
                    "code_activity": code_activity,
                    "calendar_overhead": calendar_overhead,
                    "recommendation": (
                        "Protect developer time by reducing meeting overhead. "
                        "Implement focus time blocks and reduce calendar commitments."
                    ),
                }
            )

        # Pattern 4: High Slack + High Email = Communication Fragmentation
        if slack_activity > 0.7 and email_volume > 0.7:
            patterns.append(
                {
                    "pattern_type": "communication_fragmentation",
                    "severity": "medium",
                    "description": (
                        "High activity on both Slack and email may indicate communication "
                        "fragmentation. Teams may benefit from consolidating communication channels."
                    ),
                    "slack_activity": slack_activity,
                    "email_volume": email_volume,
                    "recommendation": (
                        "Consider standardizing on a primary communication channel to reduce "
                        "fragmentation and improve clarity."
                    ),
                }
            )

        # Pattern 5: High Code + Low Communication = Potential Isolation
        if code_activity > 0.7 and slack_activity < 0.3 and email_volume < 0.3:
            patterns.append(
                {
                    "pattern_type": "potential_isolation",
                    "severity": "low",
                    "description": (
                        "High code activity but low communication may indicate teams are "
                        "working in isolation. This could lead to knowledge silos."
                    ),
                    "code_activity": code_activity,
                    "slack_activity": slack_activity,
                    "email_volume": email_volume,
                    "recommendation": (
                        "Encourage more cross-team communication while maintaining high "
                        "productivity. Consider regular sync meetings or async updates."
                    ),
                }
            )

        return patterns

    def _compute_unified_score(
        self,
        modality_scores: dict,
        slack_weight: float,
        email_weight: float,
        code_weight: float,
        calendar_weight: float,
    ) -> float:
        """Compute unified organizational health score.

        Args:
            modality_scores: Dictionary of modality scores
            slack_weight: Weight for Slack
            email_weight: Weight for email
            code_weight: Weight for code
            calendar_weight: Weight for calendar

        Returns:
            Unified health score between 0 and 1
        """
        # Extract scores
        slack_score = modality_scores.get("slack", {}).get("score", 0.5)
        email_score = modality_scores.get("email", {}).get("score", 0.5)
        code_score = modality_scores.get("code", {}).get("score", 0.5)
        calendar_score = modality_scores.get("calendar", {}).get("score", 0.5)
        network_score = modality_scores.get("network", {}).get("score", 0.5)

        # Check data availability
        slack_available = modality_scores.get("slack", {}).get("data_available", False)
        email_available = modality_scores.get("email", {}).get("data_available", False)
        code_available = modality_scores.get("code", {}).get("data_available", False)
        calendar_available = modality_scores.get("calendar", {}).get("data_available", False)

        # Adjust weights based on data availability
        total_weight = 0.0
        weighted_sum = 0.0

        if slack_available:
            weighted_sum += slack_score * slack_weight
            total_weight += slack_weight
        if email_available:
            weighted_sum += email_score * email_weight
            total_weight += email_weight
        if code_available:
            weighted_sum += code_score * code_weight
            total_weight += code_weight
        if calendar_available:
            weighted_sum += calendar_score * calendar_weight
            total_weight += calendar_weight

        # Always include network score (from graph)
        network_weight = 0.2
        weighted_sum += network_score * network_weight
        total_weight += network_weight

        # Normalize by total weight
        if total_weight > 0:
            unified_score = weighted_sum / total_weight
        else:
            unified_score = network_score  # Fallback to network score only

        return max(0.0, min(1.0, unified_score))

    def _classify_health_level(self, unified_score: float) -> str:
        """Classify organizational health level.

        Args:
            unified_score: Unified health score between 0 and 1

        Returns:
            Health level: 'excellent', 'good', 'moderate', 'poor', 'critical'
        """
        if unified_score >= 0.8:
            return "excellent"
        elif unified_score >= 0.7:
            return "good"
        elif unified_score >= 0.6:
            return "moderate"
        elif unified_score >= 0.5:
            return "poor"
        else:
            return "critical"

    def _generate_health_insights(
        self, unified_score: float, modality_scores: dict, patterns: list[dict]
    ) -> list[str]:
        """Generate natural language health insights.

        Args:
            unified_score: Unified health score
            modality_scores: Modality scores
            patterns: Detected patterns

        Returns:
            List of insight strings
        """
        insights = []

        # Overall health insight
        health_level = self._classify_health_level(unified_score)
        insights.append(
            f"Organizational Health Score: {unified_score:.2f} ({health_level.upper()})"
        )

        # Modality-specific insights
        slack_score = modality_scores.get("slack", {}).get("score", 0.5)
        if slack_score > 0.8:
            insights.append(
                f"Slack activity is high ({slack_score:.2f}), indicating strong collaboration"
            )
        elif slack_score < 0.4:
            insights.append(
                f"Slack activity is low ({slack_score:.2f}), may indicate under-communication"
            )

        code_score = modality_scores.get("code", {}).get("score", 0.5)
        if code_score > 0.8:
            insights.append(
                f"Code activity is high ({code_score:.2f}), indicating strong development productivity"
            )
        elif code_score < 0.4:
            insights.append(
                f"Code activity is low ({code_score:.2f}), may indicate low development output"
            )

        calendar_score = modality_scores.get("calendar", {}).get("coordination_overhead", 0.0)
        if calendar_score > 0.6:
            insights.append(
                f"Coordination overhead is high ({calendar_score:.1%}), indicating excessive meeting time"
            )

        # Pattern insights
        for pattern in patterns:
            insights.append(f"Pattern detected: {pattern['description']}")

        return insights

    def _generate_health_recommendations(
        self, unified_score: float, modality_scores: dict, patterns: list[dict]
    ) -> list[dict]:
        """Generate health recommendations.

        Args:
            unified_score: Unified health score
            modality_scores: Modality scores
            patterns: Detected patterns

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Overall health recommendations
        if unified_score < 0.6:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "overall_health",
                    "title": "Improve Organizational Health",
                    "description": (
                        f"Overall health score is {unified_score:.2f} (below target of 0.6). "
                        "Consider reviewing communication patterns, meeting overhead, and productivity metrics."
                    ),
                    "action": "Review and optimize cross-modal patterns",
                }
            )

        # Pattern-specific recommendations
        for pattern in patterns:
            recommendations.append(
                {
                    "priority": pattern.get("severity", "medium"),
                    "category": "cross_modal_pattern",
                    "title": f"Address: {pattern['pattern_type'].replace('_', ' ').title()}",
                    "description": pattern["description"],
                    "recommendation": pattern.get("recommendation", ""),
                    "pattern_type": pattern["pattern_type"],
                }
            )

        return recommendations
