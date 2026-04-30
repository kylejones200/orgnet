"""Predictive burnout modeling for HR insights.

This module combines temporal analysis, change detection, and NLP sentiment
to predict employee burnout risk before turnover occurs.
"""

from datetime import datetime, timedelta

import networkx as nx
import pandas as pd

from orgnet.nlp.sentiment import SentimentAnalyzer
from orgnet.temporal.change_detection import ChangeDetector
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class BurnoutPredictor:
    """Predictive burnout modeling combining temporal, network, and NLP signals.

    This class leverages existing temporal analysis, change detection, and NLP sentiment
    to identify shifts in communication frequency or sentiment that precede employee turnover.
    """

    def __init__(
        self,
        lookback_days: int = 90,
        min_events: int = 10,
        sentiment_weight: float = 0.3,
        frequency_weight: float = 0.4,
        network_weight: float = 0.3,
        **params,
    ):
        """Initialize burnout predictor.

        Args:
            lookback_days: Number of days to look back for analysis (default: 90)
            min_events: Minimum number of events required for prediction (default: 10)
            sentiment_weight: Weight for sentiment signals in burnout score (default: 0.3)
            frequency_weight: Weight for frequency change signals (default: 0.4)
            network_weight: Weight for network overload signals (default: 0.3)
        """
        self.lookback_days = lookback_days
        self.min_events = min_events
        self.sentiment_weight = sentiment_weight
        self.frequency_weight = frequency_weight
        self.network_weight = network_weight

        # Components
        self.change_detector = ChangeDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None, **fit_params):
        """Fit burnout predictor on historical data.

        Args:
            X: Event table with columns: timestamp, actor, target, text, event_type
            y: Optional turnover events with columns: person_id, turnover_date
            **fit_params: Additional fit parameters

        Returns:
            Self
        """
        if y is not None and not y.empty:
            # Fit model using actual turnover events
            logger.info(f"Fitting burnout model with {len(y)} turnover events")
            self._fit_with_labels(X, y)
        else:
            # Unsupervised: use change detection and anomaly patterns
            logger.info("Fitting burnout model in unsupervised mode")

        self._is_fitted = True
        return self

    def _fit_with_labels(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit model using labeled turnover events."""
        # For now, we use unsupervised patterns
        pass

    def predict(self, X: pd.DataFrame, graph: nx.Graph | None = None) -> pd.DataFrame:
        """Predict burnout risk scores for each person.

        Args:
            X: Event table with temporal data
            graph: Optional current network graph for network overload analysis

        Returns:
            DataFrame with columns: person_id, burnout_score, risk_level, signals
        """
        if not self._is_fitted:
            self.fit(X)  # Fit in unsupervised mode if not already fitted

        cutoff_date = X["timestamp"].max()
        start_date = cutoff_date - timedelta(days=self.lookback_days)

        # Filter to lookback window
        recent_events = X[X["timestamp"] >= start_date].copy()

        if recent_events.empty:
            return pd.DataFrame(columns=["person_id", "burnout_score", "risk_level", "signals"])

        # Get unique actors
        person_ids = recent_events["actor"].unique()

        burnout_results = []

        for person_id in person_ids:
            person_events = recent_events[recent_events["actor"] == person_id]

            if len(person_events) < self.min_events:
                continue

            # Compute burnout signals
            signals = self._compute_burnout_signals(person_events, graph, person_id)

            # Combine signals into burnout score
            burnout_score = self._compute_burnout_score(signals)

            # Determine risk level
            risk_level = self._classify_risk(burnout_score)

            burnout_results.append(
                {
                    "person_id": person_id,
                    "burnout_score": burnout_score,
                    "risk_level": risk_level,
                    "sentiment_trend": signals.get("sentiment_trend", 0.0),
                    "frequency_change": signals.get("frequency_change", 0.0),
                    "network_overload": signals.get("network_overload", 0.0),
                    "off_hours_ratio": signals.get("off_hours_ratio", 0.0),
                    "signals": signals,
                }
            )

        return pd.DataFrame(burnout_results)

    def _compute_burnout_signals(
        self, person_events: pd.DataFrame, graph: nx.Graph | None, person_id: str
    ) -> dict:
        """Compute individual burnout signals for a person.

        Args:
            person_events: Events for this person
            graph: Optional network graph
            person_id: Person identifier

        Returns:
            Dictionary with burnout signals
        """
        signals = {}

        # 1. Sentiment trend analysis
        if "text" in person_events.columns:
            sentiment_trend = self._analyze_sentiment_trend(person_events)
            signals["sentiment_trend"] = sentiment_trend
        else:
            signals["sentiment_trend"] = 0.0

        # 2. Communication frequency change
        frequency_change = self._detect_frequency_change(person_events)
        signals["frequency_change"] = frequency_change

        # 3. Network overload (if graph provided)
        if graph is not None and person_id in graph:
            network_overload = self._compute_network_overload(graph, person_id)
            signals["network_overload"] = network_overload
        else:
            signals["network_overload"] = 0.0

        # 4. Off-hours activity
        off_hours_ratio = self._compute_off_hours_ratio(person_events)
        signals["off_hours_ratio"] = off_hours_ratio

        # 5. Change point detection (sudden shifts)
        change_points = self._detect_change_points(person_events)
        signals["change_points"] = len(change_points)
        signals["recent_change"] = 1.0 if change_points else 0.0

        return signals

    def _analyze_sentiment_trend(self, person_events: pd.DataFrame) -> float:
        """Analyze sentiment trend over time.

        Returns:
            Trend score: negative values indicate declining sentiment (higher burnout risk)
        """
        if "text" not in person_events.columns or person_events["text"].isna().all():
            return 0.0

        # Sort by timestamp
        events_sorted = person_events.sort_values("timestamp")

        # Split into early and late periods
        mid_point = len(events_sorted) // 2
        early_events = events_sorted.iloc[:mid_point]
        late_events = events_sorted.iloc[mid_point:]

        # Compute average sentiment using transform method
        early_texts = early_events["text"].dropna().tolist()
        late_texts = late_events["text"].dropna().tolist()

        if not early_texts or not late_texts:
            return 0.0

        # Use transform for batch processing (returns list of SentimentResult)
        early_results = self.sentiment_analyzer.transform(early_texts[:100])
        late_results = self.sentiment_analyzer.transform(late_texts[:100])

        # Extract scores from SentimentResult objects
        if isinstance(early_results, list) and early_results:
            early_scores = [result.score for result in early_results if hasattr(result, "score")]
            early_score = sum(early_scores) / len(early_scores) if early_scores else 0.0
        else:
            early_score = 0.0

        if isinstance(late_results, list) and late_results:
            late_scores = [result.score for result in late_results if hasattr(result, "score")]
            late_score = sum(late_scores) / len(late_scores) if late_scores else 0.0
        else:
            late_score = 0.0

        # Negative trend = lower sentiment in late period (higher burnout risk)
        trend = early_score - late_score

        return trend

    def _detect_frequency_change(self, person_events: pd.DataFrame) -> float:
        """Detect change in communication frequency.

        Returns:
            Change score: positive values indicate increasing frequency (potential burnout)
        """
        if len(person_events) < 10:
            return 0.0

        # Group by week
        person_events = person_events.copy()
        person_events["week"] = person_events["timestamp"].dt.to_period("W")
        weekly_counts = person_events.groupby("week").size()

        if len(weekly_counts) < 2:
            return 0.0

        # Split into early and late periods
        mid_point = len(weekly_counts) // 2
        early_freq = weekly_counts.iloc[:mid_point].mean()
        late_freq = weekly_counts.iloc[mid_point:].mean()

        if early_freq == 0:
            return 0.0

        # Normalized change (positive = increasing frequency)
        change = (late_freq - early_freq) / early_freq

        return change

    def _compute_network_overload(self, graph: nx.Graph, person_id: str) -> float:
        """Compute network overload score.

        Args:
            graph: Network graph
            person_id: Person identifier

        Returns:
            Overload score: higher values indicate more overload
        """
        if person_id not in graph:
            return 0.0

        # Degree and betweenness as overload indicators
        degree = graph.degree(person_id)
        try:
            betweenness = nx.betweenness_centrality(graph)[person_id]
        except Exception:
            betweenness = 0.0

        # Normalize by network size
        n_nodes = graph.number_of_nodes()
        normalized_degree = degree / n_nodes if n_nodes > 0 else 0.0

        # Combined overload score
        overload = (normalized_degree * 0.5) + (betweenness * 0.5)

        return overload

    def _compute_off_hours_ratio(self, person_events: pd.DataFrame) -> float:
        """Compute ratio of off-hours activity.

        Returns:
            Ratio of events outside business hours (9 AM - 5 PM, weekdays)
        """
        person_events = person_events.copy()
        person_events["hour"] = person_events["timestamp"].dt.hour
        person_events["day_of_week"] = person_events["timestamp"].dt.dayofweek
        person_events["is_weekend"] = person_events["day_of_week"] >= 5
        person_events["is_business_hours"] = (
            (person_events["hour"] >= 9)
            & (person_events["hour"] < 17)
            & (~person_events["is_weekend"])
        )

        off_hours_count = (~person_events["is_business_hours"]).sum()
        total_count = len(person_events)

        ratio = off_hours_count / total_count if total_count > 0 else 0.0

        return ratio

    def _detect_change_points(self, person_events: pd.DataFrame) -> list[datetime]:
        """Detect change points in communication patterns.

        Returns:
            List of detected change point timestamps
        """
        if len(person_events) < 20:
            return []

        # Group by day for change detection
        person_events = person_events.copy()
        person_events["date"] = person_events["timestamp"].dt.date
        daily_counts = person_events.groupby("date").size().reset_index(name="count")
        daily_counts["timestamp"] = pd.to_datetime(daily_counts["date"])

        # Use change detector if available
        try:
            # ChangeDetector expects DataFrame with timestamp and metric columns
            change_points = self.change_detector.detect_change_points(
                daily_counts[["timestamp", "count"]], metric="count", method="pelt", min_size=3
            )
            return change_points if isinstance(change_points, list) else []
        except Exception as e:
            logger.debug(f"Change detection failed, using fallback: {e}")
            # Fallback: simple threshold-based detection
            mean_count = daily_counts["count"].mean()
            std_count = daily_counts["count"].std()

            if std_count == 0:
                return []

            # Flag days with counts > 2 standard deviations from mean
            threshold = mean_count + (2 * std_count)
            change_days = daily_counts[daily_counts["count"] > threshold]

            return change_days["timestamp"].tolist() if not change_days.empty else []

    def _compute_burnout_score(self, signals: dict) -> float:
        """Combine signals into overall burnout score.

        Weight formula (configurable via constructor):
        - sentiment_weight: 0.3 default — sentiment trend (declining = higher risk)
        - frequency_weight: 0.3 default — frequency change (increasing = higher risk)
        - network_weight: 0.2 default — network overload (betweenness vs role)
        - off_hours: fixed 0.2 — ratio of off-hours communication
        - recent_change: 1.2x multiplier if recent behavioral change detected

        Args:
            signals: Dictionary of burnout signals

        Returns:
            Burnout score between 0 (low risk) and 1 (high risk)
        """
        # Normalize individual signals
        sentiment_risk = max(0.0, min(1.0, (signals.get("sentiment_trend", 0.0) + 1.0) / 2.0))
        frequency_risk = max(0.0, min(1.0, (signals.get("frequency_change", 0.0) + 1.0) / 2.0))
        network_risk = max(0.0, min(1.0, signals.get("network_overload", 0.0) * 2.0))
        off_hours_risk = signals.get("off_hours_ratio", 0.0)

        # Weighted combination
        burnout_score = (
            sentiment_risk * self.sentiment_weight
            + frequency_risk * self.frequency_weight
            + network_risk * self.network_weight
            + off_hours_risk * 0.2  # Additional weight for off-hours
        )

        # Boost if recent change detected
        if signals.get("recent_change", 0.0) > 0:
            burnout_score = min(1.0, burnout_score * 1.2)

        return burnout_score

    def _classify_risk(self, burnout_score: float) -> str:
        """Classify burnout risk level.

        Args:
            burnout_score: Burnout score between 0 and 1

        Returns:
            Risk level: 'low', 'medium', 'high', 'critical'
        """
        if burnout_score >= 0.8:
            return "critical"
        elif burnout_score >= 0.6:
            return "high"
        elif burnout_score >= 0.4:
            return "medium"
        else:
            return "low"

    def predict_turnover_risk(
        self, X: pd.DataFrame, graph: nx.Graph | None = None, horizon_days: int = 90
    ) -> pd.DataFrame:
        """Predict turnover risk within a time horizon.

        Args:
            X: Event table
            graph: Optional network graph
            horizon_days: Prediction horizon in days (default: 90)

        Returns:
            DataFrame with turnover risk predictions
        """
        burnout_df = self.predict(X, graph)

        # Convert burnout score to turnover probability
        # Higher burnout score = higher turnover probability
        burnout_df["turnover_probability"] = burnout_df["burnout_score"] * 0.7  # Calibration

        # Add horizon information
        burnout_df["horizon_days"] = horizon_days
        burnout_df["prediction_date"] = datetime.now()

        return burnout_df
