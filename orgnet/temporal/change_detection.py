"""Change point detection for organizational networks."""

import pandas as pd
import networkx as nx
from typing import List, Dict
from datetime import datetime

from orgnet.utils.logging import get_logger

try:
    import ruptures as rpt

    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

logger = get_logger(__name__)


class ChangeDetector:
    """Detects change points in temporal network metrics."""

    def __init__(self):
        """Initialize change detector."""
        pass

    def detect_change_points(
        self,
        temporal_metrics: pd.DataFrame,
        metric: str = "density",
        method: str = "pelt",
        min_size: int = 2,
        penalty: float = 10.0,
    ) -> List[datetime]:
        """
        Detect change points in temporal metrics.

        Args:
            temporal_metrics: DataFrame with timestamp and metric columns
            metric: Metric column name to analyze
            method: Change point detection method ('pelt', 'binseg', 'window')
            min_size: Minimum segment size
            penalty: Penalty parameter

        Returns:
            List of detected change point timestamps
        """
        if not HAS_RUPTURES:
            raise ImportError("ruptures not available. Install with: pip install ruptures")

        # Prepare data
        data = temporal_metrics.sort_values("timestamp")
        signal = data[metric].values.reshape(-1, 1)

        # Detect change points using dictionary dispatch
        method_map = {
            "pelt": lambda: rpt.Pelt(model="rbf", min_size=min_size)
            .fit(signal)
            .predict(pen=penalty),
            "binseg": lambda: rpt.Binseg(model="rbf", min_size=min_size)
            .fit(signal)
            .predict(n_bkps=5),
            "window": lambda: rpt.Window(width=40, model="rbf", min_size=min_size)
            .fit(signal)
            .predict(n_bkps=5),
        }

        detector = method_map.get(method)
        if detector is None:
            raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")

        change_points = detector()
        logger.info(f"Detected {len(change_points) - 1} change points using {method} method")

        # Convert indices to timestamps
        timestamps = data["timestamp"].values
        change_timestamps = [timestamps[idx] for idx in change_points[:-1]]  # Exclude last

        return change_timestamps

    def analyze_network_evolution(self, snapshots: List[Dict]) -> pd.DataFrame:
        """
        Analyze network evolution over time using vectorized operations.

        Args:
            snapshots: List of snapshot dictionaries with 'timestamp' and 'graph'

        Returns:
            DataFrame with evolution metrics
        """
        logger.info(f"Analyzing network evolution across {len(snapshots)} snapshots")

        # Build metrics list
        evolution_data = [
            {
                "timestamp": snapshot["timestamp"],
                "nodes": snapshot["graph"].number_of_nodes(),
                "edges": snapshot["graph"].number_of_edges(),
                "density": (
                    nx.density(snapshot["graph"]) if snapshot["graph"].number_of_nodes() > 0 else 0
                ),
                "avg_clustering": (
                    nx.average_clustering(snapshot["graph"])
                    if snapshot["graph"].number_of_nodes() > 0
                    else 0
                ),
            }
            for snapshot in snapshots
        ]

        df = pd.DataFrame(evolution_data)

        # Vectorized change calculation
        df["density_change"] = df["density"].diff().fillna(0)
        df["density_change_pct"] = (df["density_change"] / df["density"].shift(1) * 100).fillna(0)

        return df
