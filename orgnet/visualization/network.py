"""Network visualization utilities."""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Optional

from orgnet.visualization.layouts import graph_layout_positions, pyvis_physics_options_json

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    from pyvis.network import Network

    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False


class NetworkVisualizer:
    """Creates visualizations of organizational networks."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize network visualizer.

        Args:
            graph: NetworkX graph
        """
        self.graph = graph

    def create_layout_positions(
        self,
        layout: str = "spring",
        *,
        weight: Optional[str] = "weight",
        **kwargs,
    ) -> Dict:
        """
        Compute node positions using :func:`orgnet.visualization.layouts.graph_layout_positions`.

        For ``spring`` layout, ``kwargs`` may include ``k`` and ``iterations`` (passed to
        NetworkX ``spring_layout``).
        """
        g = self.graph
        name = (layout or "spring").lower()
        if name == "spring":
            return graph_layout_positions(g, "spring", weight=weight, **kwargs)
        return graph_layout_positions(g, layout, **kwargs)

    def create_force_directed_layout(self, k: float = 1.0, iterations: int = 50) -> Dict:
        """Backward-compatible alias for spring layout."""
        return self.create_layout_positions("spring", k=k, iterations=iterations, weight="weight")

    def visualize_network(
        self,
        pos: Optional[Dict] = None,
        node_color: Optional[str] = None,
        node_size: Optional[str] = None,
        centrality_df: Optional[pd.DataFrame] = None,
        title: str = "Organizational Network",
        figsize: tuple = (12, 10),
        layout: str = "spring",
        node_cmap: str = "viridis",
        edge_color: str = "#cccccc",
        edge_width: float = 0.5,
    ):
        """
        Create matplotlib network visualization.

        Args:
            pos: Node positions (if None, computed via ``layout``)
            node_color: Centrality column name to color by
            node_size: Centrality column name to size by
            centrality_df: DataFrame with centrality metrics
            title: Plot title
            figsize: Figure size
            layout: Layout name for :func:`orgnet.visualization.layouts.graph_layout_positions`
            node_cmap: Matplotlib colormap name when ``node_color`` is used
            edge_color: Edge stroke color
            edge_width: Edge width

        Returns:
            Matplotlib figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. Install with: pip install matplotlib"
            )

        if pos is None:
            pos = self.create_layout_positions(layout)

        fig, ax = plt.subplots(figsize=figsize)

        # Node colors
        if node_color and centrality_df is not None:
            node_colors = []
            for node in self.graph.nodes():
                if node in centrality_df["node_id"].values:
                    value = centrality_df[centrality_df["node_id"] == node][node_color].values[0]
                    node_colors.append(value)
                else:
                    node_colors.append(0)
        else:
            node_colors = "lightblue"

        # Node sizes
        if node_size and centrality_df is not None:
            node_sizes = []
            for node in self.graph.nodes():
                if node in centrality_df["node_id"].values:
                    value = centrality_df[centrality_df["node_id"] == node][node_size].values[0]
                    node_sizes.append(100 + value * 500)
                else:
                    node_sizes.append(100)
        else:
            node_sizes = 300

        # Draw network
        node_draw_kw = {
            "node_color": node_colors,
            "node_size": node_sizes,
            "alpha": 0.7,
            "ax": ax,
        }
        if isinstance(node_colors, list):
            import matplotlib as mpl

            node_draw_kw["cmap"] = mpl.colormaps[node_cmap]

        nx.draw_networkx_nodes(self.graph, pos, **node_draw_kw)

        nx.draw_networkx_edges(
            self.graph,
            pos,
            alpha=0.35,
            width=edge_width,
            edge_color=edge_color,
            ax=ax,
        )

        # Labels (only for important nodes)
        if centrality_df is not None:
            top_nodes = centrality_df.head(20)["node_id"].values
            labels = {node: node for node in top_nodes if node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, ax=ax)

        ax.set_title(title, fontsize=16)
        ax.axis("off")

        return fig

    def create_interactive_network(
        self,
        output_path: str = "network.html",
        node_color: Optional[str] = None,
        node_size: Optional[str] = None,
        centrality_df: Optional[pd.DataFrame] = None,
        height: str = "800px",
        width: str = "100%",
        physics_preset: str = "default",
    ) -> str:
        """
        Create interactive network visualization using Pyvis.

        Args:
            output_path: Path to save HTML file
            node_color: Node attribute to color by
            node_size: Node attribute to size by
            centrality_df: DataFrame with centrality metrics
            height: Height of visualization
            width: Width of visualization
            physics_preset: Pyvis physics bundle name (``default``, ``compact``, ``spacious``).

        Returns:
            Path to saved HTML file
        """
        if not HAS_PYVIS:
            raise ImportError("pyvis not available. Install with: pip install pyvis")

        net = Network(height=height, width=width, directed=False)

        # Add nodes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]

            # Node size
            size = 10
            if node_size and centrality_df is not None:
                if node in centrality_df["node_id"].values:
                    value = centrality_df[centrality_df["node_id"] == node][node_size].values[0]
                    size = 10 + value * 30

            # Node color
            color = "#97c2fc"
            if node_color and centrality_df is not None:
                if node in centrality_df["node_id"].values:
                    value = centrality_df[centrality_df["node_id"] == node][node_color].values[0]
                    # Normalize to color scale
                    color = self._value_to_color(value)

            title = f"{node_data.get('name', node)}\n{node_data.get('department', '')}"
            net.add_node(
                node, label=node_data.get("name", node), size=size, color=color, title=title
            )

        # Add edges
        for u, v, data in self.graph.edges(data=True):
            weight = data.get("weight", 1.0)
            net.add_edge(u, v, value=weight, width=weight * 2)

        net.set_options(pyvis_physics_options_json(physics_preset))

        net.save_graph(output_path)
        return output_path

    def _value_to_color(self, value: float, colormap: str = "viridis") -> str:
        """Convert value to hex color."""
        # Normalize value to [0, 1]
        # This is simplified - would need proper normalization
        normalized = max(0, min(1, value))

        # Color mapping using threshold-based approach
        colormap_colors = {
            "viridis": [
                (0.25, "#440154"),  # Dark purple
                (0.5, "#31688e"),  # Blue
                (0.75, "#35b779"),  # Green
                (1.0, "#fde725"),  # Yellow
            ]
        }

        if colormap in colormap_colors:
            for threshold, color in colormap_colors[colormap]:
                if normalized < threshold:
                    return color
            return colormap_colors[colormap][-1][1]  # Return last color

        # Default blue scale
        intensity = int(255 * normalized)
        return f"#{intensity:02x}{intensity:02x}ff"

    def create_adjacency_heatmap(
        self, communities: Optional[Dict] = None, output_path: Optional[str] = None
    ):
        """
        Create adjacency matrix heatmap.

        Args:
            communities: Optional community assignments
            output_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.graph, weight="weight").todense()

        # Reorder by communities if provided
        if communities:
            node_order = []
            for comm_id in sorted(set(communities.values())):
                comm_nodes = [n for n, c in communities.items() if c == comm_id]
                node_order.extend(sorted(comm_nodes))

            # Reorder matrix
            node_list = list(self.graph.nodes())
            indices = [node_list.index(n) for n in node_order]
            adj_matrix = adj_matrix[np.ix_(indices, indices)]
        else:
            node_order = list(self.graph.nodes())

        # Create heatmap
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. Install with: pip install matplotlib"
            )

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(adj_matrix, cmap="viridis", aspect="auto")

        ax.set_title("Adjacency Matrix Heatmap", fontsize=16)
        plt.colorbar(im, ax=ax)

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig
