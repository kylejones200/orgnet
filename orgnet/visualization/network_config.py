"""Network visualization configuration and customization options."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

LAYOUT_OPTIONS = ("spring", "circular", "kamada_kawai", "shell", "spectral", "random")
COLOR_SCHEMES = ("viridis", "plasma", "inferno", "magma", "blues", "reds", "greens")
EDGE_STYLES = ("solid", "dashed", "dotted")


@dataclass
class NetworkVisualizationConfig:
    """Configuration for network visualization customization.

    Extends basic visualization with layout, color, and style options
    suitable for config files and programmatic control.
    """

    layout: str = "spring"
    node_size: int | float = 100
    node_size_min: int = 50
    node_size_max: int = 500
    edge_width: float = 1.0
    edge_width_min: float = 0.5
    edge_width_max: float = 5.0
    color_scheme: str = "viridis"
    show_labels: bool = True
    label_top_n: int = 20
    alpha: float = 0.7
    figsize: tuple[float, float] = (12, 10)
    # Interactive (Pyvis) options
    height: str = "800px"
    width: str = "100%"
    physics_enabled: bool = True
    gravity: float = 0.1
    spring_length: int = 100
    spring_constant: float = 0.05
    # Layout-specific params
    spring_k: float = 1.0
    spring_iterations: int = 50
    shell_nlist: list | None = None  # For shell layout: list of lists of nodes

    def __post_init__(self) -> None:
        """Validate and normalize config."""
        if self.layout not in LAYOUT_OPTIONS:
            self.layout = "spring"
        if self.color_scheme not in COLOR_SCHEMES:
            self.color_scheme = "viridis"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NetworkVisualizationConfig:
        """Create config from dictionary (e.g., config.yaml)."""
        network_config = d.get("network", d) if isinstance(d.get("network"), dict) else {}
        color_scheme = d.get("color_scheme", "viridis")
        if isinstance(network_config, dict):
            color_scheme = network_config.get("color_scheme", color_scheme)
        return cls(
            layout=network_config.get("layout", "spring"),
            node_size=network_config.get("node_size", 100),
            edge_width=network_config.get("edge_width", 1.0),
            color_scheme=color_scheme,
            figsize=tuple(network_config.get("figure_size", [12, 10]))[:2],
        )

    def get_layout_params(self) -> dict[str, Any]:
        """Get layout-specific parameters for NetworkX."""
        return {
            "spring": {"k": self.spring_k, "iterations": self.spring_iterations},
            "circular": {},
            "kamada_kawai": {},
            "shell": {"nlist": self.shell_nlist},
            "spectral": {},
            "random": {},
        }.get(self.layout, {})
