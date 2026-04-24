"""Graph layout strategies for matplotlib and static positioning."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import networkx as nx

LayoutName = str

# Presets passed to Pyvis ``set_options`` (physics block only).
PYVIS_PHYSICS_PRESETS: Dict[str, str] = {
    "default": """
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 100,
          "springConstant": 0.05
        }
      }
    }
    """,
    "compact": """
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 50,
          "springConstant": 0.08
        }
      }
    }
    """,
    "spacious": """
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -1200,
          "centralGravity": 0.05,
          "springLength": 180,
          "springConstant": 0.02
        }
      }
    }
    """,
}


def graph_layout_positions(
    graph: nx.Graph,
    layout: LayoutName = "spring",
    *,
    dim: int = 2,
    **kwargs: Any,
) -> Dict[Any, Tuple[float, ...]]:
    """
    Compute node positions for a graph.

    Supported ``layout`` values: ``spring``, ``kamada_kawai``, ``circular``,
    ``spectral``, ``planar`` (planar raises if graph is not planar).
    """
    name = (layout or "spring").lower().replace("-", "_")
    if name == "spring":
        return nx.spring_layout(graph, dim=dim, **kwargs)
    if name in ("kamada", "kamada_kawai", "kamada-kawai"):
        return nx.kamada_kawai_layout(graph, **kwargs)
    if name == "circular":
        return nx.circular_layout(graph, **kwargs)
    if name == "spectral":
        return nx.spectral_layout(graph, **kwargs)
    if name == "planar":
        return nx.planar_layout(graph, **kwargs)
    raise ValueError(
        f"Unknown layout {layout!r}. Use spring, kamada_kawai, circular, spectral, planar."
    )


def pyvis_physics_options_json(preset: str = "default") -> str:
    """Return a JSON snippet for :meth:`pyvis.network.Network.set_options`."""
    key = (preset or "default").lower()
    if key not in PYVIS_PHYSICS_PRESETS:
        keys = ", ".join(sorted(PYVIS_PHYSICS_PRESETS))
        raise ValueError(f"Unknown physics preset {preset!r}. Choose one of: {keys}")
    return PYVIS_PHYSICS_PRESETS[key]
