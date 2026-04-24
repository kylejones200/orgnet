"""
Graph primitives: prefer netsmith_rs when the compiled extension exposes kernels; otherwise
NetworkX (connectivity, WF closeness). Newman modularity stays in pure NumPy.
"""

from __future__ import annotations

import numpy as np

# Unreachable sentinel returned by netsmith_rs.shortest_paths_rust
_U64_MAX = (1 << 64) - 1


def _rust_graph_kernels():
    try:
        import netsmith_rs as ns

        if hasattr(ns, "connected_components_rust") and hasattr(ns, "shortest_paths_rust"):
            return ns
    except ImportError:
        pass
    return None


def edge_pairs_uint64_from_el(el) -> np.ndarray:
    """Stack EdgeList endpoints into shape (E, 2) uint64 for netsmith_rs."""
    u = np.asarray(el.u, dtype=np.uint64)
    v = np.asarray(el.v, dtype=np.uint64)
    return np.column_stack((u, v))


def graph_is_connected(n_nodes: int, edges_uv: np.ndarray) -> bool:
    """True iff the undirected graph has exactly one connected component."""
    if n_nodes <= 1:
        return True
    if edges_uv.size == 0:
        return False
    ns = _rust_graph_kernels()
    if ns is not None:
        ncomp, _labels = ns.connected_components_rust(int(n_nodes), edges_uv)
        return int(ncomp) == 1
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(int(n_nodes)))
    g.add_edges_from((int(a), int(b)) for a, b in np.asarray(edges_uv, dtype=np.int64))
    return nx.is_connected(g)


def closeness_centrality_wf(n_nodes: int, edges_uv: np.ndarray) -> np.ndarray:
    """
    Wasserman–Faust improved closeness (NetworkX default), one score per node index.

    Matches ``networkx.closeness_centrality(G, wf_improved=True)`` for unweighted graphs
    when nodes are labeled 0 .. n_nodes - 1 and ``edges_uv`` lists each undirected edge once.
    """
    if n_nodes <= 1:
        return np.zeros(n_nodes, dtype=np.float64)

    ns = _rust_graph_kernels()
    if ns is None:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(int(n_nodes)))
        g.add_edges_from((int(a), int(b)) for a, b in np.asarray(edges_uv, dtype=np.int64))
        cc = nx.closeness_centrality(g, wf_improved=True)
        return np.array([cc.get(i, 0.0) for i in range(int(n_nodes))], dtype=np.float64)

    len_g = float(n_nodes)
    out = np.zeros(n_nodes, dtype=np.float64)
    for src in range(n_nodes):
        raw = ns.shortest_paths_rust(int(n_nodes), edges_uv, int(src), False)
        dists = [int(x) for x in raw]
        totsp = sum(dists[v] for v in range(n_nodes) if dists[v] != _U64_MAX)
        len_sp = sum(1 for v in range(n_nodes) if dists[v] != _U64_MAX)
        if totsp > 0 and len_g > 1:
            base = (len_sp - 1.0) / float(totsp)
            out[src] = base * ((len_sp - 1.0) / (len_g - 1.0))
    return out


def weighted_modularity(
    n_nodes: int,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    weights: np.ndarray,
    community_id: np.ndarray,
) -> float:
    """
    Newman modularity for an undirected weighted graph (each edge stored once).

    ``community_id`` length ``n_nodes``; values are arbitrary community labels (hashable).
    """
    m = float(np.sum(weights))
    if m <= 0:
        return 0.0

    k = np.zeros(n_nodes, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    for i in range(w_arr.size):
        u = int(edges_u[i])
        v = int(edges_v[i])
        w = float(w_arr[i])
        k[u] += w
        k[v] += w

    twom = 2.0 * m
    w_internal: dict[int, float] = {}
    k_sum: dict[int, float] = {}

    for i in range(n_nodes):
        c = int(community_id[i])
        k_sum[c] = k_sum.get(c, 0.0) + k[i]

    for i in range(w_arr.size):
        u = int(edges_u[i])
        v = int(edges_v[i])
        w = float(w_arr[i])
        cu = int(community_id[u])
        cv = int(community_id[v])
        if cu == cv:
            w_internal[cu] = w_internal.get(cu, 0.0) + w

    q = 0.0
    for c, w_c in w_internal.items():
        k_c = k_sum.get(c, 0.0)
        q += w_c / m - (k_c / twom) ** 2
    return float(q)
