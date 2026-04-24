"""Unit tests for netsmith_rs-backed graph helpers (no full netsmith Python tree required)."""

import networkx as nx
import numpy as np
import pytest

from orgnet.three_es.business_logic.netsmith_graph_helpers import (
    closeness_centrality_wf,
    graph_is_connected,
    weighted_modularity,
)


def test_graph_is_connected_path_vs_isolated():
    n = 4
    path_edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint64)
    assert graph_is_connected(n, path_edges) is True

    two_comp = np.array([[0, 1], [2, 3]], dtype=np.uint64)
    assert graph_is_connected(n, two_comp) is False


def test_closeness_matches_networkx_wf():
    for seed in range(5):
        G = nx.gnm_random_graph(9, 18, seed=seed)
        n = G.number_of_nodes()
        mapping = {n: i for i, n in enumerate(sorted(G.nodes()))}
        H = nx.relabel_nodes(G, mapping)
        edges = np.array(list(H.edges()), dtype=np.uint64)
        exp = nx.closeness_centrality(H, wf_improved=True)
        got = closeness_centrality_wf(n, edges)
        for i in range(n):
            assert got[i] == pytest.approx(exp[i], rel=1e-9, abs=1e-9)


def test_weighted_modularity_matches_networkx():
    n = 5
    eu = np.array([0, 0, 1, 2, 3], dtype=np.int64)
    ev = np.array([1, 2, 2, 3, 4], dtype=np.int64)
    ew = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    G = nx.Graph()
    for i in range(eu.size):
        G.add_edge(int(eu[i]), int(ev[i]), weight=float(ew[i]))
    comm = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    sets = [{0, 1, 2}, {3, 4}]
    q_nx = nx.community.modularity(G, sets, weight="weight")
    q_us = weighted_modularity(n, eu, ev, ew, comm)
    assert q_us == pytest.approx(q_nx, rel=1e-9, abs=1e-9)
