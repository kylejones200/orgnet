"""Tests for GNN factory, auth/RBAC, tenancy, enterprise clients, and layouts."""

import pandas as pd
import pytest
import networkx as nx

from orgnet.auth import (
    AuthContext,
    AuthorizationError,
    is_allowed,
    require_permission,
    bearer_headers,
)
from orgnet.integrations.slack import SlackWebClient
from orgnet.integrations import microsoft_graph
from orgnet.tenancy import (
    assert_tenant_column,
    current_tenant_id,
    filter_dataframe_for_tenant,
    tenant_scope,
)
from orgnet.visualization.layouts import (
    graph_layout_positions,
    pyvis_physics_options_json,
)


def test_bearer_headers():
    h = bearer_headers("tok")
    assert h["Authorization"] == "Bearer tok"


def test_rbac_analyst_can_read_metrics():
    ctx = AuthContext(subject="u1", roles=frozenset({"analyst"}))
    assert is_allowed(ctx, "metrics.read")
    assert not is_allowed(ctx, "admin.manage")


def test_require_permission_raises():
    ctx = AuthContext(subject="u1", roles=frozenset({"viewer"}))
    with pytest.raises(AuthorizationError):
        require_permission(ctx, "admin.manage")


def test_tenant_scope_and_filter():
    df = pd.DataFrame({"tenant_id": ["a", "a", "b"], "x": [1, 2, 3]})
    assert_tenant_column(df, "tenant_id")
    with tenant_scope("a"):
        assert current_tenant_id() == "a"
        sub = filter_dataframe_for_tenant(df, "tenant_id")
        assert len(sub) == 2
        assert set(sub["x"]) == {1, 2}
    assert current_tenant_id() is None


def test_filter_requires_tenant():
    df = pd.DataFrame({"tenant_id": ["a"], "x": [1]})
    with pytest.raises(ValueError):
        filter_dataframe_for_tenant(df, "tenant_id")


def test_graph_layout_positions_spring():
    g = nx.path_graph(5)
    pos = graph_layout_positions(g, "spring", seed=0)
    assert len(pos) == 5


def test_graph_layout_positions_unknown():
    g = nx.Graph()
    with pytest.raises(ValueError):
        graph_layout_positions(g, "not_a_layout")


def test_pyvis_physics_preset():
    js = pyvis_physics_options_json("compact")
    assert "physics" in js


def test_slack_client_auth_test_mock(monkeypatch):
    def fake_json_request(method, url, **kwargs):
        assert "auth.test" in url
        return 200, {"ok": True, "team": "T"}

    monkeypatch.setattr("orgnet.integrations.slack.json_request", fake_json_request)
    c = SlackWebClient("xoxb-fake")
    out = c.auth_test()
    assert out["team"] == "T"


def test_graph_client_me_mock(monkeypatch):
    def fake_json_request(method, url, **kwargs):
        assert "/me" in url
        return 200, {"displayName": "Test"}

    monkeypatch.setattr("orgnet.integrations.microsoft_graph.json_request", fake_json_request)
    g = microsoft_graph.MicrosoftGraphClient("fake-token")
    me = g.me()
    assert me["displayName"] == "Test"


from orgnet.ml import gnn as gnn_module

HAS_TORCH_RUNTIME = gnn_module.HAS_TORCH


@pytest.mark.skipif(not HAS_TORCH_RUNTIME, reason="torch / pyg not installed")
def test_build_org_gnn_forward():
    import torch

    from orgnet.ml.gnn import build_org_gnn, graph_to_pyg_data

    g = nx.karate_club_graph()
    data = graph_to_pyg_data(g)
    for arch in ("gcn", "gat", "graphsage", "gatv2"):
        model = build_org_gnn(arch, data.x.shape[1], 16, 8, heads=2)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
        assert out.shape == (data.num_nodes, 8)


@pytest.mark.skipif(not HAS_TORCH_RUNTIME, reason="torch / pyg not installed")
def test_build_org_gnn_unknown_raises():
    from orgnet.ml.gnn import build_org_gnn

    with pytest.raises(ValueError):
        build_org_gnn("not_a_model", 4, 4, 4)
