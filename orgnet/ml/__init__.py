"""Machine learning models for organizational networks."""

from orgnet.ml.embeddings import NodeEmbedder
from orgnet.ml.link_prediction import LinkPredictor
from orgnet.ml.classification import EmailClassifier, ClassificationResult

try:
    from orgnet.ml.gnn import (
        OrgGCN,
        OrgGAT,
        OrgGATv2,
        OrgGraphSAGE,
        build_org_gnn,
    )

    __all__ = [
        "NodeEmbedder",
        "LinkPredictor",
        "EmailClassifier",
        "ClassificationResult",
        "OrgGCN",
        "OrgGAT",
        "OrgGraphSAGE",
        "OrgGATv2",
        "build_org_gnn",
    ]
except ImportError:
    # PyTorch not available
    OrgGCN = None
    OrgGAT = None
    OrgGraphSAGE = None
    OrgGATv2 = None
    build_org_gnn = None
    __all__ = ["NodeEmbedder", "LinkPredictor", "EmailClassifier", "ClassificationResult"]
