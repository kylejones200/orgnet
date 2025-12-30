"""Machine learning models for organizational networks."""

from orgnet.ml.embeddings import NodeEmbedder
from orgnet.ml.link_prediction import LinkPredictor
from orgnet.ml.classification import EmailClassifier, ClassificationResult

try:
    from orgnet.ml.gnn import OrgGCN, OrgGAT

    __all__ = [
        "NodeEmbedder",
        "LinkPredictor",
        "EmailClassifier",
        "ClassificationResult",
        "OrgGCN",
        "OrgGAT",
    ]
except ImportError:
    # PyTorch not available
    OrgGCN = None
    OrgGAT = None
    __all__ = ["NodeEmbedder", "LinkPredictor", "EmailClassifier", "ClassificationResult"]
