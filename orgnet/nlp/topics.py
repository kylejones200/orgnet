"""Topic modeling for organizational communications."""

import pandas as pd
from typing import List, Optional
from datetime import datetime

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False

try:
    from gensim import corpora, models

    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False


class TopicModeler:
    """Topic modeling for organizational text data."""

    def __init__(self, method: str = "bertopic"):
        """
        Initialize topic modeler.

        Args:
            method: Topic modeling method ('bertopic' or 'lda')
        """
        self.method = method
        self.model = None
        self.topics = None

    def fit(
        self,
        documents: List[str],
        timestamps: Optional[List[datetime]] = None,
        num_topics: Optional[int] = None,
        min_topic_size: int = 10,
    ):
        """
        Fit topic model.

        Args:
            documents: List of document texts
            timestamps: Optional timestamps for temporal analysis
            num_topics: Number of topics (for LDA, auto for BERTopic)
            min_topic_size: Minimum size for topics
        """
        method_map = {
            "bertopic": lambda: self._fit_bertopic(
                documents, timestamps, num_topics, min_topic_size
            ),
            "lda": lambda: self._fit_lda(documents, num_topics),
        }

        fitter = method_map.get(self.method)
        if fitter is None:
            raise ValueError(
                f"Unknown method: {self.method}. Choose from {list(method_map.keys())}"
            )

        fitter()

    def _fit_bertopic(
        self,
        documents: List[str],
        timestamps: Optional[List[datetime]],
        num_topics: Optional[int],
        min_topic_size: int,
    ):
        """Fit BERTopic model."""
        if not HAS_BERTOPIC:
            raise ImportError(
                "BERTopic not available. Install with: pip install bertopic sentence-transformers"
            )

        embedding_model = SentenceTransformer("all-mpnet-base-v2")

        self.model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=num_topics if num_topics else "auto",
            min_topic_size=min_topic_size,
            calculate_probabilities=True,
        )

        self.topics, probs = self.model.fit_transform(documents)

        if timestamps:
            self.topic_over_time = self.model.topics_over_time(documents, timestamps, self.topics)

    def _fit_lda(self, documents: List[str], num_topics: int):
        """Fit LDA model."""
        if not HAS_GENSIM:
            raise ImportError("Gensim not available. Install with: pip install gensim")

        # Preprocess documents
        texts = [doc.lower().split() for doc in documents]

        # Create dictionary and corpus
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Fit LDA
        self.model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha="auto",
            per_word_topics=True,
        )

        # Assign topics
        self.topics = []
        for doc in corpus:
            topic_dist = self.model.get_document_topics(doc)
            if topic_dist:
                self.topics.append(max(topic_dist, key=lambda x: x[1])[0])
            else:
                self.topics.append(-1)

    def get_topic_info(self) -> pd.DataFrame:
        """
        Get information about detected topics.

        Returns:
            DataFrame with topic information
        """
        if self.model is None:
            return pd.DataFrame()

        method_handlers = {
            "bertopic": lambda: self.model.get_topic_info(),
            "lda": lambda: pd.DataFrame(
                [
                    {
                        "Topic": topic_id,
                        "Words": ", ".join(
                            [word for word, _ in self.model.show_topic(topic_id, topn=10)]
                        ),
                        "Weights": [
                            weight for _, weight in self.model.show_topic(topic_id, topn=10)
                        ],
                    }
                    for topic_id in range(self.model.num_topics)
                ]
            ),
        }

        handler = method_handlers.get(self.method)
        if handler is None:
            raise ValueError(f"Unknown method: {self.method}")

        return handler()

    def get_topic_distribution(self) -> pd.DataFrame:
        """
        Get topic distribution across documents.

        Returns:
            DataFrame with document-topic assignments
        """
        if self.topics is None:
            return pd.DataFrame()

        return pd.DataFrame({"document_id": range(len(self.topics)), "topic": self.topics})
