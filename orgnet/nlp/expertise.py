"""Expertise inference from organizational communications."""

import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict

from orgnet.nlp.topics import TopicModeler


class ExpertiseInferencer:
    """Infers expertise domains for individuals."""

    def __init__(self, topic_modeler: TopicModeler):
        """
        Initialize expertise inferencer.

        Args:
            topic_modeler: Fitted TopicModeler instance
        """
        self.topic_modeler = topic_modeler
        self.expertise_scores: Optional[Dict] = None

    def infer_expertise(
        self, person_documents: Dict[str, List[str]], document_topics: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Infer expertise for each person based on their documents.

        Args:
            person_documents: Dictionary mapping person_id to list of document texts
            document_topics: Optional pre-computed topic assignments

        Returns:
            DataFrame with person_id, topic, and expertise_score
        """
        if self.topic_modeler.model is None:
            raise ValueError("Topic model must be fitted first")

        # Get topic distribution for each person
        person_topic_scores = defaultdict(lambda: defaultdict(float))

        for person_id, documents in person_documents.items():
            for doc in documents:
                # Get topic for this document
                topic = -1
                if document_topics is None:
                    # Need to predict topic - dictionary dispatch
                    topic_handlers = {
                        "bertopic": lambda: (
                            self.topic_modeler.model.transform([doc])[0][0]
                            if self.topic_modeler.model.transform([doc])[0]
                            else -1
                        )
                    }
                    handler = topic_handlers.get(self.topic_modeler.method)
                    if handler:
                        topic = handler()

                if topic >= 0:
                    person_topic_scores[person_id][topic] += 1.0

        # Normalize by total activity
        expertise_data = []
        for person_id, topic_counts in person_topic_scores.items():
            total = sum(topic_counts.values())
            for topic, count in topic_counts.items():
                expertise_score = count / total if total > 0 else 0.0
                expertise_data.append(
                    {
                        "person_id": person_id,
                        "topic": topic,
                        "expertise_score": expertise_score,
                        "document_count": count,
                    }
                )

        self.expertise_scores = person_topic_scores

        return pd.DataFrame(expertise_data)

    def get_top_experts(self, topic: int, top_k: int = 10) -> pd.DataFrame:
        """
        Get top experts for a specific topic.

        Args:
            topic: Topic ID
            top_k: Number of top experts

        Returns:
            DataFrame with top experts
        """
        if self.expertise_scores is None:
            return pd.DataFrame()

        experts = []
        for person_id, topic_scores in self.expertise_scores.items():
            if topic in topic_scores:
                experts.append({"person_id": person_id, "expertise_score": topic_scores[topic]})

        df = pd.DataFrame(experts)
        return df.nlargest(top_k, "expertise_score")
