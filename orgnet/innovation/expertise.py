"""Expertise Inference Engine for Internal Talent Marketplace.

This module builds upon NLP topic modeling (BERTopic) to create an automated
"Internal Talent Marketplace" that allows companies to find "hidden experts"
who are influential in technical discussions but might be invisible in the formal org chart.
"""

from collections import defaultdict

import pandas as pd

from orgnet.nlp.expertise import ExpertiseInferencer as BaseExpertiseInferencer
from orgnet.nlp.topics import TopicModeler
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class ExpertiseInferenceEngine:
    """Expertise Inference Engine for Internal Talent Marketplace.

    Builds upon NLP topic modeling to create an automated "Internal Talent Marketplace"
    that finds "hidden experts" who are influential in technical discussions.
    """

    def __init__(
        self, topic_modeler: TopicModeler | None = None, method: str = "bertopic", **params
    ):
        """Initialize expertise inference engine.

        Args:
            topic_modeler: Optional pre-fitted TopicModeler instance
            method: Topic modeling method ('bertopic' or 'lda')
        """
        self.method = method

        if topic_modeler is None:
            self.topic_modeler = TopicModeler(method=method)
        else:
            self.topic_modeler = topic_modeler

        self.expertise_inferencer: BaseExpertiseInferencer | None = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None, **fit_params):
        """Fit expertise inference engine on document corpus.

        Args:
            X: Event table or document corpus with 'text' column
            y: Optional labels (not used but required by BaseEstimator)
            **fit_params: Additional fit parameters

        Returns:
            Self
        """
        if "text" not in X.columns:
            raise ValueError("Input must contain 'text' column")

        documents = X["text"].dropna().tolist()
        timestamps = X.get("timestamp", None)

        if timestamps is not None:
            timestamps = timestamps.tolist()

        # Fit topic model
        logger.info(f"Fitting topic model on {len(documents)} documents")
        self.topic_modeler.fit(
            documents=documents,
            timestamps=timestamps,
            num_topics=fit_params.get("num_topics", None),
            min_topic_size=fit_params.get("min_topic_size", 10),
        )

        # Initialize expertise inferencer
        self.expertise_inferencer = BaseExpertiseInferencer(self.topic_modeler)

        self._is_fitted = True
        logger.info("Expertise inference engine fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform event table to expertise scores.

        Args:
            X: Event table with columns: actor, text, timestamp

        Returns:
            DataFrame with expertise scores per person per topic
        """
        if not self._is_fitted:
            raise ValueError("Must fit expertise engine before transforming")

        if "actor" not in X.columns or "text" not in X.columns:
            raise ValueError("Input must contain 'actor' and 'text' columns")

        # Group documents by person
        person_documents = defaultdict(list)
        for _, row in X.iterrows():
            person_id = row["actor"]
            text = row.get("text", "")
            if pd.notna(text) and text.strip():
                person_documents[person_id].append(str(text))

        # Infer expertise for each person
        logger.info(f"Inferring expertise for {len(person_documents)} people")
        expertise_df = self.expertise_inferencer.infer_expertise(person_documents)

        return expertise_df

    def find_experts(self, topic: int, top_k: int = 10) -> pd.DataFrame:
        """Find top experts for a specific topic (Internal Talent Marketplace).

        Args:
            topic: Topic ID
            top_k: Number of top experts to return

        Returns:
            DataFrame with top experts for this topic
        """
        if not self._is_fitted or self.expertise_inferencer is None:
            raise ValueError("Must fit expertise engine before finding experts")

        return self.expertise_inferencer.get_top_experts(topic, top_k=top_k)

    def find_hidden_experts(
        self, topic: int, org_chart: list[str] | None = None, top_k: int = 10
    ) -> pd.DataFrame:
        """Find "hidden experts" not in formal org chart (Internal Talent Marketplace).

        Args:
            topic: Topic ID
            org_chart: Optional list of person IDs in formal org chart
            top_k: Number of top experts to return

        Returns:
            DataFrame with hidden experts (experts not in org chart)
        """
        experts_df = self.find_experts(topic, top_k=top_k * 2)  # Get more to filter

        if org_chart is None or len(org_chart) == 0:
            # No org chart provided, return all experts
            return experts_df.head(top_k)

        # Filter out people in formal org chart
        hidden_experts = experts_df[~experts_df["person_id"].isin(org_chart)]

        return hidden_experts.head(top_k)

    def search_experts_by_keyword(
        self, keyword: str, topics_df: pd.DataFrame | None = None, top_k: int = 10
    ) -> pd.DataFrame:
        """Search for experts by keyword (Internal Talent Marketplace search).

        Args:
            keyword: Search keyword
            topics_df: Optional DataFrame with topic information (from topic modeler)
            top_k: Number of results to return

        Returns:
            DataFrame with experts matching keyword
        """
        if not self._is_fitted:
            raise ValueError("Must fit expertise engine before searching")

        # Find topics matching keyword (if topic modeler provides topic info)
        matching_topics = []

        if topics_df is not None and "topic" in topics_df.columns:
            # Simple keyword matching (could be enhanced with semantic search)
            matching_topics = (
                topics_df[topics_df["topic_name"].str.contains(keyword, case=False, na=False)][
                    "topic"
                ]
                .unique()
                .tolist()
            )
        elif hasattr(self.topic_modeler, "topics") and self.topic_modeler.topics is not None:
            # Try to match keyword to topic representations
            logger.warning("Topic representations not available, returning empty results")
            return pd.DataFrame()

        if not matching_topics:
            logger.warning(f"No topics found matching keyword: {keyword}")
            return pd.DataFrame()

        # Aggregate experts across matching topics
        all_experts = []
        for topic_id in matching_topics:
            topic_experts = self.find_experts(topic_id, top_k=top_k)
            if not topic_experts.empty:
                all_experts.append(topic_experts)

        if not all_experts:
            return pd.DataFrame()

        # Combine and aggregate expertise scores
        combined_df = pd.concat(all_experts, ignore_index=True)
        expert_scores = combined_df.groupby("person_id")["expertise_score"].sum().reset_index()
        expert_scores = expert_scores.sort_values("expertise_score", ascending=False)

        return expert_scores.head(top_k)

    def get_talent_marketplace_summary(self, expertise_df: pd.DataFrame | None = None) -> dict:
        """Generate summary of Internal Talent Marketplace.

        Args:
            expertise_df: Optional pre-computed expertise DataFrame

        Returns:
            Dictionary with marketplace summary statistics
        """
        if expertise_df is None:
            if (
                self.expertise_inferencer is None
                or self.expertise_inferencer.expertise_scores is None
            ):
                return {
                    "total_experts": 0,
                    "total_topics": 0,
                    "avg_expertise_per_person": 0.0,
                    "top_topics": [],
                }

            # Extract from expertise_scores
            all_person_ids = set()
            all_topics = set()

            for person_id, topic_scores in self.expertise_inferencer.expertise_scores.items():
                all_person_ids.add(person_id)
                all_topics.update(topic_scores.keys())

            total_experts = len(all_person_ids)
            total_topics = len(all_topics)

            # Compute average expertise per person
            avg_expertise = (
                sum(len(scores) for scores in self.expertise_inferencer.expertise_scores.values())
                / total_experts
                if total_experts > 0
                else 0.0
            )

            return {
                "total_experts": total_experts,
                "total_topics": total_topics,
                "avg_expertise_per_person": avg_expertise,
                "top_topics": list(all_topics)[:10],  # First 10 topics
            }

        # Use provided DataFrame
        total_experts = expertise_df["person_id"].nunique()
        total_topics = expertise_df["topic"].nunique()
        avg_expertise = len(expertise_df) / total_experts if total_experts > 0 else 0.0

        # Top topics by expert count
        top_topics = (
            expertise_df.groupby("topic")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )

        return {
            "total_experts": total_experts,
            "total_topics": total_topics,
            "avg_expertise_per_person": avg_expertise,
            "top_topics": top_topics,
        }

    def browse_talent_marketplace(
        self,
        topic_filter: int | None = None,
        min_expertise_score: float = 0.3,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """Browse talent marketplace by topic or skill.

        This creates an "Automated Talent Marketplace" where companies can
        browse experts by topic or skill area, finding hidden experts not
        visible in the formal org chart.

        Args:
            topic_filter: Optional topic ID to filter by
            min_expertise_score: Minimum expertise score to include (default: 0.3)
            top_k: Number of results to return (default: 20)

        Returns:
            DataFrame with talent marketplace results
        """
        if not self._is_fitted or self.expertise_inferencer is None:
            raise ValueError("Must fit expertise engine before browsing marketplace")

        expertise_df = self.transform(pd.DataFrame())  # Get all expertise

        if expertise_df.empty:
            return pd.DataFrame()

        # Filter by topic if specified
        if topic_filter is not None:
            expertise_df = expertise_df[expertise_df["topic"] == topic_filter]

        # Filter by minimum expertise score
        expertise_df = expertise_df[expertise_df["expertise_score"] >= min_expertise_score]

        # Sort by expertise score
        talent_marketplace = expertise_df.sort_values("expertise_score", ascending=False)

        # Add talent profile information
        talent_marketplace["talent_level"] = talent_marketplace["expertise_score"].apply(
            lambda x: (
                "expert"
                if x >= 0.7
                else ("experienced" if x >= 0.5 else ("emerging" if x >= 0.3 else "beginner"))
            )
        )

        return talent_marketplace.head(top_k)

    def match_project_to_experts(
        self,
        project_keywords: list[str],
        required_topics: list[int] | None = None,
        min_expertise_score: float = 0.4,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Match project needs to experts (Automated Talent Marketplace matching).

        This allows companies to find experts for specific projects by matching
        project keywords to inferred expertise, effectively creating a project-to-talent
        matching system.

        Args:
            project_keywords: List of keywords describing the project
            required_topics: Optional list of topic IDs required for the project
            min_expertise_score: Minimum expertise score (default: 0.4)
            top_k: Number of expert matches to return (default: 10)

        Returns:
            DataFrame with matched experts
        """
        if not self._is_fitted:
            raise ValueError("Must fit expertise engine before matching")

        # Find matching topics for project keywords
        matching_topics = set()
        for keyword in project_keywords:
            topic_experts = self.search_experts_by_keyword(keyword, top_k=5)
            if not topic_experts.empty and hasattr(topic_experts, "topic"):
                # Extract topics from experts (if available)
                matching_topics.update(topic_experts.get("topic", []))

        # If required topics specified, use those
        if required_topics:
            matching_topics = set(required_topics)

        if not matching_topics:
            logger.warning(f"No matching topics found for project keywords: {project_keywords}")
            return pd.DataFrame()

        # Get experts for matching topics
        all_experts = []
        for topic_id in matching_topics:
            experts = self.find_experts(topic_id, top_k=top_k)
            if not experts.empty:
                all_experts.append(experts)

        if not all_experts:
            return pd.DataFrame()

        # Combine and aggregate
        combined_df = pd.concat(all_experts, ignore_index=True)

        # Aggregate expertise scores across topics
        expert_scores = (
            combined_df.groupby("person_id")["expertise_score"]
            .agg(["mean", "count", "max"])
            .reset_index()
            .rename(
                columns={
                    "mean": "avg_expertise_score",
                    "count": "topic_count",
                    "max": "max_expertise_score",
                }
            )
        )

        # Calculate match score (weighted by topic count and max expertise)
        expert_scores["match_score"] = (
            expert_scores["avg_expertise_score"] * 0.6
            + (expert_scores["topic_count"] / len(matching_topics)) * 0.2
            + expert_scores["max_expertise_score"] * 0.2
        )

        # Filter by minimum expertise
        expert_scores = expert_scores[expert_scores["avg_expertise_score"] >= min_expertise_score]

        # Sort by match score
        matched_experts = expert_scores.sort_values("match_score", ascending=False)

        return matched_experts.head(top_k)

    def generate_talent_profile(self, person_id: str) -> dict:
        """Generate comprehensive talent profile for an individual.

        This creates a "talent profile" that shows all areas of expertise
        for an individual, making them discoverable in the marketplace.

        Args:
            person_id: Person identifier

        Returns:
            Dictionary with talent profile
        """
        if not self._is_fitted or self.expertise_inferencer is None:
            raise ValueError("Must fit expertise engine before generating profiles")

        expertise_df = self.transform(pd.DataFrame())  # Get all expertise

        if expertise_df.empty:
            return {
                "person_id": person_id,
                "expertise_areas": [],
                "total_topics": 0,
                "primary_expertise": None,
                "secondary_expertise": [],
            }

        # Filter to this person
        person_expertise = expertise_df[expertise_df["person_id"] == person_id]

        if person_expertise.empty:
            return {
                "person_id": person_id,
                "expertise_areas": [],
                "total_topics": 0,
                "primary_expertise": None,
                "secondary_expertise": [],
            }

        # Sort by expertise score
        person_expertise = person_expertise.sort_values("expertise_score", ascending=False)

        # Primary expertise (top score)
        primary = person_expertise.iloc[0] if not person_expertise.empty else None
        primary_expertise = (
            {
                "topic": int(primary["topic"]),
                "expertise_score": float(primary["expertise_score"]),
                "document_count": int(primary.get("document_count", 0)),
            }
            if primary is not None
            else None
        )

        # Secondary expertise (top 5)
        secondary = (
            person_expertise.head(6).tail(5)  # Skip first (primary), get next 5
            if len(person_expertise) > 1
            else pd.DataFrame()
        )
        secondary_expertise = (
            secondary[["topic", "expertise_score", "document_count"]].to_dict("records")
            if not secondary.empty
            else []
        )

        # All expertise areas
        all_expertise = person_expertise[["topic", "expertise_score", "document_count"]].to_dict(
            "records"
        )

        # Calculate talent summary
        total_topics = len(person_expertise)
        avg_expertise = person_expertise["expertise_score"].mean()
        max_expertise = person_expertise["expertise_score"].max()

        # Talent level classification
        if max_expertise >= 0.7:
            talent_level = "expert"
        elif max_expertise >= 0.5:
            talent_level = "experienced"
        elif max_expertise >= 0.3:
            talent_level = "emerging"
        else:
            talent_level = "beginner"

        return {
            "person_id": person_id,
            "expertise_areas": all_expertise,
            "total_topics": total_topics,
            "avg_expertise_score": avg_expertise,
            "max_expertise_score": max_expertise,
            "talent_level": talent_level,
            "primary_expertise": primary_expertise,
            "secondary_expertise": secondary_expertise,
            "profile_summary": (
                f"Expert in {total_topics} topics with {talent_level}-level expertise. "
                f"Primary expertise: Topic {primary_expertise['topic'] if primary_expertise else 'N/A'} "
                f"{(primary_expertise['expertise_score'] if primary_expertise else 0):.1%}"
            ),
        }

    def recommend_experts_for_need(
        self,
        skill_description: str,
        org_chart: list[str] | None = None,
        prefer_hidden: bool = True,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Recommend experts for a specific skill need (Automated Talent Marketplace recommendations).

        This is the core "talent marketplace" feature that recommends experts
        based on skill descriptions, prioritizing hidden experts not in the
        formal org chart.

        Args:
            skill_description: Description of skill/topic need
            org_chart: Optional list of person IDs in formal org chart
            prefer_hidden: If True, prefer experts not in org chart (default: True)
            top_k: Number of recommendations (default: 5)

        Returns:
            DataFrame with recommended experts
        """
        # Search for experts matching skill description
        experts_df = self.search_experts_by_keyword(skill_description, top_k=top_k * 2)

        if experts_df.empty:
            logger.warning(f"No experts found matching: {skill_description}")
            return pd.DataFrame()

        # Filter to hidden experts if requested
        if prefer_hidden and org_chart:
            hidden_experts = experts_df[~experts_df["person_id"].isin(org_chart)]
            if not hidden_experts.empty:
                experts_df = hidden_experts

        # Sort by expertise score
        recommended_experts = experts_df.sort_values("expertise_score", ascending=False)

        # Add recommendation reason
        recommended_experts["recommendation_reason"] = recommended_experts.apply(
            lambda row: (
                f"Expertise in '{skill_description}' with score {row['expertise_score']:.2f}"
            ),
            axis=1,
        )

        # Add talent profile links (for marketplace browsing)
        recommended_experts["has_profile"] = True

        return recommended_experts.head(top_k)

    def generate_marketplace_dashboard(self) -> dict:
        """Generate comprehensive talent marketplace dashboard.

        This creates a dashboard view of the entire Automated Talent Marketplace,
        showing top experts, popular topics, and marketplace statistics.

        Returns:
            Dictionary with marketplace dashboard data
        """
        if not self._is_fitted:
            raise ValueError("Must fit expertise engine before generating dashboard")

        expertise_df = self.transform(pd.DataFrame())  # Get all expertise

        if expertise_df.empty:
            return {
                "summary": "No expertise data available",
                "marketplace_stats": {},
            }

        # Marketplace statistics
        total_experts = expertise_df["person_id"].nunique()
        total_topics = expertise_df["topic"].nunique()
        total_expertise_areas = len(expertise_df)

        # Top experts (by max expertise score)
        top_experts = (
            expertise_df.sort_values("expertise_score", ascending=False)
            .groupby("person_id")
            .first()
            .head(20)
            .reset_index()
        )

        # Popular topics (by expert count)
        popular_topics = (
            expertise_df.groupby("topic")
            .agg({"person_id": "count", "expertise_score": "mean"})
            .rename(columns={"person_id": "expert_count", "expertise_score": "avg_expertise"})
            .sort_values("expert_count", ascending=False)
            .head(10)
            .reset_index()
        )

        # Expertise distribution
        expertise_distribution = {
            "expert": len(expertise_df[expertise_df["expertise_score"] >= 0.7]),
            "experienced": len(
                expertise_df[
                    (expertise_df["expertise_score"] >= 0.5)
                    & (expertise_df["expertise_score"] < 0.7)
                ]
            ),
            "emerging": len(
                expertise_df[
                    (expertise_df["expertise_score"] >= 0.3)
                    & (expertise_df["expertise_score"] < 0.5)
                ]
            ),
            "beginner": len(expertise_df[expertise_df["expertise_score"] < 0.3]),
        }

        return {
            "summary": (
                f"Automated Talent Marketplace: {total_experts} experts across {total_topics} topics"
            ),
            "marketplace_stats": {
                "total_experts": total_experts,
                "total_topics": total_topics,
                "total_expertise_areas": total_expertise_areas,
                "avg_expertise_per_person": (
                    total_expertise_areas / total_experts if total_experts > 0 else 0.0
                ),
                "avg_experts_per_topic": (
                    total_expertise_areas / total_topics if total_topics > 0 else 0.0
                ),
            },
            "top_experts": (
                top_experts[["person_id", "topic", "expertise_score"]].to_dict("records")
                if not top_experts.empty
                else []
            ),
            "popular_topics": popular_topics.to_dict("records") if not popular_topics.empty else [],
            "expertise_distribution": expertise_distribution,
        }
