"""Knowledge Graph for Problem-Solution Mapping.

This module combines Graph Construction and NLP topic modeling to create a
searchable "Knowledge Graph". This allows COOs to search for a "problem" and
instantly see which groups have already worked on "solutions."
"""

from collections import defaultdict
from datetime import datetime

import networkx as nx
import pandas as pd

from orgnet.metrics.community import CommunityDetector
from orgnet.nlp.topics import TopicModeler
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class KnowledgeGraph:
    """Knowledge Graph for Problem-Solution Mapping.

    Combines graph construction with NLP topic modeling to create a searchable
    "Knowledge Graph" where COOs can search for a "problem" and instantly see
    which groups have already worked on "solutions."

    Example: Search for "API Gateway Optimization" and find that the Platform
    Team already solved this problem with a documented solution.
    """

    def __init__(
        self,
        graph: nx.Graph,
        documents: list | None = None,
        code_commits: list | None = None,
        slack_messages: list | None = None,
        **params,
    ):
        """Initialize knowledge graph.

        Args:
            graph: Organizational network graph
            documents: Optional list of documents for topic modeling
            code_commits: Optional list of code commits for solution mapping
            slack_messages: Optional list of Slack messages for problem extraction
        """
        self.graph = graph
        self.documents = documents or []
        self.code_commits = code_commits or []
        self.slack_messages = slack_messages or []

        # Initialize components
        self.topic_modeler = TopicModeler(method="bertopic")
        self.community_detector = CommunityDetector(graph)

        # Knowledge graph structure
        self.knowledge_graph: nx.DiGraph | None = None
        self.problem_solution_map: dict[int, list[str]] = {}  # topic_id -> [solution_ids]
        self.solution_expert_map: dict[str, list[str]] = {}  # solution_id -> [expert_ids]

    def compute(
        self, graph: nx.Graph | None = None, documents: list | None = None, **kwargs
    ) -> pd.DataFrame:
        """Build knowledge graph from organizational data.

        Args:
            graph: Optional graph (uses self.graph if None)
            documents: Optional documents (uses self.documents if None)

        Returns:
            DataFrame with problem-solution mappings
        """
        if graph is not None:
            self.graph = graph

        if documents is not None:
            self.documents = documents

        # Step 1: Extract topics (problems/solutions)
        topics = self._extract_topics_from_data()

        # Step 2: Identify solutions (code commits, documents)
        solutions = self._identify_solutions()

        # Step 3: Map problems to solutions
        problem_solution_mappings = self._map_problems_to_solutions(topics, solutions)

        # Step 4: Map solutions to experts/communities
        solution_expert_mappings = self._map_solutions_to_experts(solutions)

        # Step 5: Build knowledge graph
        self.knowledge_graph = self._build_knowledge_graph(
            problem_solution_mappings, solution_expert_mappings
        )

        # Step 6: Generate searchable DataFrame
        searchable_df = self._generate_searchable_dataframe(
            problem_solution_mappings, solution_expert_mappings
        )

        return searchable_df

    def _extract_topics_from_data(self) -> dict:
        """Extract topics from digital exhaust.

        Returns:
            Dictionary with topic model results
        """
        # Combine all text sources
        all_texts = []

        # Add documents
        for doc in self.documents:
            if hasattr(doc, "text"):
                all_texts.append(doc.text)
            elif isinstance(doc, str):
                all_texts.append(doc)

        # Add Slack messages
        for msg in self.slack_messages:
            if hasattr(msg, "text"):
                all_texts.append(msg.text)
            elif isinstance(msg, str):
                all_texts.append(msg)

        # Add code commit messages
        for commit in self.code_commits:
            if hasattr(commit, "message"):
                all_texts.append(commit.message)
            elif isinstance(commit, dict) and "message" in commit:
                all_texts.append(commit["message"])

        if not all_texts:
            logger.warning("No text data available for topic extraction")
            return {"topics": [], "topic_names": []}

        # Fit topic modeler
        try:
            self.topic_modeler.fit(all_texts)
            topics = self.topic_modeler.get_topics()

            # Get topic info
            topic_info = self.topic_modeler.get_topic_info()

            return {
                "topics": topics,
                "topic_info": topic_info,
                "topic_names": [f"Topic {i}" for i in range(len(topics))],
                "texts": all_texts,
            }
        except Exception as e:
            logger.warning(f"Error extracting topics: {e}")
            return {"topics": [], "topic_names": []}

    def _identify_solutions(self) -> list[dict]:
        """Identify solutions from code commits and documents.

        Returns:
            List of solution dictionaries
        """
        solutions = []

        # Solutions from code commits (implementation)
        for commit in self.code_commits:
            solution_id = f"commit_{len(solutions)}"

            commit_message = (
                commit.message if hasattr(commit, "message") else commit.get("message", "")
            )
            author_id = (
                commit.author_id
                if hasattr(commit, "author_id")
                else commit.get("author_id", "unknown")
            )
            repository = (
                commit.repository
                if hasattr(commit, "repository")
                else commit.get("repository", "unknown")
            )

            solutions.append(
                {
                    "solution_id": solution_id,
                    "type": "code",
                    "description": commit_message,
                    "author_id": author_id,
                    "repository": repository,
                    "timestamp": (
                        commit.timestamp
                        if hasattr(commit, "timestamp")
                        else commit.get("timestamp", datetime.now())
                    ),
                }
            )

        # Solutions from documents (documentation)
        for doc in self.documents:
            solution_id = f"doc_{len(solutions)}"

            doc_text = doc.text if hasattr(doc, "text") else str(doc)
            doc_author = (
                doc.author_id if hasattr(doc, "author_id") else doc.get("author_id", "unknown")
            )

            solutions.append(
                {
                    "solution_id": solution_id,
                    "type": "documentation",
                    "description": doc_text[:500],  # First 500 chars
                    "author_id": doc_author,
                    "repository": "documents",
                    "timestamp": (
                        doc.timestamp
                        if hasattr(doc, "timestamp")
                        else doc.get("timestamp", datetime.now())
                    ),
                }
            )

        return solutions

    def _map_problems_to_solutions(
        self, topics: dict, solutions: list[dict]
    ) -> dict[int, list[str]]:
        """Map problems (topics) to solutions.

        Args:
            topics: Topic model results
            solutions: List of solution dictionaries

        Returns:
            Dictionary mapping topic_id to list of solution_ids
        """
        problem_solution_map = defaultdict(list)

        if not topics.get("topics"):
            return problem_solution_map

        # Get topic assignments for solutions
        solution_texts = [s["description"] for s in solutions]

        if not solution_texts:
            return problem_solution_map

        try:
            # Transform solutions to topics
            topic_assignments = self.topic_modeler.transform(solution_texts)

            # Map each solution to its primary topic
            for i, solution in enumerate(solutions):
                if i < len(topic_assignments):
                    topic_id = topic_assignments[i]
                    solution_id = solution["solution_id"]

                    # Only map if topic is valid (not -1)
                    if topic_id >= 0:
                        problem_solution_map[topic_id].append(solution_id)
        except Exception as e:
            logger.warning(f"Error mapping problems to solutions: {e}")

        return dict(problem_solution_map)

    def _map_solutions_to_experts(self, solutions: list[dict]) -> dict[str, list[str]]:
        """Map solutions to experts/communities.

        Args:
            solutions: List of solution dictionaries

        Returns:
            Dictionary mapping solution_id to list of expert_ids
        """
        solution_expert_map = defaultdict(list)

        # Detect communities
        communities_result = self.community_detector.detect_communities(method="louvain")
        communities = communities_result.get("communities", [])

        # Build node to community mapping
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                node_to_community[node] = comm_id

        # Map solutions to experts based on authors
        for solution in solutions:
            solution_id = solution["solution_id"]
            author_id = solution["author_id"]

            # Add author as expert
            if author_id in self.graph:
                solution_expert_map[solution_id].append(author_id)

                # Add community members as related experts
                if author_id in node_to_community:
                    comm_id = node_to_community[author_id]
                    comm_nodes = communities[comm_id] if comm_id < len(communities) else []
                    solution_expert_map[solution_id].extend(comm_nodes[:5])  # Top 5 from community

        return dict(solution_expert_map)

    def _build_knowledge_graph(
        self,
        problem_solution_map: dict[int, list[str]],
        solution_expert_map: dict[str, list[str]],
    ) -> nx.DiGraph:
        """Build knowledge graph connecting problems to solutions to experts.

        Args:
            problem_solution_map: Mapping of topic_id to solution_ids
            solution_expert_map: Mapping of solution_id to expert_ids

        Returns:
            Directed graph representing knowledge graph
        """
        KG = nx.DiGraph()

        # Add problem nodes (topics)
        for topic_id in problem_solution_map.keys():
            KG.add_node(f"problem_{topic_id}", type="problem", topic_id=topic_id)

        # Add solution nodes
        for solution_id in solution_expert_map.keys():
            KG.add_node(solution_id, type="solution", solution_id=solution_id)

        # Add expert nodes
        all_experts = set()
        for expert_list in solution_expert_map.values():
            all_experts.update(expert_list)

        for expert_id in all_experts:
            if expert_id in self.graph:
                KG.add_node(f"expert_{expert_id}", type="expert", person_id=expert_id)

        # Add problem -> solution edges
        for topic_id, solution_ids in problem_solution_map.items():
            problem_node = f"problem_{topic_id}"
            for solution_id in solution_ids:
                KG.add_edge(problem_node, solution_id, relationship="has_solution")

        # Add solution -> expert edges
        for solution_id, expert_ids in solution_expert_map.items():
            for expert_id in expert_ids:
                expert_node = f"expert_{expert_id}"
                KG.add_edge(solution_id, expert_node, relationship="implemented_by")

        logger.info(
            f"Built knowledge graph: {KG.number_of_nodes()} nodes, {KG.number_of_edges()} edges"
        )

        return KG

    def _generate_searchable_dataframe(
        self,
        problem_solution_map: dict[int, list[str]],
        solution_expert_map: dict[str, list[str]],
    ) -> pd.DataFrame:
        """Generate searchable DataFrame for problem-solution queries.

        Args:
            problem_solution_map: Mapping of topic_id to solution_ids
            solution_expert_map: Mapping of solution_id to expert_ids

        Returns:
            DataFrame with problem-solution mappings
        """
        records = []

        for topic_id, solution_ids in problem_solution_map.items():
            for solution_id in solution_ids:
                expert_ids = solution_expert_map.get(solution_id, [])

                records.append(
                    {
                        "problem_topic_id": topic_id,
                        "problem_topic_name": f"Topic {topic_id}",
                        "solution_id": solution_id,
                        "solution_type": solution_id.split("_")[0],  # "commit" or "doc"
                        "expert_count": len(expert_ids),
                        "expert_ids": expert_ids,
                        "primary_expert": expert_ids[0] if expert_ids else None,
                    }
                )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        return df

    def search_problems(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Search for problems and find existing solutions.

        This allows COOs to search for a "problem" and instantly see which
        groups have already worked on "solutions."

        Args:
            query: Problem description query (e.g., "API Gateway Optimization")
            top_k: Number of results to return (default: 10)

        Returns:
            DataFrame with problem-solution matches
        """
        if self.knowledge_graph is None:
            self.compute()

        if self.knowledge_graph is None:
            logger.warning("Knowledge graph not built")
            return pd.DataFrame()

        # Use topic modeler to find matching topics
        try:
            # Validate query
            if not query or not isinstance(query, str) or len(query.strip()) == 0:
                logger.warning("Invalid query provided")
                return pd.DataFrame()

            # Ensure topic modeler is fitted
            if not hasattr(self.topic_modeler, "model") or self.topic_modeler.model is None:
                logger.warning("Topic modeler not fitted. Cannot search without trained model.")
                return pd.DataFrame()

            # Transform query to find similar topics
            query_topics = self.topic_modeler.transform([query.strip()])

            if not query_topics or len(query_topics) == 0 or query_topics[0] < 0:
                logger.warning(f"No matching topics found for query: {query}")
                return pd.DataFrame()

            matched_topic_id = int(query_topics[0])

            # Find solutions for this topic
            problem_node = f"problem_{matched_topic_id}"

            if problem_node not in self.knowledge_graph:
                logger.warning(f"Problem node not found in knowledge graph: {problem_node}")
                return pd.DataFrame()

            # Get solutions connected to this problem
            solution_nodes = [
                node
                for node in self.knowledge_graph.successors(problem_node)
                if self.knowledge_graph.nodes[node].get("type") == "solution"
            ]

            # Get expert information for each solution
            results = []
            for solution_node in solution_nodes[:top_k]:
                solution_data = self.knowledge_graph.nodes[solution_node]
                expert_nodes = [
                    node
                    for node in self.knowledge_graph.successors(solution_node)
                    if self.knowledge_graph.nodes[node].get("type") == "expert"
                ]

                results.append(
                    {
                        "query": query,
                        "matched_topic_id": matched_topic_id,
                        "matched_topic_name": f"Topic {matched_topic_id}",
                        "solution_id": solution_node,
                        "solution_type": solution_data.get("solution_id", "").split("_")[0],
                        "expert_count": len(expert_nodes),
                        "expert_ids": [
                            self.knowledge_graph.nodes[node].get("person_id", "")
                            for node in expert_nodes
                        ],
                        "primary_expert": (
                            self.knowledge_graph.nodes[expert_nodes[0]].get("person_id", "")
                            if expert_nodes
                            else None
                        ),
                        "recommendation": (
                            f"Found existing solution: {solution_node}. "
                            f"Contact {self.knowledge_graph.nodes[expert_nodes[0]].get('person_id', 'N/A')} "
                            f"(expert in this topic) for details."
                            if expert_nodes
                            else f"Found existing solution: {solution_node}"
                        ),
                    }
                )

            return pd.DataFrame(results)

        except Exception as e:
            logger.warning(f"Error searching problems: {e}")
            return pd.DataFrame()

    def get_solution_details(self, solution_id: str) -> dict:
        """Get detailed information about a solution.

        Args:
            solution_id: Solution identifier

        Returns:
            Dictionary with solution details
        """
        if self.knowledge_graph is None:
            self.compute()

        if self.knowledge_graph is None or solution_id not in self.knowledge_graph:
            return {"error": f"Solution {solution_id} not found"}

        solution_data = self.knowledge_graph.nodes[solution_id]

        # Get experts who implemented this solution
        expert_nodes = [
            node
            for node in self.knowledge_graph.successors(solution_id)
            if self.knowledge_graph.nodes[node].get("type") == "expert"
        ]

        # Get problems this solution addresses
        problem_nodes = [
            node
            for node in self.knowledge_graph.predecessors(solution_id)
            if self.knowledge_graph.nodes[node].get("type") == "problem"
        ]

        return {
            "solution_id": solution_id,
            "solution_type": solution_data.get("solution_id", "").split("_")[0],
            "expert_count": len(expert_nodes),
            "expert_ids": [
                self.knowledge_graph.nodes[node].get("person_id", "") for node in expert_nodes
            ],
            "problem_count": len(problem_nodes),
            "problem_ids": [
                self.knowledge_graph.nodes[node].get("topic_id", -1) for node in problem_nodes
            ],
            "summary": (
                f"Solution {solution_id} addresses {len(problem_nodes)} problem(s) "
                f"and was implemented by {len(expert_nodes)} expert(s)."
            ),
        }

    def generate_knowledge_graph_report(self) -> dict:
        """Generate comprehensive knowledge graph report for COO.

        Returns:
            Dictionary with knowledge graph report
        """
        if self.knowledge_graph is None:
            self.compute()

        if self.knowledge_graph is None:
            return {
                "summary": "Knowledge graph not built",
                "total_problems": 0,
                "total_solutions": 0,
            }

        # Count nodes by type
        problem_nodes = [
            node
            for node in self.knowledge_graph.nodes()
            if self.knowledge_graph.nodes[node].get("type") == "problem"
        ]
        solution_nodes = [
            node
            for node in self.knowledge_graph.nodes()
            if self.knowledge_graph.nodes[node].get("type") == "solution"
        ]
        expert_nodes = [
            node
            for node in self.knowledge_graph.nodes()
            if self.knowledge_graph.nodes[node].get("type") == "expert"
        ]

        # Calculate solution coverage (problems with solutions)
        problems_with_solutions = sum(
            1 for node in problem_nodes if len(list(self.knowledge_graph.successors(node))) > 0
        )
        solution_coverage = problems_with_solutions / len(problem_nodes) if problem_nodes else 0.0

        return {
            "summary": (
                f"Knowledge Graph: {len(problem_nodes)} problems, "
                f"{len(solution_nodes)} solutions, {len(expert_nodes)} experts. "
                f"Solution coverage: {solution_coverage:.1%}"
            ),
            "total_problems": len(problem_nodes),
            "total_solutions": len(solution_nodes),
            "total_experts": len(expert_nodes),
            "problems_with_solutions": problems_with_solutions,
            "solution_coverage": solution_coverage,
            "avg_solutions_per_problem": (
                len(solution_nodes) / len(problem_nodes) if problem_nodes else 0.0
            ),
            "avg_experts_per_solution": (
                len(expert_nodes) / len(solution_nodes) if solution_nodes else 0.0
            ),
        }
