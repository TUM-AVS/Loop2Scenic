"""
Document retrieval with advanced filtering and ranking.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.schema import ScenarioDocument, MultimodalQuery, RetrievalResult
from ..vectorstore import MilvusVectorStore
from ..reranker import BaseReranker

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever class for querying documents with tag-based filtering.
    """

    def __init__(
        self,
        vectorstore: MilvusVectorStore,
        reranker: Optional[BaseReranker] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ):
        """
        Initialize retriever.
        
        Args:
            vectorstore: MilvusVectorStore instance
            reranker: BaseReranker instance (optional; if None, vector order is kept)
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score threshold
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.reranker = reranker

    def retrieve(
        self,
        original_query: MultimodalQuery,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            original_query: Multimodal query
            query_embedding: Query embedding
            top_k: Number of documents to retrieve (overrides default)
            filter_tags: List of tags to filter by
            metadata_filter: Additional metadata filters

        Returns:
            RetrievalResult with ranked scenarios and scores for the top hit.
        """
        k = top_k or self.top_k
        
        logger.info(f"Retrieving documents")
        
        # 1. retrieve scenario ids and vector similarity scores from Milvus
        scenario_score_pairs = self.vectorstore.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filter_tags=filter_tags,
            metadata_filter=metadata_filter,
            score_threshold=self.similarity_threshold,
        )

        if len(scenario_score_pairs) == 0:
            logger.warning("No scenarios found for query")
            return RetrievalResult(scenarios=[])

        similarity_by_id = {scenario_id: score for scenario_id, score in scenario_score_pairs}

        # 2. get the original scenario documents from the local file system
        results: List[ScenarioDocument] = []
        for scenario_id, _ in scenario_score_pairs:
            scenario = self._get_original_scenario(scenario_id)
            if scenario is None:
                logger.warning(f"Scenario not found for ID: {scenario_id}")
                continue
            results.append(scenario)
        
        logger.info(f"Retrieved {len(results)} scenarios")

        if len(results) == 0:
            return RetrievalResult(scenarios=[])

        # 3. rerank the scenarios based on the query (or keep vector order)
        if self.reranker is not None:
            scored_docs = self.reranker.rerank_with_scores(
                query=original_query,
                documents=results,
                top_k=k,
            )
            if len(scored_docs) == 0:
                logger.warning("No scenarios found after reranking")
                return RetrievalResult(scenarios=[])

            best_scenario, best_rerank_score = scored_docs[0]
            best_scenarios = [doc for doc, _ in scored_docs]
            best_similarity_score = similarity_by_id.get(best_scenario.scenario_id)
            logger.info(
                f"Best scenario: {best_scenario.scenario_id} "
                f"(similarity={best_similarity_score}, rerank={best_rerank_score})"
            )
            return RetrievalResult(
                scenarios=best_scenarios,
                best_similarity_score=best_similarity_score,
                best_rerank_score=best_rerank_score,
            )

        best_scenario = results[0]
        best_similarity_score = similarity_by_id.get(best_scenario.scenario_id)
        logger.info(
            f"Best scenario (no reranker): {best_scenario.scenario_id} "
            f"(similarity={best_similarity_score})"
        )
        return RetrievalResult(
            scenarios=results[:k],
            best_similarity_score=best_similarity_score,
            best_rerank_score=None,
        )

    def _get_original_scenario(self, scenario_id: str) -> ScenarioDocument:
        """
        Get the original scenario from the local file system.
        """
        try:
            scenario_location = Path(f"data/scenarios/{scenario_id}").resolve() # use absolute path for reranker vlm
            
            scenario_description = None
            scenario_scenic_code = None
            video_path = None
            image_path = None

            # check if the files exist, if exist, read the content
            if (scenario_location / "new_description.txt").exists():
                with open(scenario_location / "new_description.txt", "r") as f:
                    scenario_description = f.read()
            elif (scenario_location / "description.txt").exists():
                with open(scenario_location / "description.txt", "r") as f:
                    scenario_description = f.read()

            if (scenario_location / "code.scenic").exists():
                with open(scenario_location / "code.scenic", "r") as f:
                    scenario_scenic_code = f.read()

            if (scenario_location / "video.mp4").exists():
                video_path = str(scenario_location / "video.mp4")

            if (scenario_location / "image.png").exists():
                image_path = str(scenario_location / "image.png")

            logger.info(f"Scenario ID: {scenario_id}, Video Path: {video_path}, Image Path: {image_path}")
                
            return ScenarioDocument(
                scenario_id=scenario_id,
                description=scenario_description,
                scenic_code=scenario_scenic_code,
                video_path=video_path,
                image_path=image_path
            )
        except Exception as e:
            logger.error(f"Error getting original scenario for ID: {scenario_id}: {e}")
            return None
