"""
Document retrieval with advanced filtering and ranking.
"""

import logging
from typing import Dict, List, Optional, Any

from src.schema import ScenarioDocument
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
        reranker: BaseReranker,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ):
        """
        Initialize retriever.
        
        Args:
            vectorstore: MilvusVectorStore instance
            reranker: BaseReranker instance
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score threshold
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.reranker = reranker

    def retrieve(
        self,
        query: Dict[str, Any],
        top_k: Optional[int] = None,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ScenarioDocument]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query dictionary
            top_k: Number of documents to retrieve (overrides default)
            filter_tags: List of tags to filter by
            metadata_filter: Additional metadata filters
            
        Returns:
            List of scenario dictionaries
        """
        k = top_k or self.top_k
        
        logger.info(
            f"Retrieving documents for query: '{query[:50]}...' "
            f"(top_k={k}, tags={filter_tags})"
        )
        
        # 1. retrieve scenario ids from vector store via similarity search
        scenario_ids = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter_tags=filter_tags,
            metadata_filter=metadata_filter,
            score_threshold=self.similarity_threshold
        )

        if len(scenario_ids) == 0:
            logger.warning("No scenarios found for query")
            return []

        # 2. get the original scenario documents from the local file system
        results: List[ScenarioDocument] = []
        for scenario_id in scenario_ids:
            scenario = self._get_original_scenario(scenario_id)
            if scenario is None:
                logger.warning(f"Scenario not found for ID: {scenario_id}")
                continue
            results.append(scenario)
        
        logger.info(f"Retrieved {len(results)} scenarios")

        # 3. rerank the scenarios based on the query
        best_scenarios = self.reranker.rerank(
            query=query,
            documents=results,
            top_k=k # return the best k scenarios
        )

        logger.info(f"Best scenario: {best_scenarios[0].scenario_id}")

        return best_scenarios

    def _get_original_scenario(self, scenario_id: str) -> ScenarioDocument:
        """
        Get the original scenario from the local file system.
        """
        try:
            scenario_location = f"data/scenarios/{scenario_id}"
            
            scenario_description = None
            scenario_scenic_code = None
            video_path = None
            image_path = None

            # check if the files exist, if exist, read the content
            if (scenario_location / "description.txt").exists():
                with open(scenario_location / "description.txt", "r") as f:
                    scenario_description = f.read()

            if (scenario_location / "code.scenic").exists():
                with open(scenario_location / "code.scenic", "r") as f:
                    scenario_scenic_code = f.read()

            if (scenario_location / "video.mp4").exists():
                video_path = str(scenario_location / "video.mp4")

            if (scenario_location / "image.png").exists():
                image_path = str(scenario_location / "image.png")
                
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