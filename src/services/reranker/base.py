"""
Reranker module for reordering retrieved documents based on relevance.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union

from src.schema import ScenarioDocument, MultimodalQuery

class BaseReranker(ABC):
    """
    Abstract base class for rerankers.
    
    Rerankers take an initial list of documents retrieved from a vector store
    and reorder them based on more sophisticated relevance scoring.
    """

    @abstractmethod
    def rerank(
        self,
        query: MultimodalQuery,
        documents: List[ScenarioDocument],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[ScenarioDocument]:
        """
        Rerank a list of documents based on a query.
        
        Args:
            query: Multimodal query
            documents: List of documents to rerank
            top_k: Optional limit on number of documents to return after reranking
            **kwargs: Additional arguments for specific reranker implementations
            
        Returns:
            List of reranked ScenarioDocument objects, ordered by relevance (most relevant first)
        """
        pass

    @abstractmethod
    def rerank_with_scores(
        self,
        query: MultimodalQuery,
        documents: List[ScenarioDocument],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[tuple[ScenarioDocument, float]]:
        """
        Rerank documents and return them with relevance scores.
        
        Args:
            query: Multimodal query
            documents: List of documents to rerank
            top_k: Optional limit on number of documents to return
            **kwargs: Additional arguments for specific reranker implementations
            
        Returns:
            List of (ScenarioDocument, score) tuples, ordered by score (highest first)
        """
        pass
