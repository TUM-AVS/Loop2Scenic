"""
Reranker module for reordering retrieved documents based on relevance.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union

from langchain_core.documents import Document


class BaseReranker(ABC):
    """
    Abstract base class for rerankers.
    
    Rerankers take an initial list of documents retrieved from a vector store
    and reorder them based on more sophisticated relevance scoring.
    """

    @abstractmethod
    def rerank(
        self,
        query: Union[str, dict[str, Any]],
        documents: List[Document],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Rerank a list of documents based on a query.
        
        Args:
            query: Query string or multimodal query dictionary
                (e.g., {"text": "...", "image": "...", "video": "..."})
            documents: List of documents to rerank
            top_k: Optional limit on number of documents to return after reranking
            **kwargs: Additional arguments for specific reranker implementations
            
        Returns:
            List of reranked Document objects, ordered by relevance (most relevant first)
        """
        pass

    @abstractmethod
    def rerank_with_scores(
        self,
        query: Union[str, dict[str, Any]],
        documents: List[Document],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        Rerank documents and return them with relevance scores.
        
        Args:
            query: Query string or multimodal query dictionary
            documents: List of documents to rerank
            top_k: Optional limit on number of documents to return
            **kwargs: Additional arguments for specific reranker implementations
            
        Returns:
            List of (Document, score) tuples, ordered by score (highest first)
        """
        pass
