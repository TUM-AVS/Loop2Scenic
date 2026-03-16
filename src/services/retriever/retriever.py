"""
Document retrieval with advanced filtering and ranking.
"""

import logging
from typing import Dict, List, Optional, Any

from langchain_core.documents import Document

from ..vectorstore import MilvusVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever class for querying documents with tag-based filtering.
    """

    def __init__(
        self,
        vectorstore: MilvusVectorStore,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        enable_reranking: bool = False
    ):
        """
        Initialize retriever.
        
        Args:
            vectorstore: MilvusVectorStore instance
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score threshold
            enable_reranking: Whether to enable reranking (future feature)
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.enable_reranking = enable_reranking

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        return_scores: bool = False
    ) -> List[Document] | List[tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (overrides default)
            filter_tags: List of tags to filter by
            metadata_filter: Additional metadata filters
            return_scores: Whether to return similarity scores
            
        Returns:
            List of Documents or list of (Document, score) tuples
        """
        k = top_k or self.top_k
        
        logger.info(
            f"Retrieving documents for query: '{query[:50]}...' "
            f"(top_k={k}, tags={filter_tags})"
        )
        
        if return_scores:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter_tags=filter_tags,
                metadata_filter=metadata_filter
            )
            
            # Apply threshold if specified
            if self.similarity_threshold:
                results = [
                    (doc, score) for doc, score in results
                    if score >= self.similarity_threshold
                ]
            
            logger.info(f"Retrieved {len(results)} documents with scores")
            return results
        else:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter_tags=filter_tags,
                metadata_filter=metadata_filter,
                score_threshold=self.similarity_threshold
            )
            
            logger.info(f"Retrieved {len(results)} documents")
            return results

    def retrieve_by_tags(
        self,
        query: str,
        required_tags: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = False
    ) -> List[Document] | List[tuple[Document, float]]:
        """
        Retrieve documents that must contain specific tags.
        
        Args:
            query: Query string
            required_tags: List of tags that documents must have
            top_k: Number of documents to retrieve
            return_scores: Whether to return similarity scores
            
        Returns:
            List of Documents or list of (Document, score) tuples
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_tags=required_tags,
            return_scores=return_scores
        )

    def retrieve_by_source(
        self,
        query: str,
        source: str,
        top_k: Optional[int] = None,
        return_scores: bool = False
    ) -> List[Document] | List[tuple[Document, float]]:
        """
        Retrieve documents from a specific source.
        
        Args:
            query: Query string
            source: Source file or identifier
            top_k: Number of documents to retrieve
            return_scores: Whether to return similarity scores
            
        Returns:
            List of Documents or list of (Document, score) tuples
        """
        metadata_filter = {"source": source}
        
        return self.retrieve(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            return_scores=return_scores
        )