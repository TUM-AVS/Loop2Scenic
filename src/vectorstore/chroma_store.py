"""
ChromaDB vector store implementation with tag-based filtering.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from ..embedding.embedder import Embedder

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    ChromaDB-based vector store with tag filtering capabilities.
    
    This class provides methods to store, retrieve, and filter documents
    based on their embeddings and metadata tags.
    """

    def __init__(
        self,
        embedder: Embedder,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "documents",
        distance_metric: str = "cosine"
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            embedder: Embedder instance for generating embeddings
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.embedder = embedder
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize Langchain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedder.embeddings,
            collection_metadata={"hnsw:space": distance_metric}
        )
        
        logger.info(
            f"Initialized ChromaDB vector store at {persist_directory} "
            f"(collection: {collection_name}, metric: {distance_metric})"
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            ids: Optional list of IDs for the documents
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        doc_ids = self.vectorstore.add_documents(documents=documents, ids=ids)
        
        logger.info(f"Successfully added {len(doc_ids)} documents")
        
        return doc_ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_tags: List of tags to filter by (documents must have at least one)
            metadata_filter: Additional metadata filters
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar Document objects
        """
        # Build filter
        where_filter = self._build_filter(filter_tags, metadata_filter)
        
        logger.info(
            f"Searching for top {k} similar documents "
            f"(filter: {where_filter}, threshold: {score_threshold})"
        )
        
        if score_threshold is not None:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=where_filter,
                score_threshold=score_threshold
            )
            documents = [doc for doc, score in results]
        else:
            documents = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=where_filter
            )
        
        logger.info(f"Found {len(documents)} similar documents")
        
        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_tags: List of tags to filter by
            metadata_filter: Additional metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        where_filter = self._build_filter(filter_tags, metadata_filter)
        
        logger.info(f"Searching for top {k} similar documents with scores")
        
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=where_filter
        )
        
        logger.info(f"Found {len(results)} similar documents")
        
        return results

    def _build_filter(
        self,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Build a filter for ChromaDB queries.
        
        Args:
            filter_tags: List of tags to filter by
            metadata_filter: Additional metadata filters
            
        Returns:
            Filter dictionary for ChromaDB
        """
        filters = []
        
        # Add tag filters
        if filter_tags:
            # Filter for documents that have at least one of the specified tags
            tag_filters = [{"tags": {"$contains": tag}} for tag in filter_tags]
            if len(tag_filters) == 1:
                filters.append(tag_filters[0])
            else:
                filters.append({"$or": tag_filters})
        
        # Add metadata filters
        if metadata_filter:
            filters.append(metadata_filter)
        
        # Combine filters
        if not filters:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        logger.info(f"Deleting {len(ids)} documents")
        self.vectorstore.delete(ids=ids)
        logger.info("Documents deleted successfully")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        collection = self.client.get_collection(self.collection_name)
        count = collection.count()
        
        stats = {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(self.persist_directory),
            "distance_metric": self.distance_metric
        }
        
        logger.info(f"Collection stats: {stats}")
        
        return stats

    def reset_collection(self) -> None:
        """Reset (delete all documents from) the collection."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        
        # Recreate the collection
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedder.embeddings,
            collection_metadata={"hnsw:space": self.distance_metric}
        )
        
        logger.info("Collection reset successfully")
