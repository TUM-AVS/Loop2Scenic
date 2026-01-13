"""
Milvus vector store implementation with tag-based filtering.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain.schema import Document
from langchain_community.vectorstores import Milvus
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

from ..embedding.embedder import Embedder

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """
    Milvus-based vector store with tag filtering capabilities.
    
    This class provides methods to store, retrieve, and filter documents
    based on their embeddings and metadata tags.
    """

    def __init__(
        self,
        embedder: Embedder,
        collection_name: str = "documents",
        connection_args: Optional[Dict[str, Any]] = None,
        index_params: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Milvus vector store.
        
        Args:
            embedder: Embedder instance for generating embeddings
            collection_name: Name of the Milvus collection
            connection_args: Connection arguments for Milvus
                Default: {"host": "localhost", "port": "19530"}
            index_params: Index parameters for Milvus
                Default: IVF_FLAT with nlist=1024
            search_params: Search parameters for Milvus
                Default: nprobe=10
        """
        self.embedder = embedder
        self.collection_name = collection_name
        
        # Default connection args (Milvus Lite for local development)
        if connection_args is None:
            connection_args = {
                "host": "localhost",
                "port": "19530"
            }
        
        self.connection_args = connection_args
        
        # Default index params
        if index_params is None:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
        self.index_params = index_params
        
        # Default search params
        if search_params is None:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        self.search_params = search_params
        
        # Initialize Langchain Milvus wrapper
        self.vectorstore = Milvus(
            embedding_function=embedder.embeddings,
            collection_name=collection_name,
            connection_args=connection_args,
            index_params=index_params,
            search_params=search_params
        )
        
        logger.info(
            f"Initialized Milvus vector store "
            f"(collection: {collection_name}, host: {connection_args.get('host')})"
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
        
        # Ensure tags are stored as JSON-serializable format
        for doc in documents:
            if 'tags' in doc.metadata and isinstance(doc.metadata['tags'], list):
                # Milvus supports arrays, keep as list
                pass
        
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
            metadata_filter: Additional metadata filters (Milvus expression format)
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar Document objects
        """
        # Build filter expression
        expr = self._build_filter(filter_tags, metadata_filter)
        
        logger.info(
            f"Searching for top {k} similar documents "
            f"(filter: {expr}, threshold: {score_threshold})"
        )
        
        # Milvus uses expr parameter for filtering
        search_kwargs = {"expr": expr} if expr else {}
        
        if score_threshold is not None:
            # Get more results and filter by score
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k * 2,  # Get extra to account for filtering
                **search_kwargs
            )
            # Filter by threshold (Milvus cosine similarity: higher is better, range 0-1)
            documents = [doc for doc, score in results if score >= score_threshold][:k]
        else:
            documents = self.vectorstore.similarity_search(
                query=query,
                k=k,
                **search_kwargs
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
        expr = self._build_filter(filter_tags, metadata_filter)
        
        logger.info(f"Searching for top {k} similar documents with scores")
        
        search_kwargs = {"expr": expr} if expr else {}
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            **search_kwargs
        )
        
        logger.info(f"Found {len(results)} similar documents")
        
        return results

    def _build_filter(
        self,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[str] = None
    ) -> Optional[str]:
        """
        Build a filter expression for Milvus queries.
        
        Milvus uses string expressions for filtering:
        - Array contains: 'array_contains(tags, "python")'
        - OR: 'array_contains(tags, "python") || array_contains(tags, "ml")'
        - AND: 'year == 2024 && category == "tutorial"'
        
        Args:
            filter_tags: List of tags to filter by
            metadata_filter: Additional metadata filter expression
            
        Returns:
            Filter expression string for Milvus
        """
        expressions = []
        
        # Add tag filters (OR operation for multiple tags)
        if filter_tags:
            tag_exprs = [f'array_contains(tags, "{tag}")' for tag in filter_tags]
            if len(tag_exprs) == 1:
                expressions.append(tag_exprs[0])
            else:
                # Join with OR
                tag_expr = " || ".join(tag_exprs)
                expressions.append(f"({tag_expr})")
        
        # Add metadata filter
        if metadata_filter:
            if isinstance(metadata_filter, dict):
                # Convert dict to Milvus expression
                meta_exprs = []
                for key, value in metadata_filter.items():
                    if isinstance(value, str):
                        meta_exprs.append(f'{key} == "{value}"')
                    else:
                        meta_exprs.append(f'{key} == {value}')
                if meta_exprs:
                    expressions.append(" && ".join(meta_exprs))
            else:
                # Assume it's already a string expression
                expressions.append(metadata_filter)
        
        # Combine with AND
        if not expressions:
            return None
        elif len(expressions) == 1:
            return expressions[0]
        else:
            return " && ".join(f"({expr})" for expr in expressions)

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
        # Connect to Milvus
        connections.connect(**self.connection_args)
        
        # Get collection
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            collection.load()
            count = collection.num_entities
        else:
            count = 0
        
        stats = {
            "collection_name": self.collection_name,
            "document_count": count,
            "host": self.connection_args.get("host", "localhost"),
            "port": self.connection_args.get("port", "19530"),
            "index_type": self.index_params.get("index_type", "IVF_FLAT"),
            "metric_type": self.index_params.get("metric_type", "COSINE")
        }
        
        logger.info(f"Collection stats: {stats}")
        
        return stats

    def reset_collection(self) -> None:
        """Reset (delete all documents from) the collection."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        
        # Connect to Milvus
        connections.connect(**self.connection_args)
        
        # Drop the collection if it exists
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Dropped collection: {self.collection_name}")
        
        # Recreate the vectorstore (will create new collection)
        self.vectorstore = Milvus(
            embedding_function=self.embedder.embeddings,
            collection_name=self.collection_name,
            connection_args=self.connection_args,
            index_params=self.index_params,
            search_params=self.search_params
        )
        
        logger.info("Collection reset successfully")
