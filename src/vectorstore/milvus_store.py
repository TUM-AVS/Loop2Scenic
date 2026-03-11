"""
Milvus vector store implementation using native pymilvus (no LangChain wrapper).
Supports both Milvus Lite (no Docker) and Milvus Server modes.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from langchain_core.documents import Document
from pymilvus import MilvusClient, DataType

from ..embedding.embedder import Embedder

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """
    Native Milvus-based vector store with tag filtering capabilities.
    
    Uses pymilvus MilvusClient directly for better performance and full control.
    Supports both Milvus Lite (local file) and Milvus Server (Docker) modes.
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
        
        Supports two modes:
        1. Milvus Lite (no Docker): connection_args={"uri": "./milvus.db"}
        2. Milvus Server (Docker): connection_args={"host": "localhost", "port": "19530"}
        
        Args:
            embedder: Embedder instance for generating embeddings
            collection_name: Name of the Milvus collection
            connection_args: Connection arguments for Milvus
                - Lite mode: {"uri": "./path/to/milvus.db"}
                - Server mode: {"host": "localhost", "port": "19530"}
                Default: Lite mode with "./data/vector_db/milvus.db"
            index_params: Index parameters for Milvus
                Default: IVF_FLAT with nlist=1024
            search_params: Search parameters for Milvus
                Default: nprobe=10
        """
        self.embedder = embedder
        self.collection_name = collection_name
        
        # Default connection args (Milvus Lite for local development)
        if connection_args is None:
            connection_args = {"uri": "./data/vector_db/milvus.db"}
        
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
        
        # Build connection for MilvusClient
        if "uri" in connection_args:
            # Milvus Lite mode - local file database
            uri = connection_args["uri"]
            # Ensure parent directory exists
            db_path = Path(uri)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using Milvus Lite mode (no Docker): {uri}")
            # Initialize MilvusClient with URI
            self.client = MilvusClient(uri=uri)
        else:
            # Milvus Server mode - use host and port
            host = connection_args.get("host", "localhost")
            port = connection_args.get("port", "19530")
            logger.info(f"Using Milvus Server mode: {host}:{port}")
            # Initialize MilvusClient with host and port (NOT uri)
            self.client = MilvusClient(
                uri=f"http://{host}:{port}"
            )
        
        # Initialize collection
        self._init_collection()
        
        logger.info(f"Initialized Milvus vector store (collection: {collection_name})")

    def _init_collection(self):
        """Initialize or get existing collection."""
        embedding_dim = self.embedder.get_embedding_dimension()
        
        # Check if collection exists
        if self.client.has_collection(collection_name=self.collection_name):
            logger.info(f"Loaded existing collection: {self.collection_name}")
        else:
            # Create collection with schema
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            
            # Add fields
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            
            # Create index
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding", 
                index_type=self.index_params.get("index_type", "IVF_FLAT"),
                metric_type=self.index_params.get("metric_type", "COSINE"),
                params=self.index_params.get("params", {"nlist": 1024})
            )
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            
            logger.info(f"Created new collection: {self.collection_name}")

    def add_documents(
        self,
        documents: List[Dict[str, any]],
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of dictionaries with keys like:
                  {"id": "...", "embedding": "...", "metadata": "..."}
            ids: Optional list of IDs for the documents
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        def generate_id(document: Dict[str, Any]) -> str:
            """
            Generate an ID using the folder name. 
            Falls back to a UUID if no folder path is provided.
            """
            folder_path = document.get("folder_path")
            
            if folder_path:
                # Path().name cleanly extracts just the final folder name 
                # e.g., "/data/scenarios/scene_01" -> "scene_01"
                return Path(folder_path).name
                
            # Lazy fallback: Only generates a UUID if folder_path was None or empty
            return str(uuid4())
        
        ids = [generate_id(doc_dict) for doc_dict in documents]
        
        # Multimodal dictionaries (text, images, videos)
        data = []
        for doc_id, doc_dict in zip(ids, documents):
                data.append({
                    "id": doc_id,
                    "embedding": doc_dict["embedding"],
                    "metadata": doc_dict.get("metadata", {})
                })
        
        # Insert into collection
        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        
        # Flush to ensure data is persisted
        self.client.flush(collection_name=self.collection_name)
        
        logger.info(f"Successfully added {len(ids)} documents")
        
        return ids

    def similarity_search(
        self,
        query: Any,  # Can be str or Dict[str, Any]
        k: int = 5,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Search for similar documents using multimodal queries.
        
        Args:
            query: Either:
                - String: text query
                - Dictionary: multimodal query with keys like:
                  {"text": "...", "image": "...", "video": "...", "instruction": "..."}
            k: Number of results to return
            filter_tags: List of tags to filter by (documents must have at least one)
            metadata_filter: Additional metadata filters (Milvus expression format)
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar Document objects
            
        Examples:
            # Text query
            results = vector_store.similarity_search("Find me a cat")
            
            # Image query
            results = vector_store.similarity_search({"image": "path/to/cat.jpg"})
            
            # Multimodal query
            results = vector_store.similarity_search({
                "text": "Find similar images",
                "image": "reference.jpg"
            })
        """
        # Build filter expression
        expr = self._build_filter(filter_tags, metadata_filter)
        
        # Convert string query to dictionary format
        if isinstance(query, str):
            query_dict = {"text": query}
        else:
            query_dict = query
        
        logger.info(
            f"Searching for top {k} similar documents "
            f"(query type: {'text' if isinstance(query, str) else 'multimodal'}, "
            f"filter: {expr}, threshold: {score_threshold})"
        )
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query_dict)
        
        # Convert tensor to list if needed (Milvus requires list/array, not tensor)
        if hasattr(query_embedding, 'cpu'):
            # It's a PyTorch tensor
            query_embedding = query_embedding.cpu().tolist()
        elif hasattr(query_embedding, 'tolist'):
            # It's a numpy array
            query_embedding = query_embedding.tolist()
        
        # Search with extra results if threshold filtering
        search_k = k * 2 if score_threshold is not None else k
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            filter=expr if expr else "",
            limit=search_k,
            output_fields=["text", "metadata"],
            search_params=self.search_params
        )
        
        # Convert to Documents
        documents = []
        for hits in results:
            for hit in hits:
                # Apply score threshold if specified
                if score_threshold is not None and hit['distance'] < score_threshold:
                    continue
                
                doc = Document(
                    page_content=hit['entity'].get('text', ''),
                    metadata=hit['entity'].get('metadata', {})
                )
                documents.append(doc)
                
                if len(documents) >= k:
                    break
            
            if len(documents) >= k:
                break
        
        logger.info(f"Found {len(documents)} similar documents")
        
        return documents[:k]

    def similarity_search_with_score(
        self,
        query: Any,  # Can be str or Dict[str, Any]
        k: int = 5,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with similarity scores using multimodal queries.
        
        Args:
            query: Either:
                - String: text query
                - Dictionary: multimodal query with keys like:
                  {"text": "...", "image": "...", "video": "...", "instruction": "..."}
            k: Number of results to return
            filter_tags: List of tags to filter by
            metadata_filter: Additional metadata filters
            
        Returns:
            List of (Document, score) tuples
            
        Examples:
            # Text query
            results = vector_store.similarity_search_with_score("Find me a cat")
            
            # Image query
            results = vector_store.similarity_search_with_score({
                "image": "path/to/query.jpg"
            })
        """
        expr = self._build_filter(filter_tags, metadata_filter)
        
        # Convert string query to dictionary format
        if isinstance(query, str):
            query_dict = {"text": query}
        else:
            query_dict = query
        
        logger.info(
            f"Searching for top {k} similar documents with scores "
            f"(query type: {'text' if isinstance(query, str) else 'multimodal'})"
        )
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query_dict)
        
        # Convert tensor to list if needed (Milvus requires list/array, not tensor)
        if hasattr(query_embedding, 'cpu'):
            # It's a PyTorch tensor
            query_embedding = query_embedding.cpu().tolist()
        elif hasattr(query_embedding, 'tolist'):
            # It's a numpy array
            query_embedding = query_embedding.tolist()
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            filter=expr if expr else "",
            limit=k,
            output_fields=["text", "metadata"],
            search_params=self.search_params
        )
        
        # Convert to (Document, score) tuples
        doc_score_pairs = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    page_content=hit['entity'].get('text', ''),
                    metadata=hit['entity'].get('metadata', {})
                )
                score = hit['distance']  # Cosine similarity score
                doc_score_pairs.append((doc, score))
        
        logger.info(f"Found {len(doc_score_pairs)} similar documents")
        
        return doc_score_pairs

    def _build_filter(
        self,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[str] = None
    ) -> Optional[str]:
        """
        Build a filter expression for Milvus queries.
        
        Milvus uses string expressions for filtering:
        - JSON contains: 'json_contains(metadata["tags"], "python")'
        - OR: 'json_contains(metadata["tags"], "python") || json_contains(metadata["tags"], "ml")'
        - AND: 'metadata["year"] == 2024 && metadata["category"] == "tutorial"'
        
        Args:
            filter_tags: List of tags to filter by
            metadata_filter: Additional metadata filter expression
            
        Returns:
            Filter expression string for Milvus
        """
        expressions = []
        
        # Add tag filters (OR operation for multiple tags)
        if filter_tags:
            tag_exprs = [f'json_contains(metadata["tags"], "{tag}")' for tag in filter_tags]
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
                        meta_exprs.append(f'metadata["{key}"] == "{value}"')
                    else:
                        meta_exprs.append(f'metadata["{key}"] == {value}')
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
        
        # Build filter expression for deletion
        id_list_str = ', '.join([f'"{id_}"' for id_ in ids])
        expr = f"id in [{id_list_str}]"
        
        self.client.delete(
            collection_name=self.collection_name,
            filter=expr
        )
        
        logger.info("Documents deleted successfully")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        # Get collection stats
        stats_result = self.client.get_collection_stats(collection_name=self.collection_name)
        row_count = stats_result.get('row_count', 0)
        
        stats = {
            "collection_name": self.collection_name,
            "document_count": row_count,
            "index_type": self.index_params.get("index_type", "IVF_FLAT"),
            "metric_type": self.index_params.get("metric_type", "COSINE")
        }
        
        # Add connection info
        if "uri" in self.connection_args:
            stats["mode"] = "Lite"
            stats["uri"] = self.connection_args["uri"]
        else:
            stats["mode"] = "Server"
            stats["host"] = self.connection_args.get("host", "localhost")
            stats["port"] = self.connection_args.get("port", "19530")
        
        logger.info(f"Collection stats: {stats}")
        
        return stats

    def reset_collection(self) -> None:
        """Reset (delete all documents from) the collection."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        
        # Drop the collection if it exists
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
            logger.info(f"Dropped collection: {self.collection_name}")
        
        # Recreate the collection
        self._init_collection()
        
        logger.info("Collection reset successfully")
