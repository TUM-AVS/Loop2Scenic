"""
Milvus vector store implementation using native pymilvus (no LangChain wrapper).
Supports both Milvus Lite (no Docker) and Milvus Server modes.
"""

import logging
import copy
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4
from langchain_core.documents import Document
from pymilvus import MilvusClient, DataType

from src.schema import ScenarioDocument

logger = logging.getLogger(__name__)

class MilvusVectorStore:
    """
    Native Milvus-based vector store with tag filtering capabilities.
    
    Uses pymilvus MilvusClient directly for better performance and full control.
    Supports both Milvus Lite (local file) and Milvus Server (Docker) modes.
    """

    def __init__(
        self,
        embedding_dim: int,
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
            embedding_dim: The dimension of the embeddings (e.g., 384, 1536)
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
        self.embedding_dim = embedding_dim
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
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
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
                return Path(folder_path).name
                
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
        query_embedding: Any,
        k: int = 5,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[str]:
        """
        Search for similar documents using a pre-computed query embedding.

        Args:
            query_embedding: Pre-computed vector (list of floats, numpy array, or tensor)
            k: Number of results to return
            filter_tags: List of tags to filter by
            metadata_filter: Additional metadata filters
            score_threshold: Minimum similarity score threshold

        Returns:
            List of scenario IDs
        """
        expr = self._build_filter(filter_tags, metadata_filter)
        
        logger.info(
            f"Searching for top {k} similar documents "
            f"(filter: {expr}, threshold: {score_threshold})"
        )
        
        # Convert tensor to list safely
        if hasattr(query_embedding, 'cpu'):
            query_embedding = query_embedding.cpu().tolist()
        elif hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
            
        local_search_params = copy.deepcopy(self.search_params)
        
        # Inject the threshold dynamically ONLY for this specific query!
        if score_threshold is not None:
            # Ensure the nested "params" dict exists
            if "params" not in local_search_params:
                local_search_params["params"] = {}
                
            local_search_params["params"]["radius"] = score_threshold
            local_search_params["params"]["range_filter"] = 1.0

        # Perform search asking for EXACTLY k results. 
        # Milvus will apply the threshold natively.
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            filter=expr if expr else "",
            limit=k,
            output_fields=[],
            search_params=local_search_params
        )
        
        scenario_ids = []
        scores = []
        
        for hits in results:
            for hit in hits:
                scenario_id = hit.get('id')
                score = hit.get('distance')
                
                scenario_ids.append(scenario_id)
                scores.append(score)
                
                logger.info(f"Retrieved Scenario ID: {scenario_id}, Score: {score}")

        # If it found nothing, it just safely returns an empty list!
        if not scenario_ids:
            logger.warning("No scenarios met the similarity threshold.")
            return []
            
        return scenario_ids

    def similarity_search_with_score(
        self,
        query_embedding: Any,  # Now accepts raw embeddings (List[float] or Tensor)
        k: int = 5,
        filter_tags: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with similarity scores using pre-computed embeddings.
        
        Args:
            query_embedding: Pre-computed vector (list of floats, numpy array, or tensor)
            k: Number of results to return
            filter_tags: List of tags to filter by
            metadata_filter: Additional metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        expr = self._build_filter(filter_tags, metadata_filter)
        
        logger.info(f"Searching for top {k} similar documents with scores")
        
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
            # Explicitly request the "id" field alongside "metadata"
            output_fields=["id", "metadata"],
            search_params=self.search_params
        )
        
        # Convert to (Document, score) tuples
        doc_score_pairs = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    id=hit.get('id', "unknown_id"),
                    page_content="",
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
        """Build a filter expression for Milvus queries."""
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
        """Delete documents by IDs."""
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
        """Get statistics about the collection."""
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

    def get_documents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve a sample of documents from the collection for inspection."""
        logger.info(f"Retrieving up to {limit} documents from {self.collection_name}...")
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter='id != ""',
            output_fields=["id", "metadata"], 
            limit=limit
        )
        
        return results