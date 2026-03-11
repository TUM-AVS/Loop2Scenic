"""
Main RAG pipeline orchestration.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from .config import Config, get_config
from .vlm import get_vlm_service
from .ingestion import MultimodalDocumentInterpreter
from .embedding import get_embedder
from .vectorstore import MilvusVectorStore
from .retrieval import Retriever
from .generation import get_generator
from .utils import setup_logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates all components.
    
    This class provides a high-level interface for the entire RAG workflow:
    - Document ingestion and processing
    - Embedding generation
    - Vector storage with tag-based filtering
    - Document retrieval
    - Response generation
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            config: Configuration object (loads default if None)
        """
        # Load configuration
        self.config = config or get_config()
        
        # Setup logging
        setup_logging(
            level=self.config.logging.level,
            log_file=self.config.logging.file,
            log_format=self.config.logging.format
        )
        
        logger.info("Initializing RAG Pipeline")
        
        # Initialize components
        self._initialize_components()
        
        logger.info("RAG Pipeline initialized successfully")

    def _initialize_components(self):
        """Initialize all pipeline components."""
        self._initialize_vlm_service()
        self._initialize_multimodal_interpreter()
        embedding_dim = self._initialize_embedder()
        self._initialize_vector_store(embedding_dim)
        # self._initialize_retriever()
        # self._initialize_generator()

    def _initialize_vlm_service(self):
        """Initialize VLM (Vision Language Model) service."""
        logger.info("Initializing VLM Service...")

        # Define core System Prompt (Instruction) for scenario extraction
        scenario_extraction_instruction = (
            "You are an expert autonomous driving scenario analyzer. Your task is to extract "
            "the high-level logical structure of a driving scenario by analyzing the provided "
            "Scenic code, text description, image, and video.\n\n"
            "Output ONLY a valid JSON object. Do not include markdown or conversational text.\n"
        )
        
        # Dump the Pydantic config to a dictionary
        # Note: Use .model_dump() for Pydantic v2, or .dict() if you are on Pydantic v1
        vlm_kwargs = self.config.vlm.model_dump()
        
        # Override the default instruction with our specific scenario extraction prompt
        vlm_kwargs['default_instruction'] = scenario_extraction_instruction
        
        # Initialize the service
        # **vlm_kwargs will automatically map 'provider', 'temperature', etc., to the correct arguments
        self.vlm_service = get_vlm_service(**vlm_kwargs)
        
        logger.info(f"VLM Service initialized successfully with {self.config.vlm.provider}.")

    def _initialize_multimodal_interpreter(self):
        """Initialize multimodal document interpreter."""
        logger.info("Initializing Multimodal Document Interpreter...")
        self.multimodal_interpreter = MultimodalDocumentInterpreter()
        logger.info("Multimodal Document Interpreter initialized successfully.")

    def _initialize_embedder(self):
        """Initialize embedding model."""
        logger.info("Initializing Embedder...")
        
        embedder_kwargs = {}
        if hasattr(self.config.embedding, 'model_path') and self.config.embedding.model_path:
            embedder_kwargs['model_path'] = self.config.embedding.model_path
        
        self.embedder = get_embedder(
            provider=self.config.embedding.provider,
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            batch_size=self.config.embedding.batch_size,
            **embedder_kwargs
        )
        
        logger.info(f"Embedder initialized successfully with {self.config.embedding.provider}.")
        embedding_dim = self.embedder.get_embedding_dimension()
        logger.info(f"Embedding dimension: {embedding_dim}")
        return embedding_dim

    def _initialize_vector_store(self, embedding_dim: int):
        """Initialize vector store (Milvus)."""
        logger.info("Initializing Vector Store...")
        
        if not hasattr(self, 'embedder'):
            raise ValueError("Embedder must be initialized before vector store. Call _initialize_embedder() first.")
        
        # Milvus supports both Lite and Server modes
        use_lite = getattr(self.config.vector_db, "use_lite", True)
        
        if use_lite:
            # Milvus Lite (no Docker required)
            connection_args = {
                "uri": getattr(self.config.vector_db, "lite_db_path", "./data/vector_db/milvus.db")
            }
            logger.info(f"Using Milvus Lite mode (no Docker): {connection_args['uri']}")
            index_params = None
            search_params = None
        else:
            # Milvus Server (requires Docker)
            connection_args = {
                "host": getattr(self.config.vector_db, "host", "localhost"),
                "port": getattr(self.config.vector_db, "port", "19530")
            }
            logger.info(f"Using Milvus Server mode: {connection_args['host']}:{connection_args['port']}")
            
            index_params = {
                "metric_type": self.config.vector_db.distance_metric.upper(),
                "index_type": getattr(self.config.vector_db, "index_type", "IVF_FLAT"),
                "params": {"nlist": getattr(self.config.vector_db, "nlist", 1024)}
            }
            
            search_params = {
                "metric_type": self.config.vector_db.distance_metric.upper(),
                "params": {"nprobe": getattr(self.config.vector_db, "nprobe", 10)}
            }
        
        self.vectorstore = MilvusVectorStore(
            embedding_dim=embedding_dim,
            collection_name=self.config.vector_db.collection_name,
            connection_args=connection_args,
            index_params=index_params,
            search_params=search_params
        )
        
        logger.info("Vector Store initialized successfully.")

    def _initialize_retriever(self):
        """Initialize retriever."""
        logger.info("Initializing Retriever...")
        
        if not hasattr(self, 'vectorstore'):
            raise ValueError("Vector store must be initialized before retriever. Call _initialize_vector_store() first.")
        
        self.retriever = Retriever(
            vectorstore=self.vectorstore,
            top_k=self.config.retrieval.top_k,
            similarity_threshold=self.config.retrieval.similarity_threshold,
            enable_reranking=self.config.retrieval.enable_reranking
        )
        
        logger.info("Retriever initialized successfully.")

    def _initialize_generator(self):
        """Initialize LLM generator."""
        logger.info("Initializing Generator...")
        
        self.generator = get_generator(
            provider=self.config.llm.provider,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            streaming=self.config.llm.streaming
        )
        
        logger.info(f"Generator initialized successfully with {self.config.llm.provider}.")

    def interpret_scenarios(self, directory_path: Union[str, Path]):
        """
        Interpret scenarios from a directory and save the descriptions into the folder path's new_description.txt file.
        
        Args:
            directory_path: Path to the directory
        """
        scenarios_dicts = self.multimodal_interpreter.extract_from_directory(directory_path)
        for scenario_dict in scenarios_dicts:
            # 1. Get the raw string output from the VLM
            scenario_description_str = self.multimodal_interpreter.get_layer_model_description_by_vlm(
                scenario_dict, self.vlm_service
            )
            logger.info(f"The raw VLM output is: {scenario_description_str}")

            # 2. Parse the string into a Python dictionary
            try:
                scenario_data = json.loads(scenario_description_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from VLM output: {e}. Skipping this scenario.")
                logger.debug(f"Raw malformed output: {scenario_description_str}")
                continue  # Skip to the next scenario if the VLM failed to output valid JSON

            # 3. Flatten the JSON into plain text for optimal embedding
            text_parts = []
            text_parts.append(f"Scenario: {scenario_data.get('Scenario', '')}")
            text_parts.append(f"The ego vehicle is {scenario_data.get('Ego', '')}")
            
            adversarials = scenario_data.get('Adversarials', [])
            if adversarials:
                text_parts.append(f"Adversarial objects: {' '.join(adversarials)}")
            else:
                text_parts.append("There are no adversarials.")
                
            text_parts.append(f"Spatial Relation: {scenario_data.get('Spatial Relation', '')}")
            
            reqs = scenario_data.get('Requirement and restrictions', '')
            if reqs:
                text_parts.append(f"Requirements and restrictions: {reqs}")
                
            flattened_text = " ".join(text_parts)

            # 4. Ensure the folder exists
            folder_path = scenario_dict.get("folder_path")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # 5. Save the structured JSON file (for LLM context/metadata)
            json_path = Path(folder_path) / "new_description.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(scenario_data, f, indent=4)

            # 6. Save the flattened plain text file (for Vector Embedding)
            text_path = Path(folder_path) / "new_description.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(flattened_text)
                
            logger.info(f"Successfully saved JSON and Text descriptions to {folder_path}")

    def ingest_scenarios(self, directory_path: Union[str, Path]):
        """
        Ingest scenarios from a directory into the vector store.
        
        Args:
            directory_path: Path to the directory

        Returns:
            List of document IDs
        """
        self.interpret_scenarios(directory_path)
        scenarios_dicts = self.multimodal_interpreter.extract_from_directory(directory_path, use_new_description=True)
        logger.info(f"The first scenario dictionary: {scenarios_dicts[0]}")
        return scenarios_dicts

    def embed_scenarios(self, scenarios_dicts: List[Dict[str, any]]):
        """
        Embed scenarios.
        
        Args:
            scenarios_dicts: List of scenario dictionaries
        """

        # get the fields
        content_fields = ["description", "image", "video"]
        scenarios_dicts_with_content = []
        for scenario_dict in scenarios_dicts:
            scenario_dict_with_content = {}
            for field in content_fields:
                scenario_dict_with_content[field] = scenario_dict.get(field, "")
            scenarios_dicts_with_content.append(scenario_dict_with_content)
        embeddings = self.embedder.embed_documents(scenarios_dicts_with_content)
        for scenario_dict, embedding in zip(scenarios_dicts, embeddings):
            scenario_dict["embedding"] = embedding
            scenario_dict["metadata"] = scenario_dict.get("description_json", {})
        return scenarios_dicts

    def add_documents_to_vector_store(self, scenarios_dicts: List[Dict[str, any]]):
        """
        Add documents to the vector store.
        
        Args:
            scenarios_dicts: List of scenario dictionaries
        """
        doc_ids = self.vectorstore.add_documents(scenarios_dicts)
        return doc_ids

    def query_without_reranking(self, query_text: str, query_image: str, query_video: str):
        """
        Query the vector store.
        
        Args:
            query_text: Query text
            query_image: Query image
            query_video: Query video
        """
        query_dict = {
            "text": query_text,
            "image": query_image,
            "video": query_video
        }
        query_embedding = self.embedder.embed_query(query_dict)
        results = self.vectorstore.similarity_search_with_score(query_embedding)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with statistics
        """
        # 2. Get all collections
        collections = self.vectorstore.client.list_collections()
        logger.info(f"📊 Total Collections Found: {len(collections)}\n")
        logger.info("-" * 40)
        
        # 3. Loop through and get the stats for each one
        for collection_name in collections:
            # Best Practice: Flush the collection first. 
            # Milvus buffers recent inserts in memory. Flushing forces them to disk
            # so your row count is 100% accurate.
            self.vectorstore.client.flush(collection_name)
            
            # Get the stats dictionary
            stats = self.vectorstore.client.get_collection_stats(collection_name)
            
            # Extract the row_count (number of entities/documents)
            row_count = stats.get('row_count', 0)
            
            logger.info(f"Collection: '{collection_name}'")
            logger.info(f"📄 Documents : {row_count}")
            logger.info("-" * 40)

    def reset_current_collection(self):
        """Reset the vector store (delete all documents)."""
        logger.warning("Resetting vector store")
        self.vectorstore.reset_collection()
        logger.info("Vector store reset complete")
