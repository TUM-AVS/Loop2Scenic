"""
Main RAG pipeline orchestration.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np

from .schema import MultimodalQuery
from .config import Config, get_config
from .ingestion import MultimodalDocumentInterpreter
from .services import MilvusVectorStore, get_reranker, get_vlm_service, get_embedder, Retriever
from .utils import clean_and_parse_json, flatten_dsl_to_text, setup_logging

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

    def __init__(self, config: Optional[Config] = None, mode: str = "ingestion"):
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

        if mode == "ingestion":
            self._initialize_ingestion_components()
        elif mode == "get_stats":
            self._initialize_get_stats_components()
        elif mode == "query":
            self._initialize_components()
        elif mode == "mock_query": # for CUDA out of memory error, do not load embedder and reranker at the same time
            self._initialize_mock_query_components()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        logger.info("RAG Pipeline initialized successfully")

    def _initialize_get_stats_components(self):
        """Initialize get stats components."""
        embedding_dim = self._initialize_embedder()
        self._initialize_vector_store(embedding_dim)

    def _initialize_components(self):
        """Initialize all pipeline components."""
        self._initialize_vlm_service()
        self._initialize_multimodal_interpreter()
        embedding_dim = self._initialize_embedder()
        self._initialize_vector_store(embedding_dim)
        self._initialize_retriever()

    def _initialize_mock_query_components(self):
        """Initialize all pipeline components."""
        embedding_dim = 2048
        self._initialize_vector_store(embedding_dim)
        self._initialize_retriever()

    def _initialize_ingestion_components(self):
        """Initialize ingestion components."""
        self._initialize_vlm_service()
        self._initialize_multimodal_interpreter()
        embedding_dim = self._initialize_embedder()
        self._initialize_vector_store(embedding_dim)

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
        
        logger.info(f"VLM Service initialized successfully with {self.config.vlm.provider}, model: {self.config.vlm.model}")

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
            # device=self.config.embedding.device,
            # batch_size=self.config.embedding.batch_size,
            **embedder_kwargs
        )
        
        logger.info(f"Embedder initialized successfully with {self.config.embedding.provider}.")
        embedding_dim = self.embedder.dimension
        logger.info(f"Embedding dimension: {embedding_dim}")
        return embedding_dim

    def _initialize_retriever(self):
        """Initialize retriever."""
        logger.info("Initializing Reranker...")
        self.reranker = get_reranker(
            provider=self.config.reranking.provider,
            model_path=self.config.reranking.model_path,
            model_name=self.config.reranking.model_name,
        )

        logger.info("Initializing Retriever...")
        self.retriever = Retriever(
            vectorstore=self.vectorstore,
            reranker=self.reranker,
            top_k=self.config.retrieval.top_k,
            similarity_threshold=self.config.retrieval.similarity_threshold,
        )

    def _initialize_vector_store(self, embedding_dim: int):
        """Initialize vector database (Milvus) using the embedder dimension."""
        logger.info(
            f"Initializing Vector Store (Milvus "
        )
        connection_args = {
            "host": self.config.vector_db.host,
            "port": self.config.vector_db.port,
        }
        index_params = {
            "metric_type": self.config.vector_db.distance_metric.upper(),
            "index_type": self.config.vector_db.index_type,
            "params": {"nlist": self.config.vector_db.nlist},
        }
        search_params = {
            "metric_type": self.config.vector_db.distance_metric.upper(),
            "params": {"nprobe": self.config.vector_db.nprobe},
        }

        vector_db = MilvusVectorStore(
            embedding_dim=embedding_dim,
            collection_name=self.config.vector_db.collection_name,
            snippets_collection_name=self.config.vector_db.snippets_collection_name,
            connection_args=connection_args,
            index_params=index_params,
            search_params=search_params,
        )
        logger.info("Vector Store initialized.")
        self.vectorstore = vector_db

    def interpret_scenarios(self, directory_path: Union[str, Path]):
        """
        Interpret scenarios from a directory and save the descriptions into the folder path's new_description.txt file.
        
        Args:
            directory_path: Path to the directory
        """
        scenarios_dicts = self.multimodal_interpreter.extract_from_directory(directory_path)
        for scenario_dict in scenarios_dicts:
            try:
                # 1. Get the raw string output from the VLM
                scenario_description_str = self.multimodal_interpreter.get_layer_model_description_by_vlm(
                    scenario_dict, self.vlm_service
                )
                logger.info(f"The raw VLM output is: {scenario_description_str}")

                # 2. Parse the string into a Python dictionary
                try:
                    scenario_data = clean_and_parse_json(scenario_description_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from VLM output: {e}. Skipping this scenario.")
                    logger.debug(f"Raw malformed output: {scenario_description_str}")
                    continue  # Skip to the next scenario if the VLM failed to output valid JSON

                # 3. Flatten the JSON into plain text for optimal embedding
                flattened_text = flatten_dsl_to_text(scenario_data)
                if not flattened_text:
                    logger.error("Failed to flatten parsed VLM output. Skipping this scenario.")
                    continue

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
            except Exception as e:
                logger.error(f"Failed to interpret scenario: {e}")
                logger.error(f"Scenario dictionary: {scenario_dict}")
                continue

    def interpret_multimodal_query(self, multimodal_query: MultimodalQuery):
        """
        Interpret scenarios from a directory and save the descriptions into the folder path's new_description.txt file.
        
        Args:
            directory_path: Path to the directory
        """
        query_dict = multimodal_query.model_dump()
        # 1. Get the raw string output from the VLM
        scenario_description_str = self.multimodal_interpreter.get_layer_model_description_by_vlm(
            query_dict, self.vlm_service
        )
        logger.info(f"The raw VLM output is: {scenario_description_str}")

        # 2. Parse the string into a Python dictionary
        try:
            scenario_data = json.loads(scenario_description_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from VLM output: {e}.")
            logger.debug(f"Raw malformed output: {scenario_description_str}")
            return None

        # 3. Flatten the JSON into plain text for optimal embedding
        flattened_text = flatten_dsl_to_text(scenario_data)
        if not flattened_text:
            logger.error("Failed to flatten parsed VLM output.")
            return None
            
        logger.info(f"Successfully interpreted multimodal query: {flattened_text}")
        return flattened_text

    def ingest_scenarios_with_interpretation(self, directory_path: Union[str, Path]):
        """
        Ingest scenarios from a directory into the vector store.
        
        Args:
            directory_path: Path to the directory

        Returns:
            List of dictionaries, each representing a multimodal document
        """
        self.interpret_scenarios(directory_path)
        scenarios_dicts = self.multimodal_interpreter.extract_from_directory(directory_path, use_new_description=True)
        logger.info(f"The first scenario dictionary: {scenarios_dicts[0]}")
        return scenarios_dicts

    def ingest_scenarios_without_interpretation(self, directory_path: Union[str, Path]):
        """
        Ingest scenarios from a directory into the vector store, the new description file is already generated before.
        
        Args:
            directory_path: Path to the directory

        Returns:
            List of document IDs
        """
        scenarios_dicts = self.multimodal_interpreter.extract_from_directory(directory_path, use_new_description=True)
        logger.info(f"The first scenario dictionary: {scenarios_dicts[0]}")
        return scenarios_dicts

    def embed_scenarios(self, scenarios_dicts: List[Dict[str, any]]):
        """
        Embed scenarios.
        
        Args:
            scenarios_dicts: List of scenario dictionaries
        """
        return [self.embed_one_scenario(scenario_dict) for scenario_dict in scenarios_dicts]

    def embed_one_scenario(self, scenario_dict: Dict[str, any]):
        """
        Embed one scenario.

        Args:
            scenario_dict: A single scenario dictionary
        """
        scenario_dict_with_content = {
            "text": scenario_dict.get("description", ""),
            "image": scenario_dict.get("image", ""),
            "video": scenario_dict.get("video", ""),
        }
        embedding = self.embedder.encode([scenario_dict_with_content])[0]
        scenario_dict["embedding"] = embedding
        scenario_dict["metadata"] = scenario_dict.get("description_json", {})
        return scenario_dict

    def add_documents_to_vector_store(self, scenarios_dicts: List[Dict[str, any]]):
        """
        Add documents to the vector store.
        
        Args:
            scenarios_dicts: List of scenario dictionaries
        """
        doc_ids = self.vectorstore.add_documents(scenarios_dicts)
        return doc_ids

    def query_without_reranking(self, query_dict: Dict[str, Any]):
        """
        Query the vector store.
        
        Args:
            query_dict: Query dictionary, including text, image, video

        Returns:
            List of (Document, score) tuples
        """
        query_embedding = self.embedder.encode([query_dict])[0]
        results = self.vectorstore.similarity_search_with_score(query_embedding)
        return results

    def query_with_reranking(self, query_object: MultimodalQuery, interpreted_query_object: MultimodalQuery):
        """
        Query the vector store with reranking.
        
        Args:
            query_dict: Query dictionary, including text, image, video

        Returns:
            The base scenario id
        """
        query_embedding = self.embedder.encode([interpreted_query_object.model_dump()])[0]
        retrieval = self.retriever.retrieve(
            original_query=query_object,
            query_embedding=query_embedding,
        )
        base_scenario_id = retrieval.scenarios[0].scenario_id # only return the best 1 scenario
        
        logger.info(f"🔍 Found best scenario: {base_scenario_id}")
        return base_scenario_id

    def mock_query_with_reranking(self, query_dict: Dict[str, Any]):
        """
        Query the vector store with reranking.
        
        Args:
            query_dict: Query dictionary, including text, image, video

        Returns:
            The base scenario id
        """
        rng = np.random.default_rng(42)
        query_embedding = rng.standard_normal(2048, dtype=np.float32).tolist()
        retrieval = self.retriever.retrieve(
            original_query=query_dict,
            query_embedding=query_embedding,
        )
        base_scenario_id = retrieval.scenarios[0].scenario_id # only return the best 1 scenario
        
        logger.info(f"🔍 Found best scenario: {base_scenario_id}")
        return base_scenario_id

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

    def reset_current_collection(self, collection_name: str|None = None):
        """Reset the vector store (delete all documents)."""
        logger.warning("Resetting vector store")
        self.vectorstore.reset_collection(collection_name=collection_name)
        logger.info("Vector store reset complete")
