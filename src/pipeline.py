"""
Main RAG pipeline orchestration.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .config import Config, get_config
from .vlm import get_vlm_service
from .ingestion import MultimodalDocumentInterpreter
from .embedding import get_embedder
from .vectorstore import MilvusVectorStore
from .retrieval import Retriever
from .generation import get_generator, PromptTemplate
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

        logger.info("Initializing VLM Service...")

        # 1. Define your core System Prompt (Instruction) here
        scenario_extraction_instruction = (
            "You are an expert autonomous driving scenario analyzer. Your task is to extract "
            "the high-level logical structure of a driving scenario by analyzing the provided "
            "Scenic code, text description, image, and video.\n\n"
            "Output ONLY a valid JSON object. Do not include markdown or conversational text.\n"
        )
        
        # 2. Dump the Pydantic config to a dictionary
        # Note: Use .model_dump() for Pydantic v2, or .dict() if you are on Pydantic v1
        vlm_kwargs = self.config.vlm.model_dump()
        
        # 3. Override the default instruction inside the dictionary with our specific RAG prompt
        vlm_kwargs['default_instruction'] = scenario_extraction_instruction
        
        # 4. Initialize the service in ONE single step
        # **vlm_kwargs will automatically map 'provider', 'temperature', etc., to the correct arguments
        self.vlm_service = get_vlm_service(**vlm_kwargs)
        
        # 5. Log the success (accessing the provider from the config)
        logger.info(f"VLM Service initialized successfully with {self.config.vlm.provider}.")
        
        # Scenario processing
        self.multimodal_interpreter = MultimodalDocumentInterpreter()
        
        # # Embedding
        # embedder_kwargs = {}
        # if hasattr(self.config.embedding, 'model_path') and self.config.embedding.model_path:
        #     embedder_kwargs['model_path'] = self.config.embedding.model_path
        
        # self.embedder = get_embedder(
        #     provider=self.config.embedding.provider,
        #     model_name=self.config.embedding.model_name,
        #     device=self.config.embedding.device,
        #     batch_size=self.config.embedding.batch_size,
        #     **embedder_kwargs
        # )
        
        # # Vector store - Milvus only (supports both Lite and Server modes)
        # use_lite = getattr(self.config.vector_db, "use_lite", True)
        
        # if use_lite:
        #     # Milvus Lite (no Docker required)
        #     connection_args = {
        #         "uri": getattr(self.config.vector_db, "lite_db_path", "./data/vector_db/milvus.db")
        #     }
        #     logger.info(f"Using Milvus Lite mode (no Docker): {connection_args['uri']}")
        # else:
        #     # Milvus Server (requires Docker)
        #     connection_args = {
        #         "host": getattr(self.config.vector_db, "host", "localhost"),
        #         "port": getattr(self.config.vector_db, "port", "19530")
        #     }
        #     logger.info(f"Using Milvus Server mode: {connection_args['host']}:{connection_args['port']}")
            
        #     index_params = {
        #         "metric_type": self.config.vector_db.distance_metric.upper(),
        #         "index_type": getattr(self.config.vector_db, "index_type", "IVF_FLAT"),
        #         "params": {"nlist": getattr(self.config.vector_db, "nlist", 1024)}
        #     }
            
        #     search_params = {
        #         "metric_type": self.config.vector_db.distance_metric.upper(),
        #         "params": {"nprobe": getattr(self.config.vector_db, "nprobe", 10)}
        #     }
            
        #     self.vectorstore = MilvusVectorStore(
        #         embedder=self.embedder,
        #         collection_name=self.config.vector_db.collection_name,
        #         connection_args=connection_args,
        #         index_params=index_params,
        #         search_params=search_params
        #     )
        
        # # Retrieval
        # self.retriever = Retriever(
        #     vectorstore=self.vectorstore,
        #     top_k=self.config.retrieval.top_k,
        #     similarity_threshold=self.config.retrieval.similarity_threshold,
        #     enable_reranking=self.config.retrieval.enable_reranking
        # )
        
        # # Generation
        # self.generator = get_generator(
        #     provider=self.config.llm.provider,
        #     model=self.config.llm.model,
        #     temperature=self.config.llm.temperature,
        #     max_tokens=self.config.llm.max_tokens,
        #     streaming=self.config.llm.streaming
        # )

    def interpret_scenarios(self, directory_path: Union[str, Path]):
        """
        Interpret scenarios from a directory and save the descriptions into the folder path's new_description.txt file.
        
        Args:
            directory_path: Path to the directory
        """
        scenarios_dicts = self.multimodal_interpreter.extract_from_directory(directory_path)
        for scenario_dict in scenarios_dicts:
            scenario_description = self.multimodal_interpreter.get_layer_model_description_by_vlm(scenario_dict, self.vlm_service)
            logger.info(f"The scenario description is: {scenario_description}")
            
            if not os.path.exists(scenario_dict.get("folder_path")):
                os.makedirs(scenario_dict.get("folder_path"))
            new_description_path = Path(scenario_dict.get("folder_path")) / "new_description.txt"
            with open(new_description_path, "w") as f:
                f.write(scenario_description) # save the scenario description into the folder path's new_description.txt file

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
        # doc_ids = self.vectorstore.add_documents(scenarios_dicts)
        # return doc_ids

    # def ingest_documents(
    #     self,
    #     source: Union[str, Path],
    #     metadata: Optional[Dict] = None,
    #     tags: Optional[List[str]] = None,
    #     is_directory: bool = False,
    #     recursive: bool = True
    # ) -> List[str]:
    #     """
    #     Ingest documents into the vector store.
        
    #     Args:
    #         source: File or directory path
    #         metadata: Additional metadata
    #         tags: List of tags for filtering
    #         is_directory: Whether source is a directory
    #         recursive: Whether to search subdirectories (for directories)
            
    #     Returns:
    #         List of document IDs
    #     """
    #     logger.info(f"Ingesting documents from: {source}")
        
    #     # Process documents
    #     if is_directory:
    #         chunks = self.processor.process_directory(
    #             directory_path=source,
    #             metadata=metadata,
    #             tags=tags,
    #             recursive=recursive
    #         )
    #     else:
    #         chunks = self.processor.process_file(
    #             file_path=source,
    #             metadata=metadata,
    #             tags=tags
    #         )
        
    #     if not chunks:
    #         logger.warning("No chunks to ingest")
    #         return []
        
    #     # Add to vector store
    #     doc_ids = self.vectorstore.add_documents(chunks)
        
    #     logger.info(f"Ingested {len(doc_ids)} document chunks")
        
    #     return doc_ids

    # def query(
    #     self,
    #     query: str,
    #     filter_tags: Optional[List[str]] = None,
    #     top_k: Optional[int] = None,
    #     return_sources: bool = False,
    #     custom_prompt: Optional[PromptTemplate] = None
    # ) -> Union[str, Dict[str, Any]]:
    #     """
    #     Query the RAG pipeline.
        
    #     Args:
    #         query: Query string
    #         filter_tags: Optional tags to filter by
    #         top_k: Number of documents to retrieve
    #         return_sources: Whether to return source information
    #         custom_prompt: Custom prompt template
            
    #     Returns:
    #         Generated response or dictionary with response and metadata
    #     """
    #     logger.info(f"Processing query: '{query[:50]}...'")
        
    #     # Retrieve relevant documents
    #     context_docs = self.retriever.retrieve(
    #         query=query,
    #         top_k=top_k,
    #         filter_tags=filter_tags,
    #         return_scores=False
    #     )
        
    #     if not context_docs:
    #         response = "I couldn't find any relevant information to answer your question."
    #         if return_sources:
    #             return {"response": response, "sources": [], "num_sources": 0}
    #         return response
        
    #     # Generate response
    #     if return_sources:
    #         result = self.generator.generate_with_metadata(
    #             query=query,
    #             context_documents=context_docs,
    #             prompt_template=custom_prompt
    #         )
    #         return result
    #     else:
    #         response = self.generator.generate(
    #             query=query,
    #             context_documents=context_docs,
    #             prompt_template=custom_prompt
    #         )
    #         return response

    # def get_stats(self) -> Dict[str, Any]:
    #     """
    #     Get pipeline statistics.
        
    #     Returns:
    #         Dictionary with statistics
    #     """
    #     return self.vectorstore.get_collection_stats()

    # def reset(self):
    #     """Reset the vector store (delete all documents)."""
    #     logger.warning("Resetting vector store")
    #     self.vectorstore.reset_collection()
    #     logger.info("Vector store reset complete")
