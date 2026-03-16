"""
Main application entry point for ADS-MRAG system.

This module initializes the entire system using configuration from config.yaml.
"""

import logging
from typing import Optional

from src.config import get_config
from src.utils.logger import setup_logging

# Import Services
from src.services import (
    get_llm_service, 
    get_vlm_service, 
    Retriever, 
    get_reranker, 
    MilvusVectorStore, 
    get_embedder
)

# Import Agents
from src.agents import InterpreterAgent, ScenicCoderAgent, CriticAgent

# Import Workflow
from src.workflow import ScenarioWorkflow


def initialize_system(config_path: Optional[str] = None) -> ScenarioWorkflow:
    """
    Builds the entire ADS-MRAG system from the bottom up using configuration.
    
    Args:
        config_path: Optional path to YAML configuration file.
                     If None, uses default config/config.yaml
    
    Returns:
        Initialized ScenarioWorkflow instance
    """
    # Load configuration
    config = get_config(config_path)
    
    # Setup logging using config
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
        log_format=config.logging.format,
        run_name="ads_mrag_startup"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting system initialization...")
    logger.info(f"Using configuration: LLM={config.llm.provider}, VLM={config.vlm.provider}, Embedding={config.embedding.provider}")

    # ==========================================
    # LAYER 3: INITIALIZE SHARED SERVICES (Singletons)
    # ==========================================
    logger.info("Initializing Services...")
    
    # Initialize LLM service
    logger.info(f"Initializing LLM service: {config.llm.provider}")
    shared_llm_service = get_llm_service(
        provider=config.llm.provider,
        model=config.llm.model,
        api_key=config.llm.api_key,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )
    
    # Initialize VLM service
    logger.info(f"Initializing VLM service: {config.vlm.provider}")
    vlm_kwargs = {
        "provider": config.vlm.provider,
        "model": config.vlm.model,
        "api_key": config.vlm.api_key,
        "device": config.vlm.device,
        "temperature": config.vlm.temperature,
        "max_tokens": config.vlm.max_tokens,
        "fps": config.vlm.fps,
        "max_frames": config.vlm.max_frames,
        "default_instruction": config.vlm.default_instruction
    }
    # Add model_path if provided (required for Qwen3VL)
    if config.vlm.model_path:
        vlm_kwargs["model_path"] = config.vlm.model_path
    vlm_service = get_vlm_service(**vlm_kwargs)
    
    # Initialize Embedder
    logger.info(f"Initializing Embedder: {config.embedding.provider}")
    embedder_kwargs = {
        "provider": config.embedding.provider,
        "model_name": config.embedding.model_name,
        "device": config.embedding.device,
        "batch_size": config.embedding.batch_size
    }
    # Add model_path if provided (required for Qwen models)
    if config.embedding.model_path:
        embedder_kwargs["model_path"] = config.embedding.model_path
    embedder = get_embedder(**embedder_kwargs)
    
    # Initialize Vector Store (Milvus)
    logger.info(f"Initializing Vector Store (Milvus {'Lite' if config.vector_db.use_lite else 'Server'})...")
    embedding_dim = embedder.dimension
    
    # Setup connection args based on mode
    if config.vector_db.use_lite:
        connection_args = {"uri": config.vector_db.lite_db_path}
        index_params = None
        search_params = None
    else:
        connection_args = {
            "host": config.vector_db.host,
            "port": config.vector_db.port
        }
        index_params = {
            "metric_type": config.vector_db.distance_metric.upper(),
            "index_type": config.vector_db.index_type,
            "params": {"nlist": config.vector_db.nlist}
        }
        search_params = {
            "metric_type": config.vector_db.distance_metric.upper(),
            "params": {"nprobe": config.vector_db.nprobe}
        }
    
    vector_db = MilvusVectorStore(
        embedding_dim=embedding_dim,
        collection_name=config.vector_db.collection_name,
        connection_args=connection_args,
        index_params=index_params,
        search_params=search_params
    )
    
    # Initialize Reranker (if enabled)
    reranker = None
    if config.retrieval.enable_reranking:
        logger.info("Initializing Reranker...")
        reranker_kwargs = {
            "provider": config.reranking.provider,  # Currently only qwen is supported
            "device": config.reranking.device,
            "torch_dtype": config.reranking.torch_dtype,
            "instruction": config.reranking.instruction,
            "fps": config.reranking.fps
        }
        if config.reranking.model_path:
            reranker_kwargs["model_path"] = config.reranking.model_path
        if config.reranking.model_name:
            reranker_kwargs["model_name"] = config.reranking.model_name
        reranker = get_reranker(**reranker_kwargs)
    
    # Initialize Retrieval Pipeline
    logger.info("Initializing Retrieval Pipeline...")
    retrieval_pipeline = Retriever(
        vectorstore=vector_db,
        reranker=reranker,
        top_k=config.retrieval.top_k,
        similarity_threshold=config.retrieval.similarity_threshold
    )

    # ==========================================
    # LAYER 2: INITIALIZE AGENTS (Injecting Services)
    # ==========================================
    logger.info("Initializing Agents...")
    # Notice how both agents get the EXACT SAME shared_llm_service!
    interpreter_agent = InterpreterAgent(llm_service=shared_llm_service)
    scenic_coder_agent = ScenicCoderAgent(llm_service=shared_llm_service)
    critic_agent = CriticAgent(vlm_service=vlm_service)

    # ==========================================
    # LAYER 1: INITIALIZE WORKFLOW (Injecting Agents)
    # ==========================================
    logger.info("Wiring up the LangGraph Workflow...")
    workflow = ScenarioWorkflow(
        interpreter=interpreter_agent,
        coder=scenic_coder_agent,
        critic=critic_agent,
        retriever=retrieval_pipeline,
        embedder=embedder
    )

    logger.info("✅ System initialized successfully!")
    return workflow


if __name__ == "__main__":
    app_workflow = initialize_system()