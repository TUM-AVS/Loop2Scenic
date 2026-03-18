"""
Main application entry point for ADS-MRAG system.

This module initializes the entire system using configuration from config.yaml.
"""

import logging
from typing import Optional, Tuple

from src.config import get_config
from src.utils.logger import setup_logging

# Import Services
from src.services import (
    get_llm_service,
    get_vlm_service,
    Retriever,
    get_reranker,
    MilvusVectorStore,
    get_embedder,
)

# Import Agents
from src.agents import InterpreterAgent, ScenicCoderAgent, CriticAgent

# Import Workflow / schema
from src.workflow import ScenarioWorkflow
from src.schema import MultimodalQuery


def _initialize_logging(config) -> logging.Logger:
    """Configure logging and return a module logger."""
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
        log_format=config.logging.format,
        run_name="ads_mrag_startup"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting system initialization...")
    logger.info(
        f"Using configuration: "
        f"LLM={config.llm.provider}, "
        f"VLM={config.vlm.provider}, "
        f"Embedding={config.embedding.provider}"
    )
    return logger


def _initialize_llm_service(config, logger: logging.Logger):
    """Initialize shared LLM service."""
    logger.info(f"Initializing LLM service: {config.llm.provider}")
    return get_llm_service(
        provider=config.llm.provider,
        model=config.llm.model,
        api_key=config.llm.api_key,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )


def _initialize_vlm_service(config, logger: logging.Logger):
    """Initialize VLM service."""
    logger.info(f"Initializing VLM service: {config.vlm.provider}")

    vlm_kwargs = {
        "provider": config.vlm.provider,
        "model": config.vlm.model,
        "api_key": config.vlm.api_key,
        "temperature": config.vlm.temperature,
        "max_tokens": config.vlm.max_tokens,
        "default_instruction": config.vlm.default_instruction,
    }
    # Add model_path if provided (required for Qwen3VL)
    if config.vlm.model_path:
        vlm_kwargs["model_path"] = config.vlm.model_path

    vlm_service = get_vlm_service(**vlm_kwargs)
    logger.info("VLM service initialized.")
    return vlm_service


def _initialize_embedder(config, logger: logging.Logger):
    """Initialize embedding model."""
    logger.info(f"Initializing Embedder: {config.embedding.provider}")

    embedder_kwargs = {
        "provider": config.embedding.provider,
        "model_name": config.embedding.model_name,
    }
    # Add model_path if provided (required for Qwen models)
    if config.embedding.model_path:
        embedder_kwargs["model_path"] = config.embedding.model_path

    embedder = get_embedder(**embedder_kwargs)
    logger.info("Embedder initialized.")
    return embedder


def _initialize_vector_store(config, embedder, logger: logging.Logger) -> MilvusVectorStore:
    """Initialize vector database (Milvus) using the embedder dimension."""
    logger.info(
        f"Initializing Vector Store (Milvus "
        f"{'Lite' if config.vector_db.use_lite else 'Server'})..."
    )

    embedding_dim = embedder.dimension

    connection_args = {
        "host": config.vector_db.host,
        "port": config.vector_db.port,
    }
    index_params = {
        "metric_type": config.vector_db.distance_metric.upper(),
        "index_type": config.vector_db.index_type,
        "params": {"nlist": config.vector_db.nlist},
    }
    search_params = {
        "metric_type": config.vector_db.distance_metric.upper(),
        "params": {"nprobe": config.vector_db.nprobe},
    }

    vector_db = MilvusVectorStore(
        embedding_dim=embedding_dim,
        collection_name=config.vector_db.collection_name,
        connection_args=connection_args,
        index_params=index_params,
        search_params=search_params,
    )
    logger.info("Vector Store initialized.")
    return vector_db


def _initialize_reranker(config, logger: logging.Logger):
    """Initialize reranker if enabled in config."""
    if not config.retrieval.enable_reranking:
        logger.info("Reranking disabled in configuration.")
        return None

    logger.info("Initializing Reranker...")
    reranker_kwargs = {
        "provider": config.reranking.provider,  # Currently only qwen is supported
        "device": config.reranking.device,
        "instruction": config.reranking.instruction,
    }
    if config.reranking.model_path:
        reranker_kwargs["model_path"] = config.reranking.model_path
    if config.reranking.model_name:
        reranker_kwargs["model_name"] = config.reranking.model_name

    reranker = get_reranker(**reranker_kwargs)
    logger.info("Reranker initialized.")
    return reranker


def _initialize_retriever(
    config,
    vector_db: MilvusVectorStore,
    reranker,
    logger: logging.Logger,
) -> Retriever:
    """Initialize retrieval pipeline."""
    logger.info("Initializing Retrieval Pipeline...")
    retrieval_pipeline = Retriever(
        vectorstore=vector_db,
        reranker=reranker,
        top_k=config.retrieval.top_k,
        similarity_threshold=config.retrieval.similarity_threshold,
    )
    logger.info("Retrieval Pipeline initialized.")
    return retrieval_pipeline


def _initialize_agents(
    shared_llm_service,
    vlm_service,
    logger: logging.Logger,
) -> Tuple[InterpreterAgent, ScenicCoderAgent, CriticAgent]:
    """Initialize all agents."""
    logger.info("Initializing Agents...")

    interpreter_agent = InterpreterAgent(vlm_service=vlm_service)
    scenic_coder_agent = ScenicCoderAgent(llm_service=shared_llm_service)
    critic_agent = CriticAgent(vlm_service=vlm_service)

    logger.info("Agents initialized.")
    return interpreter_agent, scenic_coder_agent, critic_agent


def _initialize_workflow(
    interpreter_agent: InterpreterAgent,
    scenic_coder_agent: ScenicCoderAgent,
    critic_agent: CriticAgent,
    retrieval_pipeline: Retriever,
    embedder,
    logger: logging.Logger,
) -> ScenarioWorkflow:
    """Wire up and return the main ScenarioWorkflow."""
    logger.info("Wiring up the LangGraph Workflow...")
    workflow = ScenarioWorkflow(
        interpreter=interpreter_agent,
        coder=scenic_coder_agent,
        critic=critic_agent,
        retriever=retrieval_pipeline,
        embedder=embedder,
    )
    logger.info("Workflow initialized.")
    return workflow


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

    # Initialize logging
    logger = _initialize_logging(config)

    # ==========================================
    # LAYER 3: INITIALIZE SHARED SERVICES (Singletons)
    # ==========================================
    shared_llm_service = _initialize_llm_service(config, logger)
    vlm_service = _initialize_vlm_service(config, logger)
    embedder = _initialize_embedder(config, logger)
    vector_db = _initialize_vector_store(config, embedder, logger)
    reranker = _initialize_reranker(config, logger)
    retrieval_pipeline = _initialize_retriever(config, vector_db, reranker, logger)

    # ==========================================
    # LAYER 2: INITIALIZE AGENTS (Injecting Services)
    # ==========================================
    interpreter_agent, scenic_coder_agent, critic_agent = _initialize_agents(
        shared_llm_service=shared_llm_service,
        vlm_service=vlm_service,
        logger=logger,
    )

    # ==========================================
    # LAYER 1: INITIALIZE WORKFLOW (Injecting Agents)
    # ==========================================
    workflow = _initialize_workflow(
        interpreter_agent=interpreter_agent,
        scenic_coder_agent=scenic_coder_agent,
        critic_agent=critic_agent,
        retrieval_pipeline=retrieval_pipeline,
        embedder=embedder,
        logger=logger,
    )

    logger.info("✅ System initialized successfully!")
    return workflow


if __name__ == "__main__":
    # Example test runner using real initialized components
    workflow = initialize_system()

    # Construct a sample user query (adjust paths/text as needed)
    user_query = MultimodalQuery(
        text="Generate me a highway scenario looks like the one in this video",
        image_path=None,
        video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4",  # e.g. "data/processed/test_data/testvideo.mp4"
    )

    initial_state = {"user_query": user_query, "max_count": 3}
    config = {"configurable": {"thread_id": "app_real_components_test"}}

    logger = logging.getLogger(__name__)
    logger.info("🚀 STARTING WORKFLOW RUN WITH REAL COMPONENTS...")

    # Initial run
    for event in workflow.app.stream(initial_state, config=config):
        pass

    logger.info("🛑 GRAPH PAUSED. Pretending user clicked 'Accept'...")
    workflow.app.update_state(config, {
        "user_satisfied": True
    })

    logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY.")