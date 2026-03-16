# main.py
import logging
from src.utils.logger import setup_logging

# 1. Import Services
from src.services import get_llm_service, get_vlm_service, Retriever, get_reranker, MilvusVectorStore, get_embedder

# 2. Import Agents
from src.agents import InterpreterAgent, ScenicCoderAgent, CriticAgent

# 3. Import Workflow
from src.workflow import ScenarioWorkflow

def initialize_system() -> ScenarioWorkflow:
    """Builds the entire ADS-MRAG system from the bottom up."""
    setup_logging(run_name="ads_mrag_startup")
    logger = logging.getLogger(__name__)
    logger.info("Starting system initialization...")

    # ==========================================
    # LAYER 3: INITIALIZE SHARED SERVICES (Singletons)
    # ==========================================
    logger.info("Initializing Services...")
    # Only ONE instance of the LLM service is created
    shared_llm_service = get_llm_service(provider="gemini", api_key="YOUR_KEY")
    vlm_service = get_vlm_service(provider="gemini", api_key="YOUR_KEY")
    
    # Retrieval pipeline setup
    vector_db = MilvusVectorStore(host="localhost", port="19530")
    reranker = get_reranker(provider="qwen3vl", model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    retrieval_pipeline = Retriever(vector_db, reranker)

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
        retriever=retrieval_pipeline
    )

    logger.info("✅ System initialized successfully!")
    return workflow

if __name__ == "__main__":
    app_workflow = initialize_system()