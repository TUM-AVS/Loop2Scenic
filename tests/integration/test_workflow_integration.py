"""
Integration tests for ScenarioWorkflow.

These tests gradually replace mock components with real ones to test
component interactions while keeping tests fast and reliable.
"""

import sys
import os
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.workflow.workflow import ScenarioWorkflow  # noqa: E402
from src.agents import InterpreterAgent, ScenicCoderAgent, CriticAgent  # noqa: E402
from src.services import (  # noqa: E402
    Retriever,
    get_embedder, get_llm_service, get_vlm_service
)
from src.services.vectorstore import MilvusVectorStore  # noqa: E402

# Import mocks from unit tests
from tests.test_workflow import (  # noqa: E402
    MockRetriever,
    MockInterpreterAgent, MockScenicCoderAgent, MockCriticAgent,
    setup_mock_scenario_files, cleanup_mock_scenario_files
)


# ==========================================
# PYTEST FIXTURES
# ==========================================

@pytest.fixture(scope="function")
def mock_scenario_files():
    """Setup and teardown mock scenario files."""
    setup_mock_scenario_files()
    yield
    cleanup_mock_scenario_files()


@pytest.fixture(scope="function")
def real_embedder():
    """Create a real embedder for testing."""
    from src.config import get_config  # noqa: E402
    config = get_config()
    return get_embedder(
        provider=config.embedding.provider,
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size
    )


@pytest.fixture(scope="function")
def real_vector_store(real_embedder):
    """Create a real vector store (Milvus Lite) for testing."""
    embedding_dim = real_embedder.dimension
    
    # Use Milvus Lite for testing (no Docker needed)
    connection_args = {
        "uri": "./tests/test_data/milvus_test.db"
    }
    
    vectorstore = MilvusVectorStore(
        embedding_dim=embedding_dim,
        collection_name="test_workflow",
        connection_args=connection_args,
        index_params=None,
        search_params=None
    )
    
    yield vectorstore
    
    # Cleanup: drop collection
    try:
        vectorstore.client.drop_collection("test_workflow")
    except Exception:
        pass


@pytest.fixture(scope="function")
def real_retriever(real_vector_store):
    """Create a real retriever with test vector store."""
    from src.config import get_config  # noqa: E402
    cfg = get_config()
    return Retriever(
        vectorstore=real_vector_store,
        top_k=cfg.retrieval.top_k,
        similarity_threshold=cfg.retrieval.similarity_threshold,
        enable_reranking=cfg.retrieval.enable_reranking
    )


@pytest.fixture(scope="function")
def real_llm():
    """Create a real LLM service (if enabled via env var)."""
    use_real = os.getenv("USE_REAL_LLM", "false").lower() == "true"
    if not use_real:
        pytest.skip("Skipping real LLM test. Set USE_REAL_LLM=true to run.")
    
    from src.config import get_config  # noqa: E402
    cfg = get_config()
    return get_llm_service(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens
    )


@pytest.fixture(scope="function")
def real_vlm():
    """Create a real VLM service (if enabled via env var)."""
    use_real = os.getenv("USE_REAL_VLM", "false").lower() == "true"
    if not use_real:
        pytest.skip("Skipping real VLM test. Set USE_REAL_VLM=true to run.")
    
    from src.config import get_config  # noqa: E402
    cfg = get_config()
    return get_vlm_service(
        provider=cfg.vlm.provider,
        model=cfg.vlm.model,
        device=cfg.vlm.device,
        temperature=cfg.vlm.temperature,
        max_tokens=cfg.vlm.max_tokens
    )


# ==========================================
# INTEGRATION TESTS
# ==========================================

@pytest.mark.integration
def test_workflow_with_real_embedder(mock_scenario_files, real_embedder):
    """
    Test workflow with real embedder, all other components mocked.
    
    This tests:
    - Real embedding generation
    - Embedding → Mock retriever flow
    - Workflow state transitions
    """
    import logging
    from src.utils import setup_logging
    
    setup_logging(level="INFO", run_name="integration_test_embedder")
    logger = logging.getLogger(__name__)
    
    # Create mocks for other components
    mock_retriever = MockRetriever()
    mock_interpreter = MockInterpreterAgent()
    mock_coder = MockScenicCoderAgent()
    mock_critic = MockCriticAgent()
    
    # Create workflow with real embedder
    workflow = ScenarioWorkflow(
        interpreter=mock_interpreter,
        coder=mock_coder,
        critic=mock_critic,
        retriever=mock_retriever,
        embedder=real_embedder
    )
    
    # Test initial run
    initial_state = {
        "user_query": {"text": "highway scenario"},
        "max_count": 2,
        "messages": []
    }
    config = {"configurable": {"thread_id": "test_real_embedder"}}
    
    events = []
    for event in workflow.app.stream(initial_state, config=config):
        events.append(event)
    
    logger.info(f"✅ Workflow with real embedder completed: {len(events)} events")
    assert len(events) > 0, "Workflow should produce events"


@pytest.mark.integration
def test_workflow_with_real_retriever(mock_scenario_files, real_embedder, real_retriever):
    """
    Test workflow with real embedder and retriever.
    
    This tests:
    - Real embedding generation
    - Real vector store operations
    - Real similarity search
    - Workflow with real retrieval
    """
    import logging
    from src.utils import setup_logging
    
    setup_logging(level="INFO", run_name="integration_test_retriever")
    logger = logging.getLogger(__name__)
    
    # Add a test document to vector store first
    test_doc = {
        "text": "highway scenario with truck",
        "metadata": {"scenario_id": "test_scenario_001"}
    }
    embedding = real_embedder.encode([test_doc])[0]
    real_retriever.vectorstore.add_documents([{
        "id": "test_001",
        "embedding": embedding,
        "text": test_doc["text"],
        "metadata": test_doc["metadata"]
    }])
    
    # Create mocks for agents
    mock_interpreter = MockInterpreterAgent()
    mock_coder = MockScenicCoderAgent()
    mock_critic = MockCriticAgent()
    
    # Create workflow with real embedder and retriever
    workflow = ScenarioWorkflow(
        interpreter=mock_interpreter,
        coder=mock_coder,
        critic=mock_critic,
        retriever=real_retriever,
        embedder=real_embedder
    )
    
    # Test initial run
    initial_state = {
        "user_query": {"text": "highway scenario"},
        "max_count": 2,
        "messages": []
    }
    config = {"configurable": {"thread_id": "test_real_retriever"}}
    
    events = []
    for event in workflow.app.stream(initial_state, config=config):
        events.append(event)
    
    logger.info(f"✅ Workflow with real retriever completed: {len(events)} events")
    assert len(events) > 0, "Workflow should produce events"


@pytest.mark.integration
@pytest.mark.slow
def test_workflow_with_real_llm(mock_scenario_files, real_embedder, real_retriever, real_llm):
    """
    Test workflow with real LLM services.
    
    This tests:
    - Real DSL generation from user queries
    - Real code adaptation
    - Integration with LLM APIs
    
    Note: Requires USE_REAL_LLM=true and API keys
    """
    import logging
    from src.utils import setup_logging
    
    setup_logging(level="INFO", run_name="integration_test_llm")
    logger = logging.getLogger(__name__)
    
    # Create real agents with real LLM
    real_interpreter = InterpreterAgent(llm_service=real_llm)
    real_coder = ScenicCoderAgent(llm_service=real_llm)
    
    # Mock critic (VLM is expensive)
    mock_critic = MockCriticAgent()
    
    # Create workflow
    workflow = ScenarioWorkflow(
        interpreter=real_interpreter,
        coder=real_coder,
        critic=mock_critic,
        retriever=real_retriever,
        embedder=real_embedder
    )
    
    # Test initial run
    initial_state = {
        "user_query": {"text": "highway scenario with truck"},
        "max_count": 1,  # Limit iterations to save API costs
        "messages": []
    }
    config = {"configurable": {"thread_id": "test_real_llm"}}
    
    events = []
    for event in workflow.app.stream(initial_state, config=config):
        events.append(event)
    
    logger.info(f"✅ Workflow with real LLM completed: {len(events)} events")
    assert len(events) > 0, "Workflow should produce events"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.expensive
def test_workflow_with_real_vlm(mock_scenario_files, real_embedder, real_retriever, real_llm, real_vlm):
    """
    Test workflow with all real components except simulation.
    
    This is the most comprehensive integration test.
    Note: Very expensive, run infrequently.
    """
    import logging
    from src.utils import setup_logging
    
    setup_logging(level="INFO", run_name="integration_test_vlm")
    logger = logging.getLogger(__name__)
    
    # Create real agents
    real_interpreter = InterpreterAgent(llm_service=real_llm)
    real_coder = ScenicCoderAgent(llm_service=real_llm)
    real_critic = CriticAgent(vlm_service=real_vlm)
    
    # Create workflow with all real components
    workflow = ScenarioWorkflow(
        interpreter=real_interpreter,
        coder=real_coder,
        critic=real_critic,
        retriever=real_retriever,
        embedder=real_embedder
    )
    
    # Test initial run
    initial_state = {
        "user_query": {"text": "highway scenario"},
        "max_count": 1,  # Limit to save costs
        "messages": []
    }
    config = {"configurable": {"thread_id": "test_real_vlm"}}
    
    events = []
    for event in workflow.app.stream(initial_state, config=config):
        events.append(event)
    
    logger.info(f"✅ Workflow with real VLM completed: {len(events)} events")
    assert len(events) > 0, "Workflow should produce events"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
