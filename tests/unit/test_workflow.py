"""
Unit tests for ScenarioWorkflow with all mocked dependencies.

This test verifies that the workflow logic and state transitions work correctly
without external dependencies. All components are mocked for speed and reliability.

For integration tests with real components, see tests/integration/
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import pytest

# Add project root to path (must be before other imports)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Imports after path modification (noqa: E402 for test files)
from src.workflow.workflow import ScenarioWorkflow  # noqa: E402
from src.agents import InterpreterAgent, ScenicCoderAgent, CriticAgent  # noqa: E402
from src.services import Retriever, BaseEmbeddingModel, BaseLLMModel, BaseVLMModel  # noqa: E402


# ==========================================
# MOCK SERVICES
# ==========================================

class MockLLMModel(BaseLLMModel):
    """Mock LLM model that returns predictable JSON responses."""
    
    def __init__(self, model_name: str = "mock-llm"):
        self._model_name = model_name
    
    def chat(self, messages: List[Dict[str, str]] = None, **kwargs) -> str:
        """
        Mock chat method that handles both string and messages format.
        """
        # Handle case where agents pass a string directly
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list) and len(messages) > 0:
            # Extract content from messages format
            prompt = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
        else:
            prompt = str(messages) if messages else ""
        
        # Return mock JSON response based on prompt content
        # Note: Agents will parse this JSON, so we return valid JSON strings
        if "describe_in_layer_model" in prompt.lower() or "user_query" in prompt.lower():
            # Mock DSL generation - returns JSON string that will be parsed to dict
            return json.dumps({
                "Scenario": "highway_overtaking",
                "Ego": "sedan",
                "Adversarials": ["truck"],
                "Spatial Relation": "ego is behind truck",
                "Requirement and restrictions": "overtake safely"
            })
        elif "adapt_code" in prompt.lower():
            # Mock code adaptation - returns JSON string with scenic_code
            return json.dumps({
                "scenic_code": "scenario mock_scenario:\n    ego = Car\n    truck = Truck\n    # Adapted code"
            })
        else:
            # Default mock response
            return json.dumps({"result": "mock_response"})
    
    @property
    def model_name(self) -> str:
        return self._model_name


class MockVLMModel(BaseVLMModel):
    """Mock VLM model that returns evaluation scores and feedback."""
    
    def __init__(self, model_name: str = "mock-vlm"):
        self._model_name = model_name
        self._call_count = 0
    
    def chat(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Mock VLM chat that returns evaluation scores.
        Score increases with each call to simulate improvement.
        """
        self._call_count += 1
        
        # Return mock evaluation JSON
        # Score starts low and increases with iterations
        score = min(50.0 + (self._call_count * 15.0), 95.0)
        feedback = f"Mock evaluation feedback (iteration {self._call_count}). Score: {score}"
        
        return json.dumps({
            "score": score,
            "feedback": feedback
        })
    
    @property
    def model_name(self) -> str:
        return self._model_name


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock embedding model that returns fixed-size embeddings."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
    
    def encode(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Mock encode method that handles both dict and string inputs.
        """
        embeddings = []
        for inp in inputs:
            # Handle case where input is a string (DSL) instead of dict
            if isinstance(inp, str):
                # Generate deterministic embedding based on string hash
                import hashlib
                hash_val = int(hashlib.md5(inp.encode()).hexdigest(), 16)
                embedding = [(hash_val % 1000) / 1000.0 for _ in range(self._dimension)]
            elif isinstance(inp, dict):
                # Generate embedding from dict content
                content = str(inp)
                import hashlib
                hash_val = int(hashlib.md5(content.encode()).hexdigest(), 16)
                embedding = [(hash_val % 1000) / 1000.0 for _ in range(self._dimension)]
            else:
                # Default embedding
                embedding = [0.1] * self._dimension
            
            embeddings.append(embedding)
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension


class MockRetriever(Retriever):
    """Mock retriever that returns a fixed scenario ID."""
    
    def __init__(self):
        # Initialize with None vectorstore since we're mocking
        super().__init__(vectorstore=None, top_k=1)
        self._scenario_id = "test_scenario_001"
    
    def retrieve(self, query_embedding: List[float], **kwargs) -> str:
        """
        Mock retrieve that returns a scenario ID.
        In the actual workflow, this should return a string ID.
        """
        # Return mock scenario ID
        return self._scenario_id
    
    def set_scenario_id(self, scenario_id: str):
        """Helper to set the scenario ID to return."""
        self._scenario_id = scenario_id


# ==========================================
# MOCK AGENTS (Custom implementations that return workflow-expected types)
# ==========================================

class MockInterpreterAgent(InterpreterAgent):
    """Mock InterpreterAgent that returns dict DSL (as workflow expects)."""
    
    def __init__(self):
        mock_llm = MockLLMModel()
        super().__init__(llm_service=mock_llm)
    
    def generate_dsl(self, user_query) -> dict:
        """
        Generate DSL - returns dict directly (workflow expects this).
        Handles both string and dict user_query.
        """
        # Return mock DSL dict
        return {
            "Scenario": "highway_overtaking",
            "Ego": "sedan",
            "Adversarials": ["truck"],
            "Spatial Relation": "ego is behind truck",
            "Requirement and restrictions": "overtake safely"
        }


class MockScenicCoderAgent(ScenicCoderAgent):
    """Mock ScenicCoderAgent that returns string code (as workflow expects)."""
    
    def __init__(self):
        mock_llm = MockLLMModel()
        super().__init__(llm_service=mock_llm)
    
    def adapt_code(self, original_scenic_code: str, aim_dsl) -> str:
        """
        Adapt code - returns string directly (workflow expects this).
        Handles both string and dict aim_dsl.
        """
        # Return adapted code as string
        return f"{original_scenic_code}\n# Adapted based on: {aim_dsl}\n# Mock adaptation"


class MockCriticAgent(CriticAgent):
    """Mock CriticAgent with fixed evaluation logic."""
    
    def __init__(self):
        mock_vlm = MockVLMModel()
        super().__init__(vlm_service=mock_vlm)
        self._call_count = 0
    
    def evaluate_with_vlm(self, video_path: str, original_query) -> Tuple[float, str]:
        """
        Evaluate with VLM - returns score and feedback.
        Handles both string and dict original_query.
        """
        self._call_count += 1
        # Score increases with iterations
        score = min(50.0 + (self._call_count * 15.0), 95.0)
        feedback = f"Mock evaluation feedback (iteration {self._call_count}). Score: {score}"
        return score, feedback


def create_mock_interpreter_agent():
    """Create a mock InterpreterAgent."""
    return MockInterpreterAgent()


def create_mock_coder_agent():
    """Create a mock ScenicCoderAgent."""
    return MockScenicCoderAgent()


def create_mock_critic_agent():
    """Create a mock CriticAgent."""
    return MockCriticAgent()


# ==========================================
# TEST SETUP
# ==========================================

def setup_mock_scenario_files():
    """Create mock scenario files for testing."""
    scenario_dir = Path("data/scenarios/test_scenario_001")
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    code_file = scenario_dir / "code.scenic"
    code_file.write_text("""
scenario mock_scenario:
    ego = Car
    truck = Truck
    # Mock scenic code for testing
""")
    
    return scenario_dir


def cleanup_mock_scenario_files():
    """Clean up mock scenario files."""
    import shutil
    scenario_dir = Path("data/scenarios/test_scenario_001")
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)


# ==========================================
# WORKFLOW TEST
# ==========================================

@pytest.mark.unit
def test_workflow_run():
    """Test that the workflow can run seamlessly with mocks."""
    import logging
    from src.utils import setup_logging
    
    # Setup logging
    setup_logging(level="INFO", run_name="workflow_test")
    logger = logging.getLogger(__name__)
    
    # Create mock scenario files
    setup_mock_scenario_files()
    
    try:
        # Create mock services
        mock_embedder = MockEmbeddingModel(dimension=384)
        mock_retriever = MockRetriever()
        
        # Create mock agents
        mock_interpreter = create_mock_interpreter_agent()
        mock_coder = create_mock_coder_agent()
        mock_critic = create_mock_critic_agent()
        
        # Create workflow with mocks
        logger.info("🚀 Initializing workflow with mock services...")
        workflow = ScenarioWorkflow(
            interpreter=mock_interpreter,
            coder=mock_coder,
            critic=mock_critic,
            retriever=mock_retriever,
            embedder=mock_embedder
        )
        
        # Initial state
        initial_state = {
            "user_query": {"text": "highway scenario with truck"},
            "max_count": 3,
            "messages": []
        }
        config = {"configurable": {"thread_id": "test_workflow_1"}}
        
        logger.info("🚀 Starting workflow run...")
        
        # Run workflow until it pauses for human review
        events = []
        for event in workflow.app.stream(initial_state, config=config):
            events.append(event)
            logger.info(f"📊 Event: {list(event.keys())}")
        
        logger.info(f"✅ Workflow paused after {len(events)} events")
        
        # Simulate human review - user rejects and provides feedback
        logger.info("🛑 Simulating user rejection with feedback...")
        workflow.app.update_state(config, {
            "user_satisfied": False,
            "user_modification": {"text": "Make it rain and add more vehicles"}
        })
        
        # Resume workflow
        logger.info("🚀 Resuming workflow with user feedback...")
        for event in workflow.app.stream(None, config=config):
            events.append(event)
            logger.info(f"📊 Event: {list(event.keys())}")
        
        # Simulate user acceptance
        logger.info("🛑 Simulating user acceptance...")
        workflow.app.update_state(config, {"user_satisfied": True})
        
        # Final run
        logger.info("🚀 Final workflow run...")
        for event in workflow.app.stream(None, config=config):
            events.append(event)
            logger.info(f"📊 Event: {list(event.keys())}")
        
        logger.info(f"✅ Workflow completed successfully with {len(events)} total events")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow test failed: {e}", exc_info=True)
        return False
        
    finally:
        # Cleanup
        cleanup_mock_scenario_files()


if __name__ == "__main__":
    success = test_workflow_run()
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
