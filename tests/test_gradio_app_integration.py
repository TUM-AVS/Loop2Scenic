import logging
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Add project root to path so `src.*` imports work when running `pytest` from repo root.
import sys

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.app import ChatbotWorkflow, build_ui  # noqa: E402
from src.workflow.workflow import ScenarioWorkflow  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass
class _MockScenarioDoc:
    scenario_id: str


class MockEmbeddingModel:
    """Minimal embedding model compatible with `ScenarioWorkflow.embed_query`."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, inputs: List[Any]) -> List[List[float]]:
        # Return deterministic embeddings regardless of input type.
        return [[0.01] * self._dimension for _ in inputs]


class MockRetriever:
    """Minimal retriever compatible with `ScenarioWorkflow.retrieve_base_scenario`."""

    def __init__(self, scenario_id: str = "test_scenario_001"):
        self._scenario_id = scenario_id

    def retrieve(self, query_embedding: Any, **kwargs) -> List[_MockScenarioDoc]:
        # `ScenarioWorkflow.retrieve_base_scenario` expects a list where [0].scenario_id exists.
        return [_MockScenarioDoc(scenario_id=self._scenario_id)]


class MockInterpreterAgent:
    """Mock interpreter that returns a DSL dict (what the workflow expects)."""

    def generate_dsl(self, user_query: Any) -> Dict[str, Any]:
        return {
            "Scenario": "highway_overtaking",
            "Ego": "sedan",
            "Adversarials": ["truck"],
            "Spatial Relation": "ego is behind truck",
            "Requirement and restrictions": "overtake safely",
        }


class MockScenicCoderAgent:
    """Mock coder that returns scenic code as a string (what the workflow stores/display)."""

    def adapt_code(self, original_scenic_code: str, aim_dsl: Any) -> str:
        return f"{original_scenic_code}\n# Mock adapted code based on DSL: {aim_dsl}"


class MockCriticAgent:
    """Mock critic that returns increasing evaluation scores to trigger human review."""

    def __init__(self):
        self._call_count = 0

    def evaluate_with_vlm(self, video_path: str, original_query: str) -> Tuple[float, str]:
        self._call_count += 1
        # Ensure the workflow router reaches `human_review` when score becomes > 90.
        score_value = min(50.0 + (self._call_count * 20.0), 95.0)
        feedback = f"Mock VLM evaluation feedback (iteration={self._call_count}). score={score_value}"
        return float(score_value), feedback


def _setup_mock_scenario_files(scenario_id: str = "test_scenario_001") -> None:
    scenario_dir = project_root / "data" / "scenarios" / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "code.scenic").write_text(
        "scenario mock_scenario:\n"
        "    ego = Car\n"
        "    truck = Truck\n"
        "    # Mock scenic code for testing\n",
        encoding="utf-8",
    )


def _cleanup_mock_scenario_files(scenario_id: str = "test_scenario_001") -> None:
    import shutil

    scenario_dir = project_root / "data" / "scenarios" / scenario_id
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)


def run_mock_gradio_app(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    scenario_id: str = "test_scenario_001",
) -> None:
    """
    Manual smoke test: launch the Gradio app on localhost, but wire ScenarioWorkflow with mocks.
    """
    scenario_path = project_root / "data" / "scenarios" / scenario_id
    logger.info(f"Setting up mock scenario at {scenario_path}")
    _setup_mock_scenario_files(scenario_id)

    # Ensure relative paths inside the workflow resolve correctly.
    old_cwd = Path.cwd()
    os.chdir(project_root)

    # Build the real workflow graph, but with mocked agents/services.
    workflow = ScenarioWorkflow(
        interpreter=MockInterpreterAgent(),
        coder=MockScenicCoderAgent(),
        critic=MockCriticAgent(),
        retriever=MockRetriever(scenario_id=scenario_id),
        embedder=MockEmbeddingModel(dimension=384),
    )

    original_initialize_system = ChatbotWorkflow.initialize_system

    def _mock_initialize_system(self: ChatbotWorkflow, config_path: Any = None):
        return workflow

    try:
        # Patch initialization so `ChatbotWorkflow.process_user_input()` uses mocks.
        ChatbotWorkflow.initialize_system = _mock_initialize_system  # type: ignore[method-assign]

        demo = build_ui()
        logger.info(f"Launching Gradio on http://{host}:{port}")
        demo.launch(server_name=host, server_port=port, share=share, prevent_thread_lock=False)
    finally:
        # Restore and cleanup.
        ChatbotWorkflow.initialize_system = original_initialize_system  # type: ignore[method-assign]
        os.chdir(old_cwd)
        _cleanup_mock_scenario_files(scenario_id)


if __name__ == "__main__":
    # Make sure we actually see logs/prints when running this file directly.
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    print("Starting mock Gradio app smoke test...")
    run_mock_gradio_app()

