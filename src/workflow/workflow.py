import logging
from typing import Dict, Literal
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path

from src.schema import MultimodalQuery
from src.utils.helpers import get_scenario_document_with_scenario_id

# Add the project root (ads-mrag) to the python path
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from .scenario_workflow_state import ScenarioWorkflowState
from src.utils import find_scenic_code_with_scenario_id, run_simulation_in_carla_and_save_video, setup_logging, log_workflow_state, to_safe_string
from src.agents import InterpreterAgent, ScenicCoderAgent, CriticAgent
from src.services import Retriever, BaseEmbeddingModel

# Get a logger for this specific file
logger = logging.getLogger(__name__)

class ScenarioWorkflow:
    def __init__(
        self, 
        interpreter: InterpreterAgent, 
        coder: ScenicCoderAgent, 
        critic: CriticAgent, 
        retriever: Retriever, 
        embedder: BaseEmbeddingModel
        ):
        self.workflow = StateGraph(ScenarioWorkflowState)
        
        # Inject agents
        self.interpreter = interpreter
        self.coder = coder
        self.critic = critic
        self.retriever = retriever
        self.embedder = embedder

        # 1. add nodes
        self.workflow.add_node("embed_query", self.embed_query)
        self.workflow.add_node("retrieve_base_scenario", self.retrieve_base_scenario)
        self.workflow.add_node("run_simulation", self.run_simulation)
        self.workflow.add_node("evaluate_with_vlm", self.evaluate_with_vlm)
        self.workflow.add_node("interpret", self.interpret)
        self.workflow.add_node("adapt_code", self.adapt_code)
        self.workflow.add_node("human_review", self.human_review)

        # 2. add static edges
        self.workflow.add_edge(START, "embed_query")
        self.workflow.add_edge("embed_query", "retrieve_base_scenario")
        self.workflow.add_edge("retrieve_base_scenario", "run_simulation")
        self.workflow.add_edge("run_simulation", "evaluate_with_vlm")
        self.workflow.add_edge("interpret", "adapt_code")
        self.workflow.add_edge("adapt_code", "run_simulation")

        # 3. add dynamic edges
        self.workflow.add_conditional_edges(
            "evaluate_with_vlm",
            self.route_after_vlm_evaluation,
            {
                "human_review": "human_review",
                "interpret": "interpret",
            }
        )
        self.workflow.add_conditional_edges(
            "human_review",
            self.route_after_human_review,
            {
                "interpret": "interpret",
                "end": END,
            }
        )
        
        # 4. compile the graph with memory
        memory = MemorySaver()
        self.app = self.workflow.compile(
            checkpointer=memory, 
            interrupt_before=["human_review"] # wait for human review before going to next node
        )
        
    # ==========================================
    # ROUTING FUNCTIONS
    # ==========================================
    def route_after_vlm_evaluation(self, state: ScenarioWorkflowState) -> Literal["human_review", "interpret"]:
        score = state.get("evaluation_score", 0.0)
        count = state.get("generation_count", 0)
        max_count = state.get("max_count", 3)
        
        if score >= 0 or count >= max_count:
            logger.info(f"🚦 ROUTER: Score {score} or max count {count}/{max_count} reached. Sending to User.")
            return "human_review"
        else:
            logger.info(f"🚦 ROUTER: Score {score} is too low. Sending to Interpreter.")
            return "interpret"

    def route_after_human_review(self, state: ScenarioWorkflowState) -> Literal["end", "interpret"]:
        if state.get("user_satisfied", False):
            logger.info("🚦 ROUTER: User satisfied. Ending workflow.")
            return "end"
        else:
            logger.info("🚦 ROUTER: User rejected. Routing to Interpreter.")
            return "interpret"

    # ==========================================
    # NODE IMPLEMENTATIONS
    # ==========================================
    def embed_query(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(logger, "embed_query", state)
        
        query = state.get('user_query', None)
        if not query:
            logger.error("No user query provided")
            return {
                "user_query": None,
                "scenario_dsl": None,
                "query_embedding": None,
                "base_scenario_id": "",
                "current_scenic_code": "",
                "simulation_video_path": "",
                "evaluation_score": 0.0,
                "evaluation_feedback": "",
                "evaluation_result": None,
                "best_scenic_code": "",
                "best_score": -1.0,
                "user_satisfied": None,
                "user_modification": None,
                "generation_count": 0
            }
        dsl, flattened_text = self.interpreter.generate_dsl_from_user_query(query) # flatten text to reduce the noise caused by the formatting of the DSL
        query_to_embed = MultimodalQuery(text=flattened_text, image_path=query.image_path, video_path=query.video_path)
        query_embeddings = self.embedder.encode([query_to_embed.model_dump()])
        
        # NOTE: embedder may return a Tensor; `if not tensor` is invalid in PyTorch.
        if query_embeddings is None:
            logger.error("Failed to embed query (got None)")
            return {
                "user_query": None,
                "scenario_dsl": None,
                "query_embedding": None,
                "base_scenario_id": "",
                "current_scenic_code": "",
                "simulation_video_path": "",
                "evaluation_score": 0.0,
                "evaluation_feedback": "",
                "evaluation_result": None,
                "best_scenic_code": "",
                "best_score": -1.0,
                "user_satisfied": None,
                "user_modification": None,
                "generation_count": 0
            }

        # Handle both list-like batches and tensor-like batches
        try:
            batch_size = len(query_embeddings)
        except TypeError:
            batch_size = int(getattr(query_embeddings, "shape", [0])[0] or 0)

        if batch_size == 0:
            logger.error("Failed to embed query")
            return {
                "user_query": None,
                "scenario_dsl": None,
                "query_embedding": None,
                "base_scenario_id": "",
                "current_scenic_code": "",
                "simulation_video_path": "",
                "evaluation_score": 0.0,
                "evaluation_feedback": "",
                "evaluation_result": None,
                "best_scenic_code": "",
                "best_score": -1.0,
                "user_satisfied": None,
                "user_modification": None,
                "generation_count": 0
            }

        query_embedding = query_embeddings[0]

        logger.info("🧹 CLEANUP: Wiping previous scenario data for fresh run...")
        return {
            "scenario_dsl": dsl,
            "query_embedding": query_embedding,
            "base_scenario_id": "",
            "current_scenic_code": "",
            "simulation_video_path": "",
            "evaluation_score": 0.0,
            "evaluation_feedback": "",
            "evaluation_result": None,
            "best_scenic_code": "",
            "best_score": -1.0,
            "user_satisfied": None,
            "user_modification": None,
            "generation_count": 0
        }

    def retrieve_base_scenario(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(logger, "retrieve_base_scenario", state)
        
        user_query = state.get("user_query", None)
        query_embedding = state.get("query_embedding", None)
        # NOTE: query_embedding can be a Tensor; avoid `if not tensor` ambiguity.
        if user_query is None or query_embedding is None:
            logger.error("No user query or query embedding provided")
            return {
                "base_scenario_id": "",
                "current_scenic_code": "",
                "evaluation_result": None,
            }
            
        best_scenarios = self.retriever.retrieve(
            original_query=user_query, 
            query_embedding=query_embedding,
        )
        if not best_scenarios or len(best_scenarios) == 0:
            logger.error("No scenarios found for query")
            return {
                "base_scenario_id": "",
                "current_scenic_code": "",
                "evaluation_result": None,
            }
            
        base_scenario_id = best_scenarios[0].scenario_id # only return the best 1 scenario
        scenic_code = find_scenic_code_with_scenario_id(base_scenario_id)
        
        logger.info(f"🔍 Found best scenario: {base_scenario_id}")
        return {
            "base_scenario_id": base_scenario_id,
            "current_scenic_code": scenic_code
        }

    def run_simulation(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(logger, "run_simulation", state)
        
        scenic_code = state.get("current_scenic_code", "")
        logger.info("🎬 Running Carla Simulation...")
        video_path = run_simulation_in_carla_and_save_video(scenic_code)

        return {"simulation_video_path": video_path}

    def evaluate_with_vlm(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(logger, "evaluate_with_vlm", state)
        
        original_query = state.get("user_query", None)
        scenario_id = state.get("base_scenario_id", "")
        if not original_query or not scenario_id:
            logger.error("No original query or scenario id provided")
            return {
                "evaluation_score": 0.0,
                "evaluation_feedback": None,
                "evaluation_result": None
            }

        scenario_document = get_scenario_document_with_scenario_id(scenario_id)
        if not scenario_document:
            logger.error("No scenario document found for id")
            return {
                "evaluation_score": 0.0,
                "evaluation_feedback": None,
                "evaluation_result": None
            }
        
        score, feedback, evaluation_result = self.critic.evaluate_with_vlm(original_query, scenario_document)
        logger.info(f"📊 VLM Score: {score}")

        if feedback is None:
            logger.error("Failed to evaluate with VLM")
            return {
                "evaluation_score": 0.0,
                "evaluation_feedback": None,
                "evaluation_result": None
            }

        logger.info(f"📊 VLM Score: {score}")
        updates = {
            "evaluation_score": score,
            "evaluation_feedback": feedback,
            "evaluation_result": evaluation_result,
            "messages": [
                {"role": "assistant", "content": f"Score: {str(score)}"},
                {"role": "assistant", "content": to_safe_string(feedback)},
                {"role": "assistant", "content": f"Evaluation result: {to_safe_string(evaluation_result)}"}
            ]
        }

        if score > state.get("best_score", -1.0):
            logger.info("🏆 New best score achieved!")
            updates["best_score"] = score
            updates["best_scenic_code"] = state.get("current_scenic_code", "")

        return updates

    def interpret(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(logger, "interpret", state)

        if state.get("user_modification"):
            logger.info("🧠 Interpreting user modification into DSL...")
            feedback = state.get("user_modification")
            modified_dsl = self.interpreter.generate_dsl_from_user_feedback(feedback, state.get("scenario_dsl", {}))
            return {
                "scenario_dsl": modified_dsl,
                "messages": [
                    {"role": "assistant", "content": f"Modified DSL: {to_safe_string(modified_dsl)}"}
                ]
            }
        elif state.get("evaluation_feedback") and state.get("evaluation_result"):
            logger.info("🧠 For evaluation feedback, just return the scenario dsl from user query")
            return {
                "scenario_dsl": state.get("scenario_dsl", {}),
            }
        else:
            logger.error("No feedback to interpret")
            return {
                "scenario_dsl": None
            }

    def adapt_code(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(logger, "adapt_code", state)
        generation_count = state.get("generation_count", 0) 
        logger.info(f"🛠 Adapting Scenic code (Iteration: {generation_count + 1})")
        
        scenario_dsl = state.get("scenario_dsl", "")
        evaluation_result = state.get("evaluation_result", {})
        current_scenic_code = state.get("current_scenic_code", "")
        adapted_scenic_code = self.coder.adapt_code(current_scenic_code, evaluation_result, scenario_dsl)
        adapted_scenic_code_str = to_safe_string(adapted_scenic_code)

        logger.info(f"🛠 Adapting Scenic code: {adapted_scenic_code_str}")
        return {
            "current_scenic_code": adapted_scenic_code,
            "generation_count": generation_count + 1,
            "messages": [{"role": "assistant", "content": adapted_scenic_code_str}]
        }

    def human_review(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(logger, "human_review", state)
        
        user_satisfied = state.get("user_satisfied", False)
        
        if user_satisfied:
            logger.info("✅ Human Review: Accepted")
            return {}
        else:
            logger.info("❌ Human Review: Rejected/Modified")
            user_modification = state.get("user_modification", {"text": "Make it better."})
            return {
                "user_modification": user_modification,
                "generation_count": 0 
            }

# ==========================================
# TEST RUNNER
# ==========================================
if __name__ == "__main__":
    # 1. SETUP LOGGING FIRST
    setup_logging(level="INFO", run_name="scenic_workflow_test")
    
    workflow = ScenarioWorkflow()
    user_query = MultimodalQuery(
        text="highway scenario", 
        image_path=None, 
        video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4"
    )
    initial_state = {"user_query": user_query, "max_count": 3}
    config = {"configurable": {"thread_id": "test_1"}}
    
    logger.info("🚀 STARTING INITIAL WORKFLOW RUN...")
    for event in workflow.app.stream(initial_state, config=config):
        pass
        
    logger.info("🛑 GRAPH PAUSED. Pretending user clicked 'Reject'...")
    workflow.app.update_state(config, {
        "user_satisfied": False,
        "user_modification": MultimodalQuery(text="Make it rain", image_path=None, video_path=None)
    })
    
    logger.info("🚀 RESUMING WITH HUMAN FEEDBACK...")
    for event in workflow.app.stream(None, config=config):
        pass
        
    logger.info("🛑 GRAPH PAUSED. Pretending user clicked 'Reject'...")
    user_feedback = MultimodalQuery(
        text="Please add another car in the scenario which turns left at the intersection behind the ego vehicle as I marked with a red box in the image.", 
        image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png", 
        video_path=None
    )
    workflow.app.update_state(
        config, 
        {"user_satisfied": False,
        "user_modification": user_feedback
        }
    )

    logger.info("🚀 RESUMING WITH HUMAN FEEDBACK...")
    
    for event in workflow.app.stream(None, config=config):
        pass
    
    logger.info("🛑 GRAPH PAUSED. Pretending user clicked 'Accept'...")
    workflow.app.update_state(config, {"user_satisfied": True})
    
    logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY.")