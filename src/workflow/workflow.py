import logging
from typing import Dict, Literal
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path

from src.schema import MultimodalQuery, ScenarioDocument, ScenicScenario
from src.utils.helpers import get_scenario_document_with_scenario_id

# Add the project root (ads-mrag) to the python path
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from .scenario_workflow_state import CLEAN_STATE, MAX_COUNT, ScenarioWorkflowState
from src.utils import find_scenic_code_with_scenario_id, flatten_scenario_dsl_to_str, get_error_message_from_logs, run_simulation_in_carla_and_save_video, setup_logging, log_workflow_state, to_safe_string
from src.agents import InterpreterAgent, ScenicCoderAgent, CriticAgent
from src.services import Retriever, BaseEmbeddingModel

class ScenarioWorkflow:
    def __init__(
        self, 
        interpreter: InterpreterAgent, 
        coder: ScenicCoderAgent, 
        critic: CriticAgent, 
        retriever: Retriever, 
        embedder: BaseEmbeddingModel,
        logger: logging.Logger
        ):
        self.workflow = StateGraph(ScenarioWorkflowState)
        
        # Inject agents
        self.interpreter = interpreter
        self.coder = coder
        self.critic = critic
        self.retriever = retriever
        self.embedder = embedder
        self.logger = logger

        # 1. add nodes
        self.workflow.add_node("embed_query", self.embed_query)
        self.workflow.add_node("retrieve_base_scenario", self.retrieve_base_scenario)
        self.workflow.add_node("run_simulation", self.run_simulation)
        self.workflow.add_node("evaluate_with_vlm", self.evaluate_with_vlm)
        self.workflow.add_node("interpret", self.interpret)
        self.workflow.add_node("adapt_code", self.adapt_code)
        self.workflow.add_node("output_best_scenario", self.output_best_scenario)
        self.workflow.add_node("human_review", self.human_review)

        # 2. add static edges
        self.workflow.add_edge(START, "embed_query")
        self.workflow.add_edge("embed_query", "retrieve_base_scenario")
        self.workflow.add_edge("retrieve_base_scenario", "run_simulation")
        self.workflow.add_edge("run_simulation", "evaluate_with_vlm")
        self.workflow.add_edge("interpret", "adapt_code")
        self.workflow.add_edge("adapt_code", "run_simulation")
        self.workflow.add_edge("output_best_scenario", "human_review")

        # 3. add dynamic edges
        self.workflow.add_conditional_edges(
            "evaluate_with_vlm",
            self.route_after_vlm_evaluation,
            {
                "output_best_scenario": "output_best_scenario",
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
        count = state.get("generation_count", 0)
        max_count = MAX_COUNT

        best_score = -1.0
        best_scenario = None
        scenic_scenarios_list = state.get("scenic_scenarios_list", [])

        # if to find the best scenario should depends on if user has provided any feedback
        # if user provides feedback, which means the scenarios list should be cleaned, because that is already from the last round
        if not scenic_scenarios_list or len(scenic_scenarios_list) == 0:
            self.logger.info("🚦 ROUTER: No scenic scenarios list provided, sending to interpreter for dsl generation")
            return "interpret"

        for scenario in scenic_scenarios_list:
            if scenario.score and scenario.score > best_score and scenario.error is None:
                best_score = scenario.score
                best_scenario = scenario
        
        if count >= max_count or (best_scenario and best_score >= 70):
            self.logger.info(f"🚦 ROUTER: Best score {best_score} or max count {count}/{max_count} reached. Sending to User.")
            return "output_best_scenario"
        else:
            self.logger.info(f"🚦 ROUTER: Best score {best_score} is too low. Sending to Interpreter.")
            return "interpret"

    def route_after_human_review(self, state: ScenarioWorkflowState) -> Literal["end", "interpret"]:
        if state.get("user_satisfied", False):
            self.logger.info("🚦 ROUTER: User satisfied. Ending workflow.")
            return "end"
        else:
            self.logger.info("🚦 ROUTER: User rejected. Routing to Interpreter.")
            return "interpret"

    # ==========================================
    # NODE IMPLEMENTATIONS
    # ==========================================
    def embed_query(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(self.logger, "embed_query", state)
        
        query = state.get('user_query', None)
        if not query:
            self.logger.error("No user query provided")
            return {}
        dsl, flattened_text = self.interpreter.generate_dsl_from_user_query(query) # flatten text to reduce the noise caused by the formatting of the DSL
        query_to_embed = MultimodalQuery(text=flattened_text, image_path=query.image_path, video_path=query.video_path)
        query_embeddings = self.embedder.encode([query_to_embed.model_dump()])
        
        # NOTE: embedder may return a Tensor; `if not tensor` is invalid in PyTorch.
        if query_embeddings is None:
            self.logger.error("Failed to embed query (got None)")
            return {}

        # Handle both list-like batches and tensor-like batches
        try:
            batch_size = len(query_embeddings)
        except TypeError:
            batch_size = int(getattr(query_embeddings, "shape", [0])[0] or 0)

        if batch_size == 0:
            self.logger.error("Failed to embed query")
            return {}

        query_embedding = query_embeddings[0]

        # detect header settings
        header_settings = self.interpreter.generate_header_settings(query)
        if not header_settings:
            self.logger.info("🔍 Failed to generate header settings, use None to indicate not changing the header")
            header_settings = None # use None to indicate not changing the header

        self.logger.info("🧹 CLEANUP: Wiping previous scenario data for fresh run...")
        state = CLEAN_STATE
        state["user_query"] = query
        state["scenario_dsl"] = dsl
        state["query_embedding"] = query_embedding
        state["header_settings"] = header_settings
        return state

    def retrieve_base_scenario(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(self.logger, "retrieve_base_scenario", state)
        
        # 1. get user query and query embedding
        user_query = state.get("user_query", None)
        query_embedding = state.get("query_embedding", None)
        if user_query is None or query_embedding is None:
            self.logger.error("No user query or query embedding provided")
            return {}
            
        # 2. retrieve the base scenario
        best_scenarios = self.retriever.retrieve(
            original_query=user_query, 
            query_embedding=query_embedding,
        )
        if not best_scenarios or len(best_scenarios) == 0:
            self.logger.error("No scenarios found for query")
            return {}
            
        base_scenario_id = best_scenarios[0].scenario_id # only return the best 1 scenario
        scenic_code = find_scenic_code_with_scenario_id(base_scenario_id)

        # 3. replace the header of the scenic code when header settings are provided
        header_settings = state.get("header_settings", None) # None means not changing the header
        if header_settings:
            scenic_code = self.coder.replace_header(scenic_code, header_settings)
            self.logger.info(f"🔍 Replaced header of the scenic code with the header settings.")
        
        self.logger.info(f"🔍 Found best scenario: {base_scenario_id}")
        base_scenario_document = get_scenario_document_with_scenario_id(base_scenario_id)
        scenic_scenario = ScenicScenario(scenario_id=base_scenario_id, scenic_code=scenic_code, description=base_scenario_document.description)
        return {
            "base_scenario_id": base_scenario_id,
            "current_scenic_scenario": scenic_scenario,
            "scenic_scenarios_list": [scenic_scenario],
            "query_embedding": []
        }

    def run_simulation(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(self.logger, "run_simulation", state)
        
        # 1. get current scenario and scenic scenarios list
        scenic_scenario = state.get("current_scenic_scenario", None)
        scenic_scenarios_list = state.get("scenic_scenarios_list", [])
        if not scenic_scenario or not scenic_scenario.scenic_code or not scenic_scenario.scenario_id:
            self.logger.error("No current scenic scenario provided")
            return state

        # 2. run simulation
        self.logger.info("🎬 Running Carla Simulation...")
        video_path = run_simulation_in_carla_and_save_video(scenic_scenario.scenic_code, scenic_scenario.scenario_id)

        # 3. if simulation failed, put the error message and the score, this kind of scenario will not go to evaluate with vlm
        if not video_path:
            error_message = get_error_message_from_logs(scenic_scenario.scenario_id)
            if error_message:
                self.logger.error(f"Simulation failed with error message: {error_message}")
                scenic_scenario.error = error_message
                scenic_scenario.score = None

                for scenario in scenic_scenarios_list:
                    if scenario.scenario_id == scenic_scenario.scenario_id:
                        scenario.error = error_message
                        scenario.score = None
                        break
            else:
                self.logger.error("Simulation failed, but no error message found in logs")
                scenic_scenario.error = "Simulation failed, but no error message found in logs"
                scenic_scenario.score = None

                for scenario in scenic_scenarios_list:
                    if scenario.scenario_id == scenic_scenario.scenario_id:
                        scenario.error = "Simulation failed, but no error message found in logs"
                        scenario.score = None
                        break
        
        # 4. if simulation succeeded, update the current scenario with the video path
        else:
            scenic_scenario.error = None
            scenic_scenario.score = None
            for scenario in scenic_scenarios_list:
                if scenario.scenario_id == scenic_scenario.scenario_id:
                    scenario.error = None
                    scenario.score = None
                    break
        return {
            "scenic_scenarios_list": scenic_scenarios_list,
            "current_scenic_scenario": scenic_scenario,
        }

    def evaluate_with_vlm(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(self.logger, "evaluate_with_vlm", state)
        
        # 1. get query and current scenario, if not provided, return the state
        original_query = state.get("user_query", None) # TODO: have to be combined with the user modification (chat history)
        current_scenic_scenario = state.get("current_scenic_scenario", None)
        if not original_query or not current_scenic_scenario or not current_scenic_scenario.scenario_id:
            self.logger.error("No original query or scenario provided")
            return state
        if current_scenic_scenario.error: # if the scenario has error, skip the evaluation
            self.logger.error("Scenario has error, skipping evaluation")
            return state

        # 2. get video path and compose the scenario document
        video_path = f"temp/{current_scenic_scenario.scenario_id}/video/BEV.mp4"
        scenario_document = ScenarioDocument(
            scenario_id=current_scenic_scenario.scenario_id,
            description=current_scenic_scenario.description,
            scenic_code=current_scenic_scenario.scenic_code,
            image_path=None,
            video_path=video_path
        )
        if not scenario_document:
            self.logger.error("No scenario document found for id")
            return state
        
        # 3. evaluate with vlm
        score, feedback, evaluation_result = self.critic.evaluate_with_vlm(original_query, scenario_document)
        self.logger.info(f"📊 VLM Score: {score}")
        if feedback is None:
            self.logger.error("Failed to evaluate with VLM")
            return state

        # 4. update the current scenario with the evaluation result
        current_scenic_scenario.score = score
        current_scenic_scenario.evaluation_feedback = feedback
        current_scenic_scenario.evaluation_result = evaluation_result
        scenic_scenarios_list = state.get("scenic_scenarios_list", [])
        for scenario in scenic_scenarios_list:
            if scenario.scenario_id == current_scenic_scenario.scenario_id:
                scenario.score = score
                scenario.evaluation_feedback = feedback
                scenario.evaluation_result = evaluation_result
                break
        return {
            "scenic_scenarios_list": scenic_scenarios_list,
            "current_scenic_scenario": current_scenic_scenario,
        }

    def interpret(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(self.logger, "interpret", state)

        if state.get("user_modification"):
            self.logger.info("🧠 Interpreting user modification into DSL...")
            feedback = state.get("user_modification")
            best_scenario = state.get("best_scenario", None)
            
            # 1. if no best scenario, use the original scenario dsl for new dsl generation
            if not best_scenario: 
                self.logger.error("No best scenario provided, will use the original scenario_dsl for generation")
                modified_dsl = self.interpreter.generate_dsl_from_user_feedback(user_feedback=feedback, original_dsl=state.get("scenario_dsl", {}), scenario_to_modify=None)
            # 2. if best scenario is provided, use both original dsl and best scenario for new dsl generation
            else:
                modified_dsl = self.interpreter.generate_dsl_from_user_feedback(user_feedback=feedback, original_dsl=state.get("scenario_dsl", {}), scenario_to_modify=best_scenario)

            # 3. compose a new user query from the modified dsl for the vlm evaluation
            flattened_modified_dsl = flatten_scenario_dsl_to_str(modified_dsl)
            if not flattened_modified_dsl:
                flattened_modified_dsl = to_safe_string(modified_dsl)
            new_user_query = MultimodalQuery(
                text=flattened_modified_dsl,
                image_path=feedback.image_path,
                video_path=feedback.video_path
            )

            # 4. generate a new header settings
            new_header_settings = self.interpreter.generate_header_settings(new_user_query)
            if not new_header_settings:
                self.logger.info("🔍 Failed to generate new header settings, use None to indicate not changing the header")
                new_header_settings = None # use None to indicate not changing the header
            else:
                new_header_settings = new_header_settings
                self.logger.info(f"🔍 Generated new header settings: {new_header_settings}")
            return {
                "user_query": new_user_query,
                "header_settings": new_header_settings,
                "current_scenic_scenario": best_scenario, # use the best scenario for adapt code
                "user_modification": None, # clean the user modification for the next round generation so it will not be interpreted again
                "user_satisfied": None, # clean the user satisfied for the next round generation so it will not be interpreted again
                "best_scenario": None, # clean the best scenario for the next round generation
                "scenic_scenarios_list": [], # clean the scenarios list for the next round generation
                "scenario_dsl": modified_dsl,
                "messages": [
                    {"role": "assistant", "content": f"Modified DSL: {to_safe_string(modified_dsl)}"}
                ]
            }
        else:
            self.logger.error("No feedback to interpret, using the original user query dsl for generation")
            return {}

    def adapt_code(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(self.logger, "adapt_code", state)

        # 1. get generation count and current state
        generation_count = state.get("generation_count", 0)
        scenario_dsl = state.get("scenario_dsl", {}) # aim dsl
        current_scenic_scenario = state.get("current_scenic_scenario", None)
        if not current_scenic_scenario or not current_scenic_scenario.scenic_code or not scenario_dsl:
            self.logger.error("No current scenic scenario or scenic code or aim dsl provided")
            return state
        current_scenic_code = current_scenic_scenario.scenic_code
        current_evaluation_result = current_scenic_scenario.evaluation_result
        current_scenic_code_error = current_scenic_scenario.error
        header_settings = state.get("header_settings", None)

        # 2. adapt the code, if the code has error, call debug function, otherwise call adapt function
        adapted_scenic_code = ""
        if current_scenic_code_error or not current_evaluation_result:
            adapted_scenic_code = self.coder.debug_code(current_scenic_code, current_scenic_code_error, header_settings)
        else:
            adapted_scenic_code = self.coder.adapt_code(current_scenic_code, current_evaluation_result, scenario_dsl, header_settings)
        self.logger.info(f"🛠 Adapted Scenic code, generation count: {generation_count + 1}")

        # 3. update the scenic scenarios list with the adapted scenario
        adpated_scenario_id = f"{current_scenic_scenario.scenario_id}_adapted_{generation_count + 1}"
        adapted_scenic_scenario = ScenicScenario(scenario_id=adpated_scenario_id, scenic_code=adapted_scenic_code) # the description from previous scenario will not be used
        scenic_scenarios_list = state.get("scenic_scenarios_list", [])
        scenic_scenarios_list.append(adapted_scenic_scenario)
        return {
            "scenic_scenarios_list": scenic_scenarios_list,
            "current_scenic_scenario": adapted_scenic_scenario,
            "generation_count": generation_count + 1,
        }

    def output_best_scenario(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(self.logger, "output_best_scenario", state)
        
        scenic_scenarios_list = state.get("scenic_scenarios_list", [])
        best_scenario = None
        best_score = -1.0
        for scenario in scenic_scenarios_list:
            if scenario.score and scenario.score > best_score and scenario.error is None and scenario.scenic_code:
                best_score = scenario.score
                best_scenario = scenario
        if best_scenario:
            self.logger.info(f"🏆 Returning best scenario: {best_scenario.scenario_id}")
            return {
                "messages":[
                    {"role": "assistant", "content": f"Best scenario id is {best_scenario.scenario_id}, scenic code is {best_scenario.scenic_code}"}
                ],
                "best_scenario": best_scenario,
            }
        else:
            self.logger.error("No best scenario found")
            return {
                "messages":[
                    {"role": "assistant", "content": "Not able to generate a valid scenic code after multiple attempts."}
                ],
            }

    def human_review(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(self.logger, "human_review", state)
        
        user_satisfied = state.get("user_satisfied", False)
        
        if user_satisfied:
            self.logger.info("✅ Human Review: Accepted")
            return {}
        else:
            self.logger.info("❌ Human Review: Rejected/Modified")
            user_modification = state.get("user_modification", {"text": "Make it better."})
            return {
                "user_modification": user_modification,
                "generation_count": 0,
            }

# ==========================================
# TEST RUNNER
# ==========================================
if __name__ == "__main__":
    # 1. SETUP LOGGING FIRST
    setup_logging(level="INFO", run_name="scenic_workflow_test")

    # ScenarioWorkflow may require explicit dependency injection in some versions.
    # Fallback to app-level bootstrap if direct construction is not available.
    try:
        workflow = ScenarioWorkflow()
    except TypeError:
        from src.app import ChatbotWorkflow
        workflow = ChatbotWorkflow().initialize_system()

    user_query = MultimodalQuery(
        text="Generate me a scenario with the following requirements: The ego-vehicle encounters an obstacle blocking the lane and must perform a lane change into traffic moving in the same direction to avoid it. The obstacle may be a construction site, an accident or a parked vehicle.",
        image_path=None,
        video_path=None
    )
    initial_state = {"user_query": user_query}
    config = {"configurable": {"thread_id": "test_1"}}

    def _resume_with_feedback(feedback: MultimodalQuery | None, satisfied: bool) -> None:
        workflow.app.update_state(
            config,
            {
                "user_satisfied": satisfied,
                "user_modification": feedback,
            },
        )
        for _event in workflow.app.stream(None, config=config):
            pass

    workflow.logger.info("🚀 STARTING INITIAL WORKFLOW RUN...")
    for _event in workflow.app.stream(initial_state, config=config):
        pass

    # workflow.logger.info("🛑 GRAPH PAUSED. Pretending user clicked 'Reject' (round 1)...")
    # _resume_with_feedback(
    #     feedback=MultimodalQuery(text="Please add another car in the scenario which turns left at the intersection behind the ego vehicle as I marked with a red box in the image.", image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png", video_path=None),
    #     satisfied=False,
    # )

    workflow.logger.info("🛑 GRAPH PAUSED. Pretending user clicked 'Accept'...")
    _resume_with_feedback(feedback=None, satisfied=True)

    workflow.logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY.")