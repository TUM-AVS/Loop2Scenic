import logging
from copy import deepcopy
from typing import Any, Dict, Literal, Optional
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path

from src.schema import MultimodalQuery, ScenicScenario
from src.utils.helpers import get_scenario_document_with_scenario_id

# Add the project root (ads-mrag) to the python path
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from .scenario_workflow_state import CLEAN_STATE, ScenarioWorkflowState
from src.utils import find_scenic_code_with_scenario_id, get_error_message_from_logs, run_simulation_in_carla_and_save_video, setup_logging, log_workflow_state, to_safe_string
from src.agents import InterpreterAgent, ScenicCoderAgent, CriticAgent
from src.services import Retriever, BaseEmbeddingModel

class ScenarioWorkflow:
    def __init__(
        self, 
        interpreter: InterpreterAgent, 
        coder: ScenicCoderAgent, 
        critic: Optional[CriticAgent], 
        retriever: Retriever, 
        embedder: BaseEmbeddingModel,
        logger: logging.Logger
        ):
        self.workflow = StateGraph(ScenarioWorkflowState)
        
        # Inject agents
        self.interpreter = interpreter
        self.coder = coder
        self.critic = critic  # unused: VLM critic disabled; kept for optional re-enable
        self.retriever = retriever
        self.embedder = embedder
        self.logger = logger

        # 1. add nodes
        self.workflow.add_node("embed_query", self.embed_query)
        self.workflow.add_node("retrieve_base_scenario", self.retrieve_base_scenario)
        self.workflow.add_node("run_simulation", self.run_simulation)
        self.workflow.add_node("interpret", self.interpret)
        self.workflow.add_node("adapt_code", self.adapt_code)
        self.workflow.add_node("output_best_scenario", self.output_best_scenario)
        self.workflow.add_node("human_review", self.human_review)

        # 2. add static edges (no VLM critic: simulate → return scenario)
        self.workflow.add_edge(START, "embed_query")
        self.workflow.add_edge("embed_query", "retrieve_base_scenario")
        self.workflow.add_edge("retrieve_base_scenario", "run_simulation")
        self.workflow.add_edge("run_simulation", "output_best_scenario")
        self.workflow.add_edge("interpret", "adapt_code")
        self.workflow.add_edge("adapt_code", "run_simulation")
        self.workflow.add_edge("output_best_scenario", "human_review")

        # 3. add dynamic edges
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

    def _zero_metrics(self) -> Dict[str, float]:
        return {
            "calls": 0,
            "response_time_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _snapshot_service_metrics(self, service: Any) -> Dict[str, float]:
        if service is None or not hasattr(service, "get_metrics_snapshot"):
            return self._zero_metrics()
        snapshot = service.get_metrics_snapshot() or {}
        return {
            "calls": int(snapshot.get("calls", 0) or 0),
            "response_time_ms": float(snapshot.get("response_time_ms", 0.0) or 0.0),
            "prompt_tokens": int(snapshot.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(snapshot.get("completion_tokens", 0) or 0),
            "total_tokens": int(snapshot.get("total_tokens", 0) or 0),
        }

    def _delta_metrics(self, before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        return {
            "calls": max(0, int(after["calls"] - before["calls"])),
            "response_time_ms": max(0.0, float(after["response_time_ms"] - before["response_time_ms"])),
            "prompt_tokens": max(0, int(after["prompt_tokens"] - before["prompt_tokens"])),
            "completion_tokens": max(0, int(after["completion_tokens"] - before["completion_tokens"])),
            "total_tokens": max(0, int(after["total_tokens"] - before["total_tokens"])),
        }

    def _record_model_metrics(
        self,
        state: ScenarioWorkflowState,
        node_name: str,
        service_name: str,
        service_type: Literal["llm", "vlm"],
        delta: Dict[str, float],
    ) -> Dict[str, Any]:
        model_metrics = deepcopy(state.get("model_metrics") or {})
        totals_by_type = model_metrics.get("totals_by_type") or {
            "llm": self._zero_metrics(),
            "vlm": self._zero_metrics(),
        }
        totals_all = model_metrics.get("totals_all") or self._zero_metrics()
        by_node = model_metrics.get("by_node") or {}
        node_metrics = by_node.get(node_name) or {
            "llm": self._zero_metrics(),
            "vlm": self._zero_metrics(),
            "services": {},
        }
        services = node_metrics.get("services") or {}
        service_metrics = services.get(service_name) or self._zero_metrics()
        type_totals = totals_by_type.get(service_type) or self._zero_metrics()
        node_type_totals = node_metrics.get(service_type) or self._zero_metrics()

        for key in self._zero_metrics().keys():
            service_metrics[key] = service_metrics.get(key, 0) + delta[key]
            type_totals[key] = type_totals.get(key, 0) + delta[key]
            node_type_totals[key] = node_type_totals.get(key, 0) + delta[key]
            totals_all[key] = totals_all.get(key, 0) + delta[key]

        services[service_name] = service_metrics
        node_metrics["services"] = services
        node_metrics[service_type] = node_type_totals
        by_node[node_name] = node_metrics
        totals_by_type[service_type] = type_totals
        model_metrics["totals_by_type"] = totals_by_type
        model_metrics["totals_all"] = totals_all
        model_metrics["totals"] = totals_all
        model_metrics["by_node"] = by_node
        return model_metrics
        
    # ==========================================
    # ROUTING FUNCTIONS
    # ==========================================
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
        interpreter_vlm_before = self._snapshot_service_metrics(getattr(self.interpreter, "vlm_service", None))
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
        interpreter_vlm_after = self._snapshot_service_metrics(getattr(self.interpreter, "vlm_service", None))
        interpreter_vlm_delta = self._delta_metrics(interpreter_vlm_before, interpreter_vlm_after)
        if not header_settings:
            self.logger.info("🔍 Failed to generate header settings, use None to indicate not changing the header")
            header_settings = None # use None to indicate not changing the header

        self.logger.info("🧹 CLEANUP: Wiping previous scenario data for fresh run...")
        state = deepcopy(CLEAN_STATE)
        state["user_query"] = query
        state["scenario_dsl"] = dsl
        state["query_embedding"] = query_embedding
        state["header_settings"] = header_settings
        state["model_metrics"] = self._record_model_metrics(
            state,
            node_name="embed_query",
            service_name="interpreter_vlm",
            service_type="vlm",
            delta=interpreter_vlm_delta,
        )
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
        retrieval = self.retriever.retrieve(
            original_query=user_query, 
            query_embedding=query_embedding,
        )
        if not retrieval.scenarios:
            self.logger.error("No scenarios found for query")
            return {}
            
        base_scenario_id = retrieval.scenarios[0].scenario_id # only return the best 1 scenario
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

    def interpret(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(self.logger, "interpret", state)

        if state.get("user_modification"):
            interpreter_vlm_before = self._snapshot_service_metrics(getattr(self.interpreter, "vlm_service", None))
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

            # 3. compose a new user query for header settings generation, includes old header settings and user feedback
            new_header_settings_prompt = "Previous header settings in the chat history: \n"
            if state.get("header_settings", None):
                new_header_settings_prompt += to_safe_string(state.get("header_settings")) + "\n"

            flattened_modified_dsl = to_safe_string(modified_dsl)
            if new_header_settings_prompt.strip():
                flattened_modified_dsl = f"{new_header_settings_prompt}\n{flattened_modified_dsl}"
            if feedback.text:
                flattened_modified_dsl += f"\nUser suggestion text: {feedback.text}\n"
            new_user_query = MultimodalQuery(
                text=flattened_modified_dsl,
                image_path=feedback.image_path,
                video_path=feedback.video_path
            )

            # 4. generate a new header settings
            new_header_settings = self.interpreter.generate_header_settings(new_user_query)
            interpreter_vlm_after = self._snapshot_service_metrics(getattr(self.interpreter, "vlm_service", None))
            interpreter_vlm_delta = self._delta_metrics(interpreter_vlm_before, interpreter_vlm_after)
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
                ],
                "model_metrics": self._record_model_metrics(
                    state,
                    node_name="interpret",
                    service_name="interpreter_vlm",
                    service_type="vlm",
                    delta=interpreter_vlm_delta,
                ),
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

        # 2. adapt the code: debug on sim error; otherwise regenerate from DSL
        #    (no VLM critic — evaluation_result may be None)
        coder_llm_before = self._snapshot_service_metrics(getattr(self.coder, "llm_service", None))
        adapted_scenic_code = current_scenic_code
        adapt_warnings: list[str] = []
        try:
            if current_scenic_code_error or not current_evaluation_result:
                if hasattr(self.coder, "last_debug_failure"):
                    self.coder.last_debug_failure = None
                adapted_scenic_code = self.coder.debug_code(
                    current_scenic_code, current_scenic_code_error
                )
                debug_failure = getattr(self.coder, "last_debug_failure", None)
                if debug_failure:
                    adapt_warnings.append(str(debug_failure))
            else:
                adapted_scenic_code = self.coder.adapt_code(
                    current_scenic_code,
                    current_evaluation_result,
                    scenario_dsl,
                    header_settings,
                )
            if not adapted_scenic_code:
                warning = "Coder returned empty scenic code; keeping previous script"
                self.logger.error(warning)
                adapt_warnings.append(warning)
                adapted_scenic_code = current_scenic_code
        except Exception as exc:
            # Keep previous code so we still reach output_best_scenario / baseline fallback.
            warning = f"adapt_code/debug_code failed ({exc}); keeping previous scenic code"
            self.logger.exception(warning)
            adapt_warnings.append(warning)
            adapted_scenic_code = current_scenic_code
        self.logger.info(f"🛠 Adapted Scenic code, generation count: {generation_count + 1}")
        coder_llm_after = self._snapshot_service_metrics(getattr(self.coder, "llm_service", None))
        coder_llm_delta = self._delta_metrics(coder_llm_before, coder_llm_after)

        # 3. update the scenic scenarios list with the adapted scenario
        adpated_scenario_id = f"{current_scenic_scenario.scenario_id}_adapted_{generation_count + 1}"
        adapted_scenic_scenario = ScenicScenario(scenario_id=adpated_scenario_id, scenic_code=adapted_scenic_code) # the description from previous scenario will not be used
        scenic_scenarios_list = state.get("scenic_scenarios_list", [])
        scenic_scenarios_list.append(adapted_scenic_scenario)
        result: Dict = {
            "scenic_scenarios_list": scenic_scenarios_list,
            "current_scenic_scenario": adapted_scenic_scenario,
            "generation_count": generation_count + 1,
            "model_metrics": self._record_model_metrics(
                state,
                node_name="adapt_code",
                service_name="coder_llm",
                service_type="llm",
                delta=coder_llm_delta,
            ),
        }
        if adapt_warnings:
            result["messages"] = [
                {
                    "role": "assistant",
                    "content": f"[workflow_warning] {warning}",
                }
                for warning in adapt_warnings
            ]
        return result

    def output_best_scenario(self, state: ScenarioWorkflowState) -> Dict:
        log_workflow_state(self.logger, "output_best_scenario", state)

        scenic_scenarios_list = state.get("scenic_scenarios_list", []) or []
        current = state.get("current_scenic_scenario")

        # Prefer the scenario just simulated (no VLM critic scoring).
        if current and current.scenic_code and current.error is None:
            self.logger.info(f"🏆 Returning current scenario (no critic): {current.scenario_id}")
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            f"Best scenario id is {current.scenario_id}, "
                            f"scenic code is {current.scenic_code}"
                        ),
                    }
                ],
                "best_scenario": current,
            }

        # Prefer any scored candidate if present (legacy / optional critic).
        best_scenario = None
        best_score = -1.0
        for scenario in scenic_scenarios_list:
            if (
                scenario.score is not None
                and scenario.score > best_score
                and scenario.error is None
                and scenario.scenic_code
            ):
                best_score = scenario.score
                best_scenario = scenario

        if best_scenario:
            self.logger.info(f"🏆 Returning best scenario: {best_scenario.scenario_id}")
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            f"Best scenario id is {best_scenario.scenario_id}, "
                            f"scenic code is {best_scenario.scenic_code}"
                        ),
                    }
                ],
                "best_scenario": best_scenario,
            }

        # Fall back to the retrieved base scenario (even if sim had errors).
        fallback = self._fallback_base_scenario(state, scenic_scenarios_list)
        if fallback:
            self.logger.warning(
                "No clean current scenario; falling back to base scenario: %s "
                "(error=%s, score=%s)",
                fallback.scenario_id,
                getattr(fallback, "error", None),
                getattr(fallback, "score", None),
            )
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            f"Fell back to base scenario id {fallback.scenario_id}, "
                            f"scenic code is {fallback.scenic_code}"
                        ),
                    }
                ],
                "best_scenario": fallback,
            }

        self.logger.error("No best scenario found (and no base-scenario fallback available)")
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Not able to generate a valid scenic code after multiple attempts.",
                }
            ],
        }

    def _fallback_base_scenario(
        self,
        state: ScenarioWorkflowState,
        scenic_scenarios_list: list,
    ) -> ScenicScenario | None:
        """Prefer the retrieved base scenario; else any list entry with scenic code."""
        base_id = state.get("base_scenario_id")

        if base_id:
            for scenario in scenic_scenarios_list:
                if scenario.scenario_id == base_id and scenario.scenic_code:
                    return scenario

        for scenario in scenic_scenarios_list:
            if scenario.scenic_code:
                return scenario

        if not base_id:
            return None

        scenic_code = find_scenic_code_with_scenario_id(base_id)
        if not scenic_code:
            return None
        description = ""
        try:
            doc = get_scenario_document_with_scenario_id(base_id)
            if doc is not None:
                description = getattr(doc, "description", "") or ""
        except Exception:
            pass
        return ScenicScenario(
            scenario_id=base_id,
            scenic_code=scenic_code,
            description=description,
        )

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
        text="Vehicle A and vehicle B were both heading in the same direction on a multi-lane road in different lanes. B attempted to turn from the curb lane across the path of A onto a side street. Driver A struck illegally turning B in the driver's side.",
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