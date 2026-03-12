from typing import Any, Dict, List, Tuple, Literal
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Assuming you have this file in the same directory
from scenario_workflow_state import ScenarioWorkflowState

class ScenarioWorkflow:
    def __init__(self):
        self.workflow = StateGraph(ScenarioWorkflowState)

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
        self.workflow.add_edge("adapt_code", "run_simulation") # Added the loop-back edge!

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
            interrupt_before=["human_review"]
        )
        
    # ==========================================
    # ROUTING FUNCTIONS
    # ==========================================
    def route_after_vlm_evaluation(self, state: ScenarioWorkflowState) -> Literal["human_review", "interpret"]:
        score = state.get("evaluation_score", 0.0)
        count = state.get("generation_count", 0)
        max_count = state.get("max_count", 3)
        
        if score > 90 or count >= max_count:
            print(f"  🚦 ROUTER: Score {score} or max count {count}/{max_count} reached. Sending to User.")
            return "human_review"
        else:
            print(f"  🚦 ROUTER: Score {score} is too low. Sending to Interpreter.")
            return "interpret"

    def route_after_human_review(self, state: ScenarioWorkflowState) -> Literal["end", "interpret"]:
        if state.get("user_satisfied", False):
            print("  🚦 ROUTER: User satisfied. Ending workflow.")
            return "end"
        else:
            print("  🚦 ROUTER: User rejected. Routing to Interpreter.")
            return "interpret"

    # ==========================================
    # NODE IMPLEMENTATIONS
    # ==========================================
    def embed_query(self, state: ScenarioWorkflowState) -> Dict:
        print("🟢 NODE: embed_query")
        query = state.get('user_query', {})
        dsl = generate_dsl(query)
        query_embedding = embed_query_func(dsl)

        print("  🧹 CLEANUP: Wiping previous scenario data for fresh run...")
        return {
            "scenario_dsl": dsl,
            "query_embedding": query_embedding,
            "base_scenario_id": "",
            "current_scenic_code": "",
            "simulation_video_path": "",
            "evaluation_score": 0.0,
            "evaluation_feedback": "",
            "best_scenic_code": "",
            "best_score": -1.0,
            "user_satisfied": None,
            "user_modification": None,
            "generation_count": 0
        }

    def retrieve_base_scenario(self, state: ScenarioWorkflowState) -> Dict:
        print("🟢 NODE: retrieve_base_scenario")
        query_embedding = state["query_embedding"]
        base_scenario_id = retrieve_base_scenario_func(query_embedding)
        scenic_code = find_scenic_code_with_scenario_id(base_scenario_id)
        
        return {
            "base_scenario_id": base_scenario_id,
            "current_scenic_code": scenic_code
        }

    def run_simulation(self, state: ScenarioWorkflowState) -> Dict:
        print("🟢 NODE: run_simulation")
        scenic_code = state.get("current_scenic_code", "")
        video_path = run_simulation_in_carla_and_save_video(scenic_code)

        return {"simulation_video_path": video_path}

    def evaluate_with_vlm(self, state: ScenarioWorkflowState) -> Dict:
        print("🟢 NODE: evaluate_with_vlm")
        video_path = state.get("simulation_video_path", "")
        evaluation_score, evaluation_feedback = evaluate_with_vlm_func(video_path, state.get("generation_count", 0))

        updates = {
            "evaluation_score": evaluation_score,
            "evaluation_feedback": evaluation_feedback
        }

        # Safe dictionary get (Fix #4)
        if evaluation_score > state.get("best_score", -1.0):
            updates["best_score"] = evaluation_score
            updates["best_scenic_code"] = state.get("current_scenic_code", "")

        return updates

    def interpret(self, state: ScenarioWorkflowState) -> Dict:
        print("🟢 NODE: interpret")
        # Grab either human feedback or VLM feedback
        feedback = state.get("user_modification") or state.get("evaluation_feedback")
        scenario_dsl = generate_dsl(feedback)

        return {"scenario_dsl": scenario_dsl}

    def adapt_code(self, state: ScenarioWorkflowState) -> Dict:
        print("🟢 NODE: adapt_code")
        scenario_dsl = state.get("scenario_dsl", "")
        scenic_code = state.get("current_scenic_code", "")
        generation_count = state.get("generation_count", 0)
        
        adapted_scenic_code = adapt_code_func(scenario_dsl, scenic_code)

        return {
            "current_scenic_code": adapted_scenic_code,
            "generation_count": generation_count + 1
        }

    def human_review(self, state: ScenarioWorkflowState) -> Dict:
        print("🟢 NODE: human_review (Processing User UI Input)")
        user_satisfied = state.get("user_satisfied", False)
        
        if user_satisfied:
            return {}
        else:
            # If rejected, reset count to 0 so the system gets 3 fresh tries! (Fix #5)
            user_modification = state.get("user_modification", {"text": "Make it better."})
            return {
                "user_modification": user_modification,
                "generation_count": 0 
            }

    # ==========================================
    # DUMMY FUNCTIONS (For testing the loop)
    # ==========================================

# ==========================================
# DUMMY FUNCTIONS (For testing the loop)
# ==========================================
def generate_dsl(query: Any) -> str:
    return '{"intent": "dummy_dsl_parameters"}'

def embed_query_func(dsl: str) -> List[float]:
    return [0.1, 0.2, 0.3]

def retrieve_base_scenario_func(query_embedding: List[float]) -> str:
    return "scenario_123"

def find_scenic_code_with_scenario_id(scenario_id: str) -> str:
    return "def scenic_scenario(): pass"

def run_simulation_in_carla_and_save_video(dsl: str) -> str:
    return "/tmp/fake_video.mp4"

def evaluate_with_vlm_func(video_path: str, count: int) -> Tuple[float, str]:
    # Dummy logic: First try scores 75, second try scores 95!
    if count == 0:
        return 75.0, "The car is too slow."
    return 95.0, "Looks perfect!"

def adapt_code_func(scenario_dsl: str, scenic_code: str) -> str:
    return "# Modified code based on feedback"


# ==========================================
# TEST RUNNER (Run this file directly!)
# ==========================================
if __name__ == "__main__":
    workflow = ScenarioWorkflow()
    
    initial_state = {
        "user_query": {"text": "I want a highway scenario"},
        "max_count": 3
    }
    
    config = {"configurable": {"thread_id": "test_1"}}
    
    print("\n🚀 STARTING INITIAL WORKFLOW RUN...")
    for event in workflow.app.stream(initial_state, config=config):
        pass
        
    print("\n🛑 GRAPH PAUSED FOR HUMAN REVIEW.")
    print("Pretending user clicked 'Reject' and asked for it to rain...")
    
    # Inject user feedback
    workflow.app.update_state(config, {
        "user_satisfied": False,
        "user_modification": {"text": "Make it rain heavily"}
    })
    
    print("\n🚀 RESUMING WORKFLOW WITH HUMAN FEEDBACK...")
    for event in workflow.app.stream(None, config=config):
        pass
        
    print("\n🛑 GRAPH PAUSED FOR HUMAN REVIEW AGAIN.")
    print("Pretending user clicked 'Accept'...")
    
    workflow.app.update_state(config, {"user_satisfied": True})
    
    print("\n🚀 RESUMING WORKFLOW TO FINISH...")
    for event in workflow.app.stream(None, config=config):
        pass
        
    print("\n✅ WORKFLOW COMPLETED SUCCESSFULLY.")