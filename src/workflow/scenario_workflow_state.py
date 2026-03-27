from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from src.schema import HeaderSetting, MultimodalQuery, ScenicScenario

class ScenarioWorkflowState(TypedDict):
    """
    The central state dictionary (the "clipboard") that gets passed 
    between every node in the LangGraph workflow.
    """
    
    # --- 1. Conversation & Input ---
    # `add_messages` ensures new messages are appended, not overwritten
    messages: Annotated[list, add_messages] 
    
    # Multimodal query: text, image_path, video_path
    user_query: MultimodalQuery
    
    # --- 2. Understanding & Retrieval ---
    # The structured parameters extracted from the user's query/feedback
    header_settings: HeaderSetting | None
    scenario_dsl: Dict[str, Any]
    
    # The numerical vector used to search Milvus
    query_embedding: List[float]
    
    # The ID of the closest match found in the vector database
    base_scenario_id: str
    
    # --- 3. The Working Canvas ---
    # The code actively being tested or adapted in the current loop
    scenic_scenarios_list: List[ScenicScenario]
    current_scenic_scenario: ScenicScenario
    
    # --- 5. Best Result Tracking ---
    # Tracks the highest-scoring version in case a later adaptation breaks the code
    best_scenario: ScenicScenario
    
    # --- 6. Human-in-the-Loop ---
    # Optional fields, populated only when the user reviews the scenario
    user_satisfied: Optional[bool]
    user_modification: Optional[MultimodalQuery] # the same as user_query, can contain text, image_path, video_path
    
    # --- 7. Loop Control ---
    # Current loop iteration (starts at 0)
    generation_count: int

MAX_COUNT = 3

CLEAN_STATE: ScenarioWorkflowState = {
    "messages": [],
    "user_query": None,
    "scenario_dsl": None,
    "header_settings": None,
    "query_embedding": None,
    "base_scenario_id": None,
    "scenic_scenarios_list": [],
    "current_scenic_scenario": None,
    "best_scenario": None,
    "user_satisfied": None,
    "user_modification": None,
    "generation_count": 0,
}