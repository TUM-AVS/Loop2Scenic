from pydantic import BaseModel
from typing import Any, Dict, Optional


class ScenicScenario(BaseModel):
    """
    A scenario represented by its ID, scenic code, and score. Used for storing vlm evaluation results. Can be used for later comparison stage.
    """
    scenario_id: str
    description: Optional[str] = None
    scenic_code: str
    error: Optional[str] = None # error message from the simulation recorder
    score: Optional[float] = None
    evaluation_feedback: Optional[str] = None
    evaluation_result: Optional[Dict[str, Any]] = None