from pydantic import BaseModel
from typing import Any, Dict, Optional


class ScenicScenario(BaseModel):
    """
    A scenario represented by its ID, scenic code, and score. Used for storing vlm evaluation results. Can be used for later comparison stage.
    """
    scenario_id: str
    description: Optional[str] = None
    scenic_code: str
    error: Optional[str] = None  # error message from the simulation recorder
    score: Optional[float] = None
    # Feedback sub-object from the critic (event_verification / per_item_notes), or legacy string.
    evaluation_feedback: Optional[Any] = None
    # Full critic payload: observed_dsl + evaluation + kpi_matches + kpi_passed + feedback + score.
    # Legacy runs may still store only the flat comparison dict here.
    evaluation_result: Optional[Dict[str, Any]] = None
