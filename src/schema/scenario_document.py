from pydantic import BaseModel
from typing import Optional

class ScenarioDocument(BaseModel):
    """
    The strict data structure for a Multimodal Scenario in our MRAG system.
    This completely replaces LangChain's default Document.
    """
    # 1. Identifiers
    scenario_id: str
    
    # 2. Text Modality
    description: Optional[str] = None
    
    # 3. Code Modality
    scenic_code: Optional[str] = None
    
    # 4. Visual Modality (Path to files)
    video_path: Optional[str] = None
    image_path: Optional[str] = None