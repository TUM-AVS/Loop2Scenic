from pydantic import BaseModel
from typing import Optional

class MultimodalQuery(BaseModel):
    """
    The multimodal query schema for the MRAG system.
    """
    text: Optional[str] = None
    image_path: Optional[str] = None # path to the image file
    video_path: Optional[str] = None # path to the video file