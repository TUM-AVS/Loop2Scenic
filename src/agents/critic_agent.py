import logging
from typing import Tuple

from .base_agent import BaseAgent
from src.services import BaseVLMModel
from src.prompt import load_prompt

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    def __init__(self, vlm_service: BaseVLMModel):
        super().__init__()
        self.vlm_service = vlm_service
        self.prompt_template = load_prompt("evaluate_with_vlm")

    def process(self, state: dict) -> dict:
        return state

    def evaluate_with_vlm(self, video_path: str, original_query: str) -> Tuple[float, str]:
        formatted_prompt = self.prompt_template.format(original_query=original_query)
        response = self.vlm_service.chat(text=formatted_prompt, video=video_path)
        json_response = self._clean_and_parse_json(response)
        if json_response:
            score = json_response.get("score", 0)
            feedback = json_response.get("feedback", "")
            return score, feedback
        else:
            logger.error("Failed to parse JSON")
            return 0, "Failed to parse JSON"