import logging

from .base_agent import BaseAgent
from src.services import BaseLLMModel
from src.prompt import load_prompt
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ScenicCoderAgent(BaseAgent):
    def __init__(self, llm_service: BaseLLMModel):
        super().__init__()
        self.llm_service = llm_service
        self.prompt_template = load_prompt("adapt_code")

    def process(self, state: dict) -> dict:
        return state

    def adapt_code(self, original_scenic_code: str, aim_dsl: Dict[str, Any]) -> Dict[str, Any]:
        formatted_prompt = self.prompt_template.format(original_scenic_code=original_scenic_code, aim_dsl=aim_dsl)
        response = self.llm_service.chat(formatted_prompt)
        json_response = self._clean_and_parse_json(response)
        if json_response:
            return json_response
        else:
            logger.error("Failed to parse JSON")
            return None