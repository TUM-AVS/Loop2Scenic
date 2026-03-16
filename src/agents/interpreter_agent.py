from src.services import BaseLLMModel
from .base_agent import BaseAgent
from src.prompt import load_prompt
import logging

logger = logging.getLogger(__name__)

class InterpreterAgent(BaseAgent):
    def __init__(self, llm_service: BaseLLMModel):
        super().__init__()
        self.prompt_template = load_prompt("describe_in_layer_model")
        self.llm_service = llm_service

    def process(self, state: dict) -> dict:
        return state

    def generate_dsl(self, user_query: str) -> str:
        """
        Generate a DSL (Domain-Specific Language) in json format from the natural language user query.
        """
        formatted_prompt = self.prompt_template.format(user_query=user_query)
        response = self.llm_service.chat(formatted_prompt)
        json_response = self._clean_and_parse_json(response) # clean the response and parse it as a JSON object
        if json_response:
            return json_response
        else:
            logger.error("Failed to parse JSON")
            return None