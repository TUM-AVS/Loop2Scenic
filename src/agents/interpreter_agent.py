from typing import Any, Dict, Tuple
import logging

from src.services import BaseVLMModel
from .base_agent import BaseAgent
from src.prompt import load_prompt
from src.schema import MultimodalQuery

logger = logging.getLogger(__name__)

class InterpreterAgent(BaseAgent):
    def __init__(self, vlm_service: BaseVLMModel):
        super().__init__()
        self.prompt_template = load_prompt("describe_in_layer_model")
        self.vlm_service = vlm_service

    def process(self, state: dict) -> dict:
        return state

    def generate_dsl(self, user_query: MultimodalQuery) -> Tuple[Dict[str, Any], str]:
        """
        Generate a DSL (Domain-Specific Language) in json format from the natural language user query.
        """

        # 1. Get the content from the user query
        description = user_query.text
        image = user_query.image_path
        video = user_query.video_path
        
        # Validate that at least one content type is provided
        if not description and not image and not video:
            raise ValueError("At least one of 'description', 'image', or 'video' must be provided")

        # 2. Prepare the prompt
        formatted_prompt = self.prompt_template.format(description_text=description or "", scenic_code="")

        # 3. Call the VLM service and process response
        response = self.vlm_service.chat(text=formatted_prompt, image=image, video=video)
        json_response = self._clean_and_parse_json(response) # clean the response and parse it as a JSON object

        # 4. Flatten the DSL into text
        flattened_text = self._flatten_dsl(json_response)

        if json_response and flattened_text:
            return json_response, flattened_text
        else:
            logger.error("Failed to parse JSON")
            return None, None

    def _flatten_dsl(self, dsl: Dict[str, Any]) -> str:
        if not dsl:
            return None
        try:    
            text_parts = []
            text_parts.append(f"Scenario: {dsl.get('Scenario', '')}")
            text_parts.append(f"The ego vehicle is {dsl.get('Ego', '')}")
            
            adversarials = dsl.get('Adversarials', [])
            if adversarials:
                text_parts.append(f"Adversarial objects: {' '.join(adversarials)}")
            else:
                text_parts.append("There are no adversarials.")
                
            text_parts.append(f"Spatial Relation: {dsl.get('Spatial Relation', '')}")
            
            reqs = dsl.get('Requirement and restrictions', '')
            if reqs:
                text_parts.append(f"Requirements and restrictions: {reqs}")
                
            flattened_text = " ".join(text_parts)
                
            logger.info(f"Successfully flattened DSL into text: {flattened_text}")
            return flattened_text
        except Exception as e:
            logger.error(f"Failed to flatten DSL: {e}")
            return None