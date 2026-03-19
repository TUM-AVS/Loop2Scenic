from typing import Any, Dict, Optional, Tuple
import logging

from src.services import BaseLLMModel, BaseVLMModel
from src.services.vlm import GeminiVLModel
from .base_agent import BaseAgent
from src.prompt import load_prompt
from src.schema import MultimodalQuery, ScenarioDocument

logger = logging.getLogger(__name__)

class InterpreterAgent(BaseAgent):
    def __init__(self, vlm_service: BaseVLMModel):
        super().__init__()
        self.prompt_template = load_prompt("describe_in_layer_model")
        self.vlm_service = vlm_service

    def process(self, state: dict) -> dict:
        return state

    def generate_dsl_from_user_query(self, user_query: MultimodalQuery) -> Tuple[Dict[str, Any], str]:
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

    def generate_dsl_from_user_feedback(
        self, 
        user_feedback: MultimodalQuery, 
        dsl_to_modify: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a DSL (Domain-Specific Language) in json format from the user feedback and original scenario.
        """
        contents = []
        static_promt = load_prompt("modify_dsl_from_user_feedback").format(scenario_dsl=dsl_to_modify)
        contents.append(static_promt)

        if user_feedback.text:
            contents.append(f"User suggestion text: {user_feedback.text}")
        if user_feedback.image_path:
            contents.append("User suggestion image:")
            contents.append(self.vlm_service.load_media(user_feedback.image_path))
        if user_feedback.video_path:
            contents.append("User suggestion video:")
            contents.append(self.vlm_service.load_media(user_feedback.video_path))

        output_instructions = load_prompt("output_layer_model_format")
        contents.append(output_instructions)

        response = self.vlm_service.chat_with_content(contents)
        json_response = self._clean_and_parse_json(response)
        if json_response:
            return json_response
        else:
            logger.error("Failed to parse JSON")
            return None

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

if __name__ == "__main__":
    interpreter = InterpreterAgent(vlm_service=GeminiVLModel(model="gemini-2.5-flash"))

    # 1. generate original dsl
    user_query = MultimodalQuery(
        text="Please generate me a scenario like this.", 
        image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png", 
        video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4"
    )
    dsl, flattened_text = interpreter.generate_dsl_from_user_query(user_query)
    print(f"Successfully generated DSL: {dsl}")

    """
    result = {
        'Scenario': 'Ego vehicle approaches an intersection, waits for a traffic light, and proceeds straight after another car makes a right turn.', 
        'Ego': 'A car approaches an intersection, stops at a red light, and then drives straight through the intersection when the light turns green.', 
        'Adversarials': 
            ['A car approaches an intersection from the left and makes a right turn.'], 
        'Spatial Relation': 'The ego vehicle and an adversarial vehicle are positioned on different incoming lanes at a four-way intersection.', 
        'Requirement and restrictions': "The ego vehicle and the adversarial vehicle are initially a certain distance from the intersection. The ego vehicle's traffic light is initially red and then turns green. The scenario terminates when the ego vehicle has cleared the intersection."
        }
    """

    # 2. generate modified dsl
    user_feedback = MultimodalQuery(
        text="Please add another car in the scenario which turns left at the intersection behind the ego vehicle as I marked with a red box in the image.", 
        image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png", 
        video_path=None
    )
    modified_dsl = interpreter.generate_dsl_from_user_feedback(user_feedback, dsl)
    print(f"Successfully generated modified DSL: {modified_dsl}")

    """
    result:
    {
        'Scenario': 'Ego vehicle approaches an intersection, waits for a traffic light, and proceeds straight after another car makes a right turn, while a third car turns left from behind the ego.', 
        'Ego': 'A car approaches an intersection, stops at a red light, and then drives straight through the intersection when the light turns green.', 
        'Adversarials': 
            ['A car approaches an intersection from the left and makes a right turn.', 
             'A car approaches the intersection from behind the ego vehicle and makes a left turn.'], 
        'Spatial Relation': 'The ego vehicle and two adversarial vehicles are positioned at a four-way intersection; one adversarial car approaches from the left, and another is positioned behind the ego vehicle on the same lane.', 
        'Requirement and restrictions': "The ego vehicle and the adversarial vehicles are initially a certain distance from the intersection. The ego vehicle's traffic light is initially red and then turns green. The scenario terminates when the ego vehicle has cleared the intersection."
    }
    """