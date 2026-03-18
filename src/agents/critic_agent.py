import logging
from typing import Tuple

from src.schema import MultimodalQuery, ScenarioDocument
from src.services.vlm import GeminiVLModel

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

    def evaluate_with_vlm(self, query: MultimodalQuery, scenario: ScenarioDocument) -> Tuple[float, str, dict]:
        system_instruction = self.prompt_template
        
        # 1. Start with an empty list
        contents = []
        
        # Optional: If your wrapper expects the system instruction inside the contents array, 
        # uncomment the line below. Otherwise, passing it as the second argument is usually correct.
        # contents.append(system_instruction)

        # 2. Build Target Requirement
        requirement_text = "=== TARGET REQUIREMENT ===\n"
        if query.text:
            requirement_text += f"Text Description: {query.text}\n"
        contents.append(requirement_text)

        # Append media ONLY if it exists. 
        # NOTE: Replace `self.load_media()` with however your system actually loads files!
        if query.image_path:
            contents.append("Requirement Image:")
            contents.append(self.vlm_service.load_media(query.image_path)) # Must be a file object, not a string
            
        if query.video_path:
            contents.append("Requirement Video:")
            contents.append(self.vlm_service.load_media(query.video_path)) # Must be a file object, not a string

        # 3. Build Generated Scenario
        scenario_text = "\n=== GENERATED SCENARIO ===\n"
        if scenario.description:
            scenario_text += f"Text Description: {scenario.description}\n"
        contents.append(scenario_text)

        if scenario.image_path:
            contents.append("Scenario Image:")
            contents.append(self.vlm_service.load_media(scenario.image_path)) # Must be a file object
            
        if scenario.video_path:
            contents.append("Scenario Video:")
            contents.append(self.vlm_service.load_media(scenario.video_path)) # Must be a file object

        # 4. Add Output Instructions
        output_instructions = """
        === OUTPUT FORMAT ===
        You must respond ONLY with a valid JSON object. Do not include markdown formatting like ```json. Use the following structure:
        {
            "evaluation": {
                "scenario_match": true/false,
                "ego_behavior_match": true/false,
                "adversarials_match": true/false,
                "spatial_relation_match": true/false,
                "restrictions_match": true/false
            },
            "feedback": "<Provide a concise, natural language explanation detailing exactly what matched, what failed, and how to modify the generated scenario to fix the failures.>",
            "score": <number>
        }
        """
        contents.append(output_instructions)

        # 5. Send to VLM Service
        # Your wrapper handles the actual API call
        response = self.vlm_service.chat_with_content(contents, system_instruction)
        json_response = self._clean_and_parse_json(response)
        if json_response:
            score = json_response.get("score", 0)
            feedback = json_response.get("feedback", "")
            evaluation = json_response.get("evaluation", {})
            return score, feedback, evaluation
        else:
            logger.error("Failed to parse JSON")
            return 0, None, None

if __name__ == "__main__":
    critic_agent = CriticAgent(vlm_service=GeminiVLModel(model="gemini-2.5-flash"))
    # query
    query = MultimodalQuery(
            text="Generate me a highway scenario looks like the one in this video", 
            image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png", 
            video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4")
    
    # matched scenario
    matched_scenario = ScenarioDocument(
            scenario_id="2",
            description="The ego-vehicle is performing an unprotected left turn at an intersection, yielding to oncoming traffic. This scenario occurs at both signalized and non-signalized junctions.",  
            video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/scenarios/CARLA_Leaderboard_2/video.mp4")

    # unmatched scenario
    unmatched_scenario = ScenarioDocument(
            scenario_id="1",
            description="The ego-vehicle loses control due to bad conditions on the road and it must recover, coming back to its original lane.",  
            video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/scenarios/CARLA_Leaderboard_1/video.mp4")

    score, feedback_query, evaluation = critic_agent.evaluate_with_vlm(query, matched_scenario)
    # score, feedback_query, evaluation = critic_agent.evaluate_with_vlm(query, unmatched_scenario)
    print(score, feedback_query, evaluation)