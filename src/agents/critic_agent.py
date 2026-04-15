import logging
from typing import Any, Dict, Tuple

from google.genai import types

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

    def evaluate_with_vlm(self, query: MultimodalQuery, scenario: ScenarioDocument) -> Tuple[float, dict, dict]:
        contents = []
        
        # 1. Build Target Requirement
        contents.append(types.Part.from_text(text="** Inputs **\nTarget Requirement:\n"))
        
        if query.text:
            contents.append(types.Part.from_text(text=f"Text Description: {query.text}\n"))

        if query.image_path:
            contents.append(types.Part.from_text(text=f"Requirement Image:\n"))
            img_file = self.vlm_service.load_media(query.image_path)
            contents.append(types.Part.from_uri(file_uri=img_file.uri, mime_type=img_file.mime_type))

        if query.video_path:
            contents.append(types.Part.from_text(text="Requirement Video:\n"))
            req_file = self.vlm_service.load_media(query.video_path)
            contents.append(types.Part.from_uri(file_uri=req_file.uri, mime_type=req_file.mime_type))

        # 2. Build Generated Scenario
        contents.append(types.Part.from_text(text="\nGenerated Scenario:\n"))
        
        if scenario.description:
            contents.append(types.Part.from_text(text=f"Text Description: {scenario.description}\n"))
            
        if scenario.video_path:
            contents.append(types.Part.from_text(text="Scenario Video:\n"))
            scen_file = self.vlm_service.load_media(scenario.video_path)
            contents.append(types.Part.from_uri(file_uri=scen_file.uri, mime_type=scen_file.mime_type))

        # 3. THE RECENCY HOOK (Crucial for adherence)
        # Always end the multimodal array with a text instruction reminding it of the goal.
        final_reminder = (
            "\nBased on the videos and descriptions provided above, please execute your reasoning "
            "and output the final JSON evaluation object exactly as requested in the system instructions."
        )
        contents.append(types.Part.from_text(text=final_reminder))

        logger.info(f"🔍 Evaluate with VLM prompt is: {contents}, prompt template is: {self.prompt_template}")

        # 5. Send to VLM Service
        response = self.vlm_service.chat_with_content(contents=contents, system_instruction=self.prompt_template)
        json_response = self._clean_and_parse_json(response)
        if json_response:
            score = json_response.get("score", 0)
            feedback = json_response.get("feedback", {})
            evaluation = json_response.get("evaluation", {})
            return score, feedback, evaluation
        else:
            logger.error("Failed to parse JSON")
            return 0, {}, {}

if __name__ == "__main__":
    critic_agent = CriticAgent(vlm_service=GeminiVLModel(model="gemini-2.5-flash"))
    # query
    query = MultimodalQuery(
            text="The ego-vehicle is performing an unprotected left turn at an intersection, yielding to oncoming traffic. This scenario occurs at both signalized and non-signalized junctions.", 
            image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png")
    
    # matched scenario
    matched_scenario = ScenarioDocument(
            scenario_id="2",
            description="The ego-vehicle is performing an unprotected left turn at an intersection, yielding to oncoming traffic. This scenario occurs at both signalized and non-signalized junctions.",  
            video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/scenarios/CARLA_Leaderboard_2/video.mp4")

    # unmatched scenario
    unmatched_scenario = ScenarioDocument(
            scenario_id="1",
            description="",  
            video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/scenarios/CARLA_Leaderboard_1/video.mp4")

    # score, feedback_query, evaluation = critic_agent.evaluate_with_vlm(query, matched_scenario)
    score, feedback, evaluation = critic_agent.evaluate_with_vlm(query, unmatched_scenario)
    print(score, feedback, evaluation)
