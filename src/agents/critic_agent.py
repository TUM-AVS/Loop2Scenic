import json
import logging
from typing import Any, Dict, Optional

from google.genai import types

from src.schema import MultimodalQuery, ScenarioDocument
from src.services.vlm import GeminiVLModel

from .base_agent import BaseAgent
from src.services import BaseVLMModel
from src.prompt import load_prompt
from src.utils import to_safe_string

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    def __init__(self, vlm_service: BaseVLMModel):
        super().__init__()
        self.vlm_service = vlm_service
        self.prompt_template = load_prompt("evaluate_with_vlm")

    def process(self, state: dict) -> dict:
        return state

    def evaluate_with_vlm(
        self,
        query: MultimodalQuery,
        scenario: ScenarioDocument,
        scenario_dsl: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the 3-stage critic and return the full JSON payload:

        observed_dsl, evaluation, kpi_matches, kpi_passed, feedback, score
        """
        contents = []

        # 1. Target DSL is the requirement source (preferred).
        contents.append(types.Part.from_text(text="** Inputs **\nTarget DSL:\n"))
        if scenario_dsl:
            try:
                dsl_text = json.dumps(scenario_dsl, indent=2, ensure_ascii=False)
            except TypeError:
                dsl_text = to_safe_string(scenario_dsl)
            contents.append(types.Part.from_text(text=dsl_text + "\n"))
        else:
            logger.warning(
                "No scenario_dsl provided to critic; falling back to query modalities only"
            )
            contents.append(
                types.Part.from_text(
                    text="(No Target DSL was provided. Use the following query as a weak fallback.)\n"
                )
            )
            if query.text:
                contents.append(types.Part.from_text(text=f"Text Description: {query.text}\n"))
            if query.image_path:
                contents.append(types.Part.from_text(text="Requirement Image:\n"))
                img_file = self.vlm_service.load_media(query.image_path)
                contents.append(
                    types.Part.from_uri(file_uri=img_file.uri, mime_type=img_file.mime_type)
                )
            if query.video_path:
                contents.append(types.Part.from_text(text="Requirement Video:\n"))
                req_file = self.vlm_service.load_media(query.video_path)
                contents.append(
                    types.Part.from_uri(file_uri=req_file.uri, mime_type=req_file.mime_type)
                )

        # 2. Generated Scenario (video only).
        contents.append(types.Part.from_text(text="\n***Generated Scenario:***\n"))
        if scenario.video_path:
            contents.append(types.Part.from_text(text="***Generated Scenario BEV Video:***\n"))
            scen_file = self.vlm_service.load_media(scenario.video_path)
            contents.append(
                types.Part.from_uri(file_uri=scen_file.uri, mime_type=scen_file.mime_type)
            )

        # 3. Recency reminder
        final_reminder = (
            "\nExecute the three stages in order: "
            "(1) Perception — build Observed DSL from the Generated Scenario BEV video only; "
            "(2) Verification — compare Observed DSL to Target DSL with per-field booleans "
            "(list lengths must match Target DSL); "
            "(3) Scoring — aggregate to 7 KPIs and final score. "
            "Output the final JSON object exactly as requested in the system instructions."
        )
        contents.append(types.Part.from_text(text=final_reminder))

        response = self.vlm_service.chat_with_content(
            contents=contents, system_instruction=self.prompt_template
        )
        json_response = self._clean_and_parse_json(response)
        if json_response and isinstance(json_response, dict):
            return json_response

        logger.error("Failed to parse JSON")
        return {}


if __name__ == "__main__":
    critic_agent = CriticAgent(vlm_service=GeminiVLModel(model="gemini-2.5-flash"))
    query = MultimodalQuery(
        text=(
            "The ego-vehicle is performing an unprotected left turn at an intersection, "
            "yielding to oncoming traffic. This scenario occurs at both signalized and "
            "non-signalized junctions."
        ),
        image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png",
    )

    matched_scenario = ScenarioDocument(
        scenario_id="2",
        description=(
            "The ego-vehicle is performing an unprotected left turn at an intersection, "
            "yielding to oncoming traffic. This scenario occurs at both signalized and "
            "non-signalized junctions."
        ),
        video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/scenarios/CARLA_Leaderboard_2/video.mp4",
    )

    result = critic_agent.evaluate_with_vlm(query, matched_scenario)
    print(result)
