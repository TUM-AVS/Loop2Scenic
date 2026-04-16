
from typing import Any, Dict, Optional, Tuple
import logging

from google.genai import types

from src.schema.dsl import DSL
from src.services import BaseLLMModel, BaseVLMModel
from src.services.vlm import GeminiVLModel
from src.utils import clean_and_parse_json, to_safe_string
from .base_agent import BaseAgent
from src.prompt import load_prompt
from src.schema import MultimodalQuery, ScenarioDocument, ScenicScenario, HeaderSetting

logger = logging.getLogger(__name__)

class InterpreterAgent(BaseAgent):
    def __init__(self, vlm_service: BaseVLMModel):
        super().__init__()
        self.prompt_template = load_prompt("describe_in_layer_model")
        self.vlm_service = vlm_service

    def process(self, state: dict) -> dict:
        return state

    def generate_header_settings(self, user_query: MultimodalQuery) -> HeaderSetting | None:
        """
        Generate the header settings for the scenario based on the user query.
        Return None if the confidence is less than 0.5.
        """
        header_settings_detection_prompt = load_prompt("header_settings_detection")

        # 1. compose user query
        user_query_text = f"{user_query.text}\n"
        if user_query.image_path:
            user_query_text += f"User query image: {self.vlm_service.load_media(user_query.image_path)}\n"
        if user_query.video_path:
            user_query_text += f"User query video: {self.vlm_service.load_media(user_query.video_path)}\n"

        # 2. call the VLM service
        header_settings_detection_prompt = header_settings_detection_prompt.format(user_query=user_query_text)
        response = self.vlm_service.chat(text=header_settings_detection_prompt)

        # 3. parse the response
        json_response = self._clean_and_parse_json(response)
        logger.info(f"JSON response from header settings detection: {json_response}")
        """ json response example:
        {{
            "weather": "SoftRainNoon",
            "map_type": "rural",
            "suggested_map": "Town07",
            "time_of_day": "noon",
            "blueprint": "vehicle.tesla.model3",
            "confidence": 0.9,
            "reasoning": "Query mentions 'rainy weather', 'rural area' and 'Tesla' at noon"
        }}
        """
        if json_response:
            if json_response.get("confidence", 0.0) < 0.5:
                logger.warning("Low confidence in header settings detection, returning None to not to change the header")
                return None
            else: 
                suggested_map = json_response.get("suggested_map")
                if not suggested_map or suggested_map is None:
                    suggested_map = "Town05"
                weather = json_response.get("weather")
                if not weather or weather is None:
                    weather = "ClearNoon"
                blueprint = json_response.get("blueprint")
                if not blueprint or blueprint is None:
                    blueprint = "vehicle.lincoln.mkz_2017"
                header_settings = HeaderSetting(
                    carla_map=suggested_map,
                    map_file_path=f"../../maps/{suggested_map}.xodr",
                    weather=weather,
                    blueprint=blueprint,
                )
                return header_settings
        else:
            logger.error("Failed to parse JSON, returning None")
            return None

    def generate_dsl_from_user_query(self, user_query: MultimodalQuery) -> Tuple[DSL | None, str | None]:
        """
        Generate a DSL (Domain-Specific Language) in json format from the natural language user query.
        """

        # 1. Get the content from the user query
        description = user_query.text or ""
        image = user_query.image_path
        video = user_query.video_path
        
        # Validate that at least one content type is provided
        if not description and not image and not video:
            raise ValueError("At least one of 'description', 'image', or 'video' must be provided")

        # 2. Prepare the prompt
        system_instruction = load_prompt("describe_in_layer_model")
        contents = []
        contents.append(types.Part.from_text(text=f"** Inputs **"))
        if description:
            contents.append(types.Part.from_text(text=f"Scenario Description Text: {description}"))
        if image:
            contents.append(types.Part.from_text(text=f"Scenario Image: "))
            image_file = self.vlm_service.load_media(image)
            contents.append(types.Part.from_uri(file_uri=image_file.uri, mime_type=image_file.mime_type))
        if video:
            contents.append(types.Part.from_text(text=f"Scenario Video: "))
            video_file = self.vlm_service.load_media(video)
            contents.append(types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type))

        # 3. Call the VLM service and process response
        response = self.vlm_service.chat_with_content(contents=contents, system_instruction=system_instruction)
        json_response = clean_and_parse_json(response) # clean the response and parse it as a JSON object
        dsl = DSL(**json_response) if json_response else None

        # 4. Flatten the DSL into text
        flattened_text = self._flatten_dsl(dsl)

        if dsl and flattened_text:
            return dsl, flattened_text
        else:
            logger.error("Failed to parse JSON")
            return None, None

    def generate_dsl_from_user_feedback(
        self, 
        user_feedback: MultimodalQuery, 
        original_dsl: Optional[DSL] | None,
        scenario_to_modify: ScenicScenario | None
    ) -> Optional[DSL] | None:
        """
        Generate a DSL (Domain-Specific Language) in json format from the user feedback and the original or previous best scenario.
        """
        try:
            contents = []
            system_instruction = load_prompt("modify_dsl_from_user_feedback")
            contents.append(types.Part.from_text(text=f"\n ** Inputs ** \n"))

            # 1. deal with the original dsl and the scenario to modify
            if original_dsl is not None:
                contents.append(types.Part.from_text(text=f"** Original Scenario DSL ** \n"))
                contents.append(types.Part.from_text(text=to_safe_string(original_dsl)))
                contents.append(types.Part.from_text(text=f"\n"))

            if isinstance(scenario_to_modify, ScenicScenario):
                contents.append(types.Part.from_text(text=f"** Original Scenario Video from Bird Eye View ** \n"))
                video_path = f"temp/{scenario_to_modify.scenario_id}/video/BEV.mp4"
                video_file = self.vlm_service.load_media(video_path)
                contents.append(types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type))

            # 2. deal with the user feedback
            contents.append(types.Part.from_text(text=f"** User Modification Suggestion ** \n"))
            if user_feedback.text:
                contents.append(types.Part.from_text(text=f"User suggestion text: {user_feedback.text}"))
            if user_feedback.image_path:
                contents.append(types.Part.from_text(text=f"User suggestion image:"))
                image_file = self.vlm_service.load_media(user_feedback.image_path)
                contents.append(types.Part.from_uri(file_uri=image_file.uri, mime_type=image_file.mime_type))
            if user_feedback.video_path:
                contents.append(types.Part.from_text(text=f"User suggestion video:"))
                video_file = self.vlm_service.load_media(user_feedback.video_path)
                contents.append(types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type))

            response = self.vlm_service.chat_with_content(contents=contents, system_instruction=system_instruction)
            json_response = clean_and_parse_json(response)
            if json_response:
                return json_response
            else:
                logger.error("Failed to parse JSON, returning None")
                return None
        except Exception as e:
            logger.error(f"Failed to generate DSL from user feedback: {e}")
            return None

    def _flatten_dsl(self, dsl: DSL) -> str:
        if not dsl:
            return None
        try:    
            text_parts = []
            text_parts.append(f"Scenario: {dsl['scenario']}")
            text_parts.append(f"The ego vehicle is {dsl['ego']}")
            
            adversarials = dsl['adversarials']
            if adversarials:
                text_parts.append(f"Adversarial objects: {' '.join(map(str, adversarials))}")
            else:
                text_parts.append("There are no adversarials.")
                
            text_parts.append(f"Spatial Relation: {dsl['spatial_relation']}")
            
            reqs = dsl['requirements_and_restrictions']
            if reqs:
                text_parts.append(f"Requirements and restrictions: {reqs}")

            # road_side_structures = dsl['road_side_structures']
            # if road_side_structures:
            #     text_parts.append(f"Road side structures: {', '.join(map(str, road_side_structures))}")
            # else:
            #     text_parts.append("There are no road side structures.")

            # temporary_modifications = dsl['temporary_modifications']
            # if temporary_modifications:
            #     text_parts.append(f"Temporary modifications: {', '.join(map(str, temporary_modifications))}")
            # else:
            #     text_parts.append("There are no temporary modifications.")
                
            flattened_text = " ".join(text_parts)
                
            logger.info(f"Successfully flattened DSL into text: {flattened_text}")
            return flattened_text
        except Exception as e:
            logger.error(f"Failed to flatten DSL: {e}")
            return None

if __name__ == "__main__":
    interpreter = InterpreterAgent(vlm_service=GeminiVLModel(model="gemini-3-flash-preview"))

    def test_dsl_generation():
        # 1. generate original dsl
        user_query = MultimodalQuery(
            text="Please generate me a scenario like shown in the video, additionally, on the intersection, please add a kiosk on the right sidewalk of the ego vehicle and a street barrier in the center of the ego vehicle's lane. The scenario video is a third person view of the ego vehicle.", 
            # image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png", 
            video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4"
        )
        dsl, flattened_text = interpreter.generate_dsl_from_user_query(user_query)
        print(f"Successfully generated DSL: {dsl}")

        """
        result = {
        'scenario': 'The ego vehicle encounters an adversarial vehicle turning right at an intersection and proceeds straight after it passes.', 
        'ego': 'A car that initially waits at an intersection, then proceeds straight through it.', 
        'adversarials': ['A NPCCar that approaches the intersection and executes a right turn.'], 
        'spatial_relation': "The ego vehicle is situated at an intersection, facing a straight road that passes under an overpass, while an adversarial vehicle approaches from an adjacent lane to the ego's left.", 
        'requirements_and_restrictions': 'The traffic lights are green for the ego vehicle to proceed.', 
        
        'road_side_structures': [{'object': 'Kiosk', 'position': 'Positioned on the right-hand sidewalk relative to the ego vehicle at the intersection.'}], 
        'temporary_modifications': [{'object': 'Street Barrier', 'position': "Placed in the center of the ego vehicle's lane at the intersection."}]}
        """

        # 2. generate modified dsl
        # user_feedback = MultimodalQuery(
        #     text="Please add another car in the scenario which turns left at the intersection behind the ego vehicle as I marked with a red box in the image.", 
        #     image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png", 
        #     video_path=None
        # )
        # modified_dsl = interpreter.generate_dsl_from_user_feedback(user_feedback, dsl)
        # print(f"Successfully generated modified DSL: {modified_dsl}")

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

    def test_header_settings_generation():
        user_query = MultimodalQuery(
            text="Please generate me a scenario like this, and it is cloudy.", 
            image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png", 
        )
        header_settings = interpreter.generate_header_settings(user_query)
        print(f"Successfully generated header settings: {header_settings}")

    def test_dsl_generation_from_user_feedback():
        # 1. generate original dsl
        # user_query = MultimodalQuery(
        #     text="Please generate me a scenario like shown in the video, additionally, on the intersection, please add a kiosk on the right sidewalk of the ego vehicle and a street barrier in the center of the ego vehicle's lane. The scenario video is a third person view of the ego vehicle.", 
        #     # image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png", 
        #     video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4"
        # )
        # dsl, flattened_text = interpreter.generate_dsl_from_user_query(user_query)
        # print(f"Successfully generated DSL: {dsl}")

        """
        {'scenario': 'The ego vehicle performs a left turn at a four-way intersection after yielding to an adversarial vehicle crossing from the left.', 
        'ego': 'A car that moves straight toward an intersection, pauses to allow another vehicle to pass, and then executes a left turn.', 
        'adversarials': ["A car that enters the intersection from the left and travels across the ego vehicle's path to the right."], 
        'spatial_relation': 'The ego vehicle and the adversarial vehicle approach a four-way intersection from perpendicular directions, with the ego on a multi-lane straight road.', 
        'requirements_and_restrictions': 'The ego vehicle faces a green traffic light upon arrival at the intersection. The scenario concludes once the ego vehicle completes its left turn and travels a certain distance on the new road.', 
        'road_side_structures': [{'object': 'Kiosk', 'position': 'Positioned on the sidewalk to the right of the ego vehicle near the intersection corner.'}], 
        'temporary_modifications': [{'object': 'Street Barrier', 'position': "Located in the middle of the ego vehicle's lane at the entrance to the intersection."}], 
        'reasoning_chain': "1. Actors: The ego vehicle is the primary car viewed from a third-person perspective; the adversarial is the black car entering from the left. 2. Chronological Analysis: In the initial phase, the ego approaches the intersection. At the midpoint, the adversarial car crosses the intersection from left to right, and the ego vehicle decelerates to a stop. In the final phase, the ego vehicle turns left into the crossroad. 3. Evidence: The ego vehicle's speed decreases to zero as the adversarial car's lateral trajectory intersects its forward path, followed by the ego vehicle following a 90-degree leftward arc."}
        """
        dsl = {'scenario': 'The ego vehicle performs a left turn at a four-way intersection after yielding to an adversarial vehicle crossing from the left.', 
        'ego': 'A car that moves straight toward an intersection, pauses to allow another vehicle to pass, and then executes a left turn.', 
        'adversarials': ["A car that enters the intersection from the left and travels across the ego vehicle's path to the right."], 
        'spatial_relation': 'The ego vehicle and the adversarial vehicle approach a four-way intersection from perpendicular directions, with the ego on a multi-lane straight road.', 
        'requirements_and_restrictions': 'The ego vehicle faces a green traffic light upon arrival at the intersection. The scenario concludes once the ego vehicle completes its left turn and travels a certain distance on the new road.', 
        'road_side_structures': [{'object': 'Kiosk', 'position': 'Positioned on the sidewalk to the right of the ego vehicle near the intersection corner.'}], 
        'temporary_modifications': [{'object': 'Street Barrier', 'position': "Located in the middle of the ego vehicle's lane at the entrance to the intersection."}], 
        'reasoning_chain': "1. Actors: The ego vehicle is the primary car viewed from a third-person perspective; the adversarial is the black car entering from the left. 2. Chronological Analysis: In the initial phase, the ego approaches the intersection. At the midpoint, the adversarial car crosses the intersection from left to right, and the ego vehicle decelerates to a stop. In the final phase, the ego vehicle turns left into the crossroad. 3. Evidence: The ego vehicle's speed decreases to zero as the adversarial car's lateral trajectory intersects its forward path, followed by the ego vehicle following a 90-degree leftward arc."}

        # 2. generate modified dsl
        user_feedback = MultimodalQuery(
            text="Please add another car in the scenario which turns left at the intersection behind the ego vehicle as I marked with a red box in the image.", 
            image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/test_image_modification.png", 
            video_path=None
        )
        scenario_to_modify = ScenicScenario(
            scenario_id="test_scenario",
            scenic_code="test_scenic_code"
        )
        modified_dsl = interpreter.generate_dsl_from_user_feedback(user_feedback, dsl, scenario_to_modify)
        print(f"Successfully generated modified DSL: {modified_dsl}")

        """
        {
        'scenario': 'The ego vehicle and a following car both perform left turns at a four-way intersection after yielding to an adversarial vehicle crossing from the left.', 
        'ego': 'A car that moves straight toward an intersection, pauses to allow another vehicle to pass, and then executes a left turn.', 
        'adversarials': ["A car that enters the intersection from the left and travels across the ego vehicle's path to the right.", 'A car that follows behind the ego vehicle and executes a left turn at the intersection after the ego vehicle.'], 
        'spatial_relation': 'The ego vehicle and the first adversarial vehicle approach a four-way intersection from perpendicular directions, while the second adversarial vehicle is positioned directly behind the ego vehicle on the same multi-lane straight road.', 
        'requirements_and_restrictions': 'The ego vehicle faces a green traffic light upon arrival at the intersection. The scenario concludes once the ego vehicle completes its left turn and travels a certain distance on the new road.', 
        'road_side_structures': [{'object': 'Kiosk', 'position': 'Positioned on the sidewalk to the right of the ego vehicle near the intersection corner.'}], 
        'temporary_modifications': [{'object': 'Street Barrier', 'position': "Located in the middle of the ego vehicle's lane at the entrance to the intersection."}], 
        'reasoning_chain': "1. Actors: The ego vehicle is the primary car viewed from a third-person perspective; the first adversarial is the black car entering from the left; the second adversarial is the car added behind the ego as requested. 2. Chronological Analysis: Initially, the ego and the second adversarial approach the intersection in the same lane. At the midpoint, the first adversarial car crosses from left to right, causing the ego and the second adversarial to decelerate. In the final phase, the ego vehicle turns left, followed by the second adversarial vehicle also making a left turn. 3. Evidence: The ego vehicle's speed decreases as the first adversarial's path crosses its own. The second adversarial maintains a following distance behind the ego. Both vehicles then follow a 90-degree leftward arc into the crossroad, as seen in the original video's trajectory for the ego and the user's modification request for the second car."}
        """

    test_dsl_generation_from_user_feedback()