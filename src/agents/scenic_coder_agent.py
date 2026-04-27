import logging
import re
import time
from src.schema import HeaderSetting
from src.utils import clean_and_parse_json, setup_logging, strip_code_fence_markers
from tests.test_utils import test_video_recording

from .base_agent import BaseAgent
from src.services import BaseLLMModel, MilvusVectorStore, BaseEmbeddingModel, get_embedder, get_llm_service
from src.prompt import load_prompt
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Define a static example of what a header looks like so the LLM knows what to look for
header_format_example = """
    <header_example>
    description = "Ego vehicle performs an unprotected left turn..."
    param map = localPath('../../maps/Town05.xodr')
    param carla_map = 'Town05'
    model scenic.simulators.carla.model
    MODEL = 'vehicle.lincoln.mkz_2017'
    param weather = 'ClearNoon'
    </header_example>"""

class ScenicCoderAgent(BaseAgent):
    def __init__(self, llm_service: BaseLLMModel, vector_store: MilvusVectorStore, snippets_embedder: BaseEmbeddingModel):
        super().__init__()
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.snippets_embedder = snippets_embedder
        self.prompt_template = load_prompt("adapt_code")

    def process(self, state: dict) -> dict:
        return state

    def replace_header(self, scenic_code: str, header_settings: HeaderSetting) -> str:
        """
        Replace the header of the scenic code with the header settings.
        Used after retrieving the base scenario from the vector store.
        """
        replacements = [
            (r"(?m)^\s*param\s+map\s*=.*$",        f"param map = localPath('{header_settings.map_file_path}')"),
            (r"(?m)^\s*param\s+carla_map\s*=.*$",  f"param carla_map = '{header_settings.carla_map}'"),
            (r"(?m)^\s*MODEL\s*=.*$",              f"MODEL = '{header_settings.blueprint}'"),
            (r"(?m)^\s*param\s+weather\s*=.*$",    f"param weather = '{header_settings.weather}'"),
        ]
        for pattern, repl in replacements:
            if re.search(pattern, scenic_code):
                scenic_code = re.sub(pattern, repl, scenic_code, count=1)
            else:
                scenic_code += "\n" + repl + "\n"
        return scenic_code

    def generate_header(self, header_settings: HeaderSetting | None) -> str:
        """
        Generate the header for the scenario based on the header settings.
        """
        if not header_settings:
            return f"""
description = "No header settings provided"
param map = localPath('../../maps/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model
MODEL = 'vehicle.lincoln.mkz_2017'
param weather = 'ClearNoon'
        """
        header = f"""
description = "Using map {header_settings.map_file_path} with carla map {header_settings.carla_map} and weather {header_settings.weather}"
param map = localPath('{header_settings.map_file_path}')
param carla_map = '{header_settings.carla_map}'
model scenic.simulators.carla.model
MODEL = '{header_settings.blueprint}'
param weather = '{header_settings.weather}'
        """
        return header

    def get_snippets(self, text: str, comp_type: str) -> List[str]:
        # comp_type mapping
        if comp_type == "spatial_relation":
            comp_type = "Spatial Relation"
        elif comp_type == "ego":
            comp_type = "Ego"
        elif comp_type == "adversarials":
            comp_type = "Adversarial"
        elif comp_type == "road_side_structures":
            comp_type = "Adversarial" # same as adversarial
        elif comp_type == "temporary_modifications":
            comp_type = "Adversarial" # same as adversarial
        elif comp_type == "requirements_and_restrictions":
            comp_type = "Requirement and restrictions"
        try:
            query_embedding = self.snippets_embedder.encode([{"text": text}])[0]
            if hasattr(query_embedding, "tolist"): query_embedding = query_embedding.tolist()
            similar_snippets = self.vector_store.similarity_search_snippets(
                query_embedding=query_embedding, k=3, component_type=comp_type
            )
            results = [item["code"] for item in similar_snippets if item.get("code")]
            return results if results else []
        except Exception as e:
            logger.error(f"❌ Failed to retrieve snippets for {comp_type}: {e}")
            return []

    def build_context(self, current_state: dict) -> str:
        """Formats previously generated components to provide context for the current generation."""
        if not current_state: return "No components generated yet."
        
        context_str = "--- PREVIOUSLY GENERATED COMPONENTS (FOR CONTEXT ONLY) ---\n"
        for key, val in current_state.items():
            if isinstance(val, list):
                context_str += f"// {key}:\n" + "\n".join(val) + "\n\n"
            else:
                context_str += f"// {key}:\n{val}\n\n"
        return context_str

    def generate_and_clean(self, prompt_text: str) -> str:
        response = self.llm_service.chat([{"role": "user", "content": prompt_text.strip()}])
        logger.info(f"Response from LLM: \n{response}")
        # add a sleep to avoid rate limit
        time.sleep(1)
        if not response:
            logger.error(f"Response text was None! Finish Reason: {response.candidates[0].finish_reason}")
            logger.error(f"Raw Response: {response}")
            return ""
        else:
            return response.replace("```scenic", "").replace("```python", "").replace("```", "").strip()

    def get_prompt_for_component(self, aspect: str) -> str:
        if aspect == "adversarials":
            return load_prompt("component_generator_adv")
        elif aspect == "ego":
            return load_prompt("component_generator_ego")
        elif aspect == "requirements_and_restrictions":
            return load_prompt("component_generator_requirement")
        elif aspect == "spatial_relation":
            return load_prompt("component_generator_spatial")
        elif aspect == "road_side_structures":
            return load_prompt("component_generator_road_side_structure")
        elif aspect == "temporary_modifications":
            return load_prompt("component_generator_temporary_modification")
        else:
            logger.error(f"Invalid aspect: {aspect}")
            return load_prompt("component_generator_ego")

    def prepare_snippets(self, snippets: List[str]) -> str:
        """
        Prepare the snippets for the prompt.
        """
        prepared = "// Retrieved relevant snippets:\n"
        if snippets and len(snippets) > 0:
            for idx, snippet in enumerate(snippets):
                prepared += f"// Snippet {idx+1}:\n{snippet}\n\n"
        else:
            prepared += "// No relevant snippets found.\n\n"
        return prepared

    def generate_component(self, aspect: str, description: str, retrieved_components: Dict[str, Any]) -> str:
        """
        Generate a component based on the description and the retrieved components.
        """
        snippets = self.get_snippets(text=description, comp_type=aspect)
        context = self.build_context(retrieved_components)
        
        comp_prompt = self.get_prompt_for_component(aspect).format(
            reference_components=self.prepare_snippets(snippets),
            ready_components=context,
            user_criteria=description
        )
        new_code = self.generate_and_clean(comp_prompt)
        return new_code

    def extract_component(self, aspect: str, description: str, original_scenic_code: str, retrieved_components: Dict[str, Any]) -> str:
        """
        Extract a component based on the original scenic code and the retrieved components.
        """
        context = self.build_context(retrieved_components)
        extract_prompt = load_prompt("component_generator_extract").format(
            aspect=aspect, 
            description=description, 
            context=context, 
            original_scenic_code=original_scenic_code)
        extracted = self.generate_and_clean(extract_prompt)
        return extracted

    def adapt_code(self, original_scenic_code: str, evaluation_result: Dict[str, Any], aim_dsl: Dict[str, Any], header_settings: Any) -> str:
        """
        Adapt the original scenic code to the aim DSL using a 'Generation + Assemble' architecture.
        Each component is generated in isolation (to allow for heavy grammar instructions) 
        and then assembled into the final script.
        """
        logger.info("🚀 Starting Generation + Assemble Scenic pipeline...")

        """ The order of the components: 
        header
        spatial_relation
        ego
        adversarials
        requirements_and_restrictions
        road_side_structures
        temporary_modifications
        """
        
        # Dictionary to act as our "State" tracking the isolated components
        retrieved_components = {}

        # =========================================================
        # STEP 1: INITIALIZE HEADER
        # =========================================================
        logger.info("📄 Generating/Extracting Header...")
        retrieved_components["header"] = self.generate_header(header_settings)

        # =========================================================
        # STEP 2: COMPONENT GENERATION
        # =========================================================
        generation_order = [
            "spatial_relation", 
            "ego", 
            "adversarials", 
            "road_side_structures", 
            "temporary_modifications",
            "requirements_and_restrictions"
        ]
        
        list_aspects = ["adversarials", "road_side_structures", "temporary_modifications"]
        
        # generate the components in the first generation order
        for aspect in generation_order:
            logger.info(f"🧩 Processing Component: {aspect}")
            # deal with adversarials separately since the evaluation result is a list
            if aspect in list_aspects:
                description_list = aim_dsl.get(aspect, "")
                generated_components = []
                for idx, adversarial_result in enumerate(evaluation_result[aspect]):
                    new_code = None
                    if not adversarial_result:
                        # generate a new adversarial with description
                        description  = description_list[idx]
                        new_code = self.generate_component(aspect, description, retrieved_components)
                    else:
                        # extract the existing adversarial
                        description = description_list[idx]
                        new_code = self.extract_component(aspect, description, original_scenic_code, retrieved_components)
                    if new_code:
                        generated_components.append(new_code)
                        retrieved_components[aspect] = generated_components
                        logger.info(f"✅ Successfully generated {aspect} {idx+1}")
                    else:
                        logger.warning(f"⚠️ LLM returned empty code for {aspect} {idx+1}")
                continue
            else:
                # for those aspects that are not a list, generate a new component with the description
                target_description = aim_dsl.get(aspect, "")
                needs_modification = not evaluation_result.get(aspect, True)
                new_code = None
                if needs_modification:
                    # generate a new component with the description
                    new_code = self.generate_component(aspect, target_description, retrieved_components)
                    logger.info(f"✅ Successfully generated {aspect}")
                else:
                    # extract the existing component
                    new_code = self.extract_component(aspect, target_description, original_scenic_code, retrieved_components)
                    logger.info(f"✅ Successfully extracted {aspect}")
                if new_code:
                    retrieved_components[aspect] = new_code
                    logger.info(f"✅ Successfully generated {aspect}")
                else:
                    logger.warning(f"⚠️ LLM returned empty code for {aspect}")

        # =========================================================
        # STEP 3: ASSEMBLE SCENARIO
        # =========================================================
        logger.info("🔗 Assembling final scenario from isolated components...")
        logger.info(f"Retrieved components: {retrieved_components}")
        code_parts = []
        
        # We use a strict assembly order to ensure variables are declared before they are used
        assembly_order = ["header",             
            "spatial_relation", 
            "ego", 
            "adversarials", 
            "road_side_structures", 
            "temporary_modifications",
            "requirements_and_restrictions",]
        
        for comp_type in assembly_order:
            if comp_type not in retrieved_components:
                continue
                
            comp_data = retrieved_components[comp_type]
            
            if isinstance(comp_data, list):
                # Unpack list of adversarials
                for item in comp_data:
                    if item.strip():
                        code_parts.append(item.strip())
            else:
                # Standard component
                if comp_data.strip():
                    code_parts.append(comp_data.strip())
                    
        assembled_code = "\n\n".join(code_parts)
        
        logger.info("=" * 40)
        logger.info("🎉 Assembly complete!")
        logger.info(f"Final script length: {len(assembled_code)} characters.")
        logger.info(f"Final script: \n{assembled_code}")
        logger.info("=" * 40)

        return assembled_code

        # =========================================================
        # STEP 4: LINTING AND VALIDATION OF THE ASSEMBLED CODE
        # =========================================================
        # logger.info("🔍 Linting and validating the assembled code...")
        # validate_prompt = load_prompt("validate_scenic_code").format(
        #     assembled_code=assembled_code
        # )
        # final_code = self.generate_and_clean(validate_prompt)
        # logger.info("=" * 40)
        # logger.info("🎉 Validation complete!")
        # logger.info(f"Final script length: {len(final_code)} characters.")
        # logger.info(f"Final script: \n{final_code}")
        # logger.info("=" * 40)
        
        # return final_code

    def debug_code(self, scenic_code: str, error_message: str, header_settings: HeaderSetting | None) -> str:
        # 0. Generate the header instruction
        if not header_settings:
            header_instruction = f"""
            DO NOT modify the original header block. 
            Identify the header block (which typically contains the description, map, model, and weather parameters, similar to the example below) and keep it exactly as it is in the original code.
            {header_format_example}
            """
        else:
            new_header = self.generate_header(header_settings)
            header_instruction = f"""
            REPLACE the original header block (which typically contains the description, map, model, and weather parameters) with the exact new header provided below.
            {header_format_example}

            <new_header_to_use>
            {new_header}
            </new_header_to_use>
            """

        # 1. Format the prompt
        prompt = load_prompt("debug_scenic_code").format(
            scenic_code_to_debug=scenic_code, 
            error_message=error_message, 
            header_instruction=header_instruction
        )
        formatted_prompt = prompt.strip()

        # 2. Call the LLM service
        response = self.llm_service.chat([{"role": "user", "content": formatted_prompt}])
        fixed_code = clean_and_parse_json(response).get("fixed_code", "")
        return fixed_code

if __name__ == "__main__":
    from src.services import MilvusVectorStore
    from src.config import get_config
    config = get_config()
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
        log_format=config.logging.format,
    )
    logger.info("Running ScenicCoderAgent standalone module.")

    llm_service = get_llm_service(provider=config.llm.provider, model=config.llm.model, temperature=config.llm.temperature, max_tokens=config.llm.max_tokens)
    connection_args = {
        "host": config.vector_db.host,
        "port": config.vector_db.port,
    }
    index_params = {
        "metric_type": config.vector_db.distance_metric.upper(),
        "index_type": config.vector_db.index_type,
        "params": {"nlist": config.vector_db.nlist},
    }
    search_params = {
        "metric_type": config.vector_db.distance_metric.upper(),
        "params": {"nprobe": config.vector_db.nprobe},
    }

    vector_db = MilvusVectorStore(
        connection_args=connection_args,
        index_params=index_params,
        search_params=search_params,
    )
    snippets_embedder = get_embedder(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    agent = ScenicCoderAgent(llm_service, vector_db, snippets_embedder)

    def test_coder(mode: str = "adapt", error_message: str | None = None):
        header_settings = HeaderSetting(
            map_file_path="../../maps/Town05.xodr",
            carla_map="Town05",
            blueprint="vehicle.lincoln.mkz_2017",
            weather="MidRainyNoon"
        )
        original_scenic_code = """
        description = "Ego vehicle performs an unprotected left turn, yielding to oncoming traffic at signalized and non-signalized intersections."
param map = localPath('../../maps/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model
MODEL = 'vehicle.lincoln.mkz_2017'
param weather = 'ClearNoon'

intersection = Uniform(*filter(lambda i: any(m.type is ManeuverType.LEFT_TURN for m in i.maneuvers), network.intersections))

egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN and any(cm.type is ManeuverType.STRAIGHT for cm in m.conflictingManeuvers), intersection.maneuvers))
egoInitLane = egoManeuver.startLane
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = new OrientedPoint in egoInitLane.centerline

advManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoManeuver.conflictingManeuvers))
advInitLane = advManeuver.startLane
advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]
advSpawnPt = new OrientedPoint in advInitLane.centerline

param OPT_EGO_SPEED = Range(3, 5)
param OPT_EGO_YIELD_DIST = Range(15, 20)
OPT_EGO_DECISION_DEGREE = 35 deg

behavior EgoBehavior():
    initialDir = egoSpawnPt.heading
    try:
        do FollowTrajectoryBehavior(trajectory=egoTrajectory, target_speed=globalParameters.OPT_EGO_SPEED)
    interrupt when withinDistanceToAnyCars(self, globalParameters.OPT_EGO_YIELD_DIST):
        currentDir = self.heading
        if (abs(currentDir - initialDir) < OPT_EGO_DECISION_DEGREE):
            take SetThrottleAction(0)
            take SetBrakeAction(1)
        else:
            do FollowTrajectoryBehavior(trajectory=egoTrajectory, target_speed=globalParameters.OPT_EGO_SPEED + 2)
            abort
    terminate

ego = new Car at egoSpawnPt,
    with rolename 'hero',
    with blueprint MODEL,
    with behavior EgoBehavior()

param ADV_SPEED = Range(7, 10)

behavior AdversaryBehavior(trajectory):
    do FollowTrajectoryBehavior(target_speed=globalParameters.ADV_SPEED, trajectory=trajectory)

adversary = new Car at advSpawnPt,
    with blueprint MODEL,
    with behavior AdversaryBehavior(advTrajectory)

EGO_INIT_DIST = [10, 15]
ADV_INIT_DIST = [15, 25]
TERM_DIST = 50

monitor TrafficLights():
    freezeTrafficLights()
    while True:
        if withinDistanceToTrafficLight(ego, 100):
            setClosestTrafficLightStatus(ego, "green")
        if withinDistanceToTrafficLight(adversary, 100):
            setClosestTrafficLightStatus(adversary, "green")
        wait

require monitor TrafficLights()
require EGO_INIT_DIST[0] <= (distance from egoSpawnPt to intersection) <= EGO_INIT_DIST[1]
require ADV_INIT_DIST[0] <= (distance from advSpawnPt to intersection) <= ADV_INIT_DIST[1]
terminate when (distance from ego to egoSpawnPt) > TERM_DIST
        """

        # Spatial relation has to be consistant with the adversarial description, otherwise it will generate wrong thing.
        evaluation_result = {'adversarials': [False],
                            'ego': False,
                            'requirements_and_restrictions': True,
                            'scenario': False,
                            'spatial_relation': False,
                            'road_side_structures': [False],
                            'temporary_modifications': [False]}
        aim_dsl = { 'adversarials': [
                        {"object": "car", "behavior": "behind the ego vehicle in the beginning, then turns right at the intersection."}
                        ],
                        'ego': {"object": "car", "behavior": "stops at an intersection, then proceeds right turn at the intersection."},
                        'requirements_and_restrictions': "The traffic light for the ego vehicle must initially be red and then turn green, allowing the ego vehicle to proceed.",
                        'scenario': "The ego vehicle proceeds right turn at the intersection while an adversarial vehicle go straight at the intersection.",
                        'spatial_relation': "The ego vehicle is positioned in a lane at an intersection and make a right turn, with the  adversarial positioned behind the ego vehicle and turns right at the intersection.",
                        'road_side_structures': [
                            {"object": "kiosk", "position": "Located on the right side of the ego vehicle initial position."}
                        ],
                        'temporary_modifications': [
                            {"object": "street barrier", "position": "Located on the left lane of the ego vehicle initial position and facing the ego vehicle."}
                        ]}
        
        if mode == "adapt":
            adapted_scenic_code = agent.adapt_code(original_scenic_code, evaluation_result, aim_dsl, header_settings)
        elif mode == "debug":
            adapted_scenic_code = agent.debug_code(original_scenic_code, error_message, header_settings)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        
        logger.info(f"Adapted scenic code: \n{adapted_scenic_code}")
        with open("tests/scenic_code.txt", "w") as f:
            f.write(adapted_scenic_code)
        test_video_recording()

    def test_debug():
        error_message = """Traceback (most recent call last; use -b to show Scenic internals):
  File "/home/dellpro2/chenli/ads-mrag/ads-mrag/temp/test_scenario/code/scenic_code.scenic", line 18, in <lambda>
    adv2Maneuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN and m.startLane == egoInitLane, intersection.maneuvers))
                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
RandomControlFlowError: random values cannot be compared (and control flow cannot depend on them)"""
        original_scenic_code = """
        description = "Using map ../../maps/Town05.xodr with carla map Town05 and weather MidRainyNoon"
param map = localPath('../../maps/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model
MODEL = 'vehicle.lincoln.mkz_2017'
param weather = 'MidRainyNoon'

intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))

egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.RIGHT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
egoSpawnPt = new OrientedPoint in egoInitLane.centerline

adv1Maneuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoManeuver.conflictingManeuvers))
adv1InitLane = adv1Maneuver.startLane
adv1SpawnPt = new OrientedPoint in adv1InitLane.centerline

adv2Maneuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN and m.startLane == egoInitLane, intersection.maneuvers))
adv2InitLane = adv2Maneuver.startLane
adv2SpawnPt = new OrientedPoint behind egoSpawnPt by Range(8, 12)

egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
adv1Trajectory = [adv1InitLane, adv1Maneuver.connectingLane, adv1Maneuver.endLane]
adv2Trajectory = [adv2InitLane, adv2Maneuver.connectingLane, adv2Maneuver.endLane]

param EGO_SPEED = 7

behavior EgoBehavior(trajectory, target_speed):
    do FollowLaneBehavior(target_speed=target_speed) until self in intersection
    do FollowLaneBehavior(target_speed=0) for 2 seconds
    do TurnBehavior(trajectory=trajectory, target_speed=target_speed)
    do FollowLaneBehavior(target_speed=target_speed)

ego = new Car at egoSpawnPt,
    with blueprint MODEL,
    with rolename 'hero',
    with behavior EgoBehavior(egoTrajectory, globalParameters.EGO_SPEED)

param ADV_SPEED = Range(7, 10)

behavior AdversaryBehavior(trajectory):
    do FollowTrajectoryBehavior(target_speed=globalParameters.ADV_SPEED, trajectory=trajectory)

adversary1 = new Car at adv1SpawnPt,
    with blueprint MODEL,
    with behavior AdversaryBehavior(adv1Trajectory)

behavior Adv2Behavior(trajectory, target_speed):
    do FollowLaneBehavior(target_speed=target_speed) until self in intersection
    do TurnBehavior(trajectory=trajectory, target_speed=target_speed)
    do FollowLaneBehavior(target_speed=target_speed)

adversary2 = new Car at adv2SpawnPt,
    with blueprint MODEL,
    with behavior Adv2Behavior(adv2Trajectory, globalParameters.ADV_SPEED)

kioskSpawnPt = new OrientedPoint right of egoSpawnPt by 5
kiosk = new Prop at kioskSpawnPt,
    with blueprint 'static.prop.kiosk_01'

streetBarrierSpawnPt = new OrientedPoint left of egoSpawnPt by 3.5,
    facing toward egoSpawnPt
streetBarrier = new Prop at streetBarrierSpawnPt,
    with blueprint 'static.prop.streetbarrier',
    with allowCollisions True

EGO_INIT_DIST = [10, 15]
ADV_INIT_DIST = [15, 25]
TERM_DIST = 50

monitor TrafficLights():
    freezeTrafficLights()
    while True:
        if withinDistanceToTrafficLight(ego, 100):
            setClosestTrafficLightStatus(ego, "green")
        if withinDistanceToTrafficLight(adversary1, 100):
            setClosestTrafficLightStatus(adversary1, "green")
        wait

require monitor TrafficLights()
require EGO_INIT_DIST[0] <= (distance from egoSpawnPt to intersection) <= EGO_INIT_DIST[1]
require ADV_INIT_DIST[0] <= (distance from adv1SpawnPt to intersection) <= ADV_INIT_DIST[1]
terminate when (distance from ego to egoSpawnPt) > TERM_DIST
        """
        adapted_scenic_code = agent.debug_code(original_scenic_code, error_message, None)
        with open("tests/scenic_code.txt", "w") as f:
            f.write(adapted_scenic_code)
        test_video_recording()

    # test_coder(mode="adapt")
    test_debug()