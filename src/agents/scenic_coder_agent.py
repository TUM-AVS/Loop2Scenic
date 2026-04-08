import logging
import re

from src.schema import HeaderSetting
from src.utils import setup_logging

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

    def generate_header(self, header_settings: HeaderSetting) -> str:
        """
        Generate the header for the scenario based on the header settings.
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
        return response.replace("```scenic", "").replace("```python", "").replace("```", "").strip()

    def get_prompt_for_component(self, aspect: str) -> str:
        if aspect == "Adversarials":
            return load_prompt("component_generator_adv")
        elif aspect == "Ego":
            return load_prompt("component_generator_ego")
        elif aspect == "Requirement and restrictions":
            return load_prompt("component_generator_requirement")
        elif aspect == "Spatial Relation":
            return load_prompt("component_generator_spatial")
        else:
            logger.error(f"Invalid aspect: {aspect}")
            return load_prompt("ego_generation")

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

    def adapt_code(self, original_scenic_code: str, evaluation_result: Dict[str, Any], aim_dsl: Dict[str, Any], header_settings: Any) -> str:
        """
        Adapt the original scenic code to the aim DSL using a 'Generation + Assemble' architecture.
        Each component is generated in isolation (to allow for heavy grammar instructions) 
        and then assembled into the final script.
        """
        logger.info("🚀 Starting Generation + Assemble Scenic pipeline...")

        # Dictionary to act as our "State" tracking the isolated components
        retrieved_components = {}

        # =========================================================
        # STEP 1: INITIALIZE HEADER
        # =========================================================
        logger.info("📄 Generating/Extracting Header...")
        if header_settings:
            retrieved_components["Header"] = self.generate_header(header_settings)
        else:
            header_prompt = f"Extract the header block exactly as it is from this code. Output ONLY the extracted code:\n{original_scenic_code}"
            retrieved_components["Header"] = self.generate_and_clean(header_prompt)

        # =========================================================
        # STEP 2: COMPONENT GENERATION
        # =========================================================
        generation_order = ["Spatial Relation", "Ego", "Adversarials", "Requirement and restrictions"]
        
        for aspect in generation_order:
            logger.info(f"🧩 Processing Component: {aspect}")
            needs_modification = not evaluation_result.get(aspect, True)
            target_description = aim_dsl.get(aspect, "")
            
            # --- NON-MODIFIED EXTRACTION ---
            if not needs_modification:
                logger.info(f"⏭️ No modification needed. Extracting '{aspect}' from original code.")

                # has to add the previous generated components to the prompts
                context = self.build_context(retrieved_components)
                extract_prompt = load_prompt("component_generator_extract").format(
                    aspect=aspect,
                    context=context,
                    original_scenic_code=original_scenic_code
                )

                # For adversarials, we wrap it in a list to maintain data structure
                extracted = self.generate_and_clean(extract_prompt)
                retrieved_components[aspect] = [extracted] if aspect == "Adversarials" else extracted
                continue

            # --- MODIFIED GENERATION ---
            logger.warning(f"⚠️ Generating new '{aspect}' based on DSL requirements.")

            if aspect == "Adversarials" and isinstance(target_description, list):
                generated_adversarials = []
                
                for idx, desc in enumerate(target_description):
                    logger.info(f"🤖 Processing Adversarial {idx+1}/{len(target_description)}...")
                    
                    # 1. Get snippets for THIS specific agent's description
                    snippets = self.get_snippets(text=desc, comp_type="Adversarial")
                    
                    # 2. Re-build context so it includes any adversarials generated in previous loop iterations
                    context = self.build_context(retrieved_components)
                    
                    # 3. Format the newly optimized prompt
                    # Notice the variable names match the {placeholders} in the optimized prompt exactly
                    adv_prompt = self.get_prompt_for_component("Adversarials").format(
                        reference_components=self.prepare_snippets(snippets),
                        ready_components=context,
                        user_criteria=desc
                    )
                    logger.info(f"Adv prompt: \n{adv_prompt}")
                    
                    # 4. Send to LLM and clean the output
                    new_code = self.generate_and_clean(adv_prompt)

                    # 5. Save the result
                    if new_code:
                        generated_adversarials.append(new_code)
                        retrieved_components["Adversarials"] = generated_adversarials
                        logger.info(f"✅ Successfully generated Adversarial {idx+1}")
                    else:
                        logger.warning(f"⚠️ LLM returned empty code for Adversarial {idx+1}")
            else:
                snippets = self.get_snippets(text=target_description, comp_type=aspect)
                context = self.build_context(retrieved_components)
                
                comp_prompt = self.get_prompt_for_component(aspect).format(
                    reference_components=self.prepare_snippets(snippets),
                    ready_components=context,
                    user_criteria=target_description
                )
                logger.info(f"Comp prompt: \n{comp_prompt}")
                new_code = self.generate_and_clean(comp_prompt)
                logger.info(f"New code: {new_code}")
                retrieved_components[aspect] = new_code
                logger.info(f"✅ Generated {aspect}")

        # =========================================================
        # STEP 3: ASSEMBLE SCENARIO
        # =========================================================
        logger.info("🔗 Assembling final scenario from isolated components...")
        logger.info(f"Retrieved components: {retrieved_components}")
        code_parts = []
        
        # We use a strict assembly order to ensure variables are declared before they are used
        assembly_order = ["Header", "Spatial Relation", "Ego", "Adversarials", "Requirement and restrictions"]
        
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
                    
        final_code = "\n\n".join(code_parts)
        
        logger.info("=" * 40)
        logger.info("🎉 Assembly complete!")
        logger.info(f"Final script length: {len(final_code)} characters.")
        logger.info(f"Final script: \n{final_code}")
        logger.info("=" * 40)
        
        return final_code

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
        return response

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

    llm_service = get_llm_service(provider="gemini", model="gemini-2.5-pro")
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
        evaluation_result = {'Adversarials': False,
                            'Ego': True,
                            'Requirement and restrictions': True,
                            'Scenario': False,
                            'Spatial Relation': False}
        aim_dsl = { 'Adversarials': [ 'A car is positioned behind the ego vehicle and makes a left turn at the '
                                        'intersection.',
                                        'A car approaches an intersection from a perpendicular road and makes a left '
                                        'turn.'
                                        ],
                        'Ego': 'A car stops at an intersection, then proceeds straight.',
                        'Requirement and restrictions': 'The traffic light for the ego vehicle must initially be red and then turn green, allowing the ego vehicle to proceed.',
                        'Scenario': 'The ego vehicle proceeds straight through an intersection while an adversarial vehicle makes a left turn from a perpendicular road, and another adversarial vehicle turns left from behind the ego vehicle.',
                        'Spatial Relation': 'The ego vehicle is positioned in a lane at an intersection, with the first adversarial vehicle behind it in the same lane, and the second adversarial vehicle approaches the ego car from a perpendicular road, then makes a left turn at the intersection.'}
        
        if mode == "adapt":
            adapted_scenic_code = agent.adapt_code(original_scenic_code, evaluation_result, aim_dsl, header_settings)
        elif mode == "debug":
            adapted_scenic_code = agent.debug_code(original_scenic_code, error_message, header_settings)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        print(adapted_scenic_code)

    test_coder(mode="adapt")
    error_message = """Traceback (most recent call last; use -b to show Scenic internals):
  File "/home/dellpro2/chenli/ads-mrag/ads-mrag/temp/test_scenario/code/scenic_code.scenic", line 23, in <module>
    for lane in intersection.incomingLanes:
RandomControlFlowError: cannot iterate through a random value"""
    # test_coder(mode="debug", error_message=error_message)