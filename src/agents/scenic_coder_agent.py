import logging
import re

from src.schema import HeaderSetting
from src.utils import to_safe_string

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
            # Header Settings:
            description = "Null description"
            param map = localPath('{header_settings.map_file_path}')
            param carla_map = '{header_settings.carla_map}'
            model scenic.simulators.carla.model
            MODEL = '{header_settings.blueprint}'
            param weather = '{header_settings.weather}'
            # End of Header Settings
        """
        return header

    def adapt_code(self, original_scenic_code: str, evaluation_result: Dict[str, Any], aim_dsl: Dict[str, Any], header_settings: HeaderSetting | None) -> str:
        """
        Adapt the original scenic code to the aim DSL.
        Take the original scenic code, the evaluation result, and the aim DSL as input.
        Generate the new scenic code recursively, in each generation we put the previously generated scenic code as input to ensure compatibility.
        """

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

        # 1. Get the components needed to be modified in the aim DSL
        components_to_modify = {}
        for key, value in evaluation_result.items():
            if not value: # Assuming False means it failed evaluation
                components_to_modify[key] = aim_dsl[key] 

        # 2. Search the vector store for similar snippets
        def get_snippets(text: str, comp_type: str) -> List[str]:
            try:
                query_embedding = self.snippets_embedder.encode([{"text": text}])[0]
                    
                if hasattr(query_embedding, "tolist"):
                    query_embedding = query_embedding.tolist()

                similar_snippets = self.vector_store.similarity_search_snippets(
                    query_embedding=query_embedding, 
                    k=3, 
                    component_type=comp_type
                )
                return [item["code"] for item in similar_snippets if item.get("code")]
            except Exception as e:
                logger.error(f"Failed to retrieve/process snippets for {comp_type}: {e}")
                return []

        snippets = {}
        
        for key, value in components_to_modify.items():
            if key == "Adversarials":
                adversarial_snippets = []
                # value is a list of descriptions for different adversarial agents
                for adversarial_desc in value:
                    result = get_snippets(text=adversarial_desc, comp_type="Adversarial")
                    if result: # Only append if we actually found something
                        adversarial_snippets.append(result)
                snippets[key] = adversarial_snippets
            else:
                snippets[key] = get_snippets(text=value, comp_type=key)

        # 3. Format the few-shot examples
        few_shot_examples = ""
        if snippets:
            few_shot_examples += "Use the following verified examples to resolve the specific incompatibilities. These represent the ground-truth syntax for the target DSL:\n\n"
            
            for key, value in snippets.items():
                if not value: 
                    continue # Skip empty results
                    
                if key == "Adversarials":
                    # value is a list of lists: [["code1", "code2"], ["code3", "code4"]]
                    for idx, adversarial_group in enumerate(value):
                        few_shot_examples += f"--- Aspect: {key} (Agent {idx+1}) ---\n"
                        few_shot_examples += f"Verified DSL-compliant patterns:\n{to_safe_string(adversarial_group)}\n\n"
                else:
                    few_shot_examples += f"--- Aspect: {key} ---\n"
                    few_shot_examples += f"Verified DSL-compliant patterns:\n{to_safe_string(value)}\n\n"

        # 4. Format the prompt
        prompt = self.prompt_template.format(
            original_scenic_code=original_scenic_code, 
            header_instruction=header_instruction, 
            aim_dsl=to_safe_string(aim_dsl), 
            aspects=to_safe_string(components_to_modify), 
            few_shot_examples=few_shot_examples
        )
        
        formatted_prompt = prompt.strip()
        response = self.llm_service.chat([{"role": "user", "content": formatted_prompt}])
        
        return response

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

    llm_service = get_llm_service(provider="gemini", model="gemini-2.5-flash")
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
            map_file_path="../../maps/Town07.xodr",
            carla_map="Town07",
            blueprint="vehicle.lincoln.mkz_2017",
            weather="MidRainyNoon"
        )
        original_scenic_code = """
        description = "Ego vehicle loses control on bad road conditions and recovers to its original lane."
        param map = localPath('../../maps/Town05.xodr')
        param carla_map = 'Town05'
        model scenic.simulators.carla.model
        MODEL = 'vehicle.lincoln.mkz_2017'
        param weather = 'WetNoon'

        egoInitLane = Uniform(*network.lanes)
        egoSpawnPt = new OrientedPoint in egoInitLane.centerline

        param OPT_EGO_SPEED = Range(7, 10)
        param LOSS_CONTROL_STEER = 0.4
        param LOSS_CONTROL_DURATION = 35

        behavior EgoBehavior(speed, target_lane_sec):
            # Phase 1: Normal driving
            do FollowLaneBehavior(target_speed=speed) for Range(2, 4) seconds
            
            # Phase 2: Loss of control (simulated by a sustained steering offset)
            count = 0
            while count < globalParameters.LOSS_CONTROL_DURATION:
                take SetSteerAction(globalParameters.LOSS_CONTROL_STEER), SetThrottleAction(0.5)
                count = count + 1
            
            # Phase 3: Recovery (return to the original lane section)
            do LaneChangeBehavior(laneSectionToSwitch=target_lane_sec, target_speed=speed)
            do FollowLaneBehavior(target_speed=speed)

        egoLaneSec = egoInitLane.sections[0]

        ego = new Car at egoSpawnPt,
            with rolename 'hero',
            with blueprint MODEL,
            with behavior EgoBehavior(globalParameters.OPT_EGO_SPEED, egoLaneSec)

        require egoSpawnPt not in network.intersections
        require (distance from egoSpawnPt to intersection) > 30
        terminate after 25 seconds
        """
        evaluation_result = {'Adversarials': False,
                            'Ego': True,
                            'Requirement and restrictions': True,
                            'Scenario': False,
                            'Spatial Relation': False}
        aim_dsl = { 'Adversarials': [ 'A car approaches an intersection from a perpendicular road and makes a left '
                                        'turn.',
                                        'A car is positioned behind the ego vehicle and makes a left turn at the '
                                        'intersection.'],
                        'Ego': 'A car stops at an intersection, then proceeds straight.',
                        'Requirement and restrictions': 'The traffic light for the ego vehicle must initially be red and then turn green, allowing the ego vehicle to proceed.',
                        'Scenario': 'The ego vehicle proceeds straight through an intersection while an adversarial vehicle makes a left turn from a perpendicular road, and another adversarial vehicle turns left from behind the ego vehicle.',
                        'Spatial Relation': 'The ego vehicle is positioned in a lane at an intersection under a highway overpass, with another adversarial vehicle behind it in the same lane.'}
        
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