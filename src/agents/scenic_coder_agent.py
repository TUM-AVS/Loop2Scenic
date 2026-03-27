import logging

from src.utils import to_safe_string

from .base_agent import BaseAgent
from src.services import BaseLLMModel, MilvusVectorStore, BaseEmbeddingModel, get_embedder, get_llm_service
from src.prompt import load_prompt
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ScenicCoderAgent(BaseAgent):
    def __init__(self, llm_service: BaseLLMModel, vector_store: MilvusVectorStore, snippets_embedder: BaseEmbeddingModel):
        super().__init__()
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.snippets_embedder = snippets_embedder
        self.prompt_template = load_prompt("adapt_code")

    def process(self, state: dict) -> dict:
        return state

    def _generate_header(self, header_config: Dict[str, Any]) -> str:
        pass

    def adapt_code(self, original_scenic_code: str, evaluation_result: Dict[str, Any], aim_dsl: Dict[str, Any]) -> str:
        """
        Adapt the original scenic code to the aim DSL.
        Take the original scenic code, the evaluation result, and the aim DSL as input.
        Generate the new scenic code recursively, in each generation we put the previously generated scenic code as input to ensure compatibility.
        """

        # 1. get the components need to be modified in the aim DSL
        components_to_modify = {}
        for key, value in evaluation_result.items():
            if not value:
                components_to_modify[key] = aim_dsl[key] # get key and value from aim DSL
        

        # 2. search the vector store for similar snippets
        snippets = {}
        for key, value in components_to_modify.items():
            try:
                description_embedding = self.snippets_embedder.encode([{"text": value}])[0]
                similar_snippets = self.vector_store.similarity_search_snippets(description_embedding, k=3, component_type=key)
                snippets[key] = [item["code"] for item in similar_snippets if item.get("code")]
            except Exception as e:
                logger.error(f"Failed to encode description for {key}: {e}")
                snippets[key] = []
        print(snippets)

        # 3. format the prompt
        prompt_begin = self.prompt_template.format(original_scenic_code=original_scenic_code, aim_dsl=aim_dsl, aspects=to_safe_string(components_to_modify))

        few_shot_examples = "### REFERENCE SNIPPETS ###\n"
        for key, value in snippets.items():
            few_shot_examples += "Use the following verified examples to resolve the specific incompatibilities mentioned above. These represent the ground-truth syntax for the target DSL:\n\n"
            few_shot_example = f"""--- Aspect: {key} ---
            Verified DSL-compliant patterns:
            {to_safe_string(value)}
            """
            few_shot_examples += few_shot_example

        prompt_end = """### INSTRUCTIONS & CONSTRAINTS ###
        1. Adapt the original code to resolve all identified incompatibilities.
        2. Strictly mimic the syntax, structure, and logic demonstrated in the Reference Snippets.
        3. DO NOT hallucinate. You must only use APIs, keywords, and functions explicitly defined in the Target DSL Specification or the Reference Snippets.
        4. Ensure the final code is syntactically valid and logically sound for autonomous driving simulation.
        """

        output_format = """
        === OUTPUT FORMAT ===
        Output ONLY the raw, executable Scenic code.
        Do NOT wrap the output in markdown code blocks (e.g., absolutely no ```scenic or ```).
        Do NOT include any conversational text, introductions, or explanations before or after the code.
        """

        formatted_prompt = prompt_begin + few_shot_examples + prompt_end + output_format
        formatted_prompt = formatted_prompt.strip()
        response = self.llm_service.chat([{"role": "user", "content": formatted_prompt}])
        return response

    def debug_code(self, scenic_code: str, error_message: str) -> str:
        static_prompt = load_prompt("debug_scenic_code").format(scenic_code_to_debug=scenic_code, error_message=error_message)
        response = self.llm_service.chat([{"role": "user", "content": static_prompt}])
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
    original_scenic_code = """
    'description = "Ego vehicle performs an unprotected left turn, yielding to oncoming traffic '
                         'at signalized and non-signalized intersections."\n'
                         "param map = localPath('../../maps/Town05.xodr')\n"
                         "param carla_map = 'Town05'\n"
                         'model scenic.simulators.carla.model\n'
                         "MODEL = 'vehicle.lincoln.mkz_2017'\n"
                         "param weather = 'ClearNoon'\n"
                         '\n'
                         'intersection = Uniform(*filter(lambda i: any(m.type is ManeuverType.LEFT_TURN for m in '
                         'i.maneuvers), network.intersections))\n'
                         '\n'
                         'egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN and any(cm.type is '
                         'ManeuverType.STRAIGHT for cm in m.conflictingManeuvers), intersection.maneuvers))\n'
                         'egoInitLane = egoManeuver.startLane\n'
                         'egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]\n'
                         'egoSpawnPt = new OrientedPoint in egoInitLane.centerline\n'
                         '\n'
                         'advManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, '
                         'egoManeuver.conflictingManeuvers))\n'
                         'advInitLane = advManeuver.startLane\n'
                         'advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]\n'
                         'advSpawnPt = new OrientedPoint in advInitLane.centerline\n'
                         '\n'
                         'param OPT_EGO_SPEED = Range(3, 5)\n'
                         'param OPT_EGO_YIELD_DIST = Range(15, 20)\n'
                         'OPT_EGO_DECISION_DEGREE = 35 deg\n'
                         '\n'
                         'behavior EgoBehavior():\n'
                         '    initialDir = egoSpawnPt.heading\n'
                         '    try:\n'
                         '        do FollowTrajectoryBehavior(trajectory=egoTrajectory, '
                         'target_speed=globalParameters.OPT_EGO_SPEED)\n'
                         '    interrupt when withinDistanceToAnyCars(self, globalParameters.OPT_EGO_YIELD_DIST):\n'
                         '        currentDir = self.heading\n'
                         '        if (abs(currentDir - initialDir) < OPT_EGO_DECISION_DEGREE):\n'
                         '            take SetThrottleAction(0)\n'
                         '            take SetBrakeAction(1)\n'
                         '        else:\n'
                         '            do FollowTrajectoryBehavior(trajectory=egoTrajectory, '
                         'target_speed=globalParameters.OPT_EGO_SPEED + 2)\n'
                         '            abort\n'
                         '    terminate\n'
                         '\n'
                         'ego = new Car at egoSpawnPt,\n'
                         "    with rolename 'hero',\n"
                         '    with blueprint MODEL,\n'
                         '    with behavior EgoBehavior()\n'
                         '\n'
                         'param ADV_SPEED = Range(7, 10)\n'
                         '\n'
                         'behavior AdversaryBehavior(trajectory):\n'
                         '    do FollowTrajectoryBehavior(target_speed=globalParameters.ADV_SPEED, '
                         'trajectory=trajectory)\n'
                         '\n'
                         'adversary = new Car at advSpawnPt,\n'
                         '    with blueprint MODEL,\n'
                         '    with behavior AdversaryBehavior(advTrajectory)\n'
                         '\n'
                         'EGO_INIT_DIST = [10, 15]\n'
                         'ADV_INIT_DIST = [15, 25]\n'
                         'TERM_DIST = 50\n'
                         '\n'
                         'monitor TrafficLights():\n'
                         '    freezeTrafficLights()\n'
                         '    while True:\n'
                         '        if withinDistanceToTrafficLight(ego, 100):\n'
                         '            setClosestTrafficLightStatus(ego, "green")\n'
                         '        if withinDistanceToTrafficLight(adversary, 100):\n'
                         '            setClosestTrafficLightStatus(adversary, "green")\n'
                         '        wait\n'
                         '\n'
                         'require monitor TrafficLights()\n'
                         'require EGO_INIT_DIST[0] <= (distance from egoSpawnPt to intersection) <= EGO_INIT_DIST[1]\n'
                         'require ADV_INIT_DIST[0] <= (distance from advSpawnPt to intersection) <= ADV_INIT_DIST[1]\n'
                         'terminate when (distance from ego to egoSpawnPt) > TERM_DIST'
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
    
    adapted_scenic_code = agent.adapt_code(original_scenic_code, evaluation_result, aim_dsl)
    print(adapted_scenic_code)