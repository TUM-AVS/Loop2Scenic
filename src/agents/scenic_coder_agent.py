import logging
import re
import time
from src.schema import HeaderSetting, get_comparison_evaluation
from src.schema.header_setting import FOG_FREE_NIGHT_WEATHER, CUSTOM_NIGHT_WEATHER, WeatherParam
from src.utils import clean_and_parse_json, setup_logging, strip_code_fence_markers
from tests.test_utils import test_video_recording

from .base_agent import BaseAgent
from src.services import BaseLLMModel, MilvusVectorStore, BaseEmbeddingModel, get_embedder, get_llm_service
from src.prompt import load_prompt
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Match param weather = 'X' or a single-level { ... } block (possibly multiline).
_WEATHER_PARAM_RE = re.compile(
    r"(?ms)^\s*param\s+weather\s*=\s*(?:\{[^{}]*\}|'[^']*'|\"[^\"]*\")",
)

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

# Aspect → component_generator_* stem (without .txt)
_COMPONENT_PROMPT_STEMS = {
    "adversarials": "component_generator_adv",
    "ego": "component_generator_ego",
    "requirements_and_restrictions": "component_generator_requirement",
    "spatial_relation": "component_generator_spatial",
    "road_side_structures": "component_generator_road_side_structure",
    "temporary_modifications": "component_generator_temporary_modification",
}


def _load_codegen_flags(prompt_group: str) -> dict[str, bool]:
    """Read FLAGS.txt from a gen_eval group; legacy root → all factors on."""
    from pathlib import Path

    defaults = {"cp": True, "cot": True, "icl": True, "snippets": True}
    if not prompt_group or not prompt_group.strip():
        return defaults

    prompts_dir = Path(__file__).resolve().parent.parent / "prompt"
    flags_path = prompts_dir / prompt_group.strip().lstrip("/") / "FLAGS.txt"
    if not flags_path.is_file():
        logger.warning("No FLAGS.txt for codegen group %r; using all-on defaults", prompt_group)
        return defaults

    parsed = dict(defaults)
    for line in flags_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower()
        if key in parsed:
            parsed[key] = val.strip() in ("1", "true", "True", "yes")
    return parsed


class ScenicCoderAgent(BaseAgent):
    def __init__(
        self,
        llm_service: BaseLLMModel,
        vector_store: MilvusVectorStore,
        snippets_embedder: BaseEmbeddingModel,
        prompt_group: str = "",
        use_contextual: bool | None = None,
        use_cot: bool | None = None,
        use_icl: bool | None = None,
        use_snippet_retrieval: bool | None = None,
    ):
        super().__init__()
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.snippets_embedder = snippets_embedder
        self.prompt_template = load_prompt("adapt_code")
        self.last_debug_failure: str | None = None
        self.prompt_group = (prompt_group or "").strip().strip("/")
        flags = _load_codegen_flags(self.prompt_group)
        self.use_contextual = flags["cp"] if use_contextual is None else bool(use_contextual)
        self.use_cot = flags["cot"] if use_cot is None else bool(use_cot)
        self.use_icl = flags["icl"] if use_icl is None else bool(use_icl)
        self.use_snippet_retrieval = (
            flags["snippets"] if use_snippet_retrieval is None else bool(use_snippet_retrieval)
        )
        logger.info(
            "ScenicCoderAgent codegen ablation: group=%r cp=%s cot=%s icl=%s snippets=%s",
            self.prompt_group or "(legacy root)",
            self.use_contextual,
            self.use_cot,
            self.use_icl,
            self.use_snippet_retrieval,
        )

    def process(self, state: dict) -> dict:
        return state

    @staticmethod
    def _format_weather_assignment(weather: WeatherParam) -> str:
        """Format `param weather = ...` as a CARLA preset string or fog-free night dict."""
        if isinstance(weather, dict):
            items = ",\n    ".join(f"'{k}': {float(v)}" for k, v in weather.items())
            return f"param weather = {{\n    {items},\n}}"
        return f"param weather = '{weather}'"

    @staticmethod
    def _weather_label(weather: WeatherParam) -> str:
        if isinstance(weather, dict):
            return CUSTOM_NIGHT_WEATHER
        return str(weather)

    @classmethod
    def _resolve_weather_param(cls, weather: Any) -> WeatherParam:
        """Normalize weather: dict stays; CustomNight / *Night → fog-free dict."""
        if isinstance(weather, dict) and weather:
            return {k: float(v) for k, v in weather.items()}
        text = cls._coerce_header_field(weather, default="ClearNoon")
        lower = text.lower()
        if (
            lower == CUSTOM_NIGHT_WEATHER.lower()
            or lower.endswith("night")
            or "night" in lower
        ):
            return dict(FOG_FREE_NIGHT_WEATHER)
        return text

    def replace_header(self, scenic_code: str, header_settings: HeaderSetting) -> str:
        """
        Replace the header of the scenic code with the header settings.
        Used after retrieving the base scenario from the vector store.
        """
        map_file_path = self._coerce_header_field(
            header_settings.map_file_path, default="../../maps/Town05.xodr"
        )
        carla_map = self._coerce_header_field(header_settings.carla_map, default="Town05")
        blueprint = self._coerce_header_field(
            header_settings.blueprint, default="vehicle.lincoln.mkz_2017"
        )
        weather = self._resolve_weather_param(header_settings.weather)
        weather_assignment = self._format_weather_assignment(weather)
        replacements = [
            (r"(?m)^\s*param\s+map\s*=.*$", f"param map = localPath('{map_file_path}')"),
            (r"(?m)^\s*param\s+carla_map\s*=.*$", f"param carla_map = '{carla_map}'"),
            (r"(?m)^\s*MODEL\s*=.*$", f"MODEL = '{blueprint}'"),
        ]
        for pattern, repl in replacements:
            if re.search(pattern, scenic_code):
                scenic_code = re.sub(pattern, repl, scenic_code, count=1)
            else:
                scenic_code += "\n" + repl + "\n"
        if _WEATHER_PARAM_RE.search(scenic_code):
            scenic_code = _WEATHER_PARAM_RE.sub(weather_assignment, scenic_code, count=1)
        else:
            scenic_code += "\n" + weather_assignment + "\n"
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
        # if some of the fields are not provided, use the default values
        map_file_path = self._coerce_header_field(
            header_settings.map_file_path, default="../../maps/Town05.xodr"
        )
        carla_map = self._coerce_header_field(header_settings.carla_map, default="Town05")
        blueprint = self._coerce_header_field(
            header_settings.blueprint, default="vehicle.lincoln.mkz_2017"
        )
        weather = self._resolve_weather_param(header_settings.weather)
        weather_label = self._weather_label(weather)
        weather_assignment = self._format_weather_assignment(weather)
        header = f"""
description = "Using map { map_file_path } with carla map { carla_map } and weather {weather_label}"
param map = localPath('{map_file_path}')
param carla_map = '{carla_map}'
model scenic.simulators.carla.model
MODEL = '{blueprint}'
{weather_assignment}
        """
        return header

    @staticmethod
    def _coerce_header_field(value: Any, *, default: str) -> str:
        """Treat null / empty / literal 'None' as missing so CARLA never gets weather None."""
        if value is None:
            return default
        if isinstance(value, dict):
            return default
        text = str(value).strip()
        if not text or text.lower() in {"none", "null", "n/a", "na", "unknown"}:
            return default
        return text

    @staticmethod
    def _component_desc_to_text(description: Any) -> str:
        """Serialize a DSL component description (str, dict, or list of dicts) into embed/prompt text."""
        if description is None:
            return ""
        if isinstance(description, str):
            return description
        if isinstance(description, list):
            parts = [
                ScenicCoderAgent._component_desc_to_text(item)
                for item in description
                if item is not None and item != ""
            ]
            return "\n".join(f"- {part}" for part in parts if part)
        if isinstance(description, dict):
            obj = description.get("object", "") or ""
            detail = description.get("behavior") or description.get("position") or ""
            text = f"{obj}: {detail}".strip(": ").strip()
            if text:
                return text
            return " ".join(str(v) for v in description.values() if v)
        return str(description)

    @staticmethod
    def _list_aspect_all_matched(eval_flags: Any) -> bool:
        """True iff the whole list component can be extracted (every item matched)."""
        if eval_flags is None or eval_flags == "":
            return False
        if isinstance(eval_flags, bool):
            return eval_flags
        if isinstance(eval_flags, list):
            if not eval_flags:
                return False
            return all(bool(flag) for flag in eval_flags)
        return bool(eval_flags)

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
            query_text = self._component_desc_to_text(text)
            query_embedding = self.snippets_embedder.encode([{"text": query_text}])[0]
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
        stem = _COMPONENT_PROMPT_STEMS.get(aspect)
        if stem is None:
            logger.error(f"Invalid aspect: {aspect}")
            stem = "component_generator_ego"
        prompt_name = f"{self.prompt_group}/{stem}" if self.prompt_group else stem
        return load_prompt(prompt_name)

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
        description = self._component_desc_to_text(description)

        if self.use_snippet_retrieval:
            snippets = self.get_snippets(text=description, comp_type=aspect)
            reference_components = self.prepare_snippets(snippets)
        else:
            reference_components = ""

        if self.use_contextual:
            context = self.build_context(retrieved_components)
        else:
            context = ""

        comp_prompt = self.get_prompt_for_component(aspect).format(
            reference_components=reference_components,
            ready_components=context,
            user_criteria=description,
        )
        new_code = self.generate_and_clean(comp_prompt)
        return new_code

    def extract_component(self, aspect: str, description: str, original_scenic_code: str, retrieved_components: Dict[str, Any]) -> str:
        """
        Extract a component based on the original scenic code and the retrieved components.
        """
        description = self._component_desc_to_text(description)
        context = self.build_context(retrieved_components)
        extract_prompt = load_prompt("component_generator_extract").format(
            aspect=aspect, 
            description=description, 
            context=context, 
            original_scenic_code=original_scenic_code)
        extracted = self.generate_and_clean(extract_prompt)
        return extracted

    def adapt_code(
        self,
        original_scenic_code: str,
        evaluation_result: Dict[str, Any],
        aim_dsl: Dict[str, Any],
        header_settings: Any,
        force_generate_all: bool = False,
    ) -> str:
        """
        Adapt the original scenic code to the aim DSL using a 'Generation + Assemble' architecture.
        Each component is generated in isolation (to allow for heavy grammar instructions) 
        and then assembled into the final script.

        If force_generate_all is True, every DSL component is regenerated (no extract from
        the base scenario). Used when the first-round critic score is too low.

        List components (adversarials / road_side_structures / temporary_modifications):
        - Generate path: create each DSL item one-by-one, then aggregate into one component string.
        - Extract path: extract the whole component once from the base scenic code.

        ``evaluation_result`` may be either:
        - the full critic payload (``observed_dsl`` / ``evaluation`` / ``kpi_matches`` / ...), or
        - a legacy flat comparison dict (``scenario`` / ``ego`` / ``adversarials`` / ...).
        """
        logger.info("🚀 Starting Generation + Assemble Scenic pipeline...")
        if force_generate_all:
            logger.info(
                "🔁 force_generate_all=True: regenerating all components from DSL "
                "(skipping extract from base scenario)"
            )

        comparison = get_comparison_evaluation(evaluation_result)

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
        
        for aspect in generation_order:
            logger.info(f"🧩 Processing Component: {aspect}")
            if aspect in list_aspects:
                # DSL schema: Optional[List[Adversarial|RoadSideStructure|TemporaryModification]]
                # each item is a dict with object + behavior/position.
                raw_descriptions = aim_dsl.get(aspect)
                if raw_descriptions is None or raw_descriptions == "":
                    description_list: list = []
                elif isinstance(raw_descriptions, list):
                    description_list = [
                        item for item in raw_descriptions if item is not None and item != ""
                    ]
                elif isinstance(raw_descriptions, dict):
                    # Defensive: VLM sometimes returns a single object instead of a 1-element list.
                    description_list = [raw_descriptions]
                else:
                    description_list = [raw_descriptions]

                if not description_list:
                    logger.info(f"⏭️ Skipping {aspect}: DSL has no content for this component")
                    continue

                if force_generate_all:
                    needs_modification = True
                else:
                    needs_modification = not self._list_aspect_all_matched(
                        (comparison or {}).get(aspect)
                    )

                if needs_modification:
                    # Generate each list item one-by-one, then aggregate into one component.
                    piece_codes: list[str] = []
                    for idx, description in enumerate(description_list):
                        # Expose already-generated siblings so later items can reference them.
                        retrieved_components[aspect] = piece_codes
                        new_piece = self.generate_component(
                            aspect, description, retrieved_components
                        )
                        if new_piece:
                            piece_codes.append(new_piece)
                            logger.info(
                                "✅ Successfully generated %s item %d/%d",
                                aspect,
                                idx + 1,
                                len(description_list),
                            )
                        else:
                            logger.warning(
                                "⚠️ LLM returned empty code for %s item %d/%d",
                                aspect,
                                idx + 1,
                                len(description_list),
                            )
                    new_code = "\n\n".join(piece_codes) if piece_codes else ""
                    action = "generated"
                else:
                    # Extract the whole component once from the original scenic code.
                    new_code = self.extract_component(
                        aspect, description_list, original_scenic_code, retrieved_components
                    )
                    action = "extracted"

                if new_code:
                    retrieved_components[aspect] = new_code
                    logger.info(
                        "✅ Successfully %s aggregated %s (%d DSL item(s))",
                        action,
                        aspect,
                        len(description_list),
                    )
                else:
                    logger.warning(f"⚠️ Empty aggregated code for {aspect}")
                    retrieved_components.pop(aspect, None)
                continue

            # Scalar aspects (spatial_relation, ego, requirements_and_restrictions)
            target_description = aim_dsl.get(aspect, "")

            # The DSL explicitly allows leaving requirements_and_restrictions empty
            # when nothing applies. Skip this component entirely rather than asking
            # the LLM to invent requirements from nothing (risk of over-constraining
            # the scenario / RejectionException).
            if aspect == "requirements_and_restrictions":
                is_empty = (
                    target_description is None
                    or (isinstance(target_description, str) and not target_description.strip())
                    or (isinstance(target_description, dict) and not any(str(v).strip() for v in target_description.values()))
                )
                if is_empty:
                    logger.info(f"⏭️ Skipping {aspect}: DSL has no content for this component")
                    continue

            if force_generate_all:
                needs_modification = True
            else:
                needs_modification = not (comparison or {}).get(aspect, True)
            new_code = None
            if needs_modification:
                new_code = self.generate_component(aspect, target_description, retrieved_components)
                logger.info(f"✅ Successfully generated {aspect}")
            else:
                new_code = self.extract_component(aspect, target_description, original_scenic_code, retrieved_components)
                logger.info(f"✅ Successfully extracted {aspect}")
            if new_code:
                retrieved_components[aspect] = new_code
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
                # Backward-compatible: older per-item lists, if any remain.
                for item in comp_data:
                    if item and str(item).strip():
                        code_parts.append(str(item).strip())
            else:
                if comp_data and str(comp_data).strip():
                    code_parts.append(str(comp_data).strip())
                    
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

    def debug_code(self, scenic_code: str, error_message: str) -> str:
        # 1. Get exact what is the error component
        error_type_classification = self.generate_and_clean(load_prompt("debug_error_type_classification").format(
            scenic_code=scenic_code,
            error_message=error_message
        ))
        error_type_classification = clean_and_parse_json(error_type_classification)
        logger.info(f"Error type classification: {error_type_classification}")
        error_component = error_type_classification.get("error_component", "")
        logger.info(f"Error component: {error_component}")

        allowed_debug_components = ["spatial_relation", 
                                    "ego", 
                                    "adversarials", 
                                    "road_side_structures", 
                                    "temporary_modifications",
                                    "requirements_and_restrictions",]
        if error_component not in allowed_debug_components:
            logger.error(f"Invalid error component: {error_component}")
            return scenic_code # if the error component is not in the allowed list, return the original scenic code
        
        # 2. fix the bug using specific prompt
        full_code = None
        prompt = load_prompt(f"debug_{error_component}").format(
            scenic_code=scenic_code,
            error_message=error_message
        )
        raw_result = self.generate_and_clean(prompt)

        # Preferred path: extract full Scenic script from explicit delimiters.
        delimiter_match = re.search(
            r"<<<FULL_SCENIC_CODE_BEGIN>>>\s*([\s\S]*?)\s*<<<FULL_SCENIC_CODE_END>>>",
            raw_result or "",
        )
        if delimiter_match:
            full_code = delimiter_match.group(1).strip()
            logger.info("Extracted full scenic code from delimiter block.")
        else:
            # Backward-compatible fallback for older JSON-style prompts.
            result_json = clean_and_parse_json(raw_result)
            logger.info(f"Result JSON: {result_json}")
            full_code = result_json.get("full_code", "") if result_json else ""

        if full_code:
            self.last_debug_failure = None
            return full_code

        # Do not abort the workflow: keep the previous script so the graph can
        # continue and eventually fall back to the baseline in output_best_scenario.
        # Record the failure so callers/eval CSV can still surface it.
        self.last_debug_failure = f"Failed to fix the bug for {error_component}"
        logger.error(
            "%s; returning the previous scenic code unchanged",
            self.last_debug_failure,
        )
        return scenic_code

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