from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

from google.genai import types

from src.config import get_config
from src.services import get_vlm_service
from src.utils.helpers import clean_and_parse_json
from src.utils.logger import setup_logging

PROMPT = """
You are an expert autonomous driving scenario analyzer. Your task is to extract the high-level logical structure of a driving scenario by analyzing the provided text description, the provided image, and the attached video of the scenario.

Synthesize all these inputs to understand the scenario's layout and dynamic timeline, then convert it into a structured JSON format.

** Output Format **
Output ONLY a valid JSON object with the following structure. Do NOT wrap the output in markdown code blocks (e.g., no ```json) and do NOT include any conversational text.
    Scenario: [A single short sentence describing the overall scenario event.]

    Ego Vehicle: [Name of the Ego object]: [A short description of the behavior of the Ego object]

    Adversarial Entities: [Name of the Adversarial object]: [A short description of the behavior]

    Spatial Relation: [A single short sentence defining the road type and the relative positioning of all entities.]

    Requirements and Restrictions: [Describe any initial distance constraints and termination conditions.]

    Road Side Structures: [Name of object]: [Relative position]

    Temporary Modifications: [Name of object]: [Relative position]

    Reasoning Chain: [Write your step-by-step analysis here, covering the 3 phases and identifying the ego vehicle in 3 short sentences, max 40 words.]

** Strict Rules to Follow **
- Cross-Reference Modalities: Use the image or video to visually confirm the static entities, road types, and initial spatial layouts.
- Use the video to confirm the dynamic behaviors and trajectories of the Ego and Adversarial objects over time.
- Subject-First Descriptions: Start descriptions with the subject (e.g., "The ego vehicle travels...", "A debris object remains...").
- Separation of Concerns: Exclude spatial relations from the "Ego" and "Adversarials" components. Spatial data belongs ONLY in the "Spatial Relation" component.
- Adversarials Parsing: Create a separate array entry for each distinct adversarial object. Combine its type (from the allowed list) and behavior in a single sentence.
- Traffic Lights: Any mention of traffic lights (e.g., red lights) must be placed in the "Requirement and restrictions" component, NOT in the Ego or Adversarials components.
- Quantitative Details Only allowed for road side structures and temporary modifications: Do NOT include specific numbers, distances, speeds, or durations for all other perspectives, but can be included in the <road_side_structures> and <temporary_modifications> (e.g., say "a certain distance" instead of "50 meters").

** Allowed Ego Objects **
Car, Motorcycle, Truck.

** Allowed Adversarial Objects **
Car, NPCCar, Bicycle, Motorcycle, Truck, Pedestrian, Trash, Garbage, Container, CreasedBox, Case, Box, Gnome.

** Allowed Road Side Structure Objects **
Barrel, Bin, Box, Container, CreasedBox, Trashcan, Bench, Garden Lamp, Pergola, Plastic Chair, Table, Slide, Swing, Trampoline, Barbeque, Clothesline, Doghouse, Gnome, Watering Can, Haybale, Plantpot, Shopping Cart, Briefcase, Travelcase, Chainbarrier, Bench, Foodcart, Kiosk, Fountain, Maptable, Advertisement, Street Sign, Busstop, ATM, Mailbox, Vending Machine.

** Allowed Temporary Modifications Objects**
Street Barrier, Construction Cone, Traffic Cone, Iron Plank, Traffic Warning, Accident Warning Sign, Construction Warning Sign, Trash Bag, Cola Can, Garbage, Broken Tile, Dirt Debris.

** Reasoning and Implementation plan **
1. Identify the Actors: 
You must explicitly classify all dynamic entities in the scene as either the Ego vehicle or an Adversary vehicle based on the following strict hierarchy:
    - Explicit Labels: If the user text explicitly numbers or letters vehicles (e.g., "Vehicle A", "Vehicle B"), assign "Vehicle A" as the Ego and all others as Adversaries.
    - Camera Perspective: If no explicit labels exist, but the video/image is shot from a first-person dashboard or hood perspective, the camera-host vehicle is the Ego.
    - Spatial Inference: If the perspective is a third-person drone or traffic camera view, the vehicle located nearest to the exact center of the frame is the Ego.
    - Default Classification: Once the Ego is identified, all other active vehicles, pedestrians, or dynamic entities MUST be classified as Adversaries.

2. Conduct a Chronological Analysis: 
Break the video sequence into three strict temporal phases: Initial (start), Midpoint, and Ending. Track the exact spatial positions of the Ego and adversarials during each phase. By directly comparing how their relative positions change across these three distinct frames, deduce their dynamic behaviors (e.g., accelerating, yielding, remaining stationary).

3. Ground Conclusions in Visual Evidence: 
You must explicitly state the visual evidence that justifies your conclusions regarding dynamic behavior. Do not guess; cite specific spatial cues (e.g., "The adversarial vehicle is decelerating because its distance to the intersection visibly decreases between the Midpoint and Ending phases").
"""

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIOS_PATH = REPO_ROOT / "data" / "test"
OUTPUT_FILENAME = "text_query_description.txt"


def find_bev_video(subfolder: Path) -> Path | None:
    for file_path in subfolder.iterdir():
        if file_path.is_file() and file_path.name.lower() == "bev.mp4":
            return file_path
    return None


def _is_reasoning_chain_key(key: str) -> bool:
    return key.strip().lower().replace("_", " ") == "reasoning chain"


def remove_reasoning_chain(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if not _is_reasoning_chain_key(key)}


def _flatten_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if not value:
            return ""
        parts = []
        for key, nested in value.items():
            nested_text = _flatten_value(nested)
            if nested_text:
                parts.append(f"{key}: {nested_text}")
        return "; ".join(parts)
    if isinstance(value, list):
        parts = [_flatten_value(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value).strip()


def flatten_vlm_response_to_text(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        flat_value = _flatten_value(value)
        if flat_value:
            lines.append(f"{key}: {flat_value}")
    return "\n".join(lines)


def format_vlm_response_for_save(response: str | dict[str, Any]) -> str:
    if isinstance(response, dict):
        data = response
    elif isinstance(response, str):
        parsed = clean_and_parse_json(response)
        if not parsed:
            raise ValueError(f"Failed to parse VLM response as JSON: {response[:200]!r}")
        data = parsed
    else:
        raise ValueError(f"Unexpected VLM response type: {type(response).__name__}")

    cleaned = remove_reasoning_chain(data)
    text = flatten_vlm_response_to_text(cleaned).strip()
    if not text:
        raise ValueError("VLM response became empty after formatting")
    return text


def describe_scenario_from_video(
    vlm_service,
    video_path: Path,
    description: str,
    prompt: str = PROMPT,
) -> str:
    video = str(video_path.resolve())
    description = description.strip()

    contents = [types.Part.from_text(text="** Inputs **")]
    if description:
        contents.append(types.Part.from_text(text=f"Scenario Description Text: {description}"))
    contents.append(types.Part.from_text(text="Scenario Video: "))
    video_file = vlm_service.load_media(video)
    contents.append(types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type))

    response = vlm_service.chat_with_content(contents=contents, system_instruction=prompt)
    if isinstance(response, (str, dict)):
        return format_vlm_response_for_save(response)
    raise ValueError(f"Unexpected or empty VLM response: {response!r}")


def generate_text_query_descriptions(
    scenarios_path: Path = DEFAULT_SCENARIOS_PATH,
    config_path: Optional[str] = None,
    skip_existing: bool = True,
    prompt: str = PROMPT,
) -> dict[str, list[str]]:
    """
    Generate text_query_description.txt for each scenario subfolder from BEV.mp4.

    Returns:
        Dict with keys ``processed``, ``skipped``, and ``failed``.
    """
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Scenarios folder does not exist: {scenarios_path}")
    if not scenarios_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {scenarios_path}")

    config = get_config(config_path)
    setup_logging(
        level=config.logging.level,
        log_format=config.logging.format,
    )
    logger = logging.getLogger(__name__)

    vlm_kwargs = {
        "provider": config.vlm.provider,
        "model": config.vlm.model,
        "temperature": config.vlm.temperature,
        "max_tokens": config.vlm.max_tokens,
        "api_key": config.vlm.api_key,
    }
    if config.vlm.model_path:
        vlm_kwargs["model_path"] = config.vlm.model_path
    vlm_service = get_vlm_service(**vlm_kwargs)

    processed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for subfolder in sorted(scenarios_path.iterdir()):
        if not subfolder.is_dir():
            continue

        output_path = subfolder / OUTPUT_FILENAME
        if skip_existing and output_path.is_file():
            skipped.append(subfolder.name)
            logger.info("Skip existing output: %s", subfolder.name)
            continue

        bev_path = find_bev_video(subfolder)
        description_path = subfolder / "description.txt"
        if bev_path is None:
            failed.append(subfolder.name)
            logger.warning("Missing BEV.mp4: %s", subfolder.name)
            continue
        if not description_path.is_file():
            failed.append(subfolder.name)
            logger.warning("Missing description.txt: %s", subfolder.name)
            continue

        try:
            description_text = description_path.read_text(encoding="utf-8")
            response = describe_scenario_from_video(
                vlm_service=vlm_service,
                video_path=bev_path,
                description=description_text,
                prompt=prompt,
            )
            output_path.write_text(response.strip(), encoding="utf-8")
            processed.append(subfolder.name)
            logger.info("Saved text query description: %s", subfolder.name)
        except Exception as exc:
            failed.append(subfolder.name)
            logger.exception("Failed to process %s: %s", subfolder.name, exc)

    if failed:
        logger.error("Failed subfolders (%d): %s", len(failed), ", ".join(failed))

    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text_query_description.txt from BEV.mp4 for each scenario."
    )
    parser.add_argument(
        "--scenarios-path",
        type=Path,
        default=DEFAULT_SCENARIOS_PATH,
        help="Path to data/scenarios directory",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to config YAML",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Regenerate even when text_query_description.txt already exists",
    )
    args = parser.parse_args()

    result = generate_text_query_descriptions(
        scenarios_path=args.scenarios_path,
        config_path=args.config_path,
        skip_existing=not args.no_skip_existing,
    )
    print(
        f"Done. processed={len(result['processed'])}, "
        f"skipped={len(result['skipped'])}, failed={len(result['failed'])}"
    )
    if result["failed"]:
        print(f"Failed subfolders ({len(result['failed'])}):")
        for name in result["failed"]:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
