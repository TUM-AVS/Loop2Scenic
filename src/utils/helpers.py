"""
Helper utility functions.
"""

import logging
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional
import tiktoken
import json
import os
import subprocess

from src.schema import ScenarioDocument

logger = logging.getLogger(__name__)


def ensure_directory(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = len(encoding.encode(text))
    return num_tokens


def truncate_text(
    text: str,
    max_tokens: int,
    model: str = "gpt-3.5-turbo",
    suffix: str = "..."
) -> str:
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name for tokenizer
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate and decode
    truncated_tokens = tokens[:max_tokens - len(encoding.encode(suffix))]
    truncated_text = encoding.decode(truncated_tokens) + suffix
    
    return truncated_text


def format_metadata(metadata: dict, indent: int = 2) -> str:
    """
    Format metadata dictionary as readable string.
    
    Args:
        metadata: Metadata dictionary
        indent: Indentation spaces
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metadata.items():
        lines.append(f"{' ' * indent}{key}: {value}")
    return "\n".join(lines)


def to_safe_string(value: Any) -> str:
    """
    Convert arbitrary Python objects to a safe string representation suitable for logs/UI.

    Handles:
    - Pydantic models (BaseModel): uses model_dump_json() or model_dump()
    - Plain dict/list/tuple/set: JSON serialize (sets converted to lists)
    - Path objects: str(path)
    - bytes: UTF-8 decode with fallback to repr
    - int/float/bool/str/None: str()
    - Fallback: str(value) with error handling

    Args:
        value: Any python object

    Returns:
        String representation
    """
    try:
        # Fast path for strings
        if isinstance(value, str):
            return value

        # Bytes → decode utf-8 with fallback
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray)) else str(value)
            except Exception:
                return repr(value)

        # Path-like
        if isinstance(value, Path):
            return str(value)

        # Numerics / bool / None
        if isinstance(value, (int, float, bool)) or value is None:
            return str(value)

        # Pydantic BaseModel (v2) interface
        if hasattr(value, "model_dump_json") and callable(getattr(value, "model_dump_json")):
            try:
                return value.model_dump_json()
            except Exception:
                pass
        if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
            try:
                return json.dumps(value.model_dump(), ensure_ascii=False)
            except Exception:
                pass

        # Collections → JSON (convert sets/tuples)
        if isinstance(value, (dict, list, tuple, set)):
            try:
                json_ready = value
                if isinstance(value, set):
                    json_ready = list(value)
                elif isinstance(value, tuple):
                    json_ready = list(value)
                return json.dumps(json_ready, ensure_ascii=False)
            except Exception:
                # Fall through to generic str
                return str(value)

        # Fallback
        return str(value)
    except Exception as e:
        logger.info(f"to_safe_string fallback due to error: {e}")
        try:
            return repr(value)
        except Exception:
            return "<unserializable>"


def find_scenic_code_with_scenario_id(scenario_id: str) -> str: 
    """
    Find the scenic code for a given scenario ID.
    """
    try:
        scenario_location = f"data/scenarios/{scenario_id}/code.scenic"
        with open(scenario_location, "r") as f:
            scenic_code = f.read()
            return scenic_code
    except FileNotFoundError:
        logger.error(f"Scenario code not found for ID: {scenario_id}")
        return None
    except Exception as e:
        logger.error(f"Error finding scenic code for ID: {scenario_id}: {e}")
        return None

def get_scenario_document_with_scenario_id(scenario_id: str) -> ScenarioDocument:
    """
    Get a local scenario document for a given scenario ID.
    """
    try:
        scenario_location = Path(f"data/scenarios/{scenario_id}").resolve()
        scenario_description = scenario_location / "description.txt"
        scenario_scenic_code = scenario_location / "code.scenic"
        scenario_image = scenario_location / "image.png"
        scenario_video = scenario_location / "video.mp4"

        if scenario_description.exists() and scenario_description.is_file():
            scenario_description = scenario_description.read_text(encoding="utf-8")
        else:
            scenario_description = None

        if scenario_scenic_code.exists() and scenario_scenic_code.is_file():
            scenario_scenic_code = scenario_scenic_code.read_text(encoding="utf-8")
        else:
            scenario_scenic_code = None

        if scenario_image.exists() and scenario_image.is_file():
            image_path = str(scenario_image.resolve())
        else:
            image_path = None

        if scenario_video.exists() and scenario_video.is_file():
            video_path = str(scenario_video.resolve())
        else:
            video_path = None

        return ScenarioDocument(
            scenario_id=scenario_id,
            description=scenario_description,
            scenic_code=scenario_scenic_code,
            image_path=image_path,
            video_path=video_path
        )
    except Exception as e:
        logger.error(f"Error getting scenario document for ID: {scenario_id}: {e}")
        return None

def run_simulation_in_carla_and_save_video(scenic_code: str, scenario_id: str = "test_scenario") -> str:
    """
    Run the simulation in Carla and save the video.
    """
    # 1. save scenic code to a file
    # clean the temp/{scenario_id} directory if exists
    if os.path.exists(f"temp/{scenario_id}"):
        shutil.rmtree(f"temp/{scenario_id}")
    os.makedirs(f"temp/{scenario_id}", exist_ok=True)
    # create temp/{scenario_id}/code directory if not exists
    os.makedirs(f"temp/{scenario_id}/code", exist_ok=True)
    os.makedirs(f"temp/{scenario_id}/video", exist_ok=True)
    os.makedirs(f"temp/{scenario_id}/logs", exist_ok=True)
    with open(f"temp/{scenario_id}/code/scenic_code.scenic", "w") as f:
        f.write(scenic_code)

    # 2. run simulation and save the video
    result = subprocess.run(['src/utils/run_scenic_batch.sh', f"temp/{scenario_id}/code", '--outdir', f"temp/{scenario_id}/video", '--logdir', f"temp/{scenario_id}/logs"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to run simulation: {result.stderr}")
        return None
    # check if video really exists
    if not os.path.exists(os.path.join(f"temp/{scenario_id}/video", 'BEV.mp4')):
        logger.error(f"Video not found at {os.path.join(f"temp/{scenario_id}/video", 'BEV.mp4')}")
        return None
    video_path = os.path.join(f"temp/{scenario_id}/video", 'BEV.mp4')
    return video_path

def get_error_message_from_logs(scenario_id: str) -> Optional[str]:
    """
    Return the text block between:
      "=== RECORDING START ==="
    and
      "=== CARLA SETTINGS (post-scenic) ==="
    from the first matching log file under temp/{scenario_id}/logs.
    """
    log_dir = f"temp/{scenario_id}/logs"
    if not os.path.isdir(log_dir):
        return None
    # More permissive than only digits, matches files like:
    # scenario_id__20260326.log / scenario_id__code__...log
    pattern = re.compile(rf"^{re.escape(scenario_id)}.*\.log$")
    for file in sorted(os.listdir(log_dir)):
        if not pattern.match(file):
            continue
        path = os.path.join(log_dir, file)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if start_idx is None and "=== RECORDING START ===" in line:
                start_idx = i
                continue
            if start_idx is not None and "=== CARLA SETTINGS (post-scenic) ===" in line:
                end_idx = i
                break
        if start_idx is None:
            continue
        # If end marker missing, return everything after start marker.
        block = lines[start_idx + 1:end_idx] if end_idx is not None else lines[start_idx + 1:]
        text = "".join(block).strip()
        return text if text else None
    return None

def strip_code_fence_markers(raw_text: str) -> str:
    """
    Remove markdown code-fence wrappers and language tags from text.

    Examples removed:
    - ```json ... ```
    - ```python ... ```
    - ```scenic ... ```
    """
    text = raw_text.strip()
    fence_match = re.match(r"^```[a-zA-Z0-9_+-]*\s*([\s\S]*?)\s*```$", text)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def parse_raw_text_to_json_dict(raw_text: str | Dict[str, Any] | List[Any]) -> Dict[str, Any] | None:
    """
    Parse raw text/object into a JSON dictionary.
    """
    if isinstance(raw_text, dict):
        return raw_text

    if isinstance(raw_text, list):
        logger.error("Expected JSON object (dict), but got JSON array (list).")
        return None

    cleaned_text = strip_code_fence_markers(raw_text)

    try:
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, dict):
            return parsed
        logger.error(f"Expected JSON object (dict), but got {type(parsed).__name__}.")
        return None
    except json.JSONDecodeError:
        match = re.search(r"[\{\[][\s\S]*[\}\]]", cleaned_text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
                logger.error(f"Expected JSON object (dict), but got {type(parsed).__name__}.")
                return None
            except Exception as e:
                logger.error(f"Failed to parse extracted JSON text: {e}. Raw text: {raw_text}")
                return None
        logger.error(f"Failed to find JSON in text: {raw_text}")
        return None
    except Exception as e:
        logger.error(f"Failed to parse JSON from text: {e}. Raw text: {raw_text}")
        return None


def clean_and_parse_json(raw_text: str | Dict[str, Any] | List[Any]) -> Dict[str, Any] | None:
    """
    Clean and parse the raw text as a JSON object.
    """

    return parse_raw_text_to_json_dict(raw_text)