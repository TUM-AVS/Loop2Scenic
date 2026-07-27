"""
Helper utility functions.
"""

import logging
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import tiktoken
import json
import os
import signal
import subprocess

from src.config import get_config
from src.schema import ScenarioDocument
from src.utils.simulation import build_run_scenic_batch_command

if TYPE_CHECKING:
    from src.config import Config

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


def find_corpus_scenario_bev(scenario_id: str) -> Optional[Path]:
    """Return ``data/scenarios/<id>/BEV.mp4`` if it exists, else None."""
    if not scenario_id or "_adapted_" in scenario_id:
        return None
    from src.utils.simulation import resolve_path

    scenarios_root = (
        resolve_path(get_config().simulation.scenarios_data_dir) or "data/scenarios"
    )
    bev = Path(scenarios_root) / scenario_id / "BEV.mp4"
    return bev if bev.is_file() else None


def stage_corpus_bev_for_scenario(scenario_id: str) -> Optional[str]:
    """
    Copy a library BEV into ``temp/<id>/video/BEV.mp4`` so VLM / interpreter
    consumers keep using the same path as a live CARLA run.

    Returns the staged temp path on success, else None.
    """
    corpus_bev = find_corpus_scenario_bev(scenario_id)
    if corpus_bev is None:
        return None

    dest_dir = Path(f"temp/{scenario_id}/video")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "BEV.mp4"
    try:
        shutil.copy2(str(corpus_bev), str(dest))
    except OSError as exc:
        logger.error(
            "Failed to stage corpus BEV %s -> %s: %s", corpus_bev, dest, exc
        )
        return None
    logger.info("Staged corpus BEV for scenario %s: %s -> %s", scenario_id, corpus_bev, dest)
    return str(dest)

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


def _carla_kill_process_pattern(config: Optional["Config"] = None) -> str:
    config = config or get_config()
    return config.simulation.carla.kill_process_name_pattern


def _terminate_simulation_process_group(
    proc: subprocess.Popen,
    config: Optional["Config"] = None,
) -> None:
    """Stop run_scenic_batch.sh and its children (Scenic, recorder, same PG)."""
    carla_name_pattern = _carla_kill_process_pattern(config)

    def _list_descendant_pids(root_pid: int) -> set[int]:
        descendants: set[int] = set()
        parent_to_children: dict[int, list[int]] = {}
        proc_root = "/proc"
        try:
            for entry in os.listdir(proc_root):
                if not entry.isdigit():
                    continue
                stat_path = os.path.join(proc_root, entry, "stat")
                try:
                    with open(stat_path, "r", encoding="utf-8", errors="ignore") as f:
                        stat_line = f.read().strip()
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue

                close_idx = stat_line.rfind(")")
                if close_idx == -1:
                    continue
                after = stat_line[close_idx + 2 :].split()
                # /proc/<pid>/stat after "comm)" starts with state then ppid.
                if len(after) < 2:
                    continue
                try:
                    pid = int(entry)
                    ppid = int(after[1])
                except ValueError:
                    continue
                parent_to_children.setdefault(ppid, []).append(pid)
        except FileNotFoundError:
            return descendants

        stack = [root_pid]
        while stack:
            parent = stack.pop()
            for child in parent_to_children.get(parent, []):
                if child in descendants:
                    continue
                descendants.add(child)
                stack.append(child)
        return descendants

    def _signal_many(pids: set[int], sig: signal.Signals) -> None:
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                continue

    def _alive_pids(pids: set[int]) -> set[int]:
        alive: set[int] = set()
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.add(pid)
            except (ProcessLookupError, PermissionError):
                continue
        return alive

    def _is_carla_running_by_name() -> bool:
        try:
            result = subprocess.run(
                ["pgrep", "-f", carla_name_pattern],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _kill_carla_by_name() -> None:
        for sig in ("TERM", "KILL"):
            try:
                subprocess.run(
                    ["pkill", f"-{sig}", "-f", carla_name_pattern],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                continue
            if sig == "TERM":
                try:
                    proc.wait(timeout=3)
                except Exception:
                    pass

    if proc.poll() is not None:
        return
    if os.name == "posix":
        pgid: Optional[int] = None
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            pgid = None

        try:
            # Use resolved PGID (not always equal to proc.pid) and avoid killing our own group.
            if pgid is not None and pgid != os.getpgrp():
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except (ProcessLookupError, PermissionError):
            proc.terminate()

        descendants = _list_descendant_pids(proc.pid)
        _signal_many(descendants, signal.SIGTERM)

        try:
            proc.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            try:
                if pgid is not None and pgid != os.getpgrp():
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError):
                proc.kill()
            _signal_many(_alive_pids(descendants), signal.SIGKILL)
            proc.wait(timeout=10)

        if _is_carla_running_by_name():
            logger.warning(
                "CARLA process still running after PG/tree termination; falling back to name kill: %s",
                carla_name_pattern,
            )
            _kill_carla_by_name()
    else:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _kill_carla_processes_by_name(config: Optional["Config"] = None) -> None:
    """Kill lingering CARLA server processes by configured name pattern."""
    carla_name_pattern = _carla_kill_process_pattern(config)
    for sig in ("TERM", "KILL"):
        try:
            subprocess.run(
                ["pkill", f"-{sig}", "-f", carla_name_pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            continue


def _write_timeout_stub_log(scenario_id: str, timeout_sec: int) -> None:
    """So get_error_message_from_logs() returns a clear message after wall-clock timeout."""
    log_dir = f"temp/{scenario_id}/logs"
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{scenario_id}__simulation_timeout.log")
    body = (
        f"CARLA/Scenic subprocess exceeded {timeout_sec}s and was force-terminated.\n"
        "(run_scenic_batch.sh did not finish within the workflow timeout.)\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("Traceback (most recent call last):\n")
        f.write(f"TimeoutError: {body}\n")
        f.write("\n=== CARLA SETTINGS (post-scenic) ===\n")




def run_simulation_in_carla_and_save_video(
    scenic_code: str,
    scenario_id: str = "test_scenario",
    timeout_sec: Optional[int] = 120,
    config: Optional["Config"] = None,
) -> Optional[str]:
    """
    Run the simulation in Carla and save the video.
    """
    config = config or get_config()
    if timeout_sec is None:
        timeout_sec = config.simulation.timeout_sec

    # 1. save scenic code to a file
    if os.path.exists(f"temp/{scenario_id}"):
        shutil.rmtree(f"temp/{scenario_id}")
    os.makedirs(f"temp/{scenario_id}", exist_ok=True)
    os.makedirs(f"temp/{scenario_id}/code", exist_ok=True)
    os.makedirs(f"temp/{scenario_id}/video", exist_ok=True)
    os.makedirs(f"temp/{scenario_id}/logs", exist_ok=True)
    with open(f"temp/{scenario_id}/code/scenic_code.scenic", "w") as f:
        f.write(scenic_code)

    # 2. run simulation and save the video (bounded wall time)
    cmd = build_run_scenic_batch_command(
        config,
        f"temp/{scenario_id}/code",
        outdir=f"temp/{scenario_id}/video",
        logdir=f"temp/{scenario_id}/logs",
    )
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True, # Creates a new process group
    )
    
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        logger.error(
            "Simulation timed out after %ss for scenario %s; killing process group and CARLA",
            timeout_sec,
            scenario_id,
        )
        
        # === FIX STAGE 1: Kill the entire bash process group wrapper ===
        try:
            # os.getpgid(proc.pid) finds the group ID because of start_new_session=True
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass # Already dead
            
        # === FIX STAGE 2: Clean up any lingering CARLA instances ===
        _kill_carla_processes_by_name(config)
        
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:
            stdout, stderr = (stdout or ""), (stderr or "")
            
        _write_timeout_stub_log(scenario_id, timeout_sec)
        if stderr:
            logger.error("run_scenic_batch stderr (tail): %s", stderr[-4000:])
        return None

    if proc.returncode != 0:
        logger.error(f"Failed to run simulation: {stderr}")
        return None
        
    # check if video really exists
    if not os.path.exists(os.path.join(f"temp/{scenario_id}/video", "BEV.mp4")):
        logger.error(
            f"Video not found at {os.path.join(f'temp/{scenario_id}/video', 'BEV.mp4')}"
        )
        return None
        
    video_path = os.path.join(f"temp/{scenario_id}/video", "BEV.mp4")
    return video_path

def get_error_message_from_logs(scenario_id: str) -> Optional[str]:
    """
    Return the text block between:
      "Traceback"
    and
      "=== CARLA SETTINGS (post-scenic) ==="
    from the first matching log file under temp/{scenario_id}/logs.

    Includes the line containing "Traceback" and excludes the line containing
    "=== CARLA SETTINGS (post-scenic) ===".
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
            if start_idx is None and "Traceback" in line:
                start_idx = i
                continue
            if start_idx is not None and "=== CARLA SETTINGS (post-scenic) ===" in line:
                end_idx = i
                break
        if start_idx is None:
            continue
        # If end marker missing, return everything from Traceback to file end.
        block = lines[start_idx:end_idx] if end_idx is not None else lines[start_idx:]
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


def _unwrap_doubled_json_braces(text: str) -> str:
    """Normalize model outputs that wrap JSON in ``{{ ... }}`` instead of ``{ ... }``."""
    stripped = text.strip()
    if stripped.startswith("{{") and stripped.endswith("}}"):
        return "{" + stripped[2:-2].strip() + "}"
    return stripped


def parse_raw_text_to_json_dict(raw_text: str | Dict[str, Any] | List[Any]) -> Dict[str, Any] | None:
    """
    Parse raw text/object into a JSON dictionary.
    """
    if isinstance(raw_text, dict):
        return raw_text

    if isinstance(raw_text, list):
        logger.error("Expected JSON object (dict), but got JSON array (list).")
        return None

    cleaned_text = _unwrap_doubled_json_braces(strip_code_fence_markers(raw_text))

    try:
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, dict):
            return parsed
        logger.error(f"Expected JSON object (dict), but got {type(parsed).__name__}.")
        return None
    except json.JSONDecodeError:
        match = re.search(r"[\{\[][\s\S]*[\}\]]", cleaned_text)
        if match:
            candidate = _unwrap_doubled_json_braces(match.group(0))
            try:
                parsed = json.loads(candidate)
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


def flatten_dsl_to_text(dsl: Dict[str, Any] | None) -> str | None:
    """
    Flatten structured DSL/scenario JSON into embedding-friendly plain text.
    Supports both lowercase and title-case field names from VLM outputs.
    """
    if not dsl:
        return None

    def _format_actor(value: Any) -> str:
        if isinstance(value, dict):
            actor_object = value.get("object", "")
            actor_behavior = value.get("behavior", "")
            return f"{actor_object}: {actor_behavior}".strip(": ").strip()
        return str(value) if value is not None else ""

    try:
        text_parts = []
        text_parts.append(f"Scenario: {dsl.get('scenario', dsl.get('Scenario', ''))}")

        ego = dsl.get("ego", dsl.get("Ego", ""))
        ego_text = _format_actor(ego)
        text_parts.append(f"The ego vehicle is {ego_text}")

        adversarials = dsl.get("adversarials", dsl.get("Adversarials", []))
        if adversarials:
            adv_texts = []
            for adv in adversarials:
                adv_text = _format_actor(adv)
                if adv_text:
                    adv_texts.append(adv_text)
            if adv_texts:
                text_parts.append(f"Adversarial objects: {' | '.join(adv_texts)}")
            else:
                text_parts.append("There are no adversarials.")
        else:
            text_parts.append("There are no adversarials.")

        text_parts.append(
            f"Spatial Relation: {dsl.get('spatial_relation', dsl.get('Spatial Relation', ''))}"
        )

        reqs = dsl.get(
            "requirements_and_restrictions",
            dsl.get("Requirement and restrictions", "")
        )
        if reqs:
            text_parts.append(f"Requirements and restrictions: {reqs}")

        return " ".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to flatten DSL to text: {e}")
        return None

if __name__ == "__main__":
    with open("temp/code.scenic", "r", encoding="utf-8") as f:
        scenic_code = f.read()

    run_simulation_in_carla_and_save_video(scenic_code=scenic_code, scenario_id="test_scenario")