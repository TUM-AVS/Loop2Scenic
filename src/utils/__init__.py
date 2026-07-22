"""
Utility functions and helpers.
"""

from .logger import setup_logging, log_workflow_state
from .helpers import (
    ensure_directory,
    count_tokens,
    to_safe_string,
    find_scenic_code_with_scenario_id,
    find_corpus_scenario_bev,
    stage_corpus_bev_for_scenario,
    run_simulation_in_carla_and_save_video,
    get_error_message_from_logs,
    clean_and_parse_json,
    strip_code_fence_markers,
    flatten_dsl_to_text,
)
from .simulation import build_run_scenic_batch_command, repo_root, resolve_path

__all__ = [
    "setup_logging",
    "log_workflow_state",
    "ensure_directory",
    "count_tokens",
    "to_safe_string",
    "find_scenic_code_with_scenario_id",
    "find_corpus_scenario_bev",
    "stage_corpus_bev_for_scenario",
    "run_simulation_in_carla_and_save_video",
    "get_error_message_from_logs",
    "clean_and_parse_json",
    "strip_code_fence_markers",
    "flatten_dsl_to_text",
    "build_run_scenic_batch_command",
    "repo_root",
    "resolve_path",
]
