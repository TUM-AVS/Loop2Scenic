"""
Logging configuration.
"""

import logging
import sys
import pprint
from pathlib import Path
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo


def setup_logging(
    level: str = "INFO",
    run_name: str = "ads_mrag",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        run_name: Prefix for the auto-generated log file (if log_file is None).
        log_file: Optional log file path. If None, auto-generates timestamped file in logs/.
        log_format: Optional custom log format.
    """
    # 1. Setup Timezone and auto-generate log_file if not provided
    if log_file is None:
        tz = ZoneInfo("Europe/Berlin")
        timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/{run_name}_{timestamp}.log"

    # Default format (Added [%(name)s] so you know which file sent the log)
    if log_format is None:
        log_format = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
    
    # Convert level string to logging level for the console
    console_numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger to capture EVERYTHING (DEBUG)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to prevent duplicates
    root_logger.handlers = []
    
    # Console handler (Terminal output: keeps it clean based on 'level')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_numeric_level)
    console_formatter = logging.Formatter(log_format, datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (Log file: ALWAYS set to DEBUG to catch state dumps)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) 
    file_formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging configured (Console: {level} | File: {log_path})")


def log_workflow_state(logger_instance: logging.Logger, node_name: str, state: dict) -> None:
    """
    Utility specifically for LangGraph nodes to cleanly dump the full state to the log file.
    
    Args:
        logger_instance: The logger of the current module.
        node_name: The name of the LangGraph node currently executing.
        state: The LangGraph state dictionary.
    """
    # Prints to terminal (INFO)
    logger_instance.info(f"🟢 INVOKING NODE: {node_name}")

    state = state.copy()
    state.pop("query_embedding", None)
    
    # Saves massive state dump to the file ONLY (DEBUG)
    clean_state = pprint.pformat(state, indent=2, width=120)
    logger_instance.debug(f"--- STATE ENTRY [{node_name.upper()}] ---\n{clean_state}\n{'-'*60}")