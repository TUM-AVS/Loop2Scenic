"""
Helper utility functions.
"""

import logging
from pathlib import Path
from typing import Any

import tiktoken
import json

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
