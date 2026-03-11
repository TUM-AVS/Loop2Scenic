"""
Prompt template loader module.
"""

from pathlib import Path
from typing import List

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template by name.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        Prompt template content as string
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompt_file = _PROMPTS_DIR / f"{prompt_name}.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}. "
            f"Available prompts: {', '.join(get_available_prompts())}"
        )
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_available_prompts() -> List[str]:
    """
    Get list of available prompt template names.
    
    Returns:
        List of prompt names (without .txt extension)
    """
    return [
        f.stem 
        for f in _PROMPTS_DIR.glob("*.txt")
        if f.is_file()
    ]


__all__ = ["load_prompt", "get_available_prompts"]
