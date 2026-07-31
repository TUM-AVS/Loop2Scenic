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
        prompt_name: Path of the prompt file relative to ``src/prompt/``,
            without the ``.txt`` extension. Supports subfolders, e.g.
            ``"vlm_eval/evaluate_with_vlm"`` or ``"adapt_code"``.

    Returns:
        Prompt template content as string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    # Normalize accidental leading slashes / .txt suffix
    name = prompt_name.strip().lstrip("/")
    if name.endswith(".txt"):
        name = name[: -len(".txt")]

    prompt_file = (_PROMPTS_DIR / f"{name}.txt").resolve()
    # Prevent path escape outside the prompts directory
    try:
        prompt_file.relative_to(_PROMPTS_DIR.resolve())
    except ValueError as exc:
        raise FileNotFoundError(f"Prompt path escapes prompts dir: {prompt_name}") from exc

    if not prompt_file.is_file():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}. "
            f"Available prompts: {', '.join(get_available_prompts())}"
        )

    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_available_prompts() -> List[str]:
    """
    Get list of available prompt template names (including subfolders).

    Returns:
        List of prompt names relative to ``src/prompt/`` (without ``.txt``)
    """
    prompts = []
    for path in sorted(_PROMPTS_DIR.rglob("*.txt")):
        if path.is_file():
            prompts.append(path.relative_to(_PROMPTS_DIR).with_suffix("").as_posix())
    return prompts


__all__ = ["load_prompt", "get_available_prompts"]
