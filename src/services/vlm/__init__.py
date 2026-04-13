from .base import BaseVLMModel
from .providers import OpenAIVLModel, GeminiVLModel, Qwen3Plus

__all__ = ["BaseVLMModel", "OpenAIVLModel", "GeminiVLModel", "Qwen3Plus", "get_vlm_service"]

def get_vlm_service(provider: str, **kwargs) -> BaseVLMModel:
    """
    Get a VLM service based on the provider.
    """
    if provider == "qwen":
        return Qwen3Plus(**kwargs)
    elif provider == "openai":
        return OpenAIVLModel(**kwargs)
    elif provider == "gemini":
        return GeminiVLModel(**kwargs)