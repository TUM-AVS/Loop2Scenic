from .base import BaseVLMModel
from .providers import Qwen3VLModel, OpenAIVLModel, GeminiVLModel

__all__ = ["BaseVLMModel", "Qwen3VLModel", "OpenAIVLModel", "GeminiVLModel", "get_vlm_service"]

def get_vlm_service(provider: str, **kwargs) -> BaseVLMModel:
    """
    Get a VLM service based on the provider.
    """
    if provider == "qwen3vl":
        return Qwen3VLModel(**kwargs)
    elif provider == "openai_vision":
        return OpenAIVLModel(**kwargs)
    elif provider == "gemini":
        return GeminiVLModel(**kwargs)