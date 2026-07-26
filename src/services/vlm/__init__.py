from .base import BaseVLMModel
from .providers import OpenAIVLModel, GeminiVLModel, OllamaVLModel, Qwen3Plus

__all__ = [
    "BaseVLMModel",
    "OpenAIVLModel",
    "GeminiVLModel",
    "Qwen3Plus",
    "OllamaVLModel",
    "get_vlm_service",
]

def get_vlm_service(provider: str, **kwargs) -> BaseVLMModel:
    """
    Get a VLM service based on the provider.
    """
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == "qwen":
        return Qwen3Plus(**kwargs)
    elif normalized_provider == "openai":
        return OpenAIVLModel(**kwargs)
    elif normalized_provider == "gemini":
        return GeminiVLModel(**kwargs)
    elif normalized_provider == "ollama":
        return OllamaVLModel(**kwargs)
    else:
        raise ValueError(
            f"Unsupported VLM provider: {provider}. "
            f"Supported: qwen, openai, gemini, ollama"
        )