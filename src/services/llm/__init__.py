from .base import BaseLLMModel
from .providers import OpenAIModel, GeminiModel

__all__ = ["BaseLLMModel", "OpenAIModel", "GeminiModel", "get_llm_service"]

def get_llm_service(provider: str, **kwargs) -> BaseLLMModel:
    """
    Get a LLM service based on the provider.
    """
    if provider == "openai":
        return OpenAIModel(**kwargs)
    elif provider == "gemini":
        return GeminiModel(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: openai, gemini"
        )