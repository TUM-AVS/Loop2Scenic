from .base import BaseLLMModel
from .providers import (
    AnthropicModel,
    DeepSeekModel,
    GeminiModel,
    OllamaModel,
    OpenAIModel,
    QwenAPIModel,
)

__all__ = [
    "BaseLLMModel",
    "OpenAIModel",
    "GeminiModel",
    "QwenAPIModel",
    "DeepSeekModel",
    "AnthropicModel",
    "OllamaModel",
    "get_llm_service",
]

def get_llm_service(provider: str, **kwargs) -> BaseLLMModel:
    """
    Get a LLM service based on the provider.
    """
    normalized_provider = (provider or "").strip().lower()

    if normalized_provider == "openai":
        return OpenAIModel(**kwargs)
    elif normalized_provider == "gemini":
        return GeminiModel(**kwargs)
    elif normalized_provider in {"qwen", "qwen_api", "qwen-api"}:
        return QwenAPIModel(**kwargs)
    elif normalized_provider == "deepseek":
        return DeepSeekModel(**kwargs)
    elif normalized_provider in {"anthropic", "claude", "sonnet"}:
        return AnthropicModel(**kwargs)
    elif normalized_provider == "ollama":
        return OllamaModel(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: openai, gemini, qwen, deepseek, anthropic, ollama"
        )