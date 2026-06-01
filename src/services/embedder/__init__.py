from .base import BaseEmbeddingModel
from .providers import Qwen3VLEmbedding, HuggingFaceEmbedding, GeminiEmbedding

__all__ = ["BaseEmbeddingModel", "Qwen3VLEmbedding", "HuggingFaceEmbedding", "GeminiEmbedding", "get_embedder"]

def get_embedder(provider: str, **kwargs) -> BaseEmbeddingModel:
    """
    Get an embedder based on the provider.
    """
    if provider == "qwen":
        return Qwen3VLEmbedding(**kwargs)
    elif provider == "huggingface":
        return HuggingFaceEmbedding(**kwargs)
    elif provider in {"gemini", "google"}:
        return GeminiEmbedding(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: qwen, huggingface, gemini"
        )